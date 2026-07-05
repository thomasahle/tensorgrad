"""IR consolidation: Schwartz-Zippel value numbering modulo axis permutation.

Per-gradient adjoint walks re-derive the same downward cotangent chains, but
each gradient is derived/simplified/factored separately, so the shared chains
appear as strands computing EQUAL tensors through DIFFERENT groupings — an
algebraic equality no structural canonicalization sees. consolidate_outputs
merges them by exact mod-P value, guarded by a fresh-seed refusal gate (any
output mismatch after rewiring falls back to the input program)."""

import torch
from sympy import symbols

torch.set_num_threads(2)

from tensorgrad import Variable
import tensorgrad.functions as F
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler.consolidate import consolidate_outputs
from tensorgrad.compiler.factor import factor_outputs
from tensorgrad.compiler.stabilize import stabilize_outputs
from tensorgrad.compiler.ir import ConstNode, InputNode, toposort


def _build_resmlp(n_blocks=3):
    """Residual MLP stack with per-block layernorm-ish scaling: enough
    structure that every parameter's gradient shares the downward cotangent
    chain with every other."""
    b, d, h = symbols("b d h")
    x = Variable("x", b=b, d=d)
    t = Variable("t", b=b, d=d)
    params = {}
    hcur = x
    for i in range(n_blocks):
        w1 = params[f"w1_{i}"] = Variable(f"w1_{i}", d=d, h=h)
        w2 = params[f"w2_{i}"] = Variable(f"w2_{i}", h=h, d2=d)
        g = params[f"g_{i}"] = Variable(f"g_{i}", d=d)
        z = hcur * g
        hcur = hcur + (F.gelu(z @ w1, approximate="tanh") @ w2).rename(d2="d")
    loss = F.sum(hcur * t)
    dims = {b: 4, d: 8, h: 12}
    inputs = {"x": x, "t": t}
    return loss, params, inputs, dims


def _ncompute(outs):
    return sum(
        1
        for n in toposort([n for n, _ in outs])
        if not isinstance(n, (InputNode, ConstNode))
    )


def _pipeline(prog, dims):
    outs = factor_outputs(prog.builder, prog.outputs, dims)
    outs = stabilize_outputs(prog.builder, outs)
    return outs


def test_consolidation_shrinks_multi_gradient_program():
    loss, params, inputs, dims = _build_resmlp(3)
    grads = [loss.grad(p) for p in params.values()]
    prog = compile_to_callable(loss, *grads)
    fouts = _pipeline(prog, dims)
    before = _ncompute(fouts)
    couts = consolidate_outputs(prog.builder, fouts)
    after = _ncompute(couts)
    # The per-gradient cotangent strands must merge substantially.
    assert after < before * 0.85, f"no consolidation: {before} -> {after}"


def test_consolidated_gradients_match_autograd():
    loss, params, inputs, dims = _build_resmlp(3)
    names = list(params)
    grads = [loss.grad(params[n]) for n in names]
    prog = compile_to_callable(loss, *grads)  # consolidation active in specialize

    torch.manual_seed(0)
    d_b, d_d, d_h = 4, 8, 12
    vals = {}
    tvals = {}
    for n, var in params.items():
        shape = [ {"b": d_b, "d": d_d, "h": d_h, "d2": d_d}[e] for e in var.edges ]
        w = 0.2 * torch.randn(*shape)
        tvals[n] = w.clone().requires_grad_(True)
        vals[var] = w.refine_names(*var.edges)
    xv = torch.randn(d_b, d_d)
    tv = torch.randn(d_b, d_d)
    vals[inputs["x"]] = xv.refine_names("b", "d")
    vals[inputs["t"]] = tv.refine_names("b", "d")

    sym_dims = dict(dims)
    outs = prog(dict(vals), sym_dims)
    loss_tg, grads_tg = outs[0], outs[1:]

    # Torch reference (same math).
    with torch.enable_grad():
        h = xv
        for i in range(3):
            z = h * tvals[f"g_{i}"]
            pre = z @ tvals[f"w1_{i}"]
            act = torch.nn.functional.gelu(pre, approximate="tanh")
            h = h + act @ tvals[f"w2_{i}"]
        loss_t = (h * tv).sum()
        loss_t.backward()

    torch.testing.assert_close(loss_tg.rename(None), loss_t.detach(), rtol=1e-4, atol=1e-6)
    for n, g in zip(names, grads_tg):
        want = tvals[n].grad
        got = g.align_to(*params[n].edges).rename(None) if any(g.names) else g.rename(None)
        torch.testing.assert_close(
            got.reshape(want.shape), want, rtol=1e-3, atol=1e-6,
            msg=f"gradient mismatch for {n}",
        )
