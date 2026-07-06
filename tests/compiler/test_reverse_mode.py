"""Automatic reverse-mode contraction ordering (#29).

Symbolic gradients of parameters deep under a stack of nonlinearities lower
into FORWARD-MODE-shaped Jacobian chains: einsums separated by LinearNodes
(residual adds, LayerNorm centering, softmax-Jacobian sums) whose nodes drag
the wrt-parameter's axes through every layer above. At GPT-2 block dims the
raw lowered plan totals ~400 TB with a 59 PB rank-5 node; the compact
reverse-mode (cotangent-first) order is a legal einsum reassociation through
delta wires that the local cost model cannot reach (each single boundary
collapse improves the local score by <2%, below DIST_MARGIN).

The adjoint overrides in factor.py (INFLATE_MARGIN and friends) collapse
these chains unconditionally. This file pins:

  PLAN     the factored IR of deep-parameter gradients has no node larger
           than ~100 MB at large dims (2-layer MLP; full GPT-2-dim block)
  CORRECT  the collapsed programs match torch.autograd numerically, with
           the overrides forced on at small dims
  VERIFIED the factored DAG equals the lowered DAG at random points in
           exact mod-P arithmetic (Schwartz-Zippel fingerprints)
  LIVENESS codegen frees each intermediate at its last use (`del`), so a
           program's peak memory is its live set, not its plan total
"""

import math

import pytest
import sympy
import torch
from sympy import symbols

import tensorgrad.compiler.adjoint as adjoint_mod
import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler.factor import factor_outputs
from tensorgrad.compiler.ir import ConstNode, InputNode, toposort
from tensorgrad.compiler.stabilize import stabilize_outputs
from tensorgrad.compiler import szfp

torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def plan_numels(prog, dims):
    """Numel of every compute node in the FACTORED IR (what codegen emits),
    replicating codegen.specialize's rewrite pipeline. Nothing is executed."""
    outputs = factor_outputs(prog.builder, prog.outputs, dims)
    outputs = stabilize_outputs(prog.builder, outputs)
    outputs = factor_outputs(prog.builder, outputs, dims)
    outputs = stabilize_outputs(prog.builder, outputs)
    sizes = []
    for n in toposort([nd for nd, _ in outputs]):
        if isinstance(n, (InputNode, ConstNode)):
            continue
        numel = 1
        for d in n.dims:
            numel *= int(sympy.sympify(d).subs(dims))
        sizes.append(numel)
    return sizes


def forced_adjoint(margin=2.0):
    """Context that lowers the inflation threshold so the adjoint collapse
    (pre-pass and sweep backstop both read adjoint_mod.INFLATE_MARGIN) fires
    at test-sized dims (it is calibrated for large ones)."""

    class _Ctx:
        def __enter__(self):
            self.old = adjoint_mod.INFLATE_MARGIN
            adjoint_mod.INFLATE_MARGIN = margin

        def __exit__(self, *exc):
            adjoint_mod.INFLATE_MARGIN = self.old

    return _Ctx()


def run_compiled(tensors, values, dims):
    prog = compile_to_callable(*tensors)
    outs = prog(dict(values), dict(dims))
    return (outs,) if len(tensors) == 1 else outs, prog


# ---------------------------------------------------------------------------
# 2-layer MLP chain: the smallest reproducer
# ---------------------------------------------------------------------------


def build_mlp2():
    """loss = sum(gy * (gelu(gelu(x@w1) @ w2) @ w3)); grad wrt w1 (deepest)."""
    batch, d0, d1, d2, d3 = symbols("batch d0 d1 d2 d3")
    x = Variable("x", batch, d0)
    gy = Variable("gy", batch, d3)
    w1 = Variable("w1", d0=d0, d1=d1)
    w2 = Variable("w2", d1=d1, d2=d2)
    w3 = Variable("w3", d2=d2, d3=d3)
    h1 = F.gelu(x @ w1, approximate="tanh")
    h2 = F.gelu(h1 @ w2, approximate="tanh")
    loss = F.sum((h2 @ w3) * gy)
    return loss, (x, gy, w1, w2, w3), (batch, d0, d1, d2, d3)


def test_mlp2_deep_grad_plans_reverse_mode():
    """Before the adjoint collapse, the w1 gradient planned a 1 TB
    (d0, d2, d1, batch) outer-product node at these dims; the reverse-mode
    order needs nothing larger than the weights themselves."""
    loss, (x, gy, w1, w2, w3), (batch, d0, d1, d2, d3) = build_mlp2()
    prog = compile_to_callable(loss, loss.grad(w1))
    dims = {batch: 256, d0: 1024, d1: 1024, d2: 1024, d3: 1024}
    sizes = plan_numels(prog, dims)
    assert max(sizes) <= 4e6, f"max planned node {max(sizes):.2e} elements"
    assert sum(sizes) <= 40e6, f"total planned {sum(sizes):.2e} elements"


def test_mlp2_deep_grad_matches_autograd():
    """With the overrides forced on at small dims, the collapsed program
    still matches torch.autograd exactly (the rewrites only reassociate)."""
    loss, (x, gy, w1, w2, w3), (batch, d0, d1, d2, d3) = build_mlp2()
    dims = {batch: 5, d0: 7, d1: 6, d2: 8, d3: 4}
    torch.manual_seed(0)
    vals = {
        x: torch.randn(5, 7).rename("batch", "d0"),
        gy: torch.randn(5, 4).rename("batch", "d3"),
        w1: torch.randn(7, 6).rename("d0", "d1"),
        w2: torch.randn(6, 8).rename("d1", "d2"),
        w3: torch.randn(8, 4).rename("d2", "d3"),
    }
    with forced_adjoint():
        (lv, gv), _ = run_compiled([loss, loss.grad(w1)], vals, dims)

    w1r = vals[w1].rename(None).clone().requires_grad_(True)
    h = torch.nn.functional.gelu(vals[x].rename(None) @ w1r, approximate="tanh")
    h = torch.nn.functional.gelu(h @ vals[w2].rename(None), approximate="tanh")
    ref_loss = ((h @ vals[w3].rename(None)) * vals[gy].rename(None)).sum()
    (ref_g,) = torch.autograd.grad(ref_loss, w1r)

    torch.testing.assert_close(lv.rename(None), ref_loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        gv.align_to("d0", "d1").rename(None), ref_g, rtol=1e-4, atol=1e-5
    )


def test_factored_equals_lowered_szfp():
    """Schwartz-Zippel: the factored DAG (adjoint overrides forced, factor
    pass only — stabilize folds atoms by design) agrees with the lowered DAG
    at random evaluation points in exact mod-P arithmetic.

    The chain is built from relu + mean-centering so every constant stays
    rational: the adjoint rewrites fold weights together at compile time,
    which is exact mod P for rationals but rounds for float constants like
    gelu's 0.7978… (a fingerprint mismatch that is only fp noise)."""
    batch, d0, d1, d2 = symbols("batch d0 d1 d2")
    x = Variable("x", batch, d0)
    gy = Variable("gy", batch, d2)
    w1 = Variable("w1", d0=d0, d1=d1)
    w2 = Variable("w2", d1=d1, d2=d2)
    h1 = F.relu(x @ w1)
    h1c = h1 - F.mean(h1, dim="d1", keepdims=True)  # rational (delta - J/d1) Jacobian
    h2 = F.relu(h1c @ w2)
    loss = F.sum(h2 * gy)
    prog = compile_to_callable(loss, loss.grad(w1))
    dims = {batch: 64, d0: 96, d1: 96, d2: 96}
    with forced_adjoint():
        fouts = factor_outputs(prog.builder, prog.outputs, dims)
    assert any(a is not b for (a, _), (b, _) in zip(fouts, prog.outputs))
    for trial in range(3):
        for salt in range(50):
            try:
                res = szfp._eval_trial(list(prog.outputs) + list(fouts), (0, trial, salt))
                break
            except szfp._Retry:
                continue
        else:
            pytest.fail("szfp trial exceeded retry budget")
        k = len(prog.outputs)
        for i in range(k):
            edges_a, dims_a, arr_a = res[i]
            edges_b, dims_b, arr_b = res[k + i]
            assert edges_a == edges_b and dims_a == dims_b
            assert (arr_a == arr_b).all(), f"output {i} differs at trial {trial}"


# ---------------------------------------------------------------------------
# The milestone: a full GPT-2-dim transformer block
# ---------------------------------------------------------------------------


def build_block():
    """One pre-LN transformer block; loss = sum(block(x) * gy) / N.
    Returns (loss, params, inputs, symbols)."""
    batch, seq, d, head, hs, d_mlp = symbols("batch seq d head hs d_mlp")
    params = {}

    def param(name, **edges):
        return params.setdefault(name, Variable(name, **edges))

    x_in = Variable("x", batch, seq, d)
    gy_in = Variable("gy", batch, seq, d)
    causal_mask = Variable("causal_mask", seq=seq, key=seq)

    def layer_norm(x, name):
        x = x - F.mean(x, dim="d", keepdims=True)
        var = F.mean(x * x, dim="d", keepdims=True)
        return x / F.sqrt(var + 1e-5) * param(name + ".g", d=d) + param(name + ".b", d=d)

    def attention(x, name):
        q = x @ param(name + ".wq", d=d, head=head, hs=hs) + param(name + ".bq", head=head, hs=hs)
        k = x @ param(name + ".wk", d=d, head=head, hs=hs) + param(name + ".bk", head=head, hs=hs)
        v = x @ param(name + ".wv", d=d, head=head, hs=hs) + param(name + ".bv", head=head, hs=hs)
        k, v = k.rename(seq="key"), v.rename(seq="key")
        att = F.softmax(F.dot(q, k, dim="hs") / 8 + causal_mask, dim="key")
        y = F.dot(att, v, dim="key")
        return y @ param(name + ".wo", head=head, hs=hs, d=d) + param(name + ".bo", d=d)

    def mlp(x, name):
        h = F.gelu(x @ param(name + ".w1", d=d, d_mlp=d_mlp) + param(name + ".b1", d_mlp=d_mlp),
                   approximate="tanh")
        return h @ param(name + ".w2", d_mlp=d_mlp, d=d) + param(name + ".b2", d=d)

    xb = x_in + attention(layer_norm(x_in, "ln1"), "attn")
    y = xb + mlp(layer_norm(xb, "ln2"), "mlp")
    loss = F.sum(y * gy_in)
    return loss, params, (x_in, gy_in, causal_mask), (batch, seq, d, head, hs, d_mlp)


def test_gpt2_block_deep_grads_plan_reverse_mode():
    """THE MILESTONE. Direct compile_to_callable(loss, *grads) on one
    GPT-2-dim block (d=768, heads=12, seq=256, batch=8), grads wrt a param
    under the softmax stack (attn.wq), under gelu (mlp.w1), and at the very
    bottom (ln1.g). Lowered plan: 117 nodes, ~400 TB total, 59 PB max node.
    Factored plan must be reverse-mode compact: <= 100 MB max node."""
    loss, params, _, (batch, seq, d, head, hs, d_mlp) = build_block()
    wrt = [params["attn.wq"], params["mlp.w1"], params["ln1.g"]]
    prog = compile_to_callable(loss, *[loss.grad(p) for p in wrt])
    dims = {batch: 8, seq: 256, d: 768, head: 12, hs: 64, d_mlp: 3072}
    sizes = plan_numels(prog, dims)
    max_mb = max(sizes) * 4 / 2**20
    total_gb = sum(sizes) * 4 / 2**30
    assert max_mb <= 100, f"max planned node {max_mb:.0f} MB"
    assert total_gb <= 3.0, f"total planned {total_gb:.2f} GB"


def test_block_deep_grads_match_autograd():
    """Same block at small dims with the adjoint overrides forced on;
    loss and all three gradients vs torch.autograd."""
    loss, params, (x_in, gy_in, causal_mask), (batch, seq, d, head, hs, d_mlp) = build_block()
    B, S, D, H, HS, DM = 3, 5, 8, 2, 4, 16
    dims = {batch: B, seq: S, d: D, head: H, hs: HS, d_mlp: DM}
    torch.manual_seed(1)
    W = {}
    for name, var in params.items():
        shape = [dims[s] for s in var.shape.values()]
        last = name.rsplit(".", 1)[-1]
        W[name] = (torch.ones(shape) if last == "g" else
                   torch.zeros(shape) if last.startswith("b") else
                   0.5 * torch.randn(shape))
    xv = torch.randn(B, S, D)
    gyv = torch.randn(B, S, D)
    maskv = torch.triu(torch.full((S, S), -1e9), diagonal=1)

    wrt = ["attn.wq", "mlp.w1", "ln1.g"]
    vals = {var: W[name].rename(*var.edges) for name, var in params.items()}
    vals[x_in] = xv.rename("batch", "seq", "d")
    vals[gy_in] = gyv.rename("batch", "seq", "d")
    vals[causal_mask] = maskv.rename("seq", "key")
    with forced_adjoint():
        outs, _ = run_compiled(
            [loss] + [loss.grad(params[n]) for n in wrt], vals, dims
        )

    # torch reference
    for n in wrt:
        W[n].requires_grad_(True)

    def ln(x, g, b):
        x = x - x.mean(-1, keepdim=True)
        return x / ((x * x).mean(-1, keepdim=True) + 1e-5).sqrt() * g + b

    h = ln(xv, W["ln1.g"], W["ln1.b"])
    q = torch.einsum("bsd,dhk->bhsk", h, W["attn.wq"]) + W["attn.bq"].unsqueeze(1)
    k = torch.einsum("bsd,dhk->bhsk", h, W["attn.wk"]) + W["attn.bk"].unsqueeze(1)
    v = torch.einsum("bsd,dhk->bhsk", h, W["attn.wv"]) + W["attn.bv"].unsqueeze(1)
    att = torch.softmax(q @ k.transpose(-2, -1) / 8 + maskv, dim=-1)
    a = torch.einsum("bhsk,hkd->bsd", att @ v, W["attn.wo"]) + W["attn.bo"]
    xb = xv + a
    h2 = ln(xb, W["ln2.g"], W["ln2.b"])
    m = torch.nn.functional.gelu(h2 @ W["mlp.w1"] + W["mlp.b1"], approximate="tanh")
    yt = xb + (m @ W["mlp.w2"] + W["mlp.b2"])
    ref_loss = (yt * gyv).sum()
    ref_gs = torch.autograd.grad(ref_loss, [W[n] for n in wrt])

    torch.testing.assert_close(outs[0].rename(None), ref_loss, rtol=1e-4, atol=1e-5)
    aligns = {"attn.wq": ("d", "head", "hs"), "mlp.w1": ("d", "d_mlp"), "ln1.g": ("d",)}
    for name, got, want in zip(wrt, outs[1:], ref_gs):
        torch.testing.assert_close(
            got.align_to(*aligns[name]).rename(None), want, rtol=1e-4, atol=1e-5,
            msg=lambda m, name=name: f"grad {name}: {m}",
        )


# ---------------------------------------------------------------------------
# Multi-block stacks: per-node adjoint accumulation (#31)
# ---------------------------------------------------------------------------
#
# Gradients that cross SEVERAL stacked blocks accumulate cotangent
# contributions across different Sums (residual diamonds) and different
# gradient outputs. Rule R/M alone forked branches geometrically here and
# aborted via the growth guard, falling back to forward mode (3-layer
# gpt-nano: 522 GB planned). Stage A (_Accumulate) walks each head's region
# once, merging contributions per (node, interface signature) — these tests
# pin that multi-block programs plan reverse-mode compact and stay exact.


def build_resmlp3():
    """Three stacked residual blocks h <- h + v(gelu(h @ w)); loss =
    sum(h3 * gy). The w0/w1 gradients cross the blocks above them and share
    the upper chain — the smallest program with cross-Sum accumulation."""
    batch, d, dh = symbols("batch d dh")
    x = Variable("x", batch, d)
    gy = Variable("gy", batch, d)
    ws, h = [], x
    for i in range(3):
        wi = Variable(f"w{i}", d=d, dh=dh)
        vi = Variable(f"v{i}", dh=dh, d2=d)
        ws.append((wi, vi))
        h = h + (F.gelu(h @ wi, approximate="tanh") @ vi).rename(d2="d")
    loss = F.sum(h * gy)
    return loss, x, gy, ws, (batch, d, dh)


def test_resmlp3_deep_grads_plan_reverse_mode():
    """Deep gradients through 3 residual blocks, TWO gradient outputs
    sharing the upper chain. Reverse mode needs nothing larger than an
    activation/weight; before per-node accumulation this program fell back
    to forward mode with (batch, d, d, dh)-sized chain nodes (~0.5 TB)."""
    loss, x, gy, ws, (batch, d, dh) = build_resmlp3()
    prog = compile_to_callable(loss, loss.grad(ws[0][0]), loss.grad(ws[1][0]))
    dims = {batch: 256, d: 512, dh: 2048}
    sizes = plan_numels(prog, dims)
    assert max(sizes) <= 4e6, f"max planned node {max(sizes):.2e} elements"
    assert sum(sizes) <= 80e6, f"total planned {sum(sizes):.2e} elements"


def test_resmlp3_deep_grads_match_autograd():
    """Same stack at small dims, adjoint overrides forced on: loss plus the
    two deep gradients match torch.autograd."""
    loss, x, gy, ws, (batch, d, dh) = build_resmlp3()
    dims = {batch: 4, d: 6, dh: 9}
    torch.manual_seed(3)
    vals = {
        x: torch.randn(4, 6).rename("batch", "d"),
        gy: torch.randn(4, 6).rename("batch", "d"),
    }
    for i, (wi, vi) in enumerate(ws):
        vals[wi] = (0.5 * torch.randn(6, 9)).rename("d", "dh")
        vals[vi] = (0.5 * torch.randn(9, 6)).rename("dh", "d2")
    with forced_adjoint():
        (lv, g0, g1), _ = run_compiled(
            [loss, loss.grad(ws[0][0]), loss.grad(ws[1][0])], vals, dims
        )

    tw = [vals[wi].rename(None).clone().requires_grad_(True) for wi, _ in ws]
    h = vals[x].rename(None)
    for i, (_, vi) in enumerate(ws):
        h = h + torch.nn.functional.gelu(h @ tw[i], approximate="tanh") @ vals[vi].rename(None)
    ref_loss = (h * vals[gy].rename(None)).sum()
    ref_g0, ref_g1 = torch.autograd.grad(ref_loss, [tw[0], tw[1]])

    torch.testing.assert_close(lv.rename(None), ref_loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(g0.align_to("d", "dh").rename(None), ref_g0, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(g1.align_to("d", "dh").rename(None), ref_g1, rtol=1e-4, atol=1e-5)


def test_resmlp3_szfp_factored_equals_lowered():
    """Schwartz-Zippel pinning of the cross-block accumulation rewrites:
    relu + mean-centering keeps every constant rational (see
    test_factored_equals_lowered_szfp), so the factored DAG must equal the
    lowered DAG exactly mod P."""
    batch, d = symbols("batch d")
    x = Variable("x", batch, d)
    gy = Variable("gy", batch, d)
    ws, h = [], x
    for i in range(3):
        wi = Variable(f"w{i}", d=d, d2=d)
        ws.append(wi)
        f = F.relu(h @ wi).rename(d2="d")
        h = h + (f - F.mean(f, dim="d", keepdims=True))
    loss = F.sum(h * gy)
    prog = compile_to_callable(loss, loss.grad(ws[0]), loss.grad(ws[1]))
    dims = {batch: 64, d: 96}
    with forced_adjoint():
        fouts = factor_outputs(prog.builder, prog.outputs, dims)
    assert any(a is not b for (a, _), (b, _) in zip(fouts, prog.outputs))
    for trial in range(3):
        for salt in range(50):
            try:
                res = szfp._eval_trial(list(prog.outputs) + list(fouts), (0, trial, salt))
                break
            except szfp._Retry:
                continue
        else:
            pytest.fail("szfp trial exceeded retry budget")
        k = len(prog.outputs)
        for i in range(k):
            edges_a, dims_a, arr_a = res[i]
            edges_b, dims_b, arr_b = res[k + i]
            assert edges_a == edges_b and dims_a == dims_b
            assert (arr_a == arr_b).all(), f"output {i} differs at trial {trial}"


def test_embedding_grad_through_stack_matches_autograd():
    """A gather-fed stack: the wte gradient's one-hot carrier must ride as a
    passenger while the pushed cotangents themselves stay foldable — the
    fragmentation mode measured on multi-layer gpt-nano (pass-created
    cotangents deferred as passengers never merge)."""
    batch, seq, vocab, d, dh = symbols("batch seq vocab d dh")
    tokens = Variable("tokens", batch, seq)
    gy = Variable("gy", batch, seq, d)
    wte = Variable("wte", vocab=vocab, d=d)
    h = F.one_hot(tokens, wte.shape["vocab"], dim="vocab") @ wte
    ws = []
    for i in range(2):
        wi = Variable(f"w{i}", d=d, dh=dh)
        vi = Variable(f"v{i}", dh=dh, d2=d)
        ws.append((wi, vi))
        h = h + (F.gelu(h @ wi, approximate="tanh") @ vi).rename(d2="d")
    loss = F.sum(h * gy)

    B, S, V, D, DH = 3, 5, 7, 4, 6
    dims = {batch: B, seq: S, vocab: V, d: D, dh: DH}
    torch.manual_seed(4)
    toks = torch.randint(V, (B, S)).float()
    vals = {
        tokens: toks.rename("batch", "seq"),
        gy: torch.randn(B, S, D).rename("batch", "seq", "d"),
        wte: torch.randn(V, D).rename("vocab", "d"),
    }
    for wi, vi in ws:
        vals[wi] = (0.5 * torch.randn(D, DH)).rename("d", "dh")
        vals[vi] = (0.5 * torch.randn(DH, D)).rename("dh", "d2")
    with forced_adjoint():
        (lv, gv), _ = run_compiled([loss, loss.grad(wte)], vals, dims)

    twte = vals[wte].rename(None).clone().requires_grad_(True)
    ht = twte[toks.long()]
    for wi, vi in ws:
        ht = ht + torch.nn.functional.gelu(
            ht @ vals[wi].rename(None), approximate="tanh"
        ) @ vals[vi].rename(None)
    ref_loss = (ht * vals[gy].rename(None)).sum()
    (ref_g,) = torch.autograd.grad(ref_loss, twte)

    torch.testing.assert_close(lv.rename(None), ref_loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        gv.align_to("vocab", "d").rename(None), ref_g, rtol=1e-4, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Liveness dels
# ---------------------------------------------------------------------------


def test_codegen_frees_dead_intermediates():
    """The generated source deletes each temp at its last use, and repeated
    calls of the del-carrying program agree (dels must not free live data)."""
    loss, (x, gy, w1, w2, w3), (batch, d0, d1, d2, d3) = build_mlp2()
    dims = {batch: 4, d0: 5, d1: 6, d2: 7, d3: 3}
    torch.manual_seed(2)
    vals = {
        x: torch.randn(4, 5).rename("batch", "d0"),
        gy: torch.randn(4, 3).rename("batch", "d3"),
        w1: torch.randn(5, 6).rename("d0", "d1"),
        w2: torch.randn(6, 7).rename("d1", "d2"),
        w3: torch.randn(7, 3).rename("d2", "d3"),
    }
    (l1, g1), prog = run_compiled([loss, loss.grad(w1)], vals, dims)
    (spec,) = prog._specializations.values()
    assert "del " in spec._source
    l2, g2 = prog(dict(vals), dict(dims))
    torch.testing.assert_close(l1.rename(None), l2.rename(None), rtol=0, atol=0)
    torch.testing.assert_close(g1.rename(None), g2.rename(None), rtol=0, atol=0)
