"""Reverse-over-reverse HVP at depth (task #49).

Nested Derivative chains (grad of <grad(loss), v>) resolve innermost-first
in reverse.py: the inner gradient family is swept into plain algebra, then
the outer family sweeps THAT graph. Before this, nested chains fell back to
grad.py's forward-shaped chain stepping, whose compile time grew ~10x per
stacked softmax block (measured 0.1/0.9/12.2/132s at depths 1-4); a 3-block
GPT HVP never finished. Now depth 12 compiles in ~0.1s.

Flat programs are deliberately untouched: singleton families sweep ONLY
when the program contains a nested chain.
"""

import time

import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler.reverse import _contains_derivative, resolve_shared_gradients


def _softmax_stack(depth):
    b, d = symbols("b d")
    x = Variable("x", b=b, d=d)
    ws = [Variable(f"w{i}", d=d, d2=d) for i in range(depth)]
    h = x
    for w in ws:
        h = F.softmax((h @ w).rename(d2="d"), dim="d")
    loss = F.sum(h * h)
    return loss, x, ws, (b, d)


def _hvp_exprs(loss, wt):
    v = Variable("hvp_dir", **wt.shape)
    g = loss.grad(wt)
    hvp = F.sum(g * v).grad(wt)
    return v, hvp


def test_nested_chain_resolves_completely():
    """The sharpest pin on the mechanics: after resolution no Derivative
    survives anywhere in the outputs (inner AND outer were swept)."""
    loss, x, ws, _ = _softmax_stack(2)
    v, hvp = _hvp_exprs(loss, ws[0])
    resolved = resolve_shared_gradients((loss, hvp))
    assert not any(_contains_derivative(t) for t in resolved)


def test_hvp_stacked_softmax_deep_compiles_fast_and_correct():
    """Depth 4 took 132s on the fallback path; budget 30s makes a regression
    loud while leaving 100x headroom over the measured ~0.1s."""
    depth = 4
    loss, x, ws, (b, d) = _softmax_stack(depth)
    wt = ws[0]
    v, hvp = _hvp_exprs(loss, wt)
    t0 = time.perf_counter()
    prog = compile_to_callable(loss, hvp)
    assert time.perf_counter() - t0 < 30, "nested chain fell back to chain-rule stepping"

    B, D = 3, 4
    torch.manual_seed(0)
    vals = {
        x: torch.randn(B, D, dtype=torch.float64).rename("b", "d"),
        v: torch.randn(D, D, dtype=torch.float64).rename("d", "d2"),
    }
    for w in ws:
        vals[w] = torch.randn(D, D, dtype=torch.float64).rename("d", "d2")
    _, hvp_val = prog(vals, {b: B, d: D})

    w0 = vals[wt].rename(None).clone().requires_grad_(True)
    with torch.enable_grad():
        hr = vals[x].rename(None)
        for i, w in enumerate(ws):
            hr = torch.softmax(hr @ (w0 if i == 0 else vals[w].rename(None)), dim=1)
        lr = (hr * hr).sum()
        (gr,) = torch.autograd.grad(lr, w0, create_graph=True)
        (hvr,) = torch.autograd.grad((gr * vals[v].rename(None)).sum(), w0)
    assert torch.allclose(hvp_val.align_to("d", "d2").rename(None), hvr, rtol=1e-6, atol=1e-9)


def test_hvp_residual_mlp_wrt_weight():
    """Deep residual MLP, HVP w.r.t. the FIRST block's weight (long
    cotangent chains through the residual stream)."""
    depth = 6
    b, d, f = symbols("b d f")
    x = Variable("x", b=b, d=d)
    blocks = [
        (Variable(f"w1_{i}", d=d, f=f), Variable(f"w2_{i}", f=f, d2=d))
        for i in range(depth)
    ]
    h = x
    for w1, w2 in blocks:
        h = h + (F.tanh(h @ w1) @ w2).rename(d2="d")
    loss = F.sum(h * h)
    wt = blocks[0][0]
    v, hvp = _hvp_exprs(loss, wt)
    prog = compile_to_callable(loss, hvp)

    B, D, Fd = 4, 6, 5
    torch.manual_seed(0)
    vals = {
        x: torch.randn(B, D, dtype=torch.float64).rename("b", "d"),
        v: torch.randn(D, Fd, dtype=torch.float64).rename("d", "f"),
    }
    for w1, w2 in blocks:
        vals[w1] = torch.randn(D, Fd, dtype=torch.float64).rename("d", "f") / D**0.5
        vals[w2] = torch.randn(Fd, D, dtype=torch.float64).rename("f", "d2") / Fd**0.5
    _, hvp_val = prog(vals, {b: B, d: D, f: Fd})

    w0 = vals[wt].rename(None).clone().requires_grad_(True)
    with torch.enable_grad():
        hr = vals[x].rename(None)
        for i, (w1, w2) in enumerate(blocks):
            wa = w0 if i == 0 else vals[w1].rename(None)
            hr = hr + torch.tanh(hr @ wa) @ vals[w2].rename(None)
        lr = (hr * hr).sum()
        (gr,) = torch.autograd.grad(lr, w0, create_graph=True)
        (hvr,) = torch.autograd.grad((gr * vals[v].rename(None)).sum(), w0)
    assert torch.allclose(hvp_val.align_to("d", "f").rename(None), hvr, rtol=1e-6, atol=1e-9)


def test_flat_singleton_family_still_correct():
    """A flat single-gradient program (no nesting) keeps the classic path;
    correctness must be unaffected by the nested-mode machinery."""
    b, d = symbols("b d")
    x = Variable("x", b=b, d=d)
    w = Variable("w", d=d, d2=d)
    loss = F.sum(F.softmax((x @ w).rename(d2="d"), dim="d") ** 2)
    g = loss.grad(w)
    resolved = resolve_shared_gradients((loss, g))
    # singleton + flat: the Derivative survives resolution (grad.py's job)
    assert any(_contains_derivative(t) for t in resolved)
    prog = compile_to_callable(loss, g)
    B, D = 3, 4
    torch.manual_seed(0)
    vals = {
        x: torch.randn(B, D, dtype=torch.float64).rename("b", "d"),
        w: torch.randn(D, D, dtype=torch.float64).rename("d", "d2"),
    }
    _, g_val = prog(vals, {b: B, d: D})
    w0 = vals[w].rename(None).clone().requires_grad_(True)
    with torch.enable_grad():
        lr = (torch.softmax(vals[x].rename(None) @ w0, dim=1) ** 2).sum()
        (gr,) = torch.autograd.grad(lr, w0)
    assert torch.allclose(g_val.align_to("d", "d2").rename(None), gr, rtol=1e-6, atol=1e-9)
