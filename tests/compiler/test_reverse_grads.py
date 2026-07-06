"""Joint reverse-mode gradient resolution (compiler/reverse.py).

Parity contract: for every supported family, the jointly-resolved gradients
must equal the independent chain-rule path numerically (evaluate() oracle)
and through the compiled program. Unsupported shapes must pass through
untouched (same objects)."""

import math

import pytest
import sympy
import torch

import tensorgrad.functions as F
from tensorgrad import Delta, Derivative, Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler.reverse import resolve_shared_gradients
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)
RTOL, ATOL = 1e-4, 1e-5


def _parity(loss, params, values, dims):
    """Old-vs-new numeric parity on every gradient in the family."""
    grads = [Derivative(loss, p) for p in params]
    resolved = resolve_shared_gradients(tuple([loss] + grads))
    assert resolved[0] is loss
    for d, r in zip(grads, resolved[1:]):
        assert r is not d, f"family not resolved for {d.x.name}"
        old = d.simplify()
        new = r.simplify()
        ref = evaluate(old, dict(values), dict(dims))
        got = evaluate(new, dict(values), dict(dims))
        got = got.align_to(*ref.names) if got.names != ref.names else got
        torch.testing.assert_close(
            got.rename(None), ref.rename(None), rtol=RTOL, atol=ATOL,
            msg=lambda m: f"grad({d.x.name}): {m}",
        )


def test_linear_family():
    i, j = sympy.symbols("i j")
    x = Variable("x", i=i, j=j)
    w = Variable("w", i=i, j=j)
    loss = F.sum(x * w)
    values = {x: torch.randn(3, 4).rename("i", "j"), w: torch.randn(3, 4).rename("i", "j")}
    _parity(loss, [x, w], values, {i: 3, j: 4})


def test_matmul_chain_family():
    i, j, k = sympy.symbols("i j k")
    x = Variable("x", i=i, j=j)
    w1 = Variable("w1", j=j, k=k)
    w2 = Variable("w2", k=k)
    loss = F.sum((x @ w1) * w2) / 6
    values = {
        x: torch.randn(2, 3).rename("i", "j"),
        w1: torch.randn(3, 4).rename("j", "k"),
        w2: torch.randn(4).rename("k"),
    }
    _parity(loss, [x, w1, w2], values, {i: 2, j: 3, k: 4})


def test_nonlinear_function_family():
    """gelu + softmax + sqrt: Function VJPs through signature derivatives."""
    b, d = sympy.symbols("b d")
    x = Variable("x", b=b, d=d)
    w = Variable("w", b=b, d=d)
    h = F.gelu(x, approximate="tanh")
    s = F.softmax(h, dim="d")
    loss = F.sum(s * w)
    values = {x: torch.randn(3, 5).rename("b", "d"), w: torch.randn(3, 5).rename("b", "d")}
    _parity(loss, [x, w], values, {b: 3, d: 5})


def test_layernorm_style_family():
    """mean/sqrt/pow composition with variable reuse (x appears many times)."""
    b, d = sympy.symbols("b d")
    x = Variable("x", b=b, d=d)
    g = Variable("g", d=d)
    xc = x - F.mean(x, dim="d", keepdims=True)
    var = F.mean(xc * xc, dim="d", keepdims=True)
    y = xc / F.sqrt(var + 1e-5) * g
    loss = F.sum(y * y)
    values = {
        x: torch.randn(4, 6).rename("b", "d"),
        g: (torch.randn(6) + 2).rename("d"),
    }
    _parity(loss, [x, g], values, {b: 4, d: 6})


def test_rename_and_attention_style_family():
    """Renamed reuse (seq->key) with softmax: the mingpt attention skeleton."""
    s, d = sympy.symbols("s d")
    x = Variable("x", seq=s, d=d)
    wq = Variable("wq", d=d, h=d)
    wk = Variable("wk", d=d, h=d)
    q = x @ wq
    k = (x @ wk).rename(seq="key")
    att = F.softmax(F.dot(q, k, dim="h") / math.sqrt(4), dim="key")
    v = x.rename(seq="key")
    y = F.dot(att, v, dim="key")
    loss = F.sum(y * y)
    values = {
        x: torch.randn(3, 4).rename("seq", "d"),
        wq: torch.randn(4, 4).rename("d", "h"),
        wk: torch.randn(4, 4).rename("d", "h"),
    }
    _parity(loss, [x, wq, wk], values, {s: 3, d: 4})


def test_ce_one_hot_family():
    """cross_entropy with one_hot targets: gather/indicator plumbing."""
    b, v = sympy.symbols("b v")
    logits = Variable("logits", b=b, v=v)
    ids = Variable("ids", b=b)
    m = Variable("m", b=b)
    y = F.one_hot(ids, v, dim="v")
    loss = F.sum(F.cross_entropy(logits, y, dim="v") * m)
    values = {
        logits: torch.randn(4, 5).rename("b", "v"),
        ids: torch.randint(5, (4,)).float().rename("b"),
        m: torch.rand(4).rename("b"),
    }
    _parity(loss, [logits, m], values, {b: 4, v: 5})


def test_custom_new_names():
    i = sympy.symbols("i")
    x = Variable("x", i=i)
    w = Variable("w", i=i)
    loss = F.sum(x * w)
    d1 = Derivative(loss, x, {"i": "di"})
    d2 = Derivative(loss, w)
    r = resolve_shared_gradients((loss, d1, d2))
    assert r[1] is not d1 and set(r[1].edges) == {"di"}
    values = {x: torch.randn(3).rename("i"), w: torch.randn(3).rename("i")}
    ref = evaluate(d1.simplify(), dict(values), {i: 3})
    got = evaluate(r[1].simplify(), dict(values), {i: 3})
    torch.testing.assert_close(got.rename(None), ref.rename(None), rtol=RTOL, atol=ATOL)


def test_singleton_family_passes_through():
    i = sympy.symbols("i")
    x = Variable("x", i=i)
    loss = F.sum(x * x)
    d = Derivative(loss, x)
    r = resolve_shared_gradients((loss, d))
    assert r[1] is d  # < 2 grads: independent path


def test_nonscalar_base_passes_through():
    i = sympy.symbols("i")
    x = Variable("x", i=i)
    w = Variable("w", i=i)
    vec = x * w  # order-1 base
    d1, d2 = Derivative(vec, x), Derivative(vec, w)
    r = resolve_shared_gradients((vec, d1, d2))
    assert r[1] is d1 and r[2] is d2


def test_compiled_program_parity():
    """End to end: compiled loss+grads equal torch.autograd."""
    b, d, k = sympy.symbols("b d k")
    x = Variable("x", b=b, d=d)
    w1 = Variable("w1", d=d, k=k)
    w2 = Variable("w2", k=k)
    h = F.gelu(x @ w1, approximate="tanh")
    s = F.softmax(h, dim="k")
    loss = F.sum(s @ w2) / 4
    params = [w1, w2]
    f = compile_to_callable(loss, *[loss.grad(p) for p in params])
    values = {
        x: torch.randn(4, 3).rename("b", "d"),
        w1: torch.randn(3, 5).rename("d", "k"),
        w2: torch.randn(5).rename("k"),
    }
    outs = f(dict(values), {b: 4, d: 3, k: 5})
    xt = values[x].rename(None)
    w1t = values[w1].rename(None).clone().requires_grad_(True)
    w2t = values[w2].rename(None).clone().requires_grad_(True)
    ref = (torch.softmax(
        0.5 * (xt @ w1t) * (1 + torch.tanh(math.sqrt(2 / math.pi) * ((xt @ w1t) + 0.044715 * (xt @ w1t) ** 3))),
        dim=1,
    ) @ w2t).sum() / 4
    ref.backward()
    torch.testing.assert_close(outs[0].rename(None), ref.detach(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        outs[1].align_to("d", "k").rename(None), w1t.grad, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(outs[2].rename(None), w2t.grad, rtol=RTOL, atol=ATOL)
