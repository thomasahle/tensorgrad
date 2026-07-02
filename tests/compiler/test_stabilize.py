"""Tests for the IR stabilization pass (tensorgrad/compiler/stabilize.py, #16).

Everything here spells softmax / log-softmax / tanh out of PRIMITIVES
(exp, log, pow, sum) — exactly what a researcher writes, and exactly what
simplify({"expand_softmax": True}) produces. The expanded forms overflow
float32 at |x| >= ~89; the pass must re-fuse them into stable
softmax/log_softmax/tanh/max-shifted-logsumexp kernels, INCLUDING the shapes
they take inside gradient graphs, without changing semantics.
"""

import pytest
import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler import stabilize as stabilize_mod
from tensorgrad.tensor import Delta

RTOL, ATOL = 1e-4, 1e-5
B, V = 4, 8


# ---------------------------------------------------------------------------
# primitives-only definitions
# ---------------------------------------------------------------------------


def primitive_softmax(x, dim):
    e = F.exp(x)
    return e * F.pow(F.sum(e, dim=dim, keepdims=True), -1)


def primitive_gelu(x):
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * F.pow(x, 3))
    e, em = F.exp(inner), F.exp(-inner)
    return 0.5 * x * (1 + (e - em) * F.pow(e + em, -1))


def _source(fn, values, dims):
    fn(dict(values), dict(dims))
    (spec,) = fn._specializations.values()
    return spec._source


def _vals(seed=1, scale=200.0):
    """fp32 inputs with |x| <= scale — far beyond exp's float32 range."""
    torch.manual_seed(seed)
    xt = (torch.rand(B, V) * 2 * scale - scale).rename("b", "v")
    wt = torch.randn(B, V).rename("b", "v")
    return xt, wt


def _setup():
    b, v = symbols("b v")
    x, w = Variable("x", b, v), Variable("w", b, v)
    return b, v, x, w


# ---------------------------------------------------------------------------
# (a) exp / sum-exp ratio -> softmax
# ---------------------------------------------------------------------------


def test_primitive_softmax_forward_stable():
    b, v, x, _ = _setup()
    fn = compile_to_callable(primitive_softmax(x, "v").simplify())
    xt, _ = _vals()
    src = _source(fn, {x: xt}, {b: B, v: V})
    assert "torch.softmax" in src
    assert "torch.exp" not in src
    out = fn({x: xt}, {b: B, v: V}).rename(None)
    torch.testing.assert_close(out, torch.softmax(xt.rename(None), dim=1), rtol=RTOL, atol=ATOL)


def test_primitive_softmax_grad_stable_at_200():
    """loss = sum(softmax(x)*w); the simplified gradient contains e/Z and
    e e'/Z^2 terms split across Sum boundaries — all must re-fuse."""
    b, v, x, w = _setup()
    loss = F.sum(primitive_softmax(x, "v") * w)
    fn = compile_to_callable(loss.simplify(), loss.grad(x).simplify())
    xt, wt = _vals()
    lv, gv = fn({x: xt, w: wt}, {b: B, v: V})
    s = torch.softmax(xt.rename(None), dim=1)
    wr = wt.rename(None)
    ref_l = (s * wr).sum()
    ref_g = s * (wr - (s * wr).sum(dim=1, keepdim=True))
    assert torch.isfinite(gv.rename(None)).all()
    torch.testing.assert_close(lv.rename(None), ref_l, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(gv.rename(None), ref_g, rtol=RTOL, atol=ATOL)


def test_scaled_denominator_weight():
    """e / mean(e) = V * softmax: the sum-einsum carries a symbolic weight 1/v
    which the fusion must compensate exactly."""
    b, v, x, _ = _setup()
    e = F.exp(x)
    t = e * F.pow(F.mean(e, dim="v", keepdims=True), -1)
    fn = compile_to_callable(t.simplify())
    xt, _ = _vals()
    out = fn({x: xt}, {b: B, v: V}).rename(None)
    ref = V * torch.softmax(xt.rename(None), dim=1)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# (b) log of softmax / log of sum-exp
# ---------------------------------------------------------------------------


def test_primitive_log_softmax_stable():
    b, v, x, _ = _setup()
    fn = compile_to_callable(F.log(primitive_softmax(x, "v")).simplify())
    xt, _ = _vals()
    src = _source(fn, {x: xt}, {b: B, v: V})
    assert "torch.log_softmax" in src
    out = fn({x: xt}, {b: B, v: V}).rename(None)
    torch.testing.assert_close(out, torch.log_softmax(xt.rename(None), dim=1), rtol=RTOL, atol=ATOL)


def test_logsumexp_stable_and_single_shift():
    b, v, x, _ = _setup()
    fn = compile_to_callable(F.log(F.sum(F.exp(x), dim="v")).simplify())
    xt, _ = _vals()
    src = _source(fn, {x: xt}, {b: B, v: V})
    assert src.count("torch.amax") == 1  # max-shifted exactly once (no re-firing)
    out = fn({x: xt}, {b: B, v: V}).rename(None)
    ref = torch.logsumexp(xt.rename(None), dim=1)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_primitive_ce_loss_and_grad_stable():
    """The founding example: -sum(y*log(softmax(x)))/B from primitives. The
    gradient contains 1/softmax * softmax cancellations split across
    contraction boundaries; compiled loss+grad must match torch at |x|=200."""
    b, v, x, _ = _setup()
    y = Variable("y", b, v)
    s = primitive_softmax(x, "v")
    L = -F.sum(y * F.log(s)) / Delta(b)
    fn = compile_to_callable(L.simplify(), L.grad(x).simplify())
    xt, _ = _vals()
    torch.manual_seed(2)
    yt = torch.zeros(B, V)
    yt[torch.arange(B), torch.randint(V, (B,))] = 1.0
    lv, gv = fn({x: xt, y: yt.rename("b", "v")}, {b: B, v: V})
    xr = xt.rename(None).clone().requires_grad_(True)
    ref = -(yt * torch.log_softmax(xr, dim=1)).sum() / B
    ref.backward()
    assert torch.isfinite(gv.rename(None)).all()
    torch.testing.assert_close(lv.rename(None), ref.detach(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(gv.rename(None), xr.grad, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# (c) exp-ratio tanh
# ---------------------------------------------------------------------------


def test_exp_ratio_tanh_forward():
    b, v, x, _ = _setup()
    e, em = F.exp(x), F.exp(-x)
    t = (e - em) * F.pow(e + em, -1)
    fn = compile_to_callable(t.simplify())
    xt, _ = _vals()
    src = _source(fn, {x: xt}, {b: B, v: V})
    assert "torch.tanh" in src
    assert "torch.exp" not in src
    out = fn({x: xt}, {b: B, v: V}).rename(None)
    torch.testing.assert_close(out, torch.tanh(xt.rename(None)), rtol=RTOL, atol=ATOL)


def test_exp_ratio_sigmoid_form():
    """exp(x) / (exp(x)+exp(-x)) == sigmoid(2x) == (1+tanh(x))/2."""
    b, v, x, _ = _setup()
    e, em = F.exp(x), F.exp(-x)
    t = e * F.pow(e + em, -1)
    fn = compile_to_callable(t.simplify())
    xt, _ = _vals()
    out = fn({x: xt}, {b: B, v: V}).rename(None)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.sigmoid(2 * xt.rename(None)), rtol=RTOL, atol=ATOL)


def test_primitive_gelu_grad_stable_at_200():
    """tanh-gelu spelled with exp; the gradient divides Linear combinations of
    exp(y), exp(-y) (nested two einsums deep) by cosh powers."""
    b, v, x, w = _setup()
    loss = F.sum(primitive_gelu(x) * w)
    fn = compile_to_callable(loss.simplify(), loss.grad(x).simplify())
    xt, wt = _vals()
    src = _source(fn, {x: xt, w: wt}, {b: B, v: V})
    assert "torch.tanh" in src
    assert "torch.exp" not in src
    lv, gv = fn({x: xt, w: wt}, {b: B, v: V})
    c, a = 0.7978845608028654, 0.044715
    xr, wr = xt.rename(None), wt.rename(None)
    t = torch.tanh(c * (xr + a * xr**3))
    ref_l = (0.5 * xr * (1 + t) * wr).sum()
    ref_g = wr * (0.5 * (1 + t) + 0.5 * xr * (1 - t * t) * c * (1 + 3 * a * xr * xr))
    assert torch.isfinite(gv.rename(None)).all()
    torch.testing.assert_close(lv.rename(None), ref_l, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(gv.rename(None), ref_g, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# expand_softmax=True expansions (fused signature -> primitives -> re-fused)
# ---------------------------------------------------------------------------


def test_expand_softmax_forward_restabilized():
    b, v, x, _ = _setup()
    sm = F.softmax(x, dim="v").simplify({"expand_softmax": True})
    assert "softmax" not in repr(sm)  # really expanded to primitives
    fn = compile_to_callable(sm)
    xt, _ = _vals()
    out = fn({x: xt}, {b: B, v: V}).rename(None)
    torch.testing.assert_close(out, torch.softmax(xt.rename(None), dim=1), rtol=RTOL, atol=ATOL)


def test_expand_softmax_grad_restabilized():
    b, v, x, w = _setup()
    loss = F.sum(F.softmax(x, dim="v") * w)
    g = loss.grad(x)
    fn = compile_to_callable(
        loss.simplify({"expand_softmax": True}), g.simplify({"expand_softmax": True})
    )
    xt, wt = _vals()
    lv, gv = fn({x: xt, w: wt}, {b: B, v: V})
    s = torch.softmax(xt.rename(None), dim=1)
    wr = wt.rename(None)
    ref_g = s * (wr - (s * wr).sum(dim=1, keepdim=True))
    assert torch.isfinite(gv.rename(None)).all()
    torch.testing.assert_close(lv.rename(None), (s * wr).sum(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(gv.rename(None), ref_g, rtol=RTOL, atol=ATOL)


def test_expand_log_softmax_restabilized():
    b, v, x, _ = _setup()
    ls = F.log_softmax(x, dim="v").simplify({"expand_softmax": True})
    fn = compile_to_callable(ls)
    xt, _ = _vals()
    out = fn({x: xt}, {b: B, v: V}).rename(None)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out, torch.log_softmax(xt.rename(None), dim=1), rtol=RTOL, atol=ATOL
    )


# ---------------------------------------------------------------------------
# semantics preservation + non-vacuity
# ---------------------------------------------------------------------------


def _programs():
    b, v, x, w = _setup()
    y = Variable("y", b, v)
    sm_loss = F.sum(primitive_softmax(x, "v") * w)
    gelu_loss = F.sum(primitive_gelu(x) * w)
    ce = -F.sum(y * F.log(primitive_softmax(x, "v"))) / Delta(b)
    return b, v, x, w, y, [
        (sm_loss.simplify(), sm_loss.grad(x).simplify()),
        (gelu_loss.simplify(), gelu_loss.grad(x).simplify()),
        (ce.simplify(), ce.grad(x).simplify()),
        (F.log(F.sum(F.exp(x), dim="v")).simplify(),),
    ]


def test_stabilize_preserves_semantics_fp64():
    """At moderate |x| (fp64) the stabilized programs must agree with the raw
    expanded programs to near machine precision — the pass is pure algebra."""
    b, v, x, w, y, programs = _programs()
    torch.manual_seed(0)
    vals = {
        x: torch.randn(B, V, dtype=torch.float64).rename("b", "v"),
        w: torch.randn(B, V, dtype=torch.float64).rename("b", "v"),
        y: torch.rand(B, V, dtype=torch.float64).rename("b", "v"),
    }
    for tensors in programs:
        got = compile_to_callable(*tensors)(dict(vals), {b: B, v: V})
        assert stabilize_mod.STABILIZE
        stabilize_mod.STABILIZE = False
        try:
            raw = compile_to_callable(*tensors)(dict(vals), {b: B, v: V})
        finally:
            stabilize_mod.STABILIZE = True
        got = got if isinstance(got, tuple) else (got,)
        raw = raw if isinstance(raw, tuple) else (raw,)
        for a, r in zip(got, raw):
            torch.testing.assert_close(a.rename(None), r.rename(None), rtol=1e-10, atol=1e-10)


def test_without_stabilize_softmax_grad_overflows():
    """Non-vacuity guard: with the pass disabled, the same program really is
    non-finite at |x|=200 in fp32 (so the stability tests test something)."""
    b, v, x, w = _setup()
    loss = F.sum(primitive_softmax(x, "v") * w)
    stabilize_mod.STABILIZE = False
    try:
        fn = compile_to_callable(loss.simplify(), loss.grad(x).simplify())
        xt, wt = _vals()
        lv, gv = fn({x: xt, w: wt}, {b: B, v: V})
    finally:
        stabilize_mod.STABILIZE = True
    assert not (
        torch.isfinite(lv.rename(None)).all() and torch.isfinite(gv.rename(None)).all()
    )


def test_fused_softmax_path_untouched():
    """Programs already using the fused F.softmax signature must compile to the
    identical single-kernel form (the pass must not disturb them)."""
    b, v, x, w = _setup()
    loss = F.sum(F.softmax(x, dim="v") * w)
    fn = compile_to_callable(loss.full_simplify(), loss.grad(x).full_simplify())
    xt, wt = _vals()
    src = _source(fn, {x: xt, w: wt}, {b: B, v: V})
    assert "torch.softmax" in src and "torch.exp" not in src
    lv, gv = fn({x: xt, w: wt}, {b: B, v: V})
    s = torch.softmax(xt.rename(None), dim=1)
    wr = wt.rename(None)
    ref_g = s * (wr - (s * wr).sum(dim=1, keepdim=True))
    torch.testing.assert_close(gv.rename(None), ref_g, rtol=RTOL, atol=ATOL)
