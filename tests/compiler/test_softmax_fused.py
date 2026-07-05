"""Tests for F.softmax / F.log_softmax as PLAIN COMPOSITIONS (task #34).

F.softmax and F.log_softmax are ordinary compositions of primitives
(exp / sum / pow / log) — no fused wrapper signatures, no hand math anywhere:
derivatives of every order come from the language itself. Numerical
stability is the COMPILER's contract: the IR stabilization pass re-fuses
exp-ratio patterns into torch.softmax / log_softmax / logsumexp. Eager
evaluation of the expanded forms is exact only at moderate |x| (the exp/sum
composition overflows float32 at |x| >= ~89) — that eager instability at
extreme logits is BY DESIGN ("eager evaluation doesn't need to be smart").

Covers:
(a) softmax(x).grad(x) constructs directly and, after .simplify(), compiles to
    code that calls the native torch.softmax kernel with NO exp-expansion;
(b) compiled gradient values match torch.autograd at |logits| up to 200
    without NaN; eager values match at moderate |logits|;
(c) an attention block softmax(q@kT/sqrt(d)+mask)@v compiles (loss + grad(q))
    and matches autograd on the single fused kernel;
(d) order-1, batched (broadcast edge), and multi-dim softmax Jacobians and
    Hessians all match torch.autograd; log_softmax mirrors all of it.
"""

import math

import pytest
import torch
from sympy import symbols
from torch.autograd.functional import hessian, jacobian

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)

RTOL, ATOL = 1e-4, 1e-6


def _source(fn, values, dims):
    """Return the generated python source of the specialization used for `dims`."""
    fn(dict(values), dict(dims))  # force specialization
    (spec,) = fn._specializations.values()
    return spec._source


# ---------------------------------------------------------------------------
# (a) construction + fused compilation
# ---------------------------------------------------------------------------


def test_softmax_grad_constructs_without_simplify():
    i = symbols("i")
    x = Variable("x", i)
    g = F.softmax(x, dim="i").grad(x)  # used to raise NotImplementedError
    assert g.shape.keys() == {"i", "i_"}


def test_softmax_grad_compiles_to_native_softmax():
    b, j = symbols("b j")
    y = Variable("y", b, j)
    g = F.softmax(y, dim="j").grad(y).full_simplify()
    fn = compile_to_callable(g)
    torch.manual_seed(0)
    vals = {y: torch.randn(2, 3, names=("b", "j"))}
    src = _source(fn, vals, {b: 2, j: 3})
    assert "torch.softmax" in src
    assert "torch.exp" not in src  # no exp-expansion anywhere in the kernel


def test_softmax_is_a_plain_composition():
    """Task #34: no wrapper tier. F.softmax builds exp/sum/pow primitives
    directly — no Function node named 'softmax' exists at any stage."""
    i = symbols("i")
    x = Variable("x", i)
    sm = F.softmax(x, dim="i")
    assert "softmax" not in repr(sm)
    assert "softmax" not in repr(sm.simplify())
    # The composition agrees with torch's kernel at moderate logits (eager).
    ts = {x: torch.randn(4, names=("i",))}
    torch.testing.assert_close(
        evaluate(sm.simplify(), dict(ts), {i: 4}).rename(None),
        torch.softmax(ts[x].rename(None), dim=0),
        rtol=RTOL,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# (b) + (d) gradient values vs autograd: order-1, batched, multi-dim, |logits|<=200
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 200.0])
def test_softmax_grad_order1(scale):
    i = symbols("i")
    x = Variable("x", i)
    g = F.softmax(x, dim="i").grad(x).simplify()
    torch.manual_seed(0)
    xt = (torch.randn(5) * scale).clamp(-200, 200)
    if scale <= 1.0:
        # Moderate inputs: the derived (expanded-primitive) gradient is exact
        # even under eager evaluation.
        res = evaluate(g, {x: xt.rename("i")}, {i: 5})
    else:
        # Extreme inputs: stability is the compiler's contract — stabilize.py
        # re-fuses the derived gradient into the native softmax kernel.
        res = compile_to_callable(g)({x: xt.rename("i")}, {i: 5})
    assert not torch.isnan(res.rename(None)).any()
    ref = jacobian(lambda t: torch.softmax(t, dim=0), xt).rename("i", "i_")
    torch.testing.assert_close(
        res.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("scale", [1.0, 200.0])
def test_softmax_grad_batched_compiled(scale):
    """Broadcast (batch) edge: the Jacobian must be DIAGONAL over it."""
    b, j = symbols("b j")
    y = Variable("y", b, j)
    g = F.softmax(y, dim="j").grad(y).full_simplify()
    fn = compile_to_callable(g)
    torch.manual_seed(1)
    yt = (torch.randn(2, 3) * scale).clamp(-200, 200)
    out = fn({y: yt.rename("b", "j")}, {b: 2, j: 3})
    assert not torch.isnan(out.rename(None)).any()
    ref = jacobian(lambda t: torch.softmax(t, dim=1), yt).rename("b", "j", "b_", "j_")
    torch.testing.assert_close(
        out.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


def test_softmax_grad_multidim():
    k, el, m = symbols("k el m")
    z = Variable("z", k, el, m)
    g = F.softmax(z, dim=("k", "el")).grad(z).simplify()
    torch.manual_seed(2)
    zt = torch.randn(2, 3, 2)

    def multi_sm(t):
        return torch.softmax(t.reshape(6, 2), dim=0).reshape(2, 3, 2)

    res = evaluate(g, {z: zt.rename("k", "el", "m")}, {k: 2, el: 3, m: 2})
    ref = jacobian(multi_sm, zt).rename("k", "el", "m", "k_", "el_", "m_")
    torch.testing.assert_close(
        res.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


def test_softmax_multidim_compiles():
    """Multi-axis softmax compiles (stable max-subtracted form)."""
    k, el, m = symbols("k el m")
    z = Variable("z", k, el, m)
    sm = F.softmax(z, dim=("k", "el")).full_simplify()
    fn = compile_to_callable(sm)
    torch.manual_seed(2)
    zt = (torch.randn(2, 3, 2) * 200).clamp(-200, 200)
    out = fn({z: zt.rename("k", "el", "m")}, {k: 2, el: 3, m: 2})
    assert not torch.isnan(out.rename(None)).any()
    ref = torch.softmax(zt.reshape(6, 2), dim=0).reshape(2, 3, 2).rename("k", "el", "m")
    torch.testing.assert_close(
        out.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


# ---------------------------------------------------------------------------
# Hessians: double grad through simplify, and grad twice BEFORE simplify
# (second-order bookkeeping through the raw composition)
# ---------------------------------------------------------------------------


def test_softmax_hessian_batched():
    b, j = symbols("b j")
    y = Variable("y", b, j)
    h = F.sum(F.softmax(y, dim="j")).grad(y).grad(y).simplify()
    torch.manual_seed(3)
    yt = torch.randn(2, 3)
    res = evaluate(h, {y: yt.rename("b", "j")}, {b: 2, j: 3})
    ref = hessian(lambda t: torch.softmax(t, dim=1).sum(), yt, vectorize=True)
    ref = ref.rename("b", "j", "b_", "j_")
    torch.testing.assert_close(
        res.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


def test_softmax_second_derivative():
    """grad TWICE before simplify: the third-order tensor d^2 softmax / dx dx
    must be correct — derived from the composition, no hand Hessian."""
    i = symbols("i")
    x = Variable("x", i)
    hf = F.softmax(x, dim="i").grad(x, {"i": "i_"}).grad(x, {"i": "i__"}).simplify()
    torch.manual_seed(4)
    xt = torch.randn(4)
    res = evaluate(hf, {x: xt.rename("i")}, {i: 4})
    ref = jacobian(
        lambda t: jacobian(lambda u: torch.softmax(u, dim=0), t, create_graph=True), xt
    ).rename("i", "i_", "i__")
    torch.testing.assert_close(
        res.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


def test_log_softmax_second_derivative():
    i = symbols("i")
    x = Variable("x", i)
    hf = F.log_softmax(x, dim="i").grad(x, {"i": "i_"}).grad(x, {"i": "i__"}).simplify()
    torch.manual_seed(5)
    xt = torch.randn(4)
    res = evaluate(hf, {x: xt.rename("i")}, {i: 4})
    ref = jacobian(
        lambda t: jacobian(lambda u: torch.log_softmax(u, dim=0), t, create_graph=True), xt
    ).rename("i", "i_", "i__")
    torch.testing.assert_close(
        res.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


# ---------------------------------------------------------------------------
# log_softmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 200.0])
def test_log_softmax_value_and_grad(scale):
    b, j = symbols("b j")
    y = Variable("y", b, j)
    ls = F.log_softmax(y, dim="j")
    torch.manual_seed(6)
    yt = (torch.randn(2, 3) * scale).clamp(-200, 200)
    vals = {y: yt.rename("b", "j")}
    dims = {b: 2, j: 3}

    # Forward: eager at moderate scale only. The old fused wrapper made eager
    # forward stable at |x|=200; with F.log_softmax a plain composition that
    # property is gone BY DESIGN (task #34) — extreme inputs go through the
    # compiler, whose stabilize pass max-shifts the logsumexp.
    if scale <= 1.0:
        res = evaluate(ls.simplify(), dict(vals), dict(dims))
    else:
        res = compile_to_callable(ls.simplify())(dict(vals), dict(dims))
    ref = torch.log_softmax(yt, dim=1).rename("b", "j")
    torch.testing.assert_close(
        res.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )

    g = ls.grad(y).simplify()
    if scale <= 1.0:
        resg = evaluate(g, dict(vals), dict(dims))
    else:
        # The gradient is expanded-primitive algebra; at |x| ~ 200 its
        # stability comes from the compiler's stabilize pass.
        resg = compile_to_callable(g)(dict(vals), dict(dims))
    assert not torch.isnan(resg.rename(None)).any()
    refg = jacobian(lambda t: torch.log_softmax(t, dim=1), yt).rename("b", "j", "b_", "j_")
    torch.testing.assert_close(
        resg.align_to(*refg.names).rename(None), refg.rename(None), rtol=RTOL, atol=ATOL
    )


def test_log_softmax_forward_compiles_stable():
    """The composition x - log(sum(exp(x))) compiles to a max-shifted
    logsumexp (single torch.amax shift), exact at |x| = 200. (The fused
    wrapper used to emit a literal torch.log_softmax call for the forward;
    the shifted form is the stabilize pass's equally-exact spelling.)"""
    b, j = symbols("b j")
    y = Variable("y", b, j)
    ls = F.log_softmax(y, dim="j").simplify()
    fn = compile_to_callable(ls)
    torch.manual_seed(6)
    yt = (torch.randn(2, 3) * 200).clamp(-200, 200)
    vals = {y: yt.rename("b", "j")}
    src = _source(fn, vals, {b: 2, j: 3})
    assert src.count("torch.amax") == 1  # max-shifted exactly once
    out = fn(dict(vals), {b: 2, j: 3})
    ref = torch.log_softmax(yt, dim=1).rename("b", "j")
    torch.testing.assert_close(
        out.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


def test_log_softmax_grad_compiles_to_native_softmax():
    """The log_softmax gradient (delta - softmax) re-fuses to torch.softmax
    with no exp-expansion anywhere in the kernel."""
    b, j = symbols("b j")
    y = Variable("y", b, j)
    g = F.log_softmax(y, dim="j").grad(y).full_simplify()
    fn = compile_to_callable(g)
    torch.manual_seed(6)
    vals = {y: torch.randn(2, 3, names=("b", "j"))}
    src = _source(fn, vals, {b: 2, j: 3})
    assert "torch.softmax" in src
    assert "torch.exp" not in src


def test_log_softmax_hessian():
    b, j = symbols("b j")
    y = Variable("y", b, j)
    h = F.sum(F.log_softmax(y, dim="j")).grad(y).grad(y).simplify()
    torch.manual_seed(7)
    yt = torch.randn(2, 3)
    res = evaluate(h, {y: yt.rename("b", "j")}, {b: 2, j: 3})
    ref = hessian(lambda t: torch.log_softmax(t, dim=1).sum(), yt, vectorize=True)
    ref = ref.rename("b", "j", "b_", "j_")
    torch.testing.assert_close(
        res.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


def test_log_of_softmax_compiles_to_log_softmax():
    """log(softmax(x)) — log of the exp/sum ratio — re-fuses at compile time
    into a native torch.log_softmax kernel. (This used to be a symbolic
    rewrite on the fused signatures; it is now the stabilize pass's job.)"""
    b, j = symbols("b j")
    y = Variable("y", b, j)
    t = F.log(F.softmax(y, dim="j")).simplify()
    fn = compile_to_callable(t)
    torch.manual_seed(6)
    yt = (torch.randn(2, 3) * 200).clamp(-200, 200)
    vals = {y: yt.rename("b", "j")}
    src = _source(fn, vals, {b: 2, j: 3})
    assert "torch.log_softmax" in src
    assert "torch.exp" not in src
    out = fn(dict(vals), {b: 2, j: 3})
    ref = torch.log_softmax(yt, dim=1).rename("b", "j")
    torch.testing.assert_close(
        out.align_to(*ref.names).rename(None), ref.rename(None), rtol=RTOL, atol=ATOL
    )


def test_log_of_exp_cancels():
    b, j = symbols("b j")
    y = Variable("y", b, j)
    assert F.log(F.exp(y)).simplify() == y


# ---------------------------------------------------------------------------
# cross entropy (rewired to log_softmax)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_cross_entropy_fused_value_and_grad(scale):
    C = symbols("C")
    logits = Variable("logits", C=C)
    target = Variable("target", C=C)
    torch.manual_seed(8)
    lt = (torch.randn(5) * scale * 2).clamp(-200, 200)
    tt = torch.softmax(torch.randn(5), dim=0)
    vals = {logits: lt.rename("C"), target: tt.rename("C")}
    dims = {C: 5}

    ce = F.cross_entropy(logits, target, dim="C")
    fn = compile_to_callable(ce.simplify(), ce.grad(logits).simplify())
    val, grad = fn(dict(vals), dict(dims))
    assert not torch.isnan(val.rename(None)).any()
    assert not torch.isnan(grad.rename(None)).any()

    lr = lt.clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(lr, tt)
    loss.backward()
    torch.testing.assert_close(val.rename(None), loss.detach(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        grad.align_to("C").rename(None), lr.grad, rtol=RTOL, atol=1e-5
    )


# ---------------------------------------------------------------------------
# (c) attention block: softmax(q@kT/sqrt(d)+mask)@v, loss + grad(q)
# ---------------------------------------------------------------------------


def test_attention_block_fused_loss_and_grad():
    seq_q, seq_k, dk, dv = symbols("seq_q seq_k dk dv")
    q = Variable("q", seq_q, dk)
    k = Variable("k", seq_k, dk)
    v = Variable("v", seq_k, dv)
    m = Variable("m", seq_q, seq_k)

    dims = {seq_q: 4, seq_k: 5, dk: 4, dv: 3}
    inv_sqrt_d = 1 / math.sqrt(dims[dk])
    logits = (q @ k) * inv_sqrt_d + m
    att = F.softmax(logits, dim="seq_k") @ v
    loss = F.sum(att)
    gq = loss.grad(q)

    loss_s = loss.full_simplify()
    gq_s = gq.full_simplify()
    fn = compile_to_callable(loss_s, gq_s)

    torch.manual_seed(9)
    vals = {
        q: torch.randn(4, 4, names=("seq_q", "dk")),
        k: torch.randn(5, 4, names=("seq_k", "dk")),
        v: torch.randn(5, 3, names=("seq_k", "dv")),
        # includes -inf-like large negative masked positions
        m: (torch.randn(4, 5) * 50).rename("seq_q", "seq_k"),
    }
    lo, go = fn(dict(vals), dict(dims))
    assert not torch.isnan(lo.rename(None)).any()
    assert not torch.isnan(go.rename(None)).any()

    # autograd reference
    qr = vals[q].rename(None).clone().requires_grad_(True)
    scores = qr @ vals[k].rename(None).T * inv_sqrt_d + vals[m].rename(None)
    ref = torch.softmax(scores, dim=1) @ vals[v].rename(None)
    ref.sum().backward()
    torch.testing.assert_close(lo.rename(None), ref.sum().detach(), rtol=RTOL, atol=1e-4)
    torch.testing.assert_close(
        go.align_to("seq_q", "dk").rename(None), qr.grad, rtol=RTOL, atol=1e-5
    )

    # The exp/sum composition re-fuses into a single native softmax kernel
    # (there is no separate "expanded" form anymore: F.softmax IS the
    # expansion, and stabilize/factoring collapse it back — task #34).
    src_fused = _source(fn, vals, dims)
    assert "torch.softmax" in src_fused and "torch.exp" not in src_fused
