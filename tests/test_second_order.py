"""Second-order derivatives: the compiler must FIND the algebra.

These tests pin kernel-count ceilings, not just values: hessian(x^T A x)
must compile to the closed form A + A^T (no x-compute at all), and
hessian(logsumexp) to softmax + rank-1 correction. torch.func/jax compute
these in O(n^3); the point of symbolic differentiation + consolidation is
that the closed form falls out.
"""

import re

import sympy
import torch

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable

torch.set_num_threads(2)
n = sympy.Symbol("n")
N = 13


def _source_and_out(H, feed, dims):
    prog = compile_to_callable(H)
    out = prog(feed, dims)
    src = next(iter(prog._specializations.values()))._source
    stmts = [ln for ln in src.splitlines()
             if not ln.lstrip().startswith(("def ", "return", "del ", "#")) and ln.strip()]
    return out, stmts


def test_quadratic_hessian_is_closed_form():
    x = Variable("x", n)
    A = Variable("A", i=n, j=n)
    f = (x.rename(n="i") @ A) @ x.rename(n="j")
    H = f.grad(x, {"n": "di"}).grad(x, {"n": "dj"})
    xv, Av = torch.randn(N), torch.randn(N, N)
    out, stmts = _source_and_out(H, {x: xv.rename("n"), A: Av.rename("i", "j")}, {n: N})
    with torch.enable_grad():
        ref = torch.func.hessian(lambda xx: (xx @ Av @ xx))(xv)
    torch.testing.assert_close(out.align_to("di", "dj").rename(None), ref)
    # the closed form A + A^T: one statement, and x is never touched
    assert len(stmts) <= 2, stmts
    assert not any("x" == s.strip() for s in stmts)


def test_logsumexp_hessian_is_softmax_rank1():
    x = Variable("x", n)
    f = F.log(F.sum(F.exp(x)))
    H = f.grad(x, {"n": "di"}).grad(x, {"n": "dj"})
    xv = torch.randn(N)
    out, stmts = _source_and_out(H, {x: xv.rename("n")}, {n: N})
    with torch.enable_grad():
        ref = torch.func.hessian(lambda xx: torch.logsumexp(xx, 0))(xv)
    torch.testing.assert_close(out.align_to("di", "dj").rename(None), ref)
    # softmax + diag-minus-outer: a handful of cheap statements, ONE softmax,
    # and crucially no second exp/sum recomputation
    assert len(stmts) <= 8, stmts
    assert sum("softmax" in s for s in stmts) == 1
    assert not any("torch.exp" in s for s in stmts)


def test_mlp_hessian_matches_torch_func():
    m = sympy.Symbol("m")
    x = Variable("x", n)
    W1 = Variable("W1", n=n, m=m)
    W2 = Variable("W2", m=m)
    f = F.sum(F.tanh(x @ W1) * W2)
    H = f.grad(x, {"n": "di"}).grad(x, {"n": "dj"})
    xv, W1v, W2v = torch.randn(N), torch.randn(N, 7) / N**0.5, torch.randn(7)
    out, stmts = _source_and_out(
        H, {x: xv.rename("n"), W1: W1v.rename("n", "m"), W2: W2v.rename("m")}, {n: N, m: 7}
    )
    with torch.enable_grad():
        ref = torch.func.hessian(lambda xx: (torch.tanh(xx @ W1v) * W2v).sum())(xv)
    torch.testing.assert_close(out.align_to("di", "dj").rename(None), ref)
    # dense Hessian but shared work: tanh computed once, no n^3 contraction
    assert sum("tanh" in s for s in stmts) == 1
