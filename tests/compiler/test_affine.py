"""Tests for Affine structured tensors: re-derivations of Delta/Convolution,
and new structured tensors (shift, strided/dilated conv, basis, flatten) that
require no compiler changes — only a constraint row."""

import pytest
import torch
from sympy import symbols

torch.set_num_threads(2)

from tensorgrad import Variable, Delta
import tensorgrad.functions as F
from tensorgrad.extras.evaluate import evaluate
from tensorgrad.compiler import (
    compile_to_callable,
    Affine,
    affine_basis,
    affine_convolution,
    affine_delta,
    affine_flatten,
    affine_shift,
)
from tensorgrad.compiler import codegen_torch


def run(tensors, values, dims):
    f = compile_to_callable(*tensors)
    out = f(dict(values), dict(dims))
    return out if isinstance(out, tuple) else (out,)


def assert_named_close(a, b, rtol=1e-4, atol=1e-6):
    b = b.align_to(*a.names)
    torch.testing.assert_close(a.rename(None), b.rename(None), rtol=rtol, atol=atol)


# ---- re-derivations ---------------------------------------------------------


def test_affine_delta_equals_delta():
    i = symbols("i")
    x = Variable("x", i)
    torch.manual_seed(0)
    xv = torch.randn(5).refine_names("i")
    dims = {i: 5}
    with_delta = (x @ Delta(i, "i", "a", "b")).simplify()
    with_affine = (x @ affine_delta(i, "i", "a", "b")).simplify()
    (r1,) = run([with_delta], {x: xv}, dims)
    (r2,) = run([with_affine], {x: xv}, dims)
    assert_named_close(r1, r2)
    # and both agree with the interpreter (Affine registered as oracle type)
    assert_named_close(r2, evaluate(with_affine, {x: xv}, dict(dims)))


def test_affine_convolution_equals_f_convolution():
    w_in, k, w_out = symbols("w_in k w_out")
    x = Variable("x", w_in)
    ker = Variable("ker", k)
    torch.manual_seed(0)
    dims = {w_in: 10, k: 3, w_out: 8}
    xv = torch.randn(10).refine_names("w_in")
    kv = torch.randn(3).refine_names("k")
    old = (x @ F.Convolution(w_in, k, w_out) @ ker).full_simplify()
    new = (x @ affine_convolution(w_in, k, w_out) @ ker).full_simplify()
    (r1,) = run([old], {x: xv, ker: kv}, dims)
    (r2,) = run([new], {x: xv, ker: kv}, dims)
    assert_named_close(r1, r2)
    ref = torch.nn.functional.conv1d(xv.rename(None)[None, None], kv.rename(None)[None, None])[0, 0]
    torch.testing.assert_close(r2.rename(None), ref, rtol=1e-4, atol=1e-6)


def test_affine_conv_gradient_matches_evaluate():
    w_in, k, w_out = symbols("w_in k w_out")
    x = Variable("x", w_in)
    ker = Variable("ker", k)
    loss = F.sum((x @ affine_convolution(w_in, k, w_out) @ ker) ** 2)
    g = loss.grad(ker).full_simplify()
    gx = loss.grad(x).full_simplify()  # transpose-conv path (negative coeff)
    torch.manual_seed(0)
    dims = {w_in: 9, k: 3, w_out: 7}
    vals = {x: torch.randn(9).refine_names("w_in"), ker: torch.randn(3).refine_names("k")}
    rg, rgx = run([g, gx], vals, dims)
    assert_named_close(rg, evaluate(g, dict(vals), dict(dims)))
    assert_named_close(rgx, evaluate(gx, dict(vals), dict(dims)))


def test_standalone_affine_and_conv_materialize_identically():
    w_in, k, w_out = symbols("w_in k w_out")
    dims = {w_in: 6, k: 3, w_out: 4}
    (r_old,) = run([F.Convolution(w_in, k, w_out)], {}, dims)
    (r_new,) = run([affine_convolution(w_in, k, w_out)], {}, dims)
    assert_named_close(r_old, r_new)


# ---- new capabilities (no compiler changes needed) --------------------------


def test_shift():
    i, o = symbols("i o")
    x = Variable("x", i)
    torch.manual_seed(0)
    xv = torch.randn(8).refine_names("i")
    # (x @ S)[o] = x[o + 2], with zero padding at the boundary
    y = (x @ affine_shift(2, i=i, o=o)).simplify()
    (r,) = run([y], {x: xv}, {i: 8, o: 8})
    ref = torch.zeros(8)
    ref[:6] = xv.rename(None)[2:]
    torch.testing.assert_close(r.rename(None), ref)


def test_strided_convolution_matches_torch():
    w_in, k, w_out = symbols("w_in k w_out")
    x = Variable("x", w_in)
    ker = Variable("ker", k)
    torch.manual_seed(0)
    W, K, S = 11, 3, 2
    dims = {w_in: W, k: K, w_out: (W - K) // S + 1}
    xv = torch.randn(W).refine_names("w_in")
    kv = torch.randn(K).refine_names("k")
    y = (x @ affine_convolution(w_in, k, w_out, stride=S) @ ker).full_simplify()
    (r,) = run([y], {x: xv, ker: kv}, dims)
    ref = torch.nn.functional.conv1d(xv.rename(None)[None, None], kv.rename(None)[None, None], stride=S)[0, 0]
    torch.testing.assert_close(r.rename(None), ref, rtol=1e-4, atol=1e-6)


def test_dilated_convolution_matches_torch():
    w_in, k, w_out = symbols("w_in k w_out")
    x = Variable("x", w_in)
    ker = Variable("ker", k)
    torch.manual_seed(0)
    W, K, D = 12, 3, 2
    dims = {w_in: W, k: K, w_out: W - D * (K - 1)}
    xv = torch.randn(W).refine_names("w_in")
    kv = torch.randn(K).refine_names("k")
    y = (x @ affine_convolution(w_in, k, w_out, dilation=D) @ ker).full_simplify()
    (r,) = run([y], {x: xv, ker: kv}, dims)
    ref = torch.nn.functional.conv1d(
        xv.rename(None)[None, None], kv.rename(None)[None, None], dilation=D
    )[0, 0]
    torch.testing.assert_close(r.rename(None), ref, rtol=1e-4, atol=1e-6)


def test_basis_vector():
    i = symbols("i")
    x = Variable("x", i)
    torch.manual_seed(0)
    xv = torch.randn(6).refine_names("i")
    # <x, e_3> = x[3]
    y = (x @ affine_basis(3, i=i)).simplify()
    (r,) = run([y], {x: xv}, {i: 6})
    torch.testing.assert_close(r.rename(None), xv.rename(None)[3])
    # standalone basis vector output
    (b,) = run([affine_basis(2, i=i)], {}, {i: 5})
    torch.testing.assert_close(b.rename(None), torch.eye(5)[2])


def test_flatten_matches_reshape():
    f, i, j = symbols("f i j")
    v = Variable("v", f)
    torch.manual_seed(0)
    H, W = 3, 4
    vv = torch.randn(H * W).refine_names("f")
    # contract the flat edge: out[i, j] = v[W*i + j], i.e. reshape
    y = (v @ affine_flatten(W, f=f, i=i, j=j)).simplify()
    (r,) = run([y], {v: vv}, {f: H * W, i: H, j: W})
    assert_named_close(r, vv.rename(None).reshape(H, W).refine_names("i", "j"))
    # and the other direction: flatten an image
    img = Variable("img", i, j)
    iv = torch.randn(H, W).refine_names("i", "j")
    z = (img @ affine_flatten(W, f=f, i=i, j=j)).simplify()
    (r2,) = run([z], {img: iv}, {f: H * W, i: H, j: W})
    torch.testing.assert_close(r2.rename(None), iv.rename(None).reshape(-1))


# ---- fast path vs always-correct fallback -----------------------------------


def test_fast_path_matches_dense_fallback():
    w_in, k, w_out, i, o = symbols("w_in k w_out i o")
    x = Variable("x", w_in)
    ker = Variable("ker", k)
    exprs = [
        (x @ affine_convolution(w_in, k, w_out, stride=2) @ ker).full_simplify(),
        F.sum((x @ affine_convolution(w_in, k, w_out) @ ker) ** 2).grad(x).full_simplify(),
    ]
    torch.manual_seed(0)
    dims = {w_in: 9, k: 3, w_out: 4}
    vals = {x: torch.randn(9).refine_names("w_in"), ker: torch.randn(3).refine_names("k")}
    fast = run(exprs, vals, dims)
    old_flag = codegen_torch.AFFINE_FAST
    codegen_torch.AFFINE_FAST = False
    try:
        slow = run(exprs, vals, dims)
    finally:
        codegen_torch.AFFINE_FAST = old_flag
    for a, b in zip(fast, slow):
        assert_named_close(a, b)


def test_no_indicator_materialized_on_conv_forward():
    """The forward conv contraction must compile to pure views, no dense
    indicator and no materialized structure tensor."""
    w_in, k, w_out = symbols("w_in k w_out")
    x = Variable("x", w_in)
    ker = Variable("ker", k)
    y = (x @ affine_convolution(w_in, k, w_out) @ ker).full_simplify()
    f = compile_to_callable(y)
    torch.manual_seed(0)
    f({x: torch.randn(10).refine_names("w_in"), ker: torch.randn(3).refine_names("k")}, {w_in: 10, k: 3, w_out: 8})
    src = next(iter(f._specializations.values()))._source
    assert "as_strided" in src
    assert "_affine" not in src, f"dense indicator materialized:\n{src}"


# ---- derivative-of-convolution algebra: row-row elimination -----------------


def test_sum_of_conv_counts_solutions():
    """F.sum over a standalone conv = number of ones = k * w_out.
    Exercises free-wire summation -> range row -> drop -> orphan factors."""
    w_in, k, w_out = symbols("w_in k w_out")
    t = F.sum(affine_convolution(w_in, k, w_out)).simplify()
    dims = {w_in: 8, k: 3, w_out: 6}
    (r,) = run([t], {}, dims)
    assert_named_close(r, evaluate(t, {}, dict(dims)))
    torch.testing.assert_close(r.rename(None), torch.tensor(18.0))  # 3 * 6


def test_conv_conv_composition():
    """Two convolutions contracted over the shared input edge: the compiler
    Gaussian-eliminates the shared wire, leaving [z+y == z'+y'] — an O(ZYZ'Y')
    indicator instead of the naive O(X*ZYZ'Y') contraction."""
    x, z, y, z2, y2 = symbols("x z y z2 y2")
    c1 = affine_convolution(x=x, z=z, y=y)
    c2 = affine_convolution(x=x, z2=z2, y2=y2)
    t = (c1 @ c2).simplify()
    dims = {x: 8, z: 3, y: 6, z2: 5, y2: 4}
    (r,) = run([t], {}, dims)
    assert_named_close(r, evaluate(t, {}, dict(dims)))


def test_conv_conv_with_signal():
    """Second-derivative-style pattern: signal contracted through two
    composed convolutions."""
    x, z, y, z2, y2 = symbols("x z y z2 y2")
    v = Variable("v", y2)
    c1 = affine_convolution(x=x, z=z, y=y)
    c2 = affine_convolution(x=x, z2=z2, y2=y2)
    t = (v @ c2 @ c1).full_simplify()
    torch.manual_seed(0)
    dims = {x: 8, z: 3, y: 6, z2: 5, y2: 4}
    vals = {v: torch.randn(4).refine_names("y2")}
    (r,) = run([t], vals, dims)
    assert_named_close(r, evaluate(t, dict(vals), dict(dims)))
