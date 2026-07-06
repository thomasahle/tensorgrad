"""F.concat and F.window: sequence surgery as structural (Affine) algebra."""

import sympy
import torch

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)

b, n, m, k, j = sympy.symbols("b n m k j")


def _vals():
    return {
        Variable("x", b=b, s=n): torch.randn(2, 3).rename("b", "s"),
        Variable("y", b=b, s=m): torch.randn(2, 4).rename("b", "s"),
    }


DIMS = {b: 2, n: 3, m: 4, k: 7, j: 5}


def test_concat_matches_torch_cat():
    x = Variable("x", b=b, s=n)
    y = Variable("y", b=b, s=m)
    cat = F.concat(x, y, dim="s", size=k)
    vals = _vals()
    ref = torch.cat([v.rename(None) for v in vals.values()], dim=1)
    out = evaluate(cat.simplify(), dict(vals), DIMS)
    assert torch.equal(out.align_to("b", "s").rename(None), ref)
    out2 = compile_to_callable(cat)(dict(vals), DIMS)
    assert torch.equal(out2.align_to("b", "s").rename(None), ref)


def test_window_slices():
    x = Variable("x", b=b, s=n)
    y = Variable("y", b=b, s=m)
    cat = F.concat(x, y, dim="s", size=k)
    w = cat @ F.window(1, s=k, t=j)  # cat[1:6], output edge renamed to t
    assert set(w.edges) == {"b", "t"}
    vals = _vals()
    ref = torch.cat([v.rename(None) for v in vals.values()], dim=1)[:, 1:6]
    out = compile_to_callable(w)(dict(vals), DIMS)
    assert torch.equal(out.align_to("b", "t").rename(None), ref)


def test_window_zero_pads_when_output_larger():
    x = Variable("x", b=b, s=n)
    emb = x @ F.window(0, s=n, big=k)  # embed 3 into 7, zeros beyond
    vals = _vals()
    out = compile_to_callable(emb)(dict(vals), DIMS)
    got = out.align_to("b", "big").rename(None)
    xv = [v for kk, v in vals.items() if kk.name == "x"][0].rename(None)
    assert torch.equal(got[:, :3], xv)
    assert torch.equal(got[:, 3:], torch.zeros(2, 4))


def test_gradient_through_concat():
    x = Variable("x", b=b, s=n)
    y = Variable("y", b=b, s=m)
    cat = F.concat(x, y, dim="s", size=k)
    loss = F.sum(cat * cat)
    vals = _vals()
    g = evaluate(loss.grad(x).simplify(), dict(vals), DIMS)
    xv = [v for kk, v in vals.items() if kk.name == "x"][0]
    assert torch.allclose(g.align_to("b", "s").rename(None), 2 * xv.rename(None))


def test_window_shift_pairing():
    """tokens/targets: the same buffer windowed at offsets 0 and 1."""
    x = Variable("x", b=b, s=k)
    a = x @ F.window(0, s=k, t=j)
    c = x @ F.window(1, s=k, t=j)
    vals = {x: torch.arange(14.0).reshape(2, 7).rename("b", "s")}
    fa = compile_to_callable(a)(dict(vals), DIMS).align_to("b", "t").rename(None)
    fc = compile_to_callable(c)(dict(vals), DIMS).align_to("b", "t").rename(None)
    assert torch.equal(fc[:, :-1], fa[:, 1:])
