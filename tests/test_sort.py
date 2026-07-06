"""F.argsort (leaf) and F.sort (one_hot(argsort) contraction, derived grad)."""

import sympy
import torch

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)
b, n = sympy.symbols("b n")
DIMS = {b: 4, n: 7}


def _x():
    x = Variable("x", b=b, s=n)
    return x, {x: torch.randn(4, 7).rename("b", "s")}


def test_argsort_matches_torch():
    x, vals = _x()
    a = F.argsort(x, dim="s")
    ref = torch.argsort(vals[x].rename(None), dim=1).float()
    out = evaluate(a.simplify(), dict(vals), DIMS)
    assert torch.equal(out.align_to("b", "s").rename(None), ref)
    out2 = compile_to_callable(a)(dict(vals), DIMS)
    assert torch.equal(out2.align_to("b", "s").rename(None), ref)


def test_sort_matches_torch():
    x, vals = _x()
    s = F.sort(x, dim="s")
    ref = vals[x].rename(None).sort(dim=1).values
    out = evaluate(s.simplify(), dict(vals), DIMS)
    torch.testing.assert_close(out.align_to("b", "s").rename(None), ref)
    out2 = compile_to_callable(s)(dict(vals), DIMS)
    torch.testing.assert_close(out2.align_to("b", "s").rename(None), ref)


def test_sort_gradient_is_permutation_gather():
    """d/dx sum(sort(x) * w) = w scattered back through the sort permutation
    (= what torch autograd computes for torch.sort)."""
    x, vals = _x()
    w = Variable("w", b=b, s=n)
    wv = torch.randn(4, 7)
    loss = F.sum(F.sort(x, dim="s") * w)
    g = compile_to_callable(loss.grad(x))(dict(vals) | {w: wv.rename("b", "s")}, DIMS)

    xt = vals[x].rename(None).clone().requires_grad_(True)
    with torch.enable_grad():
        ref_loss = (xt.sort(dim=1).values * wv).sum()
        ref_loss.backward()
    torch.testing.assert_close(g.align_to("b", "s").rename(None), xt.grad)
