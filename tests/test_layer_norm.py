"""F.layer_norm: fused layer normalization (technology-mapping primitive).

Forward IS (x - mean)/sqrt(var+eps) * weight + bias over `dim` (machine-
checked against that definition); compiles to torch's native_layer_norm
kernels for forward AND the reverse backward, verified here against
torch.autograd. Fused only in reverse mode -> gradient tests use a family.
"""

import sympy
import torch

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)
b, s, d = sympy.symbols("b s d")


def test_forward_matches_definition_eager_and_compiled():
    torch.manual_seed(0)
    dims = {b: 2, s: 5, d: 8}
    x = Variable("x", b=b, seq=s, d=d)
    g = Variable("g", d=d)
    bias = Variable("bias", d=d)
    ln = F.layer_norm(x, dim="d", weight=g, bias=bias, eps=1e-5)
    assert set(ln.edges) == {"b", "seq", "d"}
    xv, gv, bv = torch.randn(2, 5, 8), torch.randn(8), torch.randn(8)
    feed = {x: xv.rename("b", "seq", "d"), g: gv.rename("d"), bias: bv.rename("d")}
    ref = torch.nn.functional.layer_norm(xv, (8,), gv, bv, 1e-5)
    for got in (evaluate(ln.simplify(), dict(feed), dims),
                compile_to_callable(ln)(dict(feed), dims)):
        torch.testing.assert_close(got.align_to("b", "seq", "d").rename(None), ref)


def test_reverse_grads_match_autograd():
    torch.manual_seed(1)
    dims = {b: 2, s: 5, d: 8}
    x = Variable("x", b=b, seq=s, d=d)
    g = Variable("g", d=d)
    bias = Variable("bias", d=d)
    w = Variable("w", b=b, seq=s, d=d)
    xv, gv, bv, wv = torch.randn(2, 5, 8), torch.randn(8), torch.randn(8), torch.randn(2, 5, 8)
    feed = {x: xv.rename("b", "seq", "d"), g: gv.rename("d"), bias: bv.rename("d"),
            w: wv.rename("b", "seq", "d")}
    loss = F.sum(F.layer_norm(x, dim="d", weight=g, bias=bias, eps=1e-5) * w)
    prog = compile_to_callable(loss.grad(x), loss.grad(g), loss.grad(bias))
    gx, gg, gb = prog(dict(feed), dims)
    xt, gt, bt = (xv.clone().requires_grad_(True), gv.clone().requires_grad_(True),
                  bv.clone().requires_grad_(True))
    with torch.enable_grad():
        o = torch.nn.functional.layer_norm(xt, (8,), gt, bt, 1e-5)
        (o * wv).sum().backward()
    torch.testing.assert_close(gx.align_to("b", "seq", "d").rename(None), xt.grad)
    torch.testing.assert_close(gg.align_to("d").rename(None), gt.grad)
    torch.testing.assert_close(gb.align_to("d").rename(None), bt.grad)
    src = next(iter(prog._specializations.values()))._source
    assert "native_layer_norm" in src
