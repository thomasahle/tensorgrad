"""F.gelu(fused=True): fused GELU technology-mapping primitive.

Forward IS the tanh-approx GELU; compiles to torch's gelu / gelu_backward
kernels for forward AND reverse backward, verified vs torch.autograd.
Fused only in reverse mode -> gradient test uses a family.
"""

import sympy
import torch

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)
b, s, d = sympy.symbols("b s d")


def test_forward_matches_torch():
    torch.manual_seed(0)
    dims = {b: 2, s: 5, d: 8}
    x = Variable("x", b=b, seq=s, d=d)
    gl = F.gelu(x, approximate="tanh", fused=True)
    xv = torch.randn(2, 5, 8)
    feed = {x: xv.rename("b", "seq", "d")}
    ref = torch.nn.functional.gelu(xv, approximate="tanh")
    for got in (evaluate(gl.simplify(), dict(feed), dims),
                compile_to_callable(gl)(dict(feed), dims)):
        torch.testing.assert_close(got.align_to("b", "seq", "d").rename(None), ref)


def test_reverse_grad_matches_autograd():
    torch.manual_seed(1)
    dims = {b: 2, s: 5, d: 8}
    x = Variable("x", b=b, seq=s, d=d)
    w = Variable("w", b=b, seq=s, d=d)
    xv, wv = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    feed = {x: xv.rename("b", "seq", "d"), w: wv.rename("b", "seq", "d")}
    loss = F.sum(F.gelu(x, approximate="tanh", fused=True) * w)
    prog = compile_to_callable(loss.grad(x), loss.grad(w))  # family -> reverse mode
    gx, _ = prog(dict(feed), dims)
    xt = xv.clone().requires_grad_(True)
    with torch.enable_grad():
        (torch.nn.functional.gelu(xt, approximate="tanh") * wv).sum().backward()
    torch.testing.assert_close(gx.align_to("b", "seq", "d").rename(None), xt.grad)
    assert "gelu_backward" in next(iter(prog._specializations.values()))._source
