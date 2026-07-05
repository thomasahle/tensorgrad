"""Orphan-wire regression tests for stabilize's ratio fusion.

Pointwise identities (Z^-1 * Z = ones; exp/softmax = sumexp) let ratio
fusion delete or shrink operands — but if the rewritten pair were the only
holders of a contracted wire, the einsum still sums over it, and dropping
the wire without paying dim(w) deletes a factor of the wire's size.
Found by the oracle property tests (compiled 1.0 where evaluate said |a|)."""

import torch
from sympy import symbols

torch.set_num_threads(2)

from tensorgrad.tensor import Variable, Product, Function
from tensorgrad.functions import _PowerFunction
import tensorgrad.functions as F
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

a, b, v = symbols("a b v")


def test_exact_cancellation_pays_orphaned_wire():
    # sum_a va^-1 * va = |a|, not 1.
    va = Variable("va", a)
    expr = Product([Function(_PowerFunction(k=-1), inputs=[va], shape_out={}), va])
    vals = {va: torch.tensor([2.0, 4.0]).refine_names("a")}
    want = evaluate(expr, dict(vals), {a: 2})
    for simplify in (False, True):
        got = compile_to_callable(expr, simplify=simplify)(dict(vals), {a: 2})
        torch.testing.assert_close(got.rename(None), want.rename(None))


def test_softmax_denominator_pays_orphaned_axis():
    # sum_v exp(x)/softmax(x) = |v| * sumexp(x): the softmax axis drops from
    # both rewritten operands and must be paid for.
    x = Variable("x", b, v)
    e = F.exp(x)
    s = e / F.sum(e, ["v"], keepdims=True)
    expr = F.sum(F.exp(x) * F.pow(s, -1), ["v"])
    torch.manual_seed(0)
    xv = torch.randn(3, 4).refine_names("b", "v")
    want = evaluate(expr, {x: xv}, {b: 3, v: 4})
    got = compile_to_callable(expr)({x: xv}, {b: 3, v: 4})
    torch.testing.assert_close(
        got.align_to(*want.names).rename(None), want.rename(None), rtol=1e-4, atol=1e-6
    )
