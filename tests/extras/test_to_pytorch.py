"""Tests for the legacy to_pytorch code generation backend."""

import torch
import torch.nn.functional as tF
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad.extras.evaluate import evaluate
from tensorgrad.extras.to_pytorch import compile_to_callable
from tensorgrad.tensor import Variable
from tensorgrad.testutils import assert_close, rand_values


def test_log_softmax_emit():
    # Regression: the log_softmax FunctionSignature (introduced when
    # cross_entropy started lowering to log_softmax) raised
    # NotImplementedError in the code generator.
    b, o = symbols("b o")
    x = Variable("x", b, o)
    expr = F.log_softmax(x, dim="o")
    vals = rand_values([x], {b: 3, o: 5})
    res = compile_to_callable(expr)(dict(vals))
    expected = tF.log_softmax(vals[x].rename(None), dim=1).rename("b", "o")
    assert_close(res, expected)


def test_cross_entropy_emit():
    # cross_entropy lowers to log_softmax; make sure the whole expression
    # compiles and matches both evaluate() and torch.
    b, o = symbols("b o")
    x = Variable("x", b, o)
    t = Variable("t", b, o)
    expr = F.cross_entropy(x, t, dim="o")
    vals = rand_values([x, t], {b: 3, o: 5})
    vals[t] = tF.softmax(vals[t].rename(None), dim=1).rename("b", "o")
    for e in (expr, expr.simplify()):
        res = compile_to_callable(e)(dict(vals))
        assert_close(res, evaluate(e, dict(vals)))
    expected = tF.cross_entropy(vals[x].rename(None), vals[t].rename(None), reduction="none").rename("b")
    assert_close(compile_to_callable(expr)(dict(vals)), expected)
