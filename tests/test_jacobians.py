import torch
from torch.autograd.functional import jacobian
from sympy import symbols

from tensorgrad.tensor import Variable
from tensorgrad.testutils import rand_values, assert_close


def test_simple_vector():
    i = symbols("i")
    x = Variable("x", i)
    ts = rand_values([x], {i: 3})
    res = x.grad(x).simplify().evaluate({x: ts[x]})
    expected = jacobian(lambda x: x, ts[x].rename(None))
    torch.testing.assert_close(res.rename(None), expected)


def test_simple_matrix():
    i, j = symbols("i j")
    x = Variable("x", i, j)
    ts = rand_values([x], {i: 3, j: 2})
    res = x.grad(x).simplify().evaluate({x: ts[x]})
    expected = jacobian(lambda x: x, ts[x].rename(None)).rename("i", "j", "i_", "j_")
    assert_close(res, expected)


def test_a_not_function_of_x():
    i = symbols("i")
    a = Variable("a")
    x = Variable("x", i)
    ts = rand_values([a, x], {i: 3})
    expr = (a / x).grad(x).simplify()
    res = expr.evaluate(ts)
    expected = jacobian(lambda x: ts[a] / x, ts[x].rename(None)).rename("i", "i_")
    assert_close(res, expected)


def test_a_not_function_of_x_matrix():
    i, j = symbols("i j")
    A = Variable("A", i, j)
    x = Variable("x", j)
    ts = rand_values([A, x], {i: 3, j: 2})
    res = (A @ x).grad(x).simplify().evaluate({A: ts[A], x: ts[x]})
    expected = jacobian(lambda x: (ts[A].rename(None) @ x), ts[x].rename(None)).rename("i", "j")
    assert_close(res, expected)
