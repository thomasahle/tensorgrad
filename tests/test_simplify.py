import pytest
from sympy import symbols
from tensorgrad.tensor import Delta, Product, Sum, Variable


def test_copy_loop():
    i = symbols("i")
    expr = Delta(i, "i, j") @ Delta(i, "i, j")
    assert expr.simplify() == Delta(i)


def test_copy_double():
    p = symbols("p")
    expr = Delta(p, "p0, p2, p0_") @ Delta(p, "p0, p0_")
    assert expr.simplify() == Delta(p, "p2")


def test_copy_trace():
    p = symbols("p")
    expr = Delta(p, "p, p1") @ Delta(p, "p, p1")
    expected = Delta(p, "p") @ Delta(p, "p")
    assert expr.simplify() == expected.simplify()


def test_copy_iso():
    # We can't just assume two copies are the same, in particular if they have
    # edge dimensions linked to different variables.
    i, j = symbols("i j")
    X = Variable("X", i, j)
    c1, c2 = X.grad(X).tensors
    assert c1 != c2


def test_copy_iso_sym():
    # If X is symmetric, X.i and X.j are the same, so the two identity matrices
    # created are the same.
    i = symbols("i")
    X = Variable("X", i=i, j=i).with_symmetries("i j")
    c1, c2 = X.grad(X).tensors
    assert c1 == c2


def test_bad_size_sym():
    # If X is symmetric, X.i and X.j are the same, so the two identity matrices
    # created are the same.
    i, j = symbols("i, j")
    with pytest.raises(ValueError):
        Variable("X", i, j).with_symmetries("i j")


def test_expand():
    i, j = symbols("i, j")
    X = Variable("X", i, j)
    a = Variable("a", i)
    b = Variable("b", j)
    expr = X @ (a + b)
    expr = expr.simplify({"expand": True})
    assert isinstance(expr, Sum)
    assert expr == Sum([Product([Delta(j, "j"), X, a]), Product([Delta(i, "i"), X, b])])
