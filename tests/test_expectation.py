import torch
from tensorgrad import Variable
from tensorgrad.extras.expectation import Expectation
from tensorgrad.tensor import Copy, Zero
from tensorgrad.utils import assert_close, rand_values


def test_names():
    x = Variable("x", "i, j")
    mu = Variable("x", "i, j")
    covar = Variable("c", "i, i2, j, j2")
    res = Expectation(x, x, mu, covar).simplify()
    assert res.is_isomorphic(mu, match_edges=True)

    # The expectation of a X transposed should be mu transposed
    xt = x.rename({"i": "j", "j": "i"})
    mut = mu.rename({"i": "j", "j": "i"})
    res = Expectation(xt, x, mu, covar).simplify()
    assert res.is_isomorphic(mut, match_edges=True)

    # The expectation of the outer product x (x) x2 should be covar if mu = 0
    zero = Zero(["i", "j"])
    x2 = x.rename({"i": "i2", "j": "j2"})
    res = Expectation(x @ x2, x, zero, covar).simplify()
    assert res.is_isomorphic(covar, match_edges=True)

    x2t = x2.rename({"i2": "j2", "j2": "i2"})
    covart = covar.rename({"i": "j", "i2": "j2", "j": "i", "j2": "i2"})
    res = Expectation(xt @ x2t, x, zero, covar).simplify()
    assert res.is_isomorphic(covart, match_edges=True)


def test_quadratic():
    X = Variable("X", "i, j")
    A = Variable("A", "j, j1")
    ts = rand_values([X, A], i=2, j=3, j1=3)
    ts[A] = ts[A] ** 2  # Make it more distinguishable

    mu = Zero("i, j", link=X)
    covar = Copy("i, k", link=X) @ Copy("j, l", link=X)

    expr = X.rename({"i": "i0"}) @ A @ X.rename({"j": "j1"})
    assert set(expr.edges) == {"i0", "i"}

    res = Expectation(expr, X, mu, covar).simplify().evaluate(ts, dims={"i": 2})
    expected = ts[A].rename(None).trace() * torch.eye(2).rename("i0", "i")
    assert_close(res, expected)


def test_quartic():
    X = Variable("X", "i, j")
    A = Variable("A", "j, j1")
    B = Variable("B", "i, i1")
    C = Variable("C", "j, j1")
    expr = (
        X.rename({"i": "i0"})
        @ A
        @ X.rename({"j": "j1"})
        @ B
        @ X.rename({"i": "i1"})
        @ C
        @ X.rename({"j": "j1"})
    )
    mu = Zero(["i", "j"])
    covar = Copy(["i", "k"]) @ Copy(["j", "l"])
    assert covar.edges == ["i", "k", "j", "l"]
    expr = Expectation(expr, X, mu, covar).simplify()

    ts = rand_values([X, A, B, C], i=2, i0=2, i1=2, j=3, j1=3)
    ts[A] = ts[A] ** 2
    ts[B] = ts[B] ** 2
    ts[C] = ts[C] ** 2

    res = expr.evaluate(ts, dims={"i": 2})
    expected = (
        ts[A].rename(None).trace() * ts[C].rename(None).trace() * ts[B].rename(None)
        + (ts[A].rename(None).T @ ts[C].rename(None)).trace() * ts[B].rename(None).T
        + (ts[A].rename(None).T @ ts[C].rename(None)).trace() * ts[B].rename(None).trace() * torch.eye(2)
    ).rename("i", "i0")
    assert_close(res, expected)
