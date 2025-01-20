import math
from sympy import symbols
import torch
from einops import einsum
from tensorgrad import Variable
from tensorgrad import functions as F
from tensorgrad.extras.expectation import Expectation
from tensorgrad import Delta, Product, Zero
from tensorgrad.tensor import Sum, Tensor
from tensorgrad.testutils import assert_close, rand_values


def simple_variables():
    i, j = symbols("i j")
    x = Variable("x", i, j)
    mu = Variable("x", i, j)
    covar = Variable("c", i, j, i2=i, j2=j).with_symmetries("i i2, j j2")
    covar_names = {"i": "i2", "j": "j2"}
    return x, mu, covar, covar_names


def test_names1():
    x, mu, covar, covar_names = simple_variables()
    res = Expectation(x, x, mu, covar, covar_names).simplify()
    assert res == mu


def test_names2():
    x, mu, covar, covar_names = simple_variables()
    # The expectation of a X transposed should be mu transposed
    xt = x.rename(i="j", j="i")
    mut = mu.rename(i="j", j="i")
    res = Expectation(xt, x, mu, covar, covar_names).simplify()
    assert res == mut


def test_names3():
    x, _, covar, covar_names = simple_variables()
    zero = Zero(**x.shape)
    # The expectation of the outer product x (x) x2 should be covar if mu = 0
    x2 = x.rename(i="i2", j="j2")
    res = Expectation(x @ x2, x, zero, covar, covar_names).simplify()
    assert res == covar


def test_names4():
    x, _, covar, covar_names = simple_variables()
    zero = Zero(**x.shape)
    # Transposing the outer product should give the transposed covariance
    xt = x.rename(i="j", j="i")
    x2t = xt.rename(i="i2", j="j2")
    covart = covar.rename(i="j", j="i", i2="j2", j2="i2")
    ex = Expectation(xt @ x2t, x, zero, covar, covar_names)
    res = ex.simplify()
    assert res == covart


def test_quadratic():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    A = Variable("A", j=j, j1=j)
    ts = rand_values([X, A], {i: 2, j: 3})
    ts[A] = ts[A] ** 2  # Make it more distinguishable

    mu = Zero(i, j)
    covar = Delta(i, "i, i_") @ Delta(j, "j, j_")

    expr = X.rename(i="i0", j="j") @ A @ X.rename(j="j1", i="i")
    assert expr.edges == {"i0", "i"}

    res = Expectation(expr, X, mu, covar, {"i": "i_", "j": "j_"}).simplify().evaluate(ts)
    expected = ts[A].rename(None).trace() * torch.eye(2).rename("i0", "i")  # trace(A) * I
    assert_close(res, expected)


def chain(*args: Tensor | str):
    if len(args) % 2 != 1:
        raise ValueError("Number of arguments must be odd")
    for i, arg in enumerate(args):
        if i % 2 == 0 and not isinstance(arg, Tensor) or i % 2 == 1 and not isinstance(arg, str):
            raise ValueError("Arguments must be alternating between tensors and strings")
    t = args[0]
    for i in range(1, len(args), 2):
        t = F.dot(t, args[i + 1], args[i])
    return t


def test_quartic():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    A = Variable("A", j=j, j1=j)
    B = Variable("B", i=i, i1=i)
    C = Variable("C", j=j, j1=j)

    # X @ A @ X.T @ B @ X.T @ C @ X
    # Maybe having a chain function like this would be useful:
    # expr = chain(("i0", "i"), X, "j", A, ("j1", "j"), X, "i", B, ("i1", "i"), X, "j", C, ("j1", "j"), X)
    expr = (
        X.rename(i="i0", j="j")
        @ A
        @ X.rename(j="j1", i="i")
        @ B
        @ X.rename(i="i1", j="j")
        @ C
        @ X.rename(j="j1", i="i")
    )

    mu = Zero(i, j)
    expr = Expectation(expr, X, mu).full_simplify()

    ts = rand_values([X, A, B, C], {i: 2, j: 3})
    ts[A] = ts[A] ** 2
    ts[B] = ts[B] ** 2
    ts[C] = ts[C] ** 2
    tA, tB, tC = ts[A].rename(None), ts[B].rename(None), ts[C].rename(None)

    res = expr.evaluate(ts)
    expected = (
        tA.trace() * tC.trace() * tB
        + (tA.T @ tC).trace() * tB.T
        + (tA @ tC).trace() * tB.trace() * torch.eye(2)
    ).rename("i0", "i")
    assert_close(res, expected)


def test_quartic2():
    # Compute E[X @ A @ X.T @ B @ X.T @ C @ X]
    # Based on https://mathematica.stackexchange.com/questions/273893

    i, j = symbols("i j")
    X = Variable("X", i, j)
    A = Variable("A", j, j1=j)
    B = Variable("B", i, i1=i)
    C = Variable("C", j, j1=j)
    expr = X.rename(i="i0") @ A @ X.rename(j="j1") @ B @ X.rename(i="i1") @ C @ X.rename(j="j1")

    # Mean and row-wise covariance
    M = Variable("M", i, j)
    Sh = Variable("Sh", j, r=j)
    S = (Sh.rename(j="j0") @ Sh).rename(j0="j", j="l")  # S = Sh @ Sh.T
    covar = Delta(i, "i, k") @ S

    assert covar.edges == {"i", "k", "j", "l"}
    expr = Expectation(expr, X, M, covar, {"i": "k", "j": "l"}).full_simplify()

    i_val, j_val = 2, 3
    ts = rand_values([X, A, B, C, M, Sh], {i: i_val, j: j_val})
    # We square A, B, and C to prevent the expectation from being 0
    ts[A] = ts[A] ** 2
    ts[B] = ts[B] ** 2
    ts[C] = ts[C] ** 2
    tA, tB, tC, tSh = ts[A].rename(None), ts[B].rename(None), ts[C].rename(None), ts[Sh].rename(None)

    m = 10**6
    X = torch.randn(m, i_val, j_val) @ tSh.T

    expected_covar = torch.cov(X.reshape(-1, j_val).T).rename("j", "l")
    assert_close(S.evaluate(ts), expected_covar, rtol=0.05, atol=1e-2)

    X += ts[M].rename(None)

    expected = (
        einsum(X, tA, X, tB, X, tC, X, "b i0 j, j j1, b i1 j1, i1 i2, b i2 j2, j2 j3, b i j3 -> b i0 i")
        .mean(0)
        .rename("i0", "i")
    )

    res = expr.evaluate(ts)
    assert_close(res, expected, rtol=0.1, atol=0.1)


def test_x():
    p0 = symbols("p0")
    S = Variable("S", p0)
    S1 = S.rename(p0="p1")
    # out_p2 = sum_{p1, p1} S_{p0} S_{p1} [p0 = p1 = p2]
    #        = S_{p2} S_{p2}
    #        = S_{p2}^2
    # E[out]_p2 = E[out_p2] = E[S_{p2}^2] = 1
    expr = Product([S, S1, Delta(p0, "p0, p1, p2")])
    expr = Expectation(expr, S)
    assert expr.simplify() == Delta(p0, "p2")


def test_triple_S():
    # This test comes from a failed attempt to give an "expected value" Strassen algorithm
    # out_{i, j, m, n, r, s}
    # = sum_p E[S_ijp S_mnp S_rsp]
    # = sum_p E[(T_ijp+[i=j=p]) (T_mnp+[m=n=p]) (T_rsp+[r=s=p])]
    # = sum_p E[
    #     [i=j=p][m=n=p][r=s=p]
    #     + [i=j=p][m=n=p] T_rsp + [m=n=p][r=s=p] T_ijp + [i=j=p][r=s=p]T_mnp
    #     + [r=s=p] T_ijp T_mnp + [m=n=p] T_ijp T_rsp + [i=j=p] T_mnp T_rsp
    #     + T_ijp T_mnp T_rsp
    #     ]
    # = [i=j=m=n=r=s]
    # + 0
    # + [r=s][i=m][j=n] + [m=n][i=r][j=s] + [i=j][m=r][n=s]
    # + 0
    i = symbols("i")
    S = Variable("S", i, j=i, p=i)
    SA = S.rename(p="p1")
    SB = S.rename(p="p2", i="m", j="n")
    W = S.rename(p="p3", i="r", j="s")
    expr = Product([SA, SB, W, Delta(i, "p1, p2, p3")])
    expr = Expectation(expr, S, Delta(i, "i, j, p"))
    expr = expr.full_simplify()
    expected = Sum(
        [
            Delta(i, "i, j, m, n, r, s"),
            Product([Delta(i, "j, i"), Delta(i, "r, m"), Delta(i, "s, n")]),
            Product([Delta(i, "r, s"), Delta(i, "n, j"), Delta(i, "m, i")]),
            Product([Delta(i, "m, n"), Delta(i, "s, j"), Delta(i, "r, i")]),
        ],
    )
    assert expr == expected


def test_strassen():
    i, k, m, p = symbols("i k m p")
    S = Variable("S", i, p)
    T = Variable("T", k, p)
    U = Variable("U", m, p)

    # The idea is to create the three strassen tensors using an element-wise product
    ST = S * T.rename(k="l")  # (p, i, l=k)
    TU = T * U.rename(m="n")
    US = U * S.rename(i="j")

    # Triple product over p
    expr = Product(
        [
            ST.rename(p="p1"),
            TU.rename(p="p2"),
            US.rename(p="p3"),
            Delta(p, "p1, p2, p3"),
        ]
    )
    expr = Expectation(Expectation(Expectation(expr, S), T), U)
    expr = expr.full_simplify()
    assert expr == Product(
        [
            Delta(i, "i, j"),
            Delta(k, "k, l"),
            Delta(m, "m, n"),
            # The product is scaled by a factor |p| which we "implement" using an empty copy tensor
            Delta(p),
        ]
    )


def test_outer_product():
    i = symbols("i")
    u = Variable("u", i)
    prods = [
        Expectation(Product([u.rename(i=f"i{i}") for i in range(k)]), u).full_simplify() for k in range(1, 5)
    ]
    assert prods[0] == Zero(i0=i)
    assert prods[1] == Delta(i, "i0, i1")
    assert prods[2] == Zero(i0=i, i1=i, i2=i)
    assert prods[3] == (F.symmetrize(Delta(i, "i0, i1") @ Delta(i, "i2, i3")) / 8).full_simplify()
