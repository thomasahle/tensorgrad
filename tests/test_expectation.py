import torch
from einops import einsum
from tensorgrad import Variable
from tensorgrad.extras.expectation import Expectation
from tensorgrad import Copy, Product, Zero
from tensorgrad.tensor import Sum
from tensorgrad.testutils import assert_close, rand_values


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
    expr = Expectation(expr, X, mu, covar).full_simplify()

    ts = rand_values([X, A, B, C], i=2, i0=2, i1=2, j=3, j1=3)
    ts[A] = ts[A] ** 2
    ts[B] = ts[B] ** 2
    ts[C] = ts[C] ** 2
    tA, tB, tC = ts[A].rename(None), ts[B].rename(None), ts[C].rename(None)

    res = expr.evaluate(ts, dims={"i": 2})
    expected = (
        tA.trace() * tC.trace() * tB
        + (tA.T @ tC).trace() * tB.T
        + (tA @ tC).trace() * tB.trace() * torch.eye(2)
    ).rename("i0", "i")
    assert_close(res, expected)


def test_quartic2():
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
    M = Variable("M", ["i", "j"])
    Sh = Variable("Sh", ["j", "l"])
    S = (Sh.rename({"j": "j0"}) @ Sh).rename({"j0": "j", "j": "l"})
    covar = Copy(["i", "k"]) @ S
    assert covar.edges == ["i", "k", "j", "l"]
    expr = Expectation(expr, X, M, covar).full_simplify()

    i, j = 2, 3
    ts = rand_values([X, A, B, C, M, Sh], i=i, i0=i, i1=i, j=j, j1=j, k=i, l=j)
    ts[A] = ts[A] ** 2
    ts[B] = ts[B] ** 2
    ts[C] = ts[C] ** 2
    tA, tB, tC, tSh = ts[A].rename(None), ts[B].rename(None), ts[C].rename(None), ts[Sh].rename(None)

    m = 10**6
    X = torch.randn(m, i, j) @ tSh.T

    expected_covar = torch.cov(X.reshape(-1, j).T).rename("j", "l")
    assert_close(S.evaluate(ts), expected_covar, rtol=0.05, atol=1e-2)

    print(f"{ts[M]=}")
    X += ts[M].rename(None)

    expected = (
        einsum(X, tA, X, tB, X, tC, X, "b i0 j, j j1, b i1 j1, i1 i2, b i2 j2, j2 j3, b i j3 -> b i0 i")
        .mean(0)
        .rename("i0", "i")
    )

    res = expr.evaluate(ts, dims={"i": i})
    assert_close(res, expected, rtol=0.05, atol=0.01)


def test_x():
    S = Variable("S", "p0")
    S1 = S.rename({"p0": "p1"})
    # out_p2 = sum_{p1, p1} S_{p0} S_{p1} [p0 = p1 = p2]
    #        = S_{p2} S_{p2}
    #        = S_{p2}^2
    # E[out]_p2 = E[out_p2] = E[S_{p2}^2] = 1
    expr = Product([S, S1, Copy("p0, p1, p2")])
    expr = Expectation(expr, S)
    assert expr.simplify() == Copy("p2")


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
    S = Variable("S", "i, j, p")
    SA = S.rename({"p": "p1"})
    SB = S.rename({"p": "p2", "i": "m", "j": "n"})
    W = S.rename({"p": "p3", "i": "r", "j": "s"})
    expr = Product([SA, SB, W, Copy("p1, p2, p3")])
    cov = Product([Copy("i, i2"), Copy("j, j2"), Copy("p, p2")])
    expr = Expectation(expr, S, Copy("i, j, p"), cov)
    expr = expr.full_simplify()
    expected = Sum(
        [
            Copy(["i", "j", "m", "n", "r", "s"]),
            Product([Copy(["j", "i"]), Copy(["r", "m"]), Copy(["s", "n"])]),
            Product([Copy(["r", "s"]), Copy(["n", "j"]), Copy(["m", "i"])]),
            Product([Copy(["m", "n"]), Copy(["s", "j"]), Copy(["r", "i"])]),
        ],
    )
    assert expr == expected


def test_strassen():
    S = Variable("S", "i, p")
    T = Variable("T", "k, p")
    U = Variable("U", "m, p")

    ST = S * T.rename({"k": "l"})
    TU = T * U.rename({"m": "n"})
    US = U * S.rename({"i": "j"})

    expr = Product(
        [
            ST.rename({"p": "p1"}),
            TU.rename({"p": "p2"}),
            US.rename({"p": "p3"}),
            Copy("p1, p2, p3"),
        ]
    )
    expr = Expectation(Expectation(Expectation(expr, S), T), U)
    expr = expr.full_simplify()
    assert expr == Product(
        [
            Copy("i, j"),
            Copy("k, l"),
            Copy("m, n"),
        ]
    )


Product(
    [
        Copy(["i", "j"]),
        Copy(["k", "l"]),
        Copy(["p_0_____", "p_"], link=Variable("U", ["n", "p_0_____"], orig=["m", "p"])),
        Copy(["p_0_____", "p_"]),
        Copy(["m", "n"]),
    ]
) == Product([Copy(["i", "j"]), Copy(["k", "l"]), Copy(["m", "n"])])
