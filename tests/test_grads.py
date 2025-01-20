import torch
from sympy import symbols
from tensorgrad import Variable
import tensorgrad.functions as F
from tensorgrad.tensor import Delta, Product, function
from tensorgrad.testutils import assert_close, rand_values


def test_simple_grad():
    """Test the gradient of a simple matrix-vector multiplication."""
    x_sym, y_sym = symbols("x y")
    x = Variable("x", x_sym)
    A = Variable("A", x_sym, y_sym)
    ts = rand_values([x, A], {x_sym: 2, y_sym: 3})
    res = (A @ x).grad(x).simplify()
    assert_close(res.evaluate(ts), ts[A].rename("x", "y"))


def test_simple_hessian():
    """Test Hessian computations for quadratic forms and Frobenius norm."""
    i = symbols("i")
    x = Variable("x", x=i)
    A = Variable("A", x=i, y=i)
    ts = rand_values([x, A], {i: 2})

    # Test quadratic form x^T A x
    quad = x @ A @ x.rename(x="y")
    res = quad.grad(x).grad(x).simplify()
    assert res == F.symmetrize(A).simplify()
    ApAt = (ts[A].rename(None) + ts[A].rename(None).T).rename("x", "x_")
    assert_close(res.evaluate(ts), ApAt)

    # Test x^T A^T A x
    quad2 = x @ A @ A @ x
    res2 = quad2.grad(x).grad(x).simplify()
    assert res2.symmetries == {frozenset({"x", "x_"})}
    tA = ts[A].rename(None)
    torch.testing.assert_close(res2.evaluate(ts).rename(None), 2 * tA @ tA.T)

    # Test Frobenius norm ||Ax - y||^2
    j = symbols("j")
    y = Variable("y", y=j)
    A = Variable("A", x=i, y=j)
    ts = rand_values([x, y, A], {i: 2, j: 3})
    frob = F.frobenius2(A @ x - y)
    hess = frob.grad(x).grad(x).simplify().evaluate(ts)
    tH = 2 * ts[A].rename(None) @ ts[A].rename(None).T
    assert_close(hess, tH.rename("x", "x_"))


def test_derivative_of_hadamard_product():
    """Test the derivative of element-wise (Hadamard) product."""
    i, j = symbols("i j")
    a = Variable("a", i, j)
    b = Variable("b", i, j)
    ts = rand_values([a, b], {i: 2, j: 3})
    res_a = (a * b).grad(a).simplify().evaluate(ts)
    res_b = (a * b).grad(b).simplify().evaluate(ts)
    torch.testing.assert_close(res_a.sum(), ts[b].sum())
    torch.testing.assert_close(res_b.sum(), ts[a].sum())


def test_f_0_0():
    """Test gradient of a scalar function with scalar input."""
    x = Variable("x")
    f = function("f", {}, (x,))
    expr = f.grad(x).simplify()
    assert set(expr.edges) == set()
    assert expr == function("D_0f", {}, (x,))


def test_f_0_1():
    """Test gradient of a scalar function with vector input."""
    i = symbols("i")
    x = Variable("x", i)
    f = function("f", {}, (x, "i"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"i"}
    assert expr == function("D_0f", {"i_": i}, (x, "i")).rename(i_="i")


def test_f_0_2():
    """Test gradient of a scalar function with matrix input."""
    b, i = symbols("b i")
    x = Variable("x", b, i)
    f = function("f", {}, (x, "i"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"b", "b_", "i"}
    x_renamed = x.rename(b="b__")
    expected = function("D_0f", {"i_": i}, (x_renamed, "i")).rename(i_="i") @ Delta(b, "b, b_, b__")
    assert expr == expected


def test_f_1_0():
    """Test gradient of a vector function with scalar input."""
    y = symbols("y")
    x = Variable("x")
    f = function("f", {"y": y}, (x,))
    expr = f.grad(x).simplify()
    assert expr.edges == {"y"}
    assert expr == function("D_0f", {"y": y}, (x,))


def test_f_1_1():
    """Test gradient of a vector function with vector input."""
    i, y = symbols("i y")
    x = Variable("x", i)
    f = function("f", {"y": y}, (x, "i"))  # (x-i-f)-y-
    assert f.edges == {"y"}
    expr = f.grad(x).simplify()
    assert expr.edges == {"y", "i"}
    expected = function("D_0f", {"y": y, "i_": i}, (x, "i")).rename(i_="i")
    assert expr == expected


def test_fxy_1_1_1():
    """Test gradient of a vector function with two vector inputs."""
    i, z = symbols("i z")
    x = Variable("x", i)
    y = Variable("y", i)
    f = function("f", {"z": z}, (x, "i"), (y, "i"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"z", "i"}
    expected = function("D_0f", {"z": z, "i_": i}, (x, "i"), (y, "i")).rename(i_="i")
    assert expr == expected


def test_fxx_1_1_1():
    """Test gradient of a vector function with repeated vector input."""
    i, z = symbols("i z")
    x = Variable("x", i)
    f = function("f", {"z": z}, (x, "i"), (x, "i"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"z", "i"}
    t1 = function("D_0f", {"z": z, "i_": i}, (x, "i"), (x, "i")).rename(i_="i")
    t2 = function("D_1f", {"z": z, "i_": i}, (x, "i"), (x, "i")).rename(i_="i")
    expected = t1 + t2
    assert expr == expected


def test_f_2_2_multi_input():
    """Test gradient of a matrix function with multiple vector inputs."""
    i, j, k, l, a, b = symbols("i j k l a b")
    x = Variable("x", i, j)
    y = Variable("y", k, l)
    f = function("f", {"a": a, "b": b}, (x, "i"), (y, "k"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"a", "b", "l", "j", "j_", "i"}
    expected = function(
        "D_0f",
        {"a": a, "b": b, "i_": i},
        (x, "i"),
        (y, "k"),
    ).rename(j="j_0", i_="i") @ Delta(j, "j, j_, j_0")
    assert expr == expected


def test_f_1_1_nested():
    """Test gradient of nested functions with vector input and output."""
    i, j, k = symbols("i j k")
    x = Variable("x", i)
    g = function("g", {"j": j}, (x, "i"))
    f = function("f", {"k": k}, (g, "j"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"k", "i"}
    expected = Product(
        [
            function("D_0f", {"k": k, "j": j}, (g, "j")),
            function("D_0g", {"j": j, "i_": i}, (x, "i")).rename(i_="i"),
        ]
    )
    assert expr.edges == expected.edges
    assert expr == expected


def test_f_2_1_nested():
    """Test gradient of nested functions with vector input and matrix output."""
    i, j, k, l = symbols("i j k l")
    x = Variable("x", i)
    g = function("g", {"j": j, "k": k}, (x, "i"))
    f = function("f", {"l": l}, (g, "j", "k"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"l", "i"}
    expected = Product(
        [
            function("D_0f", {"l": l, "j": j, "k": k}, (g, "j", "k")),
            function("D_0g", {"j": j, "k": k, "i_": i}, (x, "i")).rename(i_="i"),
        ]
    )
    assert expr == expected


def test_f_1_2_nested():
    """Test gradient of nested functions with matrix input and vector output."""
    i, j, k, l, m = symbols("i j k l m")
    x = Variable("x", i, j)
    g = function("g", {"k": k}, (x, "i", "j"))
    f = function("f", {"l": l, "m": m}, (g, "k"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"l", "m", "i", "j"}
    expected = Product(
        [
            function(
                "D_0f",
                {"l": l, "m": m, "k": k},
                (g, "k"),
            ),
            function(
                "D_0g",
                {"k": k, "i_": i, "j_": j},
                (x, "i", "j"),
            ).rename(i_="i", j_="j"),
        ]
    )
    assert expr == expected


def test_f_2_2_nested():
    """Test gradient of nested functions with matrix input and output."""
    i, j, k, l, m, n = symbols("i j k l m n")
    x = Variable("x", i, j)
    g = function("g", {"k": k, "l": l}, (x, "i", "j"))
    f = function("f", {"m": m, "n": n}, (g, "k", "l"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"m", "n", "i", "j"}
    expected = Product(
        [
            function(
                "D_0f",
                {"m": m, "n": n, "l": l, "k": k},
                (g, "k", "l"),
            ),
            function(
                "D_0g",
                {"k": k, "l": l, "j_": j, "i_": i},
                (x, "i", "j"),
            ).rename(i_="i", j_="j"),
        ]
    )
    assert expr == expected


def test_f_2_2_double_nested():
    """Test gradient of double nested functions with matrix input and output."""
    i, j, k, l, m, n, o, p = symbols("i j k l m n o p")
    x = Variable("x", i, j)
    h = function("h", {"k": k, "l": l}, (x, "i", "j"))
    g = function("g", {"m": m, "n": n}, (h, "k", "l"))
    f = function("f", {"o": o, "p": p}, (g, "m", "n"))
    expr = f.grad(x).simplify()
    assert expr.edges == {"o", "p", "i", "j"}

    expected = Product(
        [
            function(
                "D_0f",
                {"o": o, "p": p, "n": n, "m": m},
                (g, "m", "n"),
            ).rename(n="n_1", m="m_0"),
            function(
                "D_0g",
                {"m": m, "n": n, "k": k, "l": l},
                (h, "k", "l"),
            ).rename(l="l_0", k="k_1", n="n_1", m="m_0"),
            function(
                "D_0h",
                {"k": k, "l": l, "i_": i, "j_": j},
                (x, "i", "j"),
            ).rename(i_="i", j_="j", l="l_0", k="k_1"),
        ]
    )
    assert expr == expected
