import pytest
from sympy import symbols
from tensorgrad.functions import frobenius2
from tensorgrad.tensor import Variable, Function, Delta, Zero, Product, Sum, Ones, function
from tensorgrad.testutils import assert_close, rand_values


def test_x():
    x = symbols("x")
    x = Variable("x", x)
    assert x.grad(x, {"x": "x_"}).edges == {"x", "x_"}


def test_xy():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    xy = x @ y
    assert xy.grad(x, {"i": "i_"}).edges == {"i_"}
    assert xy.grad(y, {"i": "i_"}).edges == {"i_"}
    assert xy.grad(x).simplify() == y
    assert xy.grad(y).simplify() == x


def test_lstsq():
    i, j = symbols("i j")
    # ||Ax - y||_2^2
    x = Variable("x", i)
    y = Variable("y", j)
    A = Variable("A", i, j)
    Ax = A @ x
    assert Ax.edges == {"j"}
    Axmy = Ax - y
    assert Axmy.edges == {"j"}
    F = frobenius2(Axmy)
    assert F.edges == set()
    grad = F.grad(x, {"i": "i_"})
    assert grad.edges == {"i_"}


def test_identity():
    i, j = symbols("i j")
    e = Delta(i, "a", "b")
    assert e.edges == {"a", "b"}
    x = Variable("x", j)
    assert e.grad(x, {"j": "j_"}) == Zero(a=i, b=i, j_=j)


def test_zero():
    a, b, i = symbols("a b i")
    z = Zero(a, b)
    assert z.edges == {"a", "b"}
    x = Variable("x", i)
    assert z.grad(x, {"i": "i_"}) == Zero(a, b, i_=i)
    assert z != Zero(a, b, symbols("c"))


def test_variable_grad():
    i, j, k = symbols("i j k")
    x = Variable("x", i, j)
    y = Variable("y", k)
    expected = Product([Delta(i, "i, i_"), Delta(j, "j, j_")])
    assert x.grad(x, {"i": "i_", "j": "j_"}).simplify() == expected
    assert x.grad(y, {"k": "k_"}).simplify() == Zero(i, j, k_=k)


def test_contraction():
    i, j, k = symbols("i j k")
    x = Variable("x", i, j)
    y = Variable("y", j, k)
    c = Product([x, y])
    assert c.edges == {"i", "k"}
    assert c.grad(x, {"i": "i_", "j": "j_"}).simplify() == Delta(i, "i, i_") @ y.rename(j="j_")
    assert c.grad(y, {"j": "j_", "k": "k_"}).simplify() == x.rename(j="j_") @ Delta(k, "k, k_")


def test_linear_combination():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    lc = Sum([x, y], [2, -3])
    assert lc.edges == {"i"}
    assert lc.grad(x, {"i": "i_"}).simplify() == Sum([Delta(i, "i, i_")], [2])
    assert lc.grad(y, {"i": "i_"}).simplify() == Sum([Delta(i, "i, i_")], [-3])


def test_simplify():
    i, j = symbols("i j")
    x = Variable("x", i)
    y = Variable("y", i)
    z = Variable("z", i)
    zero = Zero(i)

    # Test simplifying linear combinations
    lc1 = Sum([x, y, z], [1, 2, 3])
    assert lc1.simplify() == Sum([x, y, z], [1, 2, 3])

    lc2 = Sum([x, zero], [1, 2])
    assert lc2.simplify() == x

    lc3 = Sum([x, y, z], [1, 0, 0])
    assert lc3.simplify() == x

    # Test simplifying contractions
    c1 = Product([x, y])
    assert c1.simplify() == c1

    c2 = Product([x, zero])
    assert c2.simplify() == Zero()

    c3 = x @ Delta(i, "i, j")
    assert c3.simplify() == x.rename(i="j")

    c4 = (x @ Delta(i, "i, j")) @ y
    assert c4.simplify() == Product([x.rename(i="j"), y])


def test_multiplication():
    i, j, k = symbols("i j k")
    x = Variable("x", i, j)
    y = Variable("y", j, k)
    z = x * y
    assert isinstance(z, Product)
    assert z.edges == {"i", "j", "k"}


def test_subtraction():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    z = x - y
    assert isinstance(z, Sum)
    assert len(z.tensors) == 2
    assert z.weights == [1, -1]
    assert z.edges == {"i"}


def test_inner_product_grad():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    z = x @ y
    assert z.edges == set()
    assert z.grad(x).simplify() == y.rename(i="i_")
    assert z.grad(y).simplify() == x.rename(i="i_")


def test_gradient_variable_self():
    x = symbols("x")
    x_var = Variable("x", x)
    assert x_var.grad(x_var, {"x": "x_"}).edges == {"x", "x_"}

    y = symbols("y")
    y_var = Variable("y", y)
    assert x_var.grad(y_var, {"y": "y_"}) == Zero(x, y_=y)

    z = x_var + y_var
    assert set(z.grad(x_var, {"x": "x_"}).edges) == {"x_", "x", "y"}

    # Gradient through an identity tensor should return Zero when the variable does not match
    x_var = Variable("x", x)
    I = Delta(x, "x, x_")
    result = I.grad(x_var)
    assert isinstance(result, Zero)

    # Gradient of a Zero tensor with respect to any variable should be Zero
    z = Zero(x)
    assert isinstance(z.grad(x_var), Zero)

    # Complex operation gradient should correctly handle the combination of operations
    z = (x_var + y_var) @ x_var - y_var
    assert set(z.grad(x_var).edges) == {"x", "y"}


def test_square_ip():
    i = symbols("i")
    x = Variable("x", i)
    x2 = x @ x
    assert x2.edges == set()
    assert x2.grad(x).simplify() == Sum([x.rename(i="i_")], [2])


def test_square_xAAx():
    x, y = symbols("x y")
    x_var = Variable("x", x)
    A = Variable("A", x, y)
    Ax = A @ x_var
    xAAx = Ax @ Ax
    assert xAAx.edges == set()
    assert xAAx.grad(x_var).edges == {"x"}


def test_hessian():
    x, y = symbols("x y")
    x_var = Variable("x", x)
    y_var = Variable("y", y)
    A = Variable("A", x, y)
    F = frobenius2(A @ x_var - y_var)
    hess = F.grad(x_var).grad(x_var)
    assert set(hess.edges) == {"x", "x_"}


def test_square_grad():
    i = symbols("i")
    x = Variable("x", i)
    y = x * x
    assert y.edges == {"i"}
    assert set(y.grad(x).edges) == {"i", "i_"}


def test_quadratic_grad():
    i, j = symbols("i j")
    x = Variable("x", i)
    A = Variable("A", j, i)
    y = frobenius2(A @ x)
    assert y.edges == set()
    assert y.grad(x).edges == {"i"}


def test_equality():
    i, j = symbols("i j")
    p1 = Product(
        [
            Variable("A", j, i_=i),
            Variable("A", j, i),
            Variable("x", i),
        ]
    )
    p2 = Product(
        [
            Variable("A", j, i),
            Variable("x", i),
            Variable("A", j, i_=i),
        ]
    )
    assert p1 == p2


def test_func_grad():
    x = symbols("x")
    x_var = Variable("x", x)
    f = function("f", {}, (x_var, "x"))
    assert f.grad(x_var, {"x": "x_"}).edges == {"x_"}


def test_two_func_grad():
    x, y = symbols("x y")
    x_var = Variable("x", x)
    v = function("v", {"y": y}, (x_var, "x"))
    f = function("f", {}, (v, "y"))
    assert f.grad(x_var).edges == {"x"}


def test_matrix_grad():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    assert X.grad(X) == Delta(i, "i, i_") @ Delta(j, "j, j_")


def test_broadcasting():
    x, y = symbols("x y")
    x_var = Variable("x", x)
    y_var = Variable("y", y)
    z = x_var + y_var
    assert set(z.edges) == {"x", "y"}
    assert set(z.grad(x_var).simplify().edges) == {"x", "y", "x_"}


def test_simplify_ones():
    x, y = symbols("x y")
    x_var = Variable("x", x)
    broadcasted = Product([Product([x_var, Ones(y)]), Ones()])
    assert broadcasted.edges == {"x", "y"}
    assert broadcasted.simplify() == Product([x_var, Delta(y, "y")])


def test_simplify_ones_deeper():
    x, y = symbols("x y")
    x_var = Variable("x", x)
    broadcasted = Product([Product([Product([x_var, Ones(y)]), Ones()]), Ones()])
    assert broadcasted.edges == {"x", "y"}
    assert broadcasted.simplify() == Product([x_var, Delta(y, "y")])


def test_broadcasting2():
    x, y = symbols("x y")
    x_var = Variable("x", x)
    y_var = Variable("y", y)
    z = frobenius2(x_var + y_var)
    assert z.edges == set()
    assert z.grad(x_var).edges == {"x"}
    assert z.grad(x_var).simplify().edges == {"x"}
    actual = z.grad(x_var)
    expected = 2 * ((x_var.rename(x="x_") + y_var) @ Delta(y, "y"))
    assert actual.simplify() == expected.simplify()


def test_broadcast_zero_rank_ones():
    x, y = symbols("x y")
    x_var = Variable("x", x)
    y_var = Variable("y", y)
    t = x_var + y_var
    assert "Ones()" not in repr(t.simplify())


def test_pseudo_linear_gradient():
    i, j = symbols("i j")
    x = Variable("x", i)
    A = function("A", {"i": i, "j": j}, (x, "i"))
    expr = (A @ x).grad(x).simplify()
    assert expr.edges == {"j", "i"}
    D_0A = function("D_0A", {"i": i, "j": j, "i_": i}, (x, "i")).rename(i_="i", i="i_")
    expected = (D_0A @ x.rename(i="i_") + A).simplify()
    assert expr == expected


def test_hash():
    i, j, k = symbols("i j k")
    # Variables with different (original) edge names are essentially different types
    xi = Variable("x", i)
    xj = Variable("x", j)
    assert hash(xi) != hash(xj)

    # Inner product and outer product should have different hashes
    assert hash(Product([xi, xi])) != hash(Product([xi, xj]))
    # Ordering does not affect hash
    assert hash(Product([xi, xj])) == hash(Product([xj, xi]))

    # Reordering of edges should not affect the basic hash
    assert hash(Variable("x", i, j)) == hash(Variable("x", j, i))

    # Again: Different (original) names are different types
    assert hash(Variable("x", i, j)) != hash(Variable("x", i, k))


def test_equal():
    x, y = symbols("x y")
    v = Variable("x", x)
    cp = Delta(y, "y")
    assert (cp @ v).is_isomorphic(v @ cp)

    s = Sum(
        [
            Product([Variable("x", x), Delta(y, "y")]),
            Product([Variable("y", y), Delta(x, "x_")]),
        ]
    )
    assert (cp @ s).is_isomorphic(s @ cp)


def test_transpose():
    i = symbols("i")
    x = Variable("x", i, j=i)
    ts = rand_values([x], {i: 3})

    expr = x + x.rename(i="j", j="i")

    res = expr.evaluate(ts)
    expected = ts[x].rename(None) + ts[x].rename(None).T
    assert_close(res, expected.rename("i", "j"))
    res2 = expr.simplify().evaluate(ts)
    assert_close(res2, expected.rename("i", "j"))


def test_transpose_mismatched():
    i, j = symbols("i j")
    for symbols_tuple in [(i, j), (j, i)]:
        x = Variable("x", *symbols_tuple)
        with pytest.raises(ValueError):
            _ = x + x.rename(i="j", j="i")
