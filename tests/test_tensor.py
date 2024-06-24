import pytest
from tensorgrad.functions import frobenius2
from tensorgrad.tensor import Variable, Function, Copy, Zero, Product, Sum, Ones
from tensorgrad.testutils import assert_close, rand_values


def test_x():
    x = Variable("x", ["x"])
    assert x.grad(x, ["x_"]).edges == ["x", "x_"]


def test_xy():
    x = Variable("x", ["i"])
    y = Variable("y", ["i"])
    xy = x @ y
    assert xy.grad(x, ["i_"]).edges == ["i_"]
    assert xy.grad(y, ["i_"]).edges == ["i_"]


def test_lstsq():
    # ||Ax - y||_2^2
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    A = Variable("A", ["x", "y"])
    Ax = A @ x
    assert Ax.edges == ["y"]
    Axmy = Ax - y
    assert Axmy.edges == ["y"]
    F = frobenius2(Axmy)
    assert F.edges == []
    grad = F.grad(x, ["x_"])
    assert grad.edges == ["x_"]


def test_identity():
    e = Copy(["a", "b"])
    assert e.edges == ["a", "b"]
    assert e.grad(Variable("x", ["x"]), ["x_"]) == Zero(["a", "b", "x_"])


def test_zero():
    z = Zero(["a", "b"])
    assert z.edges == ["a", "b"]
    assert z.grad(Variable("x", ["x"]), ["x_"]) == Zero(["a", "b", "x_"])
    # assert z == Zero(["b", "a"])
    assert z != Zero(["a", "b", "c"])


def test_variable_grad():
    x = Variable("x", ["i", "j"])
    y = Variable("y", ["k"])
    assert x.grad(x, ["i_", "j_"]) == Product([Copy(["i", "i_"]), Copy(["j", "j_"])])
    assert x.grad(y, ["k_"]) == Zero(["i", "j", "k_"])


def test_contraction():
    x = Variable("x", ["i", "j"])
    y = Variable("y", ["j", "k"])
    c = Product([x, y])
    assert c.edges == ["i", "k"]
    assert c.contractions == ["j"]
    assert c.grad(x, ["i_", "j_"]).simplify() == Copy(["i", "i_"]) @ y.rename({"j": "j_"})
    assert c.grad(y, ["j_", "k_"]).simplify() == x.rename({"j": "j_"}) @ Copy(["k", "k_"])


def test_linear_combination():
    x = Variable("x", ["i"])
    y = Variable("y", ["i"])
    lc = Sum([x, y], [2, -3])
    assert lc.edges == ["i"]
    assert lc.grad(x, ["i_"]).simplify() == Sum([Copy(["i", "i_"])], [2])
    assert lc.grad(y, ["i_"]).simplify() == Sum([Copy(["i", "i_"])], [-3])


def test_simplify():
    x = Variable("x", ["i"])
    y = Variable("y", ["i"])
    z = Variable("z", ["i"])
    zero = Zero(["i"])

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
    assert c2.simplify() == Zero([])

    c3 = x @ Copy(["i", "j"])
    assert c3.simplify() == x.rename({"i": "j"})

    c4 = (x @ Copy(["i", "j"])) @ y
    assert c4.simplify() == Product([x.rename({"i": "j"}), y])


def test_multiplication():
    x = Variable("x", ["i", "j"])
    y = Variable("y", ["j", "k"])
    z = x * y
    assert isinstance(z, Product)
    assert z.edges == ["i", "k", "j"]


def test_subtraction():
    x = Variable("x", ["i"])
    y = Variable("y", ["i"])
    z = x - y
    assert isinstance(z, Sum)
    assert len(z.tensors) == 2
    assert z.weights == [1, -1]
    assert z.edges == ["i"]


def test_inner_product_grad():
    x = Variable("x", ["i"])
    y = Variable("y", ["i"])
    z = x @ y
    assert z.edges == []
    assert z.grad(x).simplify() == y.rename({"i": "i_"})
    assert z.grad(y).simplify() == x.rename({"i": "i_"})


def test_gradient_variable_self():
    x = Variable("x", ["x"])
    assert x.grad(x, ["x_"]).edges == ["x", "x_"]

    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    assert x.grad(y, ["y_"]) == Zero(["x", "y_"])

    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    z = x + y
    assert set(z.grad(x, ["x_"]).edges) == {"x_", "x", "y"}

    # Gradient of a contraction operation with respect to one of its operands should adjust edges appropriately
    x = Variable("x", ["i"])
    y = Variable("y", ["i"])
    z = x @ y
    assert z.grad(x).edges == ["i_"]

    # Gradient through an identity tensor should return Zero when the variable does not match
    x = Variable("x", ["x"])
    I = Copy(["x", "x_"])
    result = I.grad(x)
    assert isinstance(result, Zero)

    # Gradient of a Zero tensor with respect to any variable should be Zero
    x = Variable("x", ["x"])
    z = Zero(["x"])
    assert isinstance(z.grad(x), Zero)

    # Complex operation gradient should correctly handle the combination of operations
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    z = (x + y) @ x - y
    assert set(z.grad(x).edges) == {"x_", "y"}


def test_square_ip():
    x = Variable("x", ["i"])
    x2 = x @ x
    assert x2.edges == []
    assert x2.grad(x).simplify() == Sum([x.rename({"i": "i_"})], [2])


def test_square_xAAx():
    x = Variable("x", ["x"])
    A = Variable("A", ["x", "y"])
    Ax = A @ x
    xAAx = Ax @ Ax
    assert xAAx.edges == []
    assert xAAx.grad(x).edges == ["x_"]


def test_hessian():
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    A = Variable("A", ["x", "y"])
    F = frobenius2(A @ x - y)
    hess = F.grad(x).grad(x)
    assert set(hess.edges) == {"x_", "x__"}


def test_square_grad():
    x = Variable("x", ["i"])
    y = x * x
    assert y.edges == ["i"]
    assert set(y.grad(x).edges) == {"i", "i_"}


def test_quadratic_grad():
    x = Variable("x", ["i"])
    A = Variable("A", ["j", "i"])
    y = frobenius2(A @ x)
    assert y.edges == []
    assert y.grad(x).edges == ["i_"]


def test_equality():
    p1 = Product(
        [
            Variable("A", ["j", "i_"], ["j", "i"]),
            Variable("A", ["j", "i"], ["j", "i"]),
            Variable("x", ["i"], ["i"]),
        ]
    )
    p2 = Product(
        [
            Variable("A", ["j", "i"], ["j", "i"]),
            Variable("x", ["i"], ["i"]),
            Variable("A", ["j", "i_"], ["j", "i"]),
        ]
    )
    assert p1 == p2


def test_func_grad():
    # Gradient of a function with respect to its variable should adjust edges appropriately
    x = Variable("x", ["x"])
    f = Function("f", [], (x, "x"))
    assert f.grad(x, ["x_"]).edges == ["x_"]


def test_two_func_grad():
    # Gradient of a function with respect to its variable should adjust edges appropriately
    x = Variable("x", ["x"])
    v = Function("v", ["y"], (x, "x"))
    f = Function("f", [], (v, "y"))
    assert f.grad(x).edges == ["x_"]


def test_matrix_grad():
    X = Variable("X", ["i", "j"])
    assert X.grad(X) == Copy(["i", "i_"]) @ Copy(["j", "j_"])


def test_broadcasting():
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    z = x + y
    assert set(z.edges) == {"x", "y"}
    assert set(z.grad(x).simplify().edges) == {"x", "y", "x_"}


def test_simplify_ones():
    x = Variable("x", ["x"])
    broadcasted = Product([Product([x, Ones(["y"])]), Ones([])])
    assert broadcasted.edges == ["x", "y"]
    assert broadcasted.simplify() == Product([x, Copy(["y"])])


def test_simplify_ones_deeper():
    x = Variable("x", ["x"])
    broadcasted = Product([Product([Product([x, Ones(["y"])]), Ones([])]), Ones([])])
    assert broadcasted.edges == ["x", "y"]
    assert broadcasted.simplify() == Product([x, Copy(["y"])])


def test_broadcasting2():
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    z = frobenius2(x + y)
    assert z.edges == []
    assert z.grad(x).edges == ["x_"]
    assert z.grad(x).simplify().edges == ["x_"]
    actual = z.grad(x)
    # { 2 (x1+y1) + 2 (x1+y2) }
    # { 2 (x2+y1) + 2 (x2+y2) }
    expected = 2 * ((x.rename({"x": "x_"}) + y) @ Copy(["y"]))
    assert actual.simplify() == expected.simplify()


def test_broadcast_zero_rank_ones():
    # Tests for a bug where the simplification would leave dangling zero-rank Ones tensors
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    t = x + y
    assert "Ones([])" not in repr(t.simplify())


def test_pseudo_linear_gradient():
    # Derivation of Peyman Milanfar’s gradient, d[A(x)x]
    x = Variable("x", ["i"])
    A = Function("A", ["i", "j"], (x, "i"))
    expr = (A @ x).grad(x).simplify()
    assert set(expr.edges) == {"j", "i_"}
    D_0A = Function("D_0A", ["i", "j", "i_"], (x, "i"))
    assert expr == (D_0A @ x + A.rename({"i": "i_"})).simplify()


def test_hash():
    # Variables with different (original) edge names are eseentially different types
    xi = Variable("x", ["i"])
    xj = Variable("x", ["j"])
    assert hash(xi) != hash(xj)

    # Inner product and outer product should have different hashes
    assert hash(Product([xi, xi])) != hash(Product([xi, xj]))
    # Ordering does not affect hash
    assert hash(Product([xi, xj])) == hash(Product([xj, xi]))

    # Reordering of edges should not affect the basic hash
    assert hash(Variable("x", ["i", "j"])) == hash(Variable("x", ["j", "i"]))

    # Again: Different (original) names are different types
    assert hash(Variable("x", ["i", "j"])) != hash(Variable("x", ["i", "k"]))


def test_equal():
    v = Variable("x", ["x"])
    assert Product([Copy(["y"]), v]).is_isomorphic(Product([v, Copy(["y"])]))

    s = Sum(
        [
            Product([Variable("x", ["x"], ["x_"]), Copy(["y"])]),
            Product([Variable("y", ["y"], ["y"]), Copy(["x_"])]),
        ],
        (1, 1),
    )

    assert Product([Copy(["y"]), s]).is_isomorphic(Product([s, Copy(["y"])]))


def test_size0():
    v = Variable("x", ["x"])
    shapes = {v: {"x": 3}}

    assert v._compute_edge_dims(shapes)[id(v)] == {"x": 3, ("Original", "x"): 3}

    c = Copy(["x", "y"])
    assert Product([c, v])._compute_edge_dims(shapes)[id(c)] == {"x": 3, "y": 3}
    assert Product([v, c])._compute_edge_dims(shapes)[id(c)] == {"x": 3, "y": 3}


def test_size1():
    v = Variable("x", ["i0"])
    shapes = {v: {"i0": 3}}

    t0 = Product([v, Copy(["i1", "i2"])])
    t1 = Product([Copy(["i0", "i1"])])

    assert (t0 @ t1)._compute_edge_dims(shapes)[id(t0)] == {"i0": 3, "i1": 3, "i2": 3}


def test_size2():
    v = Variable("x", ["i0"])
    shapes = {v: {"i0": 3}}
    t0 = Product([v, Copy(["i1", "i2"])])
    t1 = Product([Copy(["i0", "i1"]), Copy(["i2", "i3"])])

    assert (t0 @ t1)._compute_edge_dims(shapes)[id(t1)] == {
        "i0": 3,
        "i1": 3,
        "i2": 3,
        "i3": 3,
    }


def test_size3():
    v = Variable("x", ["i0"])
    shapes = {v: {"i0": 3}}
    t0 = Product([v, Copy(["i1", "i2"]), Copy(["i3", "i4"])])
    t1 = Product([Copy(["i0", "i1"]), Copy(["i2", "i3"]), Copy(["i4", "i5"])])
    assert (t0 @ t1)._compute_edge_dims(shapes)[id(t1)] == {
        "i0": 3,
        "i1": 3,
        "i2": 3,
        "i3": 3,
        "i4": 3,
        "i5": 3,
    }


def test_sizes():
    v = Variable("x", ["i0"])
    shapes = {v: {"i0": 3}}
    t0 = v
    t1 = Copy([])
    for i in range(0, 10, 2):
        t1 = t1 @ Copy([f"i{i}", f"i{i+1}"])
        t0 = t0 @ Copy([f"i{i+1}", f"i{i+2}"])
        assert (t0 @ t1)._compute_edge_dims(shapes)[id(t0)] == {f"i{j}": 3 for j in range(0, i + 3)}


def test_transpose():
    x = Variable("x", "i, j")
    ts = rand_values([x], i=3, j=3)

    expr = x + x.rename({"i": "j", "j": "i"})

    res = expr.evaluate(ts)
    expected = ts[x].rename(None) + ts[x].rename(None).T
    assert_close(res, expected.rename("i", "j"))
    res2 = expr.simplify().evaluate(ts)
    assert_close(res2, expected.rename("i", "j"))


def test_transpose_mismatched():
    for string in ["i, j", "j, i"]:
        x = Variable("x", string)
        expr = x + x.rename({"i": "j", "j": "i"})
        ts = rand_values([x], i=3, j=2)
        with pytest.raises(ValueError):
            expr.evaluate(ts)
        with pytest.raises(ValueError):
            expr.simplify().evaluate(ts)
