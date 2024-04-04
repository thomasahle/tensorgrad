from functions import frobenius2
from tensor import Variable, Function, Copy, Zero, Product, Sum, Ones


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
    assert z == Zero(["b", "a"])
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
    assert c.grad(x, ["i_", "j_"]).simplify() == Product([Copy(["i", "i_"]), y.rename({"j": "j_"})])
    assert c.grad(y, ["j_", "k_"]).simplify() == Product([x.rename({"j": "j_"}), Copy(["k", "k_"])])


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
    assert any(isinstance(t, Copy) and t.edges == ["j", "j_", "j__"] for t in z.tensors)


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
    print(y)
    print("y grad", y.grad(x))
    assert y.edges == ["i"]
    # assert y.grad(x).simplify() == Sum([x, Identity(["i", "i_"])], [2, 2])
    assert set(y.grad(x).edges) == {"i", "i___"}


def test_quadratic_grad():
    x = Variable("x", ["i"])
    A = Variable("A", ["j", "i"])
    y = frobenius2(A @ x)
    assert y.edges == []
    print("y grad", y.grad(x).simplify())
    # To do this test well, we need graph isomorphism testing. Or maybe we can just use edge ordering?
    # assert y.grad(x).simplify() == A @ A @ x
    assert y.grad(x).edges == ["i_"]


def test_func_grad():
    # Gradient of a function with respect to its variable should adjust edges appropriately
    x = Variable("x", ["x"])
    f = Function("f", [x], ["x"], [])
    assert f.grad(x).edges == ["x_"]


def test_two_func_grad():
    # Gradient of a function with respect to its variable should adjust edges appropriately
    x = Variable("x", ["x"])
    v = Function("v", [x], ["x"], ["y"])
    f = Function("f", [v], ["y"], [])
    assert f.grad(x).edges == ["x_"]


def test_matrix_grad():
    X = Variable("X", ["i", "j"])
    assert X.grad(X) == Copy(["i", "i_"]) @ Copy(["j", "j_"])


def test_broadcasting():
    x = Variable("x", ["x"])
    y = Variable("y", ["y"])
    z = x + y
    assert set(z.edges) == {"x", "y"}
    print(z.grad(x))
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
    print(f"{actual.simplify()=}")
    print(f"{expected.simplify()=}")
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
    A = Function("A", [x], ["i"], ["i", "j"])
    expr = (A @ x).grad(x).simplify()
    assert set(expr.edges) == {"j", "i_"}
    Ad = Function("A_d0", [x], ["i"], ["i", "j", "i_"])
    assert expr == (Ad @ x + A.rename({"i": "i_"})).simplify()
