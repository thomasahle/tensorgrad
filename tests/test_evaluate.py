import random
import torch
from tensorgrad.tensor import (
    Copy,
    Derivative,
    Function,
    FunctionInfo,
    Ones,
    Product,
    Sum,
    Variable,
    Zero,
)
import tensorgrad.functions as F
from tensorgrad.testutils import assert_close, generate_random_tensor_expression, rand_values


def test_copy():
    copy_tensor = Copy(["i", "j"])
    result = copy_tensor.evaluate({}, dims={"i": 3, "j": 3})
    expected = torch.eye(3).rename("i", "j")
    assert_close(result, expected)


def test_zero():
    zero_tensor = Zero(["i", "j"])
    result = zero_tensor.evaluate({}, dims={"i": 2, "j": 3})
    expected = torch.zeros(2, 3).rename("i", "j")
    assert_close(result, expected)


def test_ones():
    ones_tensor = Ones(["i", "j"])
    result = ones_tensor.evaluate({}, dims={"i": 2, "j": 3})
    expected = torch.ones(2, 3).rename("i", "j")
    assert_close(result, expected)


def test_product():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])
    product_tensor = a @ b
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    result = product_tensor.evaluate({a: t_a, b: t_b})
    expected = t_a @ t_b
    assert_close(result, expected)


def test_sum_tensor():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["i", "j"])
    sum_tensor = Sum([a, b], weights=[2, 3])
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(2, 3, names=("i", "j"))
    result = sum_tensor.evaluate({a: t_a, b: t_b})
    expected = 2 * t_a + 3 * t_b
    assert_close(result, expected)


def test_derivative():
    a = Variable("a", ["i"])
    b = Variable("b", ["i"])
    product_tensor = Product([a, b])
    derivative_tensor = Derivative(product_tensor, a, ["j"]).simplify()
    t_a = torch.randn(3, names=("i",))
    t_b = torch.randn(3, names=("i",))
    result = derivative_tensor.evaluate({a: t_a, b: t_b})
    expected = t_b.rename("j")
    assert_close(result, expected)


def test_rename():
    a = Variable("a", ["i", "j"])
    renamed_tensor = a.rename({"i": "k", "j": "l"})
    t_a = torch.randn(2, 3, names=("i", "j"))
    result = renamed_tensor.evaluate({a: t_a})
    expected = t_a.rename("k", "l")
    torch.testing.assert_close(result.rename(None), expected.rename(None))


def test_nested_product_and_sum():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])
    c = Variable("c", ["i", "k"])
    d = Variable("d", ["i", "m"])

    expr = (a @ b + c) @ d

    # Create random input tensors
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    t_c = torch.randn(2, 4, names=("i", "k"))
    t_d = torch.randn(2, 6, names=("i", "m"))

    # Evaluate the tensor expression
    result = expr.evaluate({a: t_a, b: t_b, c: t_c, d: t_d})
    # Compare with the expected result
    expected = (t_a @ t_b + t_c).transpose("k", "i") @ t_d
    assert_close(result, expected)


def test_derivative_of_product():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])
    c = Variable("c", ["k", "l"])

    # Create a tensor expression: a @ b @ c
    expr = Product([a, b, c])

    # Take the derivative with respect to b
    derivative_expr = Derivative(expr, b)

    # Create random input tensors
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    t_c = torch.randn(4, 5, names=("k", "l"))

    # Evaluate the derivative expression
    result = derivative_expr.simplify().evaluate({a: t_a, b: t_b, c: t_c})

    # Compare with the expected result
    expected = torch.einsum("ij,kl->ijkl", t_a.rename(None), t_c.rename(None)).rename("i", "j_", "k_", "l")
    assert_close(result, expected)


def test_function_evaluation():
    a = Variable("a", ["i"])
    b = Variable("b", ["i"])

    expr = Function(
        FunctionInfo(
            "element_wise_product",
            eval=lambda x, y: x * y,
        ),
        ["i"],
        (a, "i"),
        (b, "i"),
    )

    # Create random input tensors
    t_a = torch.randn(3, names=("i",))
    t_b = torch.randn(3, names=("i",))

    # Evaluate the function expression
    result = expr.evaluate({a: t_a, b: t_b})

    # Compare with the expected result
    expected = t_a * t_b
    assert_close(result, expected)


def test_simplify_product_with_zero():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])

    # Create a tensor expression: a @ 0
    expr = Product([a, Zero(["j", "k"])])

    # Simplify the expression
    simplified_expr = expr.simplify()

    # Check if the simplified expression is a Zero tensor
    assert isinstance(simplified_expr, Zero)
    assert simplified_expr.edges == ["i", "k"]


def test_simplify_sum_of_products():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])
    c = Variable("c", ["i", "j"])
    d = Variable("d", ["j", "k"])
    values = rand_values([a, b, c, d], i=2, j=3, k=4)

    assert a != b
    assert a @ b != c @ d

    expr = (a @ b) + (c @ d)
    simplified_expr = expr.simplify()

    # Check if the simplified expression is a single Product
    assert isinstance(simplified_expr, Sum)
    assert len(simplified_expr.tensors) == 2
    assert all(isinstance(t, Product) for t in simplified_expr.tensors)

    assert_close(expr.evaluate(values), simplified_expr.evaluate(values))


def test_trace():
    a = Variable("a", ["i", "j"])
    trace_tensor = F.trace(a)
    t_a = torch.randn(3, 3, names=("i", "j"))
    result = trace_tensor.evaluate({a: t_a})
    expected = t_a.rename(None).trace()
    assert_close(result, expected)


def test_random_small():
    torch.manual_seed(42)
    random.seed(42)
    for _ in range(100):
        expr, expected, variables = generate_random_tensor_expression(5)
        print(f"{expr=}")
        result = expr.evaluate(variables)
        assert_close(result, expected)
        result2 = expr.simplify().evaluate(variables)
        assert_close(result2, expected)


def test_rand2():
    x = Variable("x", ["a"])
    y = Variable("y", ["a", "b"])
    z = Variable("z", ["a", "b", "c"])
    ts = rand_values([x, y, z], a=2, b=3, c=3)
    expr = x @ z + y

    res = expr.evaluate(ts)
    expected_xy = torch.einsum("a,abc->bc", ts[x].rename(None), ts[z].rename(None))
    expected = expected_xy.reshape(1, 3, 3) + ts[y].rename(None).reshape(2, 3, 1)
    assert_close(res, expected.rename("a", "b", "c"))


def test_rand0():
    # A strange test that the random generator came up with.
    # Originally it failed because b != c, which is a requirement for the identity matrix Copy(["b, c"])
    x = Variable("x", ["a", "b", "c"])
    y = Variable("y", ["a", "b", "c"])
    expr = Sum(
        [
            Product([Copy(["a"]), Copy(["b", "c"])]),
            Sum([x, y]),
        ],
    )
    a, b, c = 2, 3, 3
    ts = rand_values([x, y], a=a, b=b, c=c)
    res = expr.evaluate(ts, dims={"a": a, "b": b, "c": c})
    assert_close(
        res,
        ts[x] + ts[y] + torch.eye(b).reshape(1, b, c).rename("a", "b", "c"),
    )


def test_rand1():
    x = Variable("x", ["a"], ["a"])
    z = Variable("z", ["a"], ["a"])
    expr = (x + z) + (z + z)
    ts = rand_values([x, z], a=2)
    res = expr.evaluate(ts)
    assert_close(res, 1 * ts[x] + 3 * ts[z])


def test_linked_variable():
    x = Variable("x", ["a"])
    ts = rand_values([x], a=3)
    expr = x.grad(x).simplify()
    assert_close(expr.evaluate(ts), torch.eye(3).rename("a", "a_"))

    X = Variable("x", ["a", "b"])
    ts = rand_values([X], a=3, b=2)
    expr = X.grad(X).simplify()
    assert_close(expr.evaluate(ts), torch.eye(6).reshape(3, 2, 3, 2).rename("a", "b", "a_", "b_"))


def test_pow0_linked_variable():
    x = Variable("x", ["a"])
    ts = rand_values([x], a=3)
    expr = F.pow(x, 0).simplify()
    assert_close(expr.evaluate(ts), torch.ones(3).rename("a"))

    X = Variable("x", ["a", "b"])
    ts = rand_values([X], a=3, b=2)
    expr = F.pow(X, 0).simplify()
    assert_close(expr.evaluate(ts), torch.ones(3, 2).rename("a", "b"))


def test_unfold():
    data = Variable("data", ["b", "cin", "win", "hin"])
    unfold = F.Unfold(["win", "hin"], ["kw", "kh"], ["wout", "hout"])
    expr = data @ unfold

    b, cin, win, hin, kw, kh = 2, 3, 4, 5, 2, 2
    ts = rand_values([data], b=b, cin=cin, win=win, hin=hin, kw=kw, kh=kh)
    res = expr.evaluate(ts, dims={"kw": kw, "kh": kh})
    expected = (
        torch.nn.functional.unfold(ts[data].rename(None), (2, 2))
        .reshape(b, cin, kw, kh, win - kw + 1, hin - kh + 1)
        .rename("b", "cin", "kw", "kh", "wout", "hout")
    )

    assert_close(res, expected)
