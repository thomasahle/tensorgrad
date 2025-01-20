import random
import pytest
from sympy import symbols
import torch
from tensorgrad.tensor import (
    Delta,
    Derivative,
    FunctionSignature,
    Function,
    Ones,
    Product,
    Sum,
    Variable,
    Zero,
)
import tensorgrad.functions as F
from tensorgrad.testutils import assert_close, random_tensor_expr, rand_values
import torch.nn.functional as tF


def test_copy():
    i = symbols("i")
    copy_tensor = Delta(i, "i, j")
    result = copy_tensor.evaluate({}, dims={i: 3})
    expected = torch.eye(3).rename("i", "j")
    assert_close(result, expected)


def test_zero():
    i, j = symbols("i j")
    zero_tensor = Zero(i, j)
    result = zero_tensor.evaluate({}, dims={i: 2, j: 3})
    expected = torch.zeros(2, 3).rename("i", "j")
    assert_close(result, expected)


def test_ones():
    i, j = symbols("i j")
    ones_tensor = Ones(i, j)
    result = ones_tensor.evaluate({}, dims={i: 2, j: 3})
    expected = torch.ones(2, 3).rename("i", "j")
    assert_close(result, expected)


def test_product():
    i, j, k = symbols("i j k")
    a = Variable("a", i, j)
    b = Variable("b", j, k)
    product_tensor = a @ b
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    result = product_tensor.evaluate({a: t_a, b: t_b})
    expected = t_a @ t_b
    assert_close(result, expected)


def test_sum_tensor():
    i, j = symbols("i j")
    a = Variable("a", i, j)
    b = Variable("b", i, j)
    sum_tensor = Sum([a, b], weights=[2, 3])
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(2, 3, names=("i", "j"))
    result = sum_tensor.evaluate({a: t_a, b: t_b})
    expected = 2 * t_a + 3 * t_b
    assert_close(result, expected)


def test_derivative():
    i = symbols("i")
    a = Variable("a", i)
    b = Variable("b", i)
    product_tensor = Product([a, b])
    derivative_tensor = Derivative(product_tensor, a, {"i": "i_prime"}).simplify()
    t_a = torch.randn(3, names=("i",))
    t_b = torch.randn(3, names=("i",))
    result = derivative_tensor.evaluate({a: t_a, b: t_b})
    expected = t_b.rename("i_prime")
    assert_close(result, expected)


def test_rename():
    i, j = symbols("i j")
    a = Variable("a", i, j)
    renamed_tensor = a.rename(i="k", j="l")
    t_a = torch.randn(2, 3, names=("i", "j"))
    result = renamed_tensor.evaluate({a: t_a})
    expected = t_a.rename("k", "l")
    assert_close(result, expected)


def test_nested_product_and_sum():
    i, j, k, m = symbols("i j k m")
    a = Variable("a", i, j)
    b = Variable("b", j, k)
    c = Variable("c", i, k)
    d = Variable("d", i, m)

    expr = (a @ b + c) @ d

    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    t_c = torch.randn(2, 4, names=("i", "k"))
    t_d = torch.randn(2, 6, names=("i", "m"))

    result = expr.evaluate({a: t_a, b: t_b, c: t_c, d: t_d})
    expected = (t_a @ t_b + t_c).transpose("k", "i") @ t_d
    assert_close(result, expected)


def test_derivative_of_product():
    i, j, k, l = symbols("i j k l")
    a = Variable("a", i, j)
    b = Variable("b", j, k)
    c = Variable("c", k, l)

    expr = Product([a, b, c])
    derivative_expr = Derivative(expr, b)

    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    t_c = torch.randn(4, 5, names=("k", "l"))

    result = derivative_expr.simplify().evaluate({a: t_a, b: t_b, c: t_c})
    expected = torch.einsum("ij,kl->ijkl", t_a.rename(None), t_c.rename(None)).rename("i", "j", "k", "l")
    assert_close(result, expected)


def test_function_evaluation():
    i = symbols("i")
    a = Variable("a", i)
    b = Variable("b", i)

    class ElementWiseProduct(FunctionSignature):
        def __init__(self):
            super().__init__(
                "element_wise_product",
                edges={"i"},
                inputs=[{"i"}, {"i"}],
            )

        def eval(self, a, b):
            return a * b

    expr = Function(ElementWiseProduct(), [a, b], {"i": i})

    t_a = torch.randn(3, names=("i",))
    t_b = torch.randn(3, names=("i",))

    result = expr.evaluate({a: t_a, b: t_b})
    expected = t_a * t_b
    assert_close(result, expected)


def test_simplify_product_with_zero():
    i, j, k = symbols("i j k")
    a = Variable("a", i, j)

    expr = Product([a, Zero(j, k)])
    simplified_expr = expr.simplify()

    assert isinstance(simplified_expr, Zero)
    assert simplified_expr.edges == {"i", "k"}


def test_simplify_sum_of_products():
    i, j, k = symbols("i j k")
    a = Variable("a", i, j)
    b = Variable("b", j, k)
    c = Variable("c", i, j)
    d = Variable("d", j, k)
    values = rand_values([a, b, c, d], {i: 2, j: 3, k: 4})

    assert a != b
    assert a @ b != c @ d

    expr = (a @ b) + (c @ d)
    simplified_expr = expr.simplify()

    assert isinstance(simplified_expr, Sum)
    assert len(simplified_expr.tensors) == 2
    assert all(isinstance(t, Product) for t in simplified_expr.tensors)

    assert_close(expr.evaluate(values), simplified_expr.evaluate(values))


def test_trace_square():
    i = symbols("i")
    a = Variable("a", i=i, j=i)
    trace_tensor = F.trace(a)
    t_a = torch.randn(3, 3, names=("i", "j"))
    result = trace_tensor.evaluate({a: t_a})
    expected = t_a.rename(None).trace()
    assert_close(result, expected)


def test_trace_rectangular():
    i, j = symbols("i j")
    a = Variable("a", i, j)
    with pytest.raises(ValueError):
        F.trace(a)


def test_diag_rectangular():
    i, j = symbols("i j")
    a = Variable("a", i, j)
    with pytest.raises(ValueError):
        F.diag(a, ["i"])


@pytest.mark.parametrize("max_depth", range(2, 4))
@pytest.mark.parametrize("max_dim", range(1, 5))
def test_random_small(max_depth, max_dim):
    torch.manual_seed(42)
    random.seed(42)
    for _ in range(30):
        expr, expected, variables = random_tensor_expr(max_depth=max_depth, max_dim=max_dim)
        # expr, expected, variables = random_tensor_expr(max_depth=max_depth, max_dim=max_dim)
        result = expr.evaluate(variables)
        assert_close(result, expected, atol=1e-2, rtol=1e-2)
        result2 = expr.simplify().evaluate(variables)
        assert_close(result2, expected, atol=1e-2, rtol=1e-2)


def test_rand2():
    a, b, c = symbols("a b c")
    x = Variable("x", a)
    y = Variable("y", a, b)
    z = Variable("z", a, b, c)
    ts = rand_values([x, y, z], {a: 2, b: 3, c: 3})
    expr = x @ z + y

    res = expr.evaluate(ts)
    expected_xy = torch.einsum("a,abc->bc", ts[x].rename(None), ts[z].rename(None))
    expected = expected_xy.reshape(1, 3, 3) + ts[y].rename(None).reshape(2, 3, 1)
    assert_close(res, expected.rename("a", "b", "c"))


def test_rand0():
    # A strange test that the random generator came up with.
    # Originally it failed because b != c, which is a requirement for the identity matrix Copy(["b, c"])
    a, b = symbols("a b")
    x = Variable("x", a, b, c=b)
    y = Variable("y", a, b, c=b)
    expr = Sum(
        [
            Product([Delta(a, "a"), Delta(b, "b, c")]),
            Sum([x, y]),
        ],
    )
    a_val, b_val, c_val = 2, 3, 3
    ts = rand_values([x, y], {a: a_val, b: b_val})
    res = expr.evaluate(ts, dims={a: a_val, b: b_val})
    expected = ts[x] + ts[y] + torch.eye(b_val).reshape(1, b_val, c_val).rename("a", "b", "c")
    assert_close(res, expected)


def test_rand1():
    a = symbols("a")
    x = Variable("x", a, link=a)
    z = Variable("z", a, link=a)
    expr = (x + z) + (z + z)
    ts = rand_values([x, z], {a: 2})
    res = expr.evaluate(ts)
    assert_close(res, 1 * ts[x] + 3 * ts[z])


def test_linked_variable():
    a, b = symbols("a b")
    x = Variable("x", a)
    ts = rand_values([x], {a: 3})
    expr = x.grad(x).simplify()
    assert_close(expr.evaluate(ts), torch.eye(3).rename("a", "a_"))

    X = Variable("x", a, b)
    ts = rand_values([X], {a: 3, b: 2})
    expr = X.grad(X).simplify()
    assert_close(expr.evaluate(ts), torch.eye(6).reshape(3, 2, 3, 2).rename("a", "b", "a_", "b_"))


def test_pow0_linked_variable():
    a, b = symbols("a b")
    x = Variable("x", a)
    ts = rand_values([x], {a: 3})
    expr = F.pow(x, 0).simplify()
    assert_close(expr.evaluate(ts), torch.ones(3).rename("a"))

    X = Variable("x", a, b)
    ts = rand_values([X], {a: 3, b: 2})
    expr = F.pow(X, 0).simplify()
    assert_close(expr.evaluate(ts), torch.ones(3, 2).rename("a", "b"))


def test_unfold():
    b, cin, win, hin, kw, kh, wout, hout = symbols("b cin win hin kw kh wout hout")
    data = Variable("data", b, cin, win, hin)
    unfold = F.Convolution(win, kw, wout) @ F.Convolution(hin, kh, hout)
    expr = data @ unfold

    b_val, cin_val, win_val, hin_val, kw_val, kh_val = 2, 3, 4, 5, 2, 2
    ts = rand_values([data], {b: b_val, cin: cin_val, win: win_val, hin: hin_val})
    # res = expr.evaluate(ts, dims={kw: kw_val, kh: kh_val})
    res = expr.evaluate(ts, dims={kw: kw_val, kh: kh_val})
    expected = (
        torch.nn.functional.unfold(ts[data].rename(None), (2, 2))
        .reshape(b_val, cin_val, kw_val, kh_val, win_val - kw_val + 1, hin_val - kh_val + 1)
        .rename("b", "cin", "kw", "kh", "wout", "hout")
    )

    assert_close(res, expected)


def test_nn():
    batch, dim1, dim2, out = symbols("batch dim1 dim2 out")
    data = Variable("data", batch, dim1)
    targets = Variable("targets", batch, out)
    shapes = {
        batch: 3,
        dim1: 4,
        dim2: 5,
        out: 10,
    }
    layer1 = Variable("lin", dim1, dim2)
    layer2 = Variable("lin", dim2, out)

    # Random initialization
    parameters = rand_values([data, targets, layer1, layer2], shapes)

    # Make model
    x = data
    x = F.relu(x @ layer1)
    x = x @ layer2
    y = F.mean(F.cross_entropy(x, targets, dim="out"), dim="batch")

    # Symbolic backprop
    grad_layer1 = y.grad(layer1).full_simplify()
    grad_layer2 = y.grad(layer2).full_simplify()

    # Numerical evaluation
    tx = parameters[data]
    tx = torch.relu(tx @ parameters[layer1])
    tx = tx @ parameters[layer2]
    tloss = tF.cross_entropy(tx, parameters[targets].rename(None))

    # Copy parameters to avoid caching, evaluate initial loss
    params0 = {t: v.clone() for t, v in parameters.items()}
    loss0 = y.evaluate(params0, shapes)
    assert_close(tloss, loss0)

    # Gradient descent
    parameters[layer1] -= 1e-1 * grad_layer1.evaluate(params0, shapes).align_to("dim1", "dim2")
    parameters[layer2] -= 1e-1 * grad_layer2.evaluate(params0, shapes).align_to("dim2", "out")

    # Copy parameters to avoid caching, evaluate new loss
    params1 = {t: v.clone() for t, v in parameters.items()}
    loss1 = y.evaluate(params1, shapes)

    assert loss1 < loss0, "Loss should decrease after one step of gradient descent"
