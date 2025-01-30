import random

import pytest
import torch
import torch.nn.functional as tF
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad.extras.to_pytorch import compile_to_callable
from tensorgrad.tensor import (
    Delta,
    Derivative,
    Function,
    FunctionSignature,
    Ones,
    Product,
    Sum,
    Variable,
    Zero,
)
from tensorgrad.extras.evaluate import evaluate
from tensorgrad.extras.evaluate import evaluate_function
from tensorgrad.testutils import assert_close, rand_values, random_tensor_expr


def test_copy():
    i = symbols("i")
    copy_tensor = Delta(i, "i, j")
    result = evaluate(copy_tensor, {}, dims={i: 3})
    expected = torch.eye(3).rename("i", "j")
    assert_close(result, expected)


def test_zero():
    i, j = symbols("i j")
    zero_tensor = Zero(i, j)
    result = evaluate(zero_tensor, {}, dims={i: 2, j: 3})
    expected = torch.zeros(2, 3).rename("i", "j")
    assert_close(result, expected)


def test_ones():
    i, j = symbols("i j")
    ones_tensor = Ones(i, j)
    result = evaluate(ones_tensor, {}, dims={i: 2, j: 3})
    expected = torch.ones(2, 3).rename("i", "j")
    assert_close(result, expected)


def test_product():
    i, j, k = symbols("i j k")
    a = Variable("a", i, j)
    b = Variable("b", j, k)
    product_tensor = a @ b
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    result = evaluate(product_tensor, {a: t_a, b: t_b})
    expected = t_a @ t_b
    assert_close(result, expected)


def test_sum_tensor():
    i, j = symbols("i j")
    a = Variable("a", i, j)
    b = Variable("b", i, j)
    sum_tensor = Sum([a, b], weights=[2, 3])
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(2, 3, names=("i", "j"))
    result = evaluate(sum_tensor, {a: t_a, b: t_b})
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
    result = evaluate(derivative_tensor, {a: t_a, b: t_b})
    expected = t_b.rename("i_prime")
    assert_close(result, expected)


def test_rename():
    i, j = symbols("i j")
    a = Variable("a", i, j)
    renamed_tensor = a.rename(i="k", j="l")
    t_a = torch.randn(2, 3, names=("i", "j"))
    result = evaluate(renamed_tensor, {a: t_a})
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

    result = evaluate(expr, {a: t_a, b: t_b, c: t_c, d: t_d})
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

    result = evaluate(derivative_expr.simplify(), {a: t_a, b: t_b, c: t_c})
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

    evaluate_function.register(ElementWiseProduct, lambda _, a, b: a * b)

    expr = Function(ElementWiseProduct(), [a, b], {"i": i})

    t_a = torch.randn(3, names=("i",))
    t_b = torch.randn(3, names=("i",))

    result = evaluate(expr, {a: t_a, b: t_b})
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

    assert_close(evaluate(expr, values), evaluate(simplified_expr, values))


def test_trace_square():
    i = symbols("i")
    a = Variable("a", i=i, j=i)
    trace_tensor = F.trace(a)
    t_a = torch.randn(3, 3, names=("i", "j"))
    result = evaluate(trace_tensor, {a: t_a})
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
        for expr_ in [expr, expr.simplify()]:
            result = evaluate(expr_, variables)
            result_code = compile_to_callable(expr_, verbose=True)(variables)
            for val in [result, result_code]:
                assert_close(val, expected, atol=1e-2, rtol=1e-2)


def test_rand2():
    a, b, c = symbols("a b c")
    x = Variable("x", a)
    y = Variable("y", a, b)
    z = Variable("z", a, b, c)
    ts = rand_values([x, y, z], {a: 2, b: 3, c: 3})
    expr = x @ z + y

    res = evaluate(expr, ts)
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
    res = evaluate(expr, ts, dims={a: a_val, b: b_val})
    expected = ts[x] + ts[y] + torch.eye(b_val).reshape(1, b_val, c_val).rename("a", "b", "c")
    assert_close(res, expected)


def test_rand1():
    a = symbols("a")
    x = Variable("x", a, link=a)
    z = Variable("z", a, link=a)
    expr = (x + z) + (z + z)
    ts = rand_values([x, z], {a: 2})
    res = evaluate(expr, ts)
    assert_close(res, 1 * ts[x] + 3 * ts[z])


def test_linked_variable():
    a, b = symbols("a b")
    x = Variable("x", a)
    ts = rand_values([x], {a: 3})
    expr = x.grad(x).simplify()
    assert_close(evaluate(expr, ts), torch.eye(3).rename("a", "a_"))

    X = Variable("x", a, b)
    ts = rand_values([X], {a: 3, b: 2})
    expr = X.grad(X).simplify()
    assert_close(evaluate(expr, ts), torch.eye(6).reshape(3, 2, 3, 2).rename("a", "b", "a_", "b_"))


def test_pow0_linked_variable():
    a, b = symbols("a b")
    x = Variable("x", a)
    ts = rand_values([x], {a: 3})
    expr = F.pow(x, 0).simplify()
    assert_close(evaluate(expr, ts), torch.ones(3).rename("a"))

    X = Variable("x", a, b)
    ts = rand_values([X], {a: 3, b: 2})
    expr = F.pow(X, 0).simplify()
    assert_close(evaluate(expr, ts), torch.ones(3, 2).rename("a", "b"))


def test_unfold():
    b, cin, win, hin, kw, kh, wout, hout = symbols("b cin win hin kw kh wout hout")
    data = Variable("data", b, cin, win, hin)
    unfold = F.Convolution(win, kw, wout) @ F.Convolution(hin, kh, hout)
    expr = data @ unfold

    b_val, cin_val, win_val, hin_val, kw_val, kh_val = 2, 3, 4, 5, 2, 2
    ts = rand_values([data], {b: b_val, cin: cin_val, win: win_val, hin: hin_val})
    # res = evaluate(expr,ts, dims={kw: kw_val, kh: kh_val})
    res = evaluate(expr, ts, dims={kw: kw_val, kh: kh_val})
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
    loss0 = evaluate(y, params0, shapes)
    assert_close(tloss, loss0)

    # Gradient descent
    parameters[layer1] -= 1e-1 * evaluate(grad_layer1, params0, shapes).align_to("dim1", "dim2")
    parameters[layer2] -= 1e-1 * evaluate(grad_layer2, params0, shapes).align_to("dim2", "out")

    # Copy parameters to avoid caching, evaluate new loss
    params1 = {t: v.clone() for t, v in parameters.items()}
    loss1 = evaluate(y, params1, shapes)

    assert loss1 < loss0, "Loss should decrease after one step of gradient descent"
