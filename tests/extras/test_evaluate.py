import random

import pytest
import torch
import torch.nn.functional as tF
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler import compile_to_callable as compile_aot
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
    assert len(simplified_expr.terms) == 2
    assert all(isinstance(t, Product) for t in simplified_expr.terms)

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
    compiler_divergences = []
    for _ in range(30):
        expr, expected, variables = random_tensor_expr(max_depth=max_depth, max_dim=max_dim)
        for tag, expr_ in [("raw", expr), ("simplified", expr.simplify())]:
            # The interpreter (this file's subject) must always match ground truth.
            result = evaluate(expr_, variables)
            assert_close(result, expected, atol=1e-2, rtol=1e-2)
            # Cross-check the AOT compiler against the same ground truth.
            result_code = compile_to_callable(expr_)(variables)
            try:
                assert_close(result_code, expected, atol=1e-2, rtol=1e-2)
            except AssertionError:
                compiler_divergences.append(tag)
    # Historically xfailed on a factoring bug: splice_child orphaned a
    # broadcast-only child wire that the parent einsum contracted, and
    # Builder.einsum silently dropped it -- losing a factor of dim(wire)
    # (task #33; regression-pinned in tests/compiler/test_factor.py).
    assert not compiler_divergences, (
        f"compiler diverged from ground truth: {len(compiler_divergences)} "
        f"({set(compiler_divergences)})"
    )


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


# ---------------------------------------------------------------------------
# Order-0 Delta scalar factors.
# Regression tests: evaluate() used to silently drop order-0 Delta factors
# (scalars equal to their symbol's size) when Product evaluation filtered
# Delta tensors out of the einsum contraction, e.g.
#   evaluate(Delta(a) * (x @ x)) == x @ x   instead of   |a| * (x @ x).
# The compiler backend folds order-0 deltas into einsum weights, so we
# cross-check against it on the same expressions.
# ---------------------------------------------------------------------------


def _order0_setup():
    a, i = symbols("a i")
    x = Variable("x", i=i)
    xv = torch.tensor([1.0, 2.0, 1.5], names=("i",))  # x @ x == 7.25
    dims = {a: 4, i: 3}
    return a, i, x, xv, dims


def _check_order0(expr, expected, values, dims):
    """Check evaluate on expr and expr.simplify(), and cross-check the compiler."""
    for e in (expr, expr.simplify()):
        res = evaluate(e, dict(values), dims=dict(dims))
        assert_close(res, expected)
        compiled = compile_aot(e)(dict(values), dict(dims))
        assert_close(compiled, expected)


def test_order0_delta_alone():
    a, i, x, xv, dims = _order0_setup()
    res = evaluate(Delta(a), {}, dims=dims)
    assert res.item() == 4.0
    # Float dtype, to match every other evaluate path and the compiler backend
    assert res.dtype == torch.float32


def test_order0_delta_times_contraction():
    # The original oracle bug: the |a| = 4 factor was dropped from the product
    a, i, x, xv, dims = _order0_setup()
    expr = Delta(a) * (x @ x)
    _check_order0(expr, torch.tensor(29.0), {x: xv}, dims)


def test_order0_delta_only_product():
    a, i, x, xv, dims = _order0_setup()
    expr = Product([Delta(a), Delta(a), Delta(i)])
    _check_order0(expr, torch.tensor(48.0), {}, dims)


def test_order0_delta_power():
    a, i, x, xv, dims = _order0_setup()
    expr = F.pow(Delta(a), 2) * (x @ x)
    _check_order0(expr, torch.tensor(116.0), {x: xv}, dims)
    expr = F.pow(Delta(a), -1) * (x @ x)
    _check_order0(expr, torch.tensor(7.25 / 4), {x: xv}, dims)


def test_order0_delta_in_sum_and_weights():
    a, i, x, xv, dims = _order0_setup()
    # Plain sum
    _check_order0(Delta(a) + (x @ x), torch.tensor(11.25), {x: xv}, dims)
    # Weighted sum: 2*4 + 3*7.25
    _check_order0(Sum([Delta(a), x @ x], [2, 3]), torch.tensor(29.75), {x: xv}, dims)
    # Nested in a Sum inside a Product: (2 - |a|) * x
    expr = Product([Sum([Ones(), Delta(a)], [2, -1]), x])
    _check_order0(expr, (-2.0 * xv.rename(None)).rename("i"), {x: xv}, dims)


def test_order0_delta_broadcast_to_vector():
    a, i, x, xv, dims = _order0_setup()
    expr = Delta(a) * x
    _check_order0(expr, (4.0 * xv.rename(None)).rename("i"), {x: xv}, dims)


def test_order0_delta_with_identity():
    # All remaining factors are Deltas (merge_copies=False path)
    a, i, x, xv, dims = _order0_setup()
    expr = Product([Delta(a), Delta(i, "i", "j")])
    expected = (4.0 * torch.eye(3)).rename("i", "j")
    _check_order0(expr, expected, {}, dims)


def test_order0_delta_repeated_output():
    # Repeated-output einsum branch: out[j,k] = |a| * x_j * [j == k]
    a, i, x, xv, dims = _order0_setup()
    expr = Product([Delta(a), x, Delta(i, "i", "j", "k")])
    expected = (4.0 * torch.diag(xv.rename(None))).rename("j", "k")
    for e in (expr, expr.simplify()):
        res = evaluate(e, {x: xv}, dims=dict(dims))
        assert_close(res.align_to("j", "k"), expected)


# ---------------------------------------------------------------------------
# Property test: scalar-heavy random expressions, evaluate vs an independent
# reference and vs the compiler backend.
# ---------------------------------------------------------------------------


def _random_scalar_expr(rng, depth, x, xv, size_syms):
    """Return (order-0 tensorgrad expr, python float reference)."""
    if depth <= 0:
        choice = rng.randrange(6)
        if choice == 0:
            sym, val = rng.choice(size_syms)
            return Delta(sym), float(val)
        if choice == 1:
            return Ones(), 1.0
        if choice == 2:
            return Product([]), 1.0
        if choice == 3:
            return x @ x, float(xv.rename(None) @ xv.rename(None))
        if choice == 4:
            sym, val = rng.choice(size_syms)
            k = rng.choice([-1, 2])
            return F.pow(Delta(sym), k), float(val) ** k
        return F.sum(x), float(xv.rename(None).sum())
    kind = rng.choice(["sum", "prod", "pow1"])
    if kind == "sum":
        n = rng.randint(2, 3)
        pairs = [_random_scalar_expr(rng, depth - 1, x, xv, size_syms) for _ in range(n)]
        weights = [rng.choice([-2, -1, 1, 2]) for _ in range(n)]
        return Sum([p[0] for p in pairs], weights), sum(w * p[1] for w, p in zip(weights, pairs))
    if kind == "prod":
        p1 = _random_scalar_expr(rng, depth - 1, x, xv, size_syms)
        p2 = _random_scalar_expr(rng, depth - 1, x, xv, size_syms)
        return Product([p1[0], p2[0]]), p1[1] * p2[1]
    # pow with k=1 wraps composites in a _PowerFunction without growing magnitude
    p = _random_scalar_expr(rng, depth - 1, x, xv, size_syms)
    return F.pow(p[0], 1), p[1]


def test_random_scalar_heavy_vs_compiler():
    rng = random.Random(0)
    torch.manual_seed(0)
    a, b, i = symbols("a b i")
    dims = {a: 2, b: 5, i: 3}
    x = Variable("x", i=i)
    size_syms = [(a, 2), (b, 5), (i, 3)]
    n_compiler_checked = 0
    for trial in range(50):
        xv = torch.randn(3, names=("i",))
        expr, ref = _random_scalar_expr(rng, rng.randint(1, 2), x, xv, size_syms)
        if rng.random() < 0.4:
            # Broadcast the scalar onto a vector output
            expr = expr * x
            expected = (ref * xv.rename(None)).rename("i")
        else:
            expected = torch.tensor(ref)
        for e in (expr, expr.simplify()):
            res = evaluate(e, {x: xv}, dims=dict(dims))
            assert_close(res, expected, rtol=1e-3, atol=5e-2)
            # The historical scalar-subgraph gaps (factor._hoist empty-max,
            # codegen float operand) are fixed: every trial is strictly checked.
            compiled = compile_aot(e)({x: xv}, dict(dims))
            n_compiler_checked += 1
            assert_close(compiled, expected, rtol=1e-3, atol=5e-2)
    assert n_compiler_checked == 100, f"only {n_compiler_checked} compiler cross-checks ran"
