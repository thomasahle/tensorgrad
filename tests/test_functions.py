from functools import partial
import itertools
import pytest
import torch
from torch.autograd.functional import jacobian, hessian
import torch.nn.functional as tF
from sympy import symbols
from tensorgrad import Variable
import tensorgrad.functions as F
from tensorgrad.tensor import Delta
from tensorgrad.testutils import rand_values, assert_close


@pytest.mark.parametrize("dims", [1, 2])
@pytest.mark.parametrize(
    "functions",
    [
        (F.relu, tF.relu),
        (F.exp, torch.exp),
        (F.log, torch.log),
        (F.abs, torch.abs),
        (F.gt0, lambda x: torch.where(x > 0, 1.0, 0.0)),
        (F.sign, torch.sign),
        (F.sqrt, torch.sqrt),
        (F.tanh, torch.tanh),
        (F.sigmoid, torch.sigmoid),
        (lambda x: F.pow(x, -1), lambda x: x.pow(-1)),
        (lambda x: F.pow(x, 0), lambda x: x.pow(0)),
        (lambda x: F.pow(x, 1), lambda x: x.pow(1)),
        (lambda x: F.pow(x, 2), lambda x: x.pow(2)),
        (lambda x: F.pow(x, 3), lambda x: x.pow(3)),
        (lambda x: F.pow(x, 4), lambda x: x.pow(4)),
        (partial(F.softmax, dim="i"), partial(tF.softmax, dim=-1)),
        (
            partial(F.max_grad, dim="i"),
            partial(jacobian, lambda t: t.amax(-1).sum()),
        ),
    ],
)
def test_all_elementwise(dims, functions):
    tg_fun, torch_fun = functions
    if dims == 1:
        i = symbols("i")
        x = Variable("x", i)
        ts = rand_values([x], {i: 3})
        batch = []
    elif dims == 2:
        b, i = symbols("b i")
        x = Variable("x", b, i)
        ts = rand_values([x], {b: 2, i: 3})
        batch = ["b"]
    ts[x] = 1 + ts[x].abs()  # Some functions require positive inputs
    tX = ts[x].rename(None)

    # Create the expression
    expr = tg_fun(x)

    # Basic evaluation
    result = expr.evaluate(ts)
    expected = torch_fun(tX).rename(*batch, "i")
    assert_close(result, expected)

    # Evaluation of function expanded
    expanded = expr.simplify({"expand_functions": True})
    result2 = expanded.evaluate(ts)
    assert_close(result2, expected)

    # Evaluation of jacobian
    tg_jac = expanded.grad(x).simplify().evaluate(ts)
    if dims == 1:
        torch_jac = jacobian(torch_fun, tX).rename("i", "i_")
    elif dims == 2:
        torch_jac = jacobian(torch_fun, tX).rename("b", "i", "b_", "i_")
    assert_close(tg_jac, torch_jac)

    tg_hessian = F.sum(expanded).grad(x).grad(x).simplify().evaluate(ts)
    torch_hessian = hessian(lambda x: torch_fun(x).sum(), tX, vectorize=True)
    torch_hessian = (
        torch_hessian.rename("b", "i", "b_", "i_") if dims == 2 else torch_hessian.rename("i", "i_")
    )
    assert_close(tg_hessian, torch_hessian)


# TODO: We're not testing any "real" (R^n, R^n) -> R functions here. The closest thing
# is cross_entropy, but it doesn't use a Function object, just an expression.
# If we were to add such functions, we might run into this issue with Function not supporting
# shared broadcasted edges...
@pytest.mark.parametrize(
    "functions",
    [
        (lambda x, y: x + y, lambda x, y: x + y),
        (lambda x, y: x - y, lambda x, y: x - y),
        (lambda x, y: x * y, lambda x, y: x * y),
        (lambda x, y: x / y, lambda x, y: x / y),
        (lambda x, y: x**2 + y**3, lambda x, y: x**2 + y**3),
        (lambda x, y: (x - y) ** 3, lambda x, y: (x - y) ** 3),
        (F.gt, lambda x, y: (torch.sign(x - y) + 1) / 2),
        (F.maximum, torch.maximum),
        (partial(F.cross_entropy, dim="C"), partial(tF.cross_entropy, reduction="none")),
    ],
)
def test_bivariate_functions(functions):
    tg_fun, torch_fun = functions

    N, C = symbols("N C")
    x = Variable("logits", N=N, C=C)
    y = Variable("target", N=N, C=C)
    ts = rand_values([x, y], {N: 3, C: 4})

    # We convert all values to [0, 1], since some functions,
    # like cross entropy, expects that.
    ts[x] = ts[x].softmax(dim=1)
    ts[y] = ts[y].softmax(dim=1)
    tx = ts[x].rename(None)
    ty = ts[y].rename(None)

    # We're interested in functions (R^n, R^n) -> R.
    # For the kind that's (R^n, R^n) -> R^n we sum over the last dimension.
    # if tg_fun(x, y).edges == {"N", "C"}:
    #     wide_tg, wide_torch = tg_fun, torch_fun
    #     tg_fun = lambda x, y: F.sum(wide_tg(x, y), dim="C")  # noqa: E731
    #     torch_fun = lambda x, y: torch.sum(wide_torch(x, y), dim=1)  # noqa: E731

    # Basic evaluation.
    # Note that each evaluate call stores the sub-values seen so far in the dict.
    # This is normally a fine performance optimization, but it's annoying when debugging
    # in tests, so we take copies of `ts` each time we use it.
    expr = tg_fun(x, y)
    result = expr.evaluate(ts.copy())
    names = ("N", "C") if len(expr.edges) == 2 else ("N",)
    expected = torch_fun(tx, ty).rename(*names)
    assert_close(result, expected)

    # Evaluation of function expanded
    expanded = expr.simplify({"expand_functions": True})
    result2 = expanded.evaluate(ts.copy())
    assert_close(result2, expected)

    # Jacobians
    tg_jac_x = expanded.grad(x).simplify().evaluate(ts.copy())
    tg_jac_y = expanded.grad(y).simplify().evaluate(ts.copy())
    jac_x, jac_y = jacobian(torch_fun, (tx, ty))
    names = ("N", "C", "N_", "C_") if len(expr.edges) == 2 else ("N", "N_", "C")
    assert_close(tg_jac_x, jac_x.rename(*names))
    assert_close(tg_jac_y, jac_y.rename(*names))

    # Hessians,
    # For pytorch, the functions must return single values, so we sum them.
    torch_hessians = hessian((lambda x, y: torch_fun(x, y).sum()), (tx, ty))
    for i, j in itertools.product(range(2), repeat=2):
        v1, v2 = [x, y][i], [x, y][j]
        my_hessian_expr = F.sum(expanded).grad(v1).grad(v2).simplify()
        my_hessian = my_hessian_expr.evaluate(ts.copy())
        assert_close(my_hessian, torch_hessians[i][j].rename("N", "C", "N_", "C_"))


def test_softmax_grad_mat():
    i, j = symbols("i j")
    A = Variable("A", i=i, j=j)
    ts = rand_values([A], {i: 3, j: 2})
    res = F.softmax(A, {"i": i}).simplify({"expand_functions": True}).grad(A).simplify().evaluate(ts)
    expected = jacobian(lambda A: tF.softmax(A, dim=0), ts[A].rename(None)).rename("i", "j", "i_", "j_")
    assert_close(res, expected)


def test_ce():
    N, C = symbols("N C")
    logits = Variable("logits", N=N, C=C)
    target = Variable("target", N=N, C=C)
    ts = rand_values([logits, target], {N: 3, C: 3})
    ts[target] = ts[target].softmax(dim=1)
    ce = F.cross_entropy(logits, target, dim="C").simplify()
    res = ce.evaluate(ts)
    expected = tF.cross_entropy(
        ts[logits].rename(None),
        ts[target].rename(None),
        reduction="none",
    ).rename("N")
    assert_close(res, expected)


def test_ce_grad():
    C = symbols("C")
    logits = Variable("logits", C=C)
    target = Variable("target", C=C)
    ts = rand_values([logits, target], {C: 3})
    ce = F.cross_entropy(logits, target, {"C": C})
    my_jac_logits = ce.grad(logits).simplify().evaluate(ts)
    my_jac_targets = ce.grad(target).simplify().evaluate(ts)
    jac_logits, jac_targets = [
        jac.rename("C")
        for jac in jacobian(
            lambda x, y: tF.cross_entropy(x, y),
            (ts[logits].rename(None), ts[target].rename(None)),
        )
    ]
    assert_close(my_jac_logits, jac_logits)
    assert_close(my_jac_targets, jac_targets)


def test_ce_hess():
    C = symbols("C")
    logits = Variable("logits", C=C)
    target = Variable("target", C=C)
    ts = rand_values([logits, target], {C: 3})
    ce = F.cross_entropy(logits, target, dim="C")
    my_hessians = [
        [
            ce.grad(logits).grad(logits).simplify().evaluate(ts.copy()),
            ce.grad(logits).grad(target).simplify().evaluate(ts.copy()),
        ],
        [
            ce.grad(target).grad(logits).simplify().evaluate(ts.copy()),
            torch.zeros(3, 3).rename("C", "C_"),
        ],
    ]
    torch_hessians = hessian(
        lambda x, y: tF.cross_entropy(x, y),
        (ts[logits].rename(None), ts[target].rename(None)),
    )
    for i in range(2):
        for j in range(2):
            assert_close(my_hessians[i][j], torch_hessians[i][j].rename("C", "C_"))


def test_frobenius2():
    a, b, c = symbols("a b c")
    t = torch.randn(2, 3, 4, names=("a", "b", "c"))
    v = Variable("t", a=a, b=b, c=c)
    frob = F.frobenius2(v)
    res = frob.evaluate({v: t})
    expected = (t * t).sum()
    assert_close(res, expected)


def test_diag():
    a, b = symbols("a b")
    v = Variable("v", a=a)
    mat = F.diag(v, {"a": a, "b": b})
    t = torch.randn(2, names=("a",))
    res = mat.evaluate({v: t})
    expected = torch.diag(t.rename(None)).rename("a", "b")
    assert_close(res, expected)


def test_kronecker():
    i, j, k, l = symbols("i j k l")
    a = Variable("a", i=i, j=j)
    b = Variable("b", k=k, l=l)
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(4, 5, names=("k", "l"))
    result = F.kronecker(a, b).evaluate({a: t_a, b: t_b})
    expected = (
        torch.kron(t_a.rename(None), t_b.rename(None))
        .reshape(2, 4, 3, 5)
        .rename("i", "k", "j", "l")
        .align_to("i", "j", "k", "l")
    )
    torch.testing.assert_close(result.rename(None), expected.rename(None))


def test_sum():
    i, j = symbols("i j")
    a = Variable("a", i=i, j=j)
    t_a = torch.randn(2, 3, names=("i", "j"))

    # Test sum over one dimension
    result = F.sum(a, {"i": i}).evaluate({a: t_a})
    expected = t_a.sum(dim="i")
    torch.testing.assert_close(result.rename(None), expected.rename(None))

    # Test sum over multiple dimensions
    result = F.sum(a, {"i": i, "j": j}).evaluate({a: t_a})
    expected = t_a.sum(dim=("i", "j"))
    torch.testing.assert_close(result.rename(None), expected.rename(None))


def test_mean():
    i, j = symbols("i j")
    a = Variable("a", i=i, j=j)
    t_a = torch.randn(2, 3, names=("i", "j"))

    # Test mean over one dimension
    result1 = F.mean(a, ["i"]).evaluate({a: t_a})
    result2 = F.mean(a, ["i"]).simplify().evaluate({a: t_a})
    expected = t_a.mean(dim="i")
    assert_close(result1, expected)
    assert_close(result2, expected)

    # Test mean over multiple dimensions
    result1 = F.mean(a, ["i", "j"]).evaluate({a: t_a})
    result2 = F.mean(a, ["i", "j"]).simplify().evaluate({a: t_a})
    expected = t_a.mean(dim=("i", "j"))
    assert_close(result1, expected)
    assert_close(result2, expected)


def test_pow():
    i, j = symbols("i j")
    a = Variable("a", i=i, j=j)
    t_a = torch.randn(2, 3, names=("i", "j")).abs()
    result = F.pow(a, -1).evaluate({a: t_a})
    expected = torch.pow(t_a.rename(None), -1).rename("i", "j")
    assert_close(result, expected)


def test_pow_cancel_1():
    #      S - T - * - j
    # (S - T)^(-1) ┘
    i, j = symbols("i j")
    S = Variable("S", i)
    T = Variable("T", i, j)
    ST = S @ T
    expr = ST * F.pow(ST, -1)
    assert expr.full_simplify() == Delta(j, "j")


def test_pow_cancel_1b():
    i, j = symbols("i j")
    S = Variable("S", i)
    expr = S * S * F.pow(S, -1)
    assert expr.full_simplify() == S


def test_pow_cancel_2():
    i, j = symbols("i j")
    S = Variable("S", i)
    T = Variable("T", i, j)
    ST = S @ T
    expr = F.pow(ST, 2) * F.pow(ST, -1)
    assert expr.full_simplify() == ST


def test_pow_cancel_3():
    i, j = symbols("i j")
    S = Variable("S", i)
    T = Variable("T", i, j)
    ST = S @ T
    expr = F.pow(ST, 3) * F.pow(ST, -1)
    assert expr.full_simplify() == F.pow(ST, 2)


def test_pow_multi():
    S = Variable("S")
    x = torch.tensor(1.0)
    expr = F.pow(S, 5)
    val = 1
    for n in range(5, -1, -1):
        assert expr.simplify().evaluate({S: x}) == val
        expr = expr.grad(S)
        val *= n


def test_trace():
    i = symbols("i")
    x = Variable("x", i=i, j=i)
    ts = rand_values([x], {i: 3})
    res = F.trace(x).simplify().evaluate({x: ts[x]})
    expected = ts[x].rename(None).trace()
    assert_close(res, expected)


def test_rnn():
    input_dim, hidden_dim, hidden_dim_out = symbols("input_dim hidden_dim hidden_dim_out")
    x = Variable("x", input_dim)
    h = Variable("h", hidden_dim)
    W_ih = Variable("W_ih", input_dim, hidden_dim_out)
    W_hh = Variable("W_hh", hidden_dim, hidden_dim_out)
    b = Variable("b", hidden_dim_out)

    next_h = F.tanh(W_ih @ x + W_hh @ h + b).rename(hidden_dim_out="hidden_dim")
    loss = F.sum(next_h)
    grad = loss.grad(W_hh)

    # TODO: Test output better


def test_flatten():
    """
    Test Flatten. If Flatten has input_edges=[i, j], output_edge='k',
    it means we flatten i,j into a single dimension k.
    """

    i, j, k = symbols("i j k")
    flatten = F.Reshape(i, j, k)
    X = Variable("X", i, j)

    assert flatten.edges == {"i", "j", "k"}
    assert (X @ flatten).shape == {"k": k}

    ts = rand_values([X], {i: 3, j: 2})

    result = (X @ flatten).evaluate(ts, {k: 6})
    expected = ts[X].rename(None).reshape(-1).rename("k")
    assert_close(result, expected)


def test_reshape():
    i, j, k, l = symbols("i j k l")
    flatten = F.Reshape(i, j, k, l)
    X = Variable("X", i, j)

    assert flatten.edges == {"i", "j", "k", "l"}
    assert (X @ flatten).shape == {"k": k, "l": l}
    sizes = {i: 3, j: 2, k: 1, l: 6}

    ts = rand_values([X], sizes)

    result = (X @ flatten).evaluate(ts, sizes)
    expected = ts[X].rename(None).reshape(1, 6).rename("k", "l")
    assert_close(result, expected)


@pytest.mark.parametrize(
    "x_vals, eps_vals",
    [
        ([2.0, 3.0, 4.0], [0.1, -0.2, 0.05]),
        ([1.0, 2.0, 5.0], [0.01, -0.05, 0.3]),
    ],
)
def test_taylor(x_vals, eps_vals):
    """
    Test the n=1 Taylor expansion of log(x) at x+eps by comparing it to
    log(x+eps). We don't directly compare the expansions; rather, we check
    that for small eps, the difference between the linear approximation
    and the true log(x+eps) is small, implying correct 'behavior.'
    """
    i = symbols("i")
    x = Variable("x", i=i)
    eps = Variable("eps", i=i)
    expr = F.taylor(F.log(x), x, eps, n=1)

    t_x = torch.tensor(x_vals, names=("i",))
    t_eps = torch.tensor(eps_vals, names=("i",))
    out_approx = expr.simplify().evaluate({x: t_x, eps: t_eps})
    out_true = (t_x.rename(None) + t_eps.rename(None)).log().rename("i")

    # We check behavior by ensuring the difference is reasonably small,
    # not by direct equality.
    diff = (out_approx - out_true).abs().max()
    assert diff < 1e-2, f"Taylor approximation differs too much ({diff.item()})"


def test_symmetrize():
    """
    Test symmetrization by verifying that summing the symmetrized tensor
    along all permutations yields the same total as summing a manual
    permutation-based expansion. The behavior we check is that the
    aggregated values match the manual permutations.
    """
    i = symbols("i")
    X = Variable("X", i, j=i, k=i)
    symX = F.symmetrize(X)

    manual_sum = 0
    keys = ["i", "j", "k"]
    for perm in itertools.permutations(keys):
        manual_sum += X.rename(**dict(zip(keys, perm)))

    assert symX.simplify() == manual_sum.simplify()


def test_anti_symmetrize():
    i = symbols("i")
    X = Variable("X", i, j=i)
    symX = F.symmetrize(X, signed=True)
    manual_sum = X - X.rename(j="i", i="j")
    assert symX.simplify() == manual_sum.simplify()


def test_dot():
    """
    Test F.dot(t1, t2, dims) by comparing it to a manual sum-of-products
    along the specified dimension, verifying the result, not the shape
    or edges themselves.
    """
    i, j = symbols("i j")
    A = Variable("A", i=i, j=j)
    B = Variable("B", i=i, j=j)
    tA = torch.randn(2, 3, names=("i", "j"))
    tB = torch.randn(2, 3, names=("i", "j"))

    # Contract along 'j'
    expr = F.dot(A, B, ["j"])
    out = expr.evaluate({A: tA, B: tB})

    # Behavior check: sum-of-products along 'j':
    expected = (tA * tB).sum(dim="j")
    assert_close(out, expected)


def test_pairwise_distance():
    """
    Test F.pairwise_distance(t1, t2, dims) by comparing it to a manual
    L2 distance along 'dims'. We confirm correct numeric behavior.
    """
    i = symbols("i")
    X = Variable("X", i=i)
    Y = Variable("Y", i=i)

    tX = torch.tensor([1.0, 2.0, 3.0], names=("i",))
    tY = torch.tensor([2.0, 2.0, 2.0], names=("i",))
    expr = F.pairwise_distance(X, Y, ["i"])
    out = expr.evaluate({X: tX, Y: tY})

    # Behavior check: manual sum of squared differences
    expected = (tX.rename(None) - tY.rename(None)).pow(2).sum()
    assert_close(out, expected)


def test_max_grad():
    """
    Test F.gt0 by verifying that it yields 1 where inputs > 0 else 0.
    Behavior check: an elementwise threshold function.
    """
    i, j = symbols("i j")
    X = Variable("X", i, j)
    assert_close(
        F.max_grad(X, dim="j").evaluate(
            {
                X: torch.tensor(
                    [
                        [-1.0, 0.0, 1.0, 2.0],
                        [1.0, 0.0, 1.0, -2.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ],
                    names=("i", "j"),
                )
            }
        ),
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.5, 0.0, 0.5, 0.0],
                [1 / 3, 0.0, 1 / 3, 1 / 3],
                [0.25, 0.25, 0.25, 0.25],
            ],
            names=("i", "j"),
        ),
    )


def test_max_grad2():
    """
    Verify that the subgradient of max(t) along a dimension matches PyTorch’s
    partial derivative for ties, i.e. each maximal element gets 1/k if there
    are k ties.

    We'll do 1D examples. Compare to PyTorch's max(...) + backward() approach.
    """

    i = symbols("i")
    X = Variable("X", i=i)

    # We'll test multiple arrays, each with ties or no ties:
    data_cases = [
        # No ties: only one max
        ([1.0, 3.0, 2.0], [0.0, 1.0, 0.0]),
        # Ties among two elements
        ([2.0, 2.0, 1.0], [0.5, 0.5, 0.0]),
        # Ties among all three
        ([5.0, 5.0, 5.0], [1 / 3, 1 / 3, 1 / 3]),
        # Negative & positive ties
        ([-1.0, 2.0, 2.0], [0.0, 0.5, 0.5]),
    ]

    for inp, expected_subgrad in data_cases:
        tX = torch.tensor(inp, names=("i",)).requires_grad_(True)

        # PyTorch approach: y = x.max(dim=0) => shape=() => then sum => 1D scalar
        # But we want subgradient w.r.t x =>
        # We'll replicate the dimension approach:
        val = tX.rename(None).amax(dim=0)  # a scalar
        y = val.sum()  # just .sum() so we can backward
        y.backward()

        # PyTorch's x.grad has the subgradient. But note:
        #  - ties get 1/k each
        #  - the result is plain .grad, shape [3], with no naming
        # We'll rename it for comparison:
        py_grad = tX.grad.rename("i")

        # Now do the same in tensorgrad:
        # The derivative is: F.max(X, "i").grad(X)
        expr = F.max(X, "i").grad(X).simplify()
        tg_grad = expr.evaluate({X: tX.detach()})  # pass the detached version

        # Compare numeric results
        assert_close(tg_grad, py_grad)

        # Also optionally check direct subgradient array:
        # e.g. for [2,2,1], subgradient => [0.5, 0.5, 0]
        manual = torch.tensor(expected_subgrad, names=("i",))
        assert_close(tg_grad, manual)


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dims", [(), ("i",), ("j",), ("i", "j")])
def test_max(keepdim, dims):
    i, j = symbols("i j")
    X = Variable("X", i, j)
    ts = rand_values([X], {i: 3, j: 4})
    names = ts[X].names
    tX = ts[X].rename(None)
    if not keepdim:
        names = tuple(n for n in names if n not in dims and dims)

    # Test max operation
    res = F.max(X, dims, keepdim=keepdim).evaluate(ts)
    adims = tuple(ts[X].names.index(d) for d in dims)
    expected = tX.amax(dim=adims, keepdim=keepdim)
    expected = expected.rename(*names)
    assert_close(res, expected)

    # Test gradients using autograd
    # Note that if we used F.sum/torch.sum we would get a factor i*j off,
    # since our version of keepdim also _keeps the size of the dimensions_
    # where pytorch just uses dummy/size-1 dimensions.
    res_max = F.max(X, dims, keepdim=keepdim)
    res = F.mean(res_max).grad(X).simplify().evaluate(ts)
    x = tX.clone().detach().requires_grad_(True)
    y = x.amax(dim=adims, keepdim=keepdim).rename(*names)
    y.mean().backward()
    expected = x.grad.rename("i", "j")
    assert_close(res, expected)


def test_inverse():
    i = symbols("i")
    A = Variable("A", i, j=i)
    N = 3
    ts = rand_values([A], {i: N})
    tA = ts[A].rename(None)

    res = F.inverse(A).evaluate(ts)
    assert_close(res, tA.inverse().rename("j", "i"))

    assert abs((A @ F.inverse(A)).evaluate(ts) - N) < 1e-5

    # This simplification is not currently supported
    # assert (A @ F.inverse(A)).simplify() == Copy(i)

    assert F.inverse(F.inverse(A)).simplify() == A

    J = jacobian((lambda x: x.inverse().T), tA).rename("i", "j", "i_", "j_")
    myJ = F.inverse(A).grad(A).simplify().evaluate(ts)
    assert_close(myJ, J)
