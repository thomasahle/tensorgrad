import pytest
import torch
from torch.autograd.functional import jacobian, hessian
import torch.nn.functional as tF
from sympy import symbols
from tensorgrad import Variable, Function
import tensorgrad.functions as F
from tensorgrad.tensor import Copy, Tensor
from tensorgrad.testutils import rand_values, assert_close


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


def test_einsum():
    i, j, k, l = symbols("i j k l")
    a = Variable("a", i=i, j=j)
    b = Variable("b", j=j, k=k)
    c = Variable("c", k=k, l=l)
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    t_c = torch.randn(4, 5, names=("k", "l"))

    # Test basic einsum
    res = F.einsum([a, b], {"i": i, "k": k}).evaluate({a: t_a, b: t_b})
    expected = torch.einsum("ij,jk->ik", t_a.rename(None), t_b.rename(None)).rename("i", "k")
    assert_close(res, expected)

    # Test einsum with multiple tensors
    res = F.einsum([a, b, c], {"i": i, "l": l}).evaluate({a: t_a, b: t_b, c: t_c})
    expected = torch.einsum("ij,jk,kl->il", t_a.rename(None), t_b.rename(None), t_c.rename(None)).rename(
        "i", "l"
    )
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
    print(expr.full_simplify())
    assert expr.full_simplify() == Copy(j, "j")


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


def test_log():
    i, j = symbols("i j")
    a = Variable("a", i=i, j=j)
    t_a = torch.randn(2, 3, names=("i", "j")).abs()
    result = F.log(a).evaluate({a: t_a})
    expected = torch.log(t_a.rename(None)).rename("i", "j")
    assert_close(result, expected)


def test_log_grad():
    i = symbols("i")
    v = Variable("v", i=i)
    t_v = torch.randn(3, names=("i",)).abs()
    assert F.log(v).edges == {"i"}
    jacobian = F.log(v).grad(v).simplify()
    assert jacobian.edges == {"i", "i_"}
    result = jacobian.evaluate({v: t_v})
    expected = torch.diag(torch.pow(t_v.rename(None), -1)).rename("i", "i_")
    assert_close(result, expected)


def test_exp():
    i, j = symbols("i j")
    a = Variable("a", i=i, j=j)
    t_a = torch.randn(2, 3, names=("i", "j"))
    result = F.exp(a).evaluate({a: t_a})
    expected = torch.exp(t_a.rename(None)).rename("i", "j")
    assert_close(result, expected)


def test_relu_jac():
    i = symbols("i")
    x = Variable("x", i=i)
    ts = rand_values([x], {i: 3})
    expr = F.relu(x).grad(x).simplify()
    res = expr.evaluate({x: ts[x]})
    expected = jacobian(lambda x: tF.relu(x), ts[x].rename(None)).rename("i", "i_")
    assert_close(res, expected)


def test_softmax():
    i, j = symbols("i j")
    A = Variable("A", i=i, j=j)
    ts = rand_values([A], {i: 3, j: 2})
    res = F.softmax(A, {"i": i}).evaluate({A: ts[A]})
    expected = tF.softmax(ts[A].rename(None), dim=0).rename("i", "j")
    assert_close(res, expected)


def test_softmax_jac():
    i = symbols("i")
    x = Variable("x", i=i)
    ts = rand_values([x], {i: 3})
    expr = F.softmax(x, {"i": i}).grad(x).simplify()
    print(expr)
    res = expr.evaluate({x: ts[x]})
    expected = jacobian(lambda x: tF.softmax(x), ts[x].rename(None)).rename("i", "i_")
    assert_close(res, expected)


def test_softmax_grad():
    i = symbols("i")
    x = Variable("x", i=i)
    ts = rand_values([x], {i: 3})
    res = F.sum(F.softmax(x, {"i": i}), {"i": i}).grad(x).simplify().evaluate({x: ts[x]})
    expected = jacobian(lambda x: tF.softmax(x).sum(), ts[x].rename(None)).rename("i_")
    assert_close(res, expected)


def test_softmax_hess():
    i = symbols("i")
    x = Variable("x", i=i)
    ts = rand_values([x], {i: 3})
    res = (
        F.sum(F.softmax(x, ["i"]), ["i"])
        .grad(x)
        .grad(x)
        .simplify({"sum_combine_terms": False})
        .evaluate({x: ts[x]})
    )
    expected = hessian(lambda x: tF.softmax(x).sum(), ts[x].rename(None)).rename("i_", "i__")
    assert_close(res, expected)


def test_softmax_hess2():
    i = symbols("i")
    x = Variable("x", i=i)
    targets = Variable("targets", i=i)
    ts = rand_values([x, targets], {i: 3})
    res = (F.softmax(x, {"i": i}) @ targets).grad(x).grad(x).simplify().evaluate(ts)
    expected = hessian(lambda x: tF.softmax(x) @ ts[targets], ts[x].rename(None)).rename("i_", "i__")
    assert_close(res, expected)


def test_softmax_grad_mat():
    i, j = symbols("i j")
    A = Variable("A", i=i, j=j)
    ts = rand_values([A], {i: 3, j: 2})
    res = F.softmax(A, {"i": i}).grad(A).simplify().evaluate(ts)
    expected = jacobian(lambda A: tF.softmax(A, dim=0), ts[A].rename(None)).rename("i", "j", "i_", "j_")
    assert_close(res, expected)


def test_ce():
    N, C = symbols("N C")
    logits = Variable("logits", N=N, C=C)
    target = Variable("target", N=N, C=C)
    ts = rand_values([logits, target], {N: 3, C: 3})
    ts[target] = ts[target].softmax(dim=1)
    ce = F.cross_entropy(logits, target, {"C": C}).simplify()
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
        jac.rename("C_")
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
    ce = F.cross_entropy(logits, target, {"C": C})
    my_hessians = [
        [
            ce.grad(logits).grad(logits).simplify().evaluate(ts.copy()),
            ce.grad(logits).grad(target).simplify().evaluate(ts.copy()),
        ],
        [
            ce.grad(target).grad(logits).simplify().evaluate(ts.copy()),
            torch.zeros(3, 3).rename("C_", "C__"),
        ],
    ]
    torch_hessians = hessian(
        lambda x, y: tF.cross_entropy(x, y),
        (ts[logits].rename(None), ts[target].rename(None)),
    )
    for i in range(2):
        for j in range(2):
            assert_close(my_hessians[i][j], torch_hessians[i][j].rename("C_", "C__"))


def test_pow_hess():
    i = symbols("i")
    x = Variable("x", i=i)
    ts = rand_values([x], {i: 3})
    ce = F.sum(F.pow(x, 3))
    my_hessian = ce.grad(x).grad(x).simplify().evaluate(ts.copy())
    torch_hessian = hessian(lambda x: x.pow(3).sum(), ts[x].rename(None)).rename("i_", "i__")
    assert_close(my_hessian, torch_hessian)


def test_pow_hess2():
    i = symbols("i")
    x = Variable("x", i=i)
    y = Variable("y", i=i)
    ts = rand_values([x, y], {i: 2})
    ce = F.sum(F.pow((x - y), 3))
    my_hessians = [
        [
            ce.grad(x).grad(x).simplify().evaluate(ts.copy()),
            ce.grad(x).grad(y).simplify().evaluate(ts.copy()),
        ],
        [
            ce.grad(y).grad(x).simplify().evaluate(ts.copy()),
            ce.grad(y).grad(y).simplify().evaluate(ts.copy()),
        ],
    ]
    torch_hessians = hessian(
        lambda x, y: torch.pow(x - y, 3).sum(),
        (ts[x].rename(None), ts[y].rename(None)),
    )
    for i in range(2):
        for j in range(2):
            names = my_hessians[i][j].names
            assert_close(my_hessians[i][j], torch_hessians[i][j].rename(*names))


def test_trace():
    i = symbols("i")
    x = Variable("x", i=i, j=i)
    ts = rand_values([x], {i: 3})
    res = F.trace(x).simplify().evaluate({x: ts[x]})
    expected = ts[x].rename(None).trace()
    assert_close(res, expected)


def test_attention():
    seq, dim, inner, head = symbols("seq dim inner head")
    X = Variable("X", seq, dim)
    W_q = Variable("W_q", dim, inner, head)
    W_k = Variable("W_k", dim, inner, head)
    W_v = Variable("W_v", dim, inner, head)
    W_o = Variable("W_o", dim, inner, head)
    query = (W_q @ X).rename(seq="seq_q")
    key = (W_k @ X).rename(seq="seq_k")
    value = (W_v @ X).rename(seq="seq_k")

    logits = F.dot(query, key, ["inner"])
    attention_scores = Function("softmax", {"seq_k": seq}, (logits, "seq_k"))
    expr = F.dot(value, attention_scores, ["seq_k"])
    expr = F.dot(W_o, expr, ["inner", "head"])
    expr = expr.grad(X)

    # TODO: Test output better


def test_relu_hessian():
    batch, in_dim, out_dim = symbols("batch in_dim out_dim")
    X = Variable("X", batch=batch, in_dim=in_dim)
    W = Variable("W", in_dim=in_dim, out_dim=out_dim)
    b = Variable("b", out_dim=out_dim)

    Y = F.relu(X @ W + b)
    loss = F.sum(Y * Y)
    grad = loss.grad(W)
    hessian = grad.grad(W)
    # TODO: Test output better


def test_batch_norm():
    batch, features = symbols("batch features")
    X = Variable("X", batch, features)
    gamma = Variable("gamma", features)
    beta = Variable("beta", features)
    epsilon = 1

    mean = F.mean(X, ["batch"])
    var = F.mean((X - mean) ** 2, ["batch"])
    X_norm = (X - mean) / F.sqrt(var + epsilon)
    Y = gamma * X_norm + beta

    loss = F.sum(Y)
    grad = loss.grad(gamma)

    # TODO: Test output better


def test_l1_reg():
    in_dim, out_dim = symbols("in_dim out_dim")
    W = Variable("W", in_dim=in_dim, out_dim=out_dim)
    l1_reg = F.sum(F.abs(W))
    grad = l1_reg.grad(W)

    # TODO: Test output better


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


@pytest.mark.parametrize("keepdim", [False, True])
def test_max(keepdim):
    i, j = symbols("i j")
    X = Variable("X", i, j)
    ts = rand_values([X], {i: 3, j: 4})
    tX = ts[X].rename(None)

    for dims in [(), ["i"], ["j"], ["i", "j"]]:
        res = F.max(X, dims, keepdim=keepdim).evaluate(ts)
        adims = tuple(ts[X].names.index(d) for d in dims)
        expected = tX.amax(adims, keepdim=keepdim)
        names = ts[X].names if keepdim else tuple(n for n in ts[X].names if n not in dims)
        # Note that dims == () is the same as dims == ("i", "j") in torch.amax.
        expected = expected if dims == () else expected.rename(*names)
        assert_close(res, expected)

    res = F.max(X, keepdim=keepdim).grad(X).simplify().evaluate(ts)
    expected = (tX == tX.amax()).float()
    assert_close(res, expected)

    res = F.max(X, "i", keepdim=keepdim).grad(X).simplify().evaluate(ts)
    expected = (tX == tX.amax(dim=0).values).float()
    assert_close(res, expected)

    res = F.max(X, "j", keepdim=keepdim).grad(X).simplify().evaluate(ts)
    expected = (tX == tX.amax(dim=1).values[:, None]).float()
    assert_close(res, expected)

    res = F.max(X, ("i", "j"), keepdim=keepdim).grad(X).simplify().evaluate(ts)
    expected = torch.ones_like(tX)
    assert_close(res, expected)
