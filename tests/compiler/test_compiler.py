"""Differential test suite for the new AOT compiler (tensorgrad.compiler).

Every test compares compile_to_callable(...) against the reference oracle
tensorgrad.extras.evaluate.evaluate(...) on small random named tensors
(all dims <= 8), plus a few direct torch cross-checks (conv1d/autograd).

Known-broken cases are marked pytest.xfail(strict=False) with a comment
naming the failure mode, so they flip to PASS automatically once fixed.
"""

import inspect
from fractions import Fraction

import pytest
import sympy
import torch
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Delta, Ones, Product, Sum, Variable, Zero
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)

RTOL, ATOL = 1e-4, 1e-6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rand_named(var: Variable, dims: dict, positive: bool = False) -> torch.Tensor:
    """Random named tensor matching a Variable's shape."""
    if var.order == 0:
        return torch.randn([])
    edges, sizes = zip(*var.shape.items())
    t = torch.randn([dims[s] for s in sizes])
    if positive:
        t = t.abs() + 0.3  # bounded away from 0 for log/sqrt/pow(-k)
    return t.rename(*edges)


def assert_compiles_like_evaluate(tensors, values, dims, rtol=RTOL, atol=ATOL, fn=None):
    """Compile `tensors` (already simplified) and compare each output to evaluate().

    Handles the named-tensor edge-order difference: evaluate() may return
    outputs with a different edge ORDER than the compiler, so the reference
    is aligned to the compiled output's names before comparison.
    Returns the compiled callable (for reuse / introspection).
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    f = fn if fn is not None else compile_to_callable(*tensors)
    outs = f(dict(values), dict(dims))
    if len(tensors) == 1:
        outs = (outs,)
    for idx, (t, out) in enumerate(zip(tensors, outs)):
        ref = evaluate(t, dict(values), dict(dims))
        if out.dim() > 0:
            ref = ref.align_to(*out.names)
        torch.testing.assert_close(
            out.rename(None),
            # .to(out.dtype): argmax/equal come back int64 from evaluate but
            # float from the compiler; values must still agree exactly.
            ref.rename(None).to(out.dtype),
            rtol=rtol,
            atol=atol,
            msg=lambda m, idx=idx: f"output {idx}: {m}",
        )
    return f


def _simplify(t, mode):
    if mode == "full":
        return t.full_simplify()
    if mode == "simple":
        return t.simplify()
    if mode == "noexpand":
        return t.simplify({"expand_functions": False})
    return t


def run_case(tensors, variables, simplify="full", values=None, dims=None, positive=False):
    dims = dims if dims is not None else DIMS
    torch.manual_seed(0)
    if values is None:
        values = {v: rand_named(v, dims, positive=positive) for v in variables}
    simped = [_simplify(t, simplify) for t in tensors]
    assert_compiles_like_evaluate(simped, values, dims)


# ---------------------------------------------------------------------------
# Shared symbols / variables (core algebra)
# ---------------------------------------------------------------------------

i, j, k, l = sympy.symbols("i j k l")
DIMS = {i: 3, j: 4, k: 5, l: 6}

A = Variable("A", i, j)
A2 = Variable("A2", i, j)
B = Variable("B", j, k)
C = Variable("C", k, l)
x = Variable("x", i)
y = Variable("y", j)
Asq = Variable("A", **{"i": i, "j": i})
Bsq = Variable("B", **{"j": i, "i": i})


# ===========================================================================
# 1. Core algebra: contractions, deltas, sums, renames, symmetry
# ===========================================================================

CORE_CASES = {
    # --- plain contractions ---
    "matvec": lambda: ([A @ y], [A, y]),
    "matmat": lambda: ([A @ B], [A, B]),
    "chain_ABC": lambda: ([A @ B @ C], [A, B, C]),
    "outer_product": lambda: ([Product([x, y])], [x, y]),
    "frobenius2": lambda: ([F.frobenius2(A)], [A]),
    "trace_via_delta": lambda: ([Asq @ Delta(i, "i", "j")], [Asq]),
    "trace_fn": lambda: ([F.trace(Asq)], [Asq]),
    "AAT_repeated_var": lambda: ([A @ A.rename(i="k")], [A]),
    "xx_inner": lambda: ([x @ x], [x]),
    "dot_xx": lambda: ([F.dot(x, x, dim="i")], [x]),
    # --- delta hyperedges / diagonal embedding ---
    "delta3_full_contraction": lambda: (
        [Delta(i, "a", "b", "c") @ Variable("xa", a=i) @ Variable("yb", b=i) @ Variable("zc", c=i)],
        [Variable("xa", a=i), Variable("yb", b=i), Variable("zc", c=i)],
    ),
    "delta3_diag_embedding": lambda: (
        [Delta(i, "a", "b", "c") @ Variable("zc", c=i)],
        [Variable("zc", c=i)],
    ),
    "diag_fn": lambda: ([F.diag(x.rename(i="a"), ["a", "b"])], [x]),
    "diag_times_vector": lambda: (
        [F.diag(x.rename(i="a"), ["a", "b"]) @ Variable("yb", b=i)],
        [x, Variable("yb", b=i)],
    ),
    "trace_of_product": lambda: (
        [F.trace(Asq @ Bsq.rename(i="k").rename(j="i", k="j"))],
        [Asq, Bsq],
    ),
    "order4_delta_embedding": lambda: (
        [Variable("X", i) @ Delta(i, "i", "j", "k", "l")],
        [Variable("X", i)],
    ),
    # --- scalar broadcast / reductions / Ones ---
    "x_plus_2": lambda: ([x + 2], [x]),
    "A_plus_1p5": lambda: ([A + 1.5], [A]),
    "sum_all": lambda: ([F.sum(A)], [A]),
    "sum_dim_i": lambda: ([F.sum(A, dim="i")], [A]),
    "mean_all": lambda: ([F.mean(A)], [A]),
    "mean_dim_j": lambda: ([F.mean(A, dim="j")], [A]),
    "ones_standalone": lambda: ([Ones(i=i)], []),
    "x_times_ones_broadcast": lambda: ([Product([x, Ones(j=j)])], [x]),
    # --- sums and weights ---
    "sum_div_3": lambda: ([(A + A2) / 3], [A, A2]),
    "single_div_3": lambda: ([A / 3], [A]),
    "nested_sums": lambda: ([(A + (A2 - A) / 2) + A2 / 4], [A, A2]),
    "x_plus_ones": lambda: ([x + Ones(i=i)], [x]),
    "subtraction": lambda: ([A - A2], [A, A2]),
    "scalar_constant": lambda: ([Sum([Product([])], [2.5])], []),
    "scalar_constant_times_A": lambda: ([Sum([Product([])], [2.5]) @ A], [A]),
    "negation": lambda: ([-A], [A]),
    "fraction_weights": lambda: ([Sum([A, A2], [Fraction(1, 3), Fraction(2, 7)])], [A, A2]),
    # --- renames and gradients through renames ---
    "rename_in_product": lambda: (
        [x.rename(i="a") @ Variable("B2", a=i, b=k)],
        [x, Variable("B2", a=i, b=k)],
    ),
    "grad_xx_renamed_edges": lambda: ([(x @ x).grad(x)], [x]),
    "grad_frob_wrt_A": lambda: ([F.frobenius2(A @ y).grad(A)], [A, y]),
    "double_rename": lambda: (
        [x.rename(i="p").rename(p="a") @ Variable("C2", a=i, b=l)],
        [x, Variable("C2", a=i, b=l)],
    ),
    # --- Zero alongside live outputs ---
    "zero_among_outputs": lambda: ([x - x, A @ y, Zero(i=i, j=j)], [x, A, y]),
    # --- multiple outputs sharing structure ---
    "loss_plus_grad_shared": lambda: _loss_grad_shared(),
    "three_outputs": lambda: ([A @ B, A @ B @ C, F.frobenius2(A @ B)], [A, B, C]),
    # --- examples/main.py projection ---
    "projection_I_minus_uuT": lambda: (
        [Delta(i, "i", "j") - Variable("u", i) @ Variable("u", i).rename(i="j") / Delta(i)],
        [Variable("u", i)],
    ),
    "trace_of_projection": lambda: (
        [F.trace(Delta(i, "i", "j") - Variable("u", i) @ Variable("u", i).rename(i="j") / Delta(i))],
        [Variable("u", i)],
    ),
}


def _loss_grad_shared():
    W = Variable("W", **{"o": j, "in": i})
    X = Variable("X", **{"in": i, "b": k})
    Y = Variable("Y", **{"o": j, "b": k})
    frob = F.frobenius2(W @ X - Y)
    return [frob, frob.grad(W)], [W, X, Y]


@pytest.mark.parametrize("case", CORE_CASES.keys())
def test_core_algebra(case):
    tensors, variables = CORE_CASES[case]()
    run_case(tensors, variables)


def test_identity_collapse_simplify_only():
    # Delta(j,'j','a') @ B collapses the identity; passes with .simplify()
    Bja = Variable("B", **{"a": j, "b": k})
    run_case([Delta(j, "j", "a") @ Bja], [Bja], simplify="simple")


@pytest.mark.parametrize("expr_kind", ["Sx", "quadratic"])
def test_symmetric_variable(expr_kind):
    torch.manual_seed(0)
    S = Variable("S", **{"i": i, "j": i}).with_symmetries("i j")
    Sv = torch.randn(DIMS[i], DIMS[i])
    Sv = ((Sv + Sv.T) / 2).rename("i", "j")
    xv = torch.randn(DIMS[i], names=["i"])
    if expr_kind == "Sx":
        expr = S @ x.rename(i="j")
    else:
        expr = x.rename(i="i") @ S @ x.rename(i="j")
    run_case([expr], [S, x], values={S: Sv, x: xv})


# ===========================================================================
# 2. Functions: elementwise ops, desugared ops, softmax family, gradients
# ===========================================================================

fi, fj, fb, fc, fo = symbols("fi fj fb fc fo")
FDIMS = {fi: 3, fj: 4, fb: 3, fc: 4, fo: 5}
fx = Variable("fx", fi, fj)
fy = Variable("fy", fi, fj)
fv = Variable("fv", fi)


def _fvals(positive=False):
    torch.manual_seed(0)
    xt = torch.randn(3, 4, names=("fi", "fj"))
    if positive:
        xt = (xt.rename(None).abs() + 0.3).rename("fi", "fj")
    yt = torch.randn(3, 4, names=("fi", "fj"))
    vt = torch.randn(3, names=("fi",))
    return {fx: xt, fy: yt, fv: vt}


ELEMENTWISE = [
    ("pow_-2", lambda t: F.pow(t, -2), True),
    ("pow_-1", lambda t: F.pow(t, -1), True),
    ("pow_2", lambda t: F.pow(t, 2), False),
    ("pow_3", lambda t: F.pow(t, 3), False),
    ("pow_half", lambda t: F.pow(t, Fraction(1, 2)), True),
    ("exp", F.exp, False),
    ("log", F.log, True),
    ("relu", F.relu, False),
    ("sign", F.sign, False),
    ("abs", F.abs, False),
    # gt0 at generic nonzero inputs only: compiler uses (x>=0), evaluate uses
    # (x>0) -- they disagree exactly at 0 (measure-zero semantic difference).
    ("gt0", F.gt0, False),
]


@pytest.mark.parametrize("name,fn,needs_pos", ELEMENTWISE, ids=[e[0] for e in ELEMENTWISE])
def test_elementwise(name, fn, needs_pos):
    vals = _fvals(positive=needs_pos)
    # value, elementwise product, and full contraction
    run_case([fn(fx)], [], values=vals, dims=FDIMS)
    run_case([fn(fx) * fy], [], values=vals, dims=FDIMS)
    run_case([fn(fx) @ fy], [], values=vals, dims=FDIMS)


@pytest.mark.parametrize(
    "name,fn,needs_pos",
    [("tanh", F.tanh, False), ("sigmoid", F.sigmoid, False), ("sqrt", F.sqrt, True)],
    ids=["tanh", "sigmoid", "sqrt"],
)
def test_desugared_value_and_grad(name, fn, needs_pos):
    vals = _fvals(positive=needs_pos)
    run_case([fn(fx)], [], values=vals, dims=FDIMS)
    run_case([F.sum(fn(fx)).grad(fx)], [], values=vals, dims=FDIMS)


def test_softmax_expanded():
    run_case([F.softmax(fx, dim="fj")], [], values=_fvals(), dims=FDIMS)


def test_softmax_unexpanded_native():
    # Unexpanded Function node survives and compiles to native torch.softmax.
    sm = F.softmax(fx, dim="fj")
    run_case([sm], [], values=_fvals(), dims=FDIMS, simplify="noexpand")


def test_softmax_two_axis_expanded():
    run_case([F.softmax(fx, dim=("fi", "fj"))], [], values=_fvals(), dims=FDIMS)


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda: F.argmax(fx, dim="fj"),
        lambda: F.equal(fx, fy),
        lambda: F.equal(fx, fx),
        lambda: F.max(fx, dim="fj"),
        lambda: F.max(fx),
    ],
    ids=["argmax", "equal_xy", "equal_xx", "max_dim", "max_all"],
)
def test_native_ops_noexpand(expr_fn):
    # argmax/equal/max compile as native torch ops without expansion.
    # (argmax dtype: compiler returns float-cast, evaluate int64; the helper
    # casts the reference to the compiled dtype before comparing.)
    run_case([expr_fn()], [], values=_fvals(), dims=FDIMS, simplify="noexpand")


def test_grad_relu_matmul():
    X = Variable("X", fb, fi)
    W = Variable("W", fi, fo)
    torch.manual_seed(0)
    vals = {X: torch.randn(3, 3, names=("fb", "fi")), W: torch.randn(3, 5, names=("fi", "fo"))}
    run_case([F.sum(F.relu(X @ W)).grad(W)], [], values=vals, dims=FDIMS)


@pytest.mark.parametrize(
    "expr_fn,positive",
    [
        (lambda: F.sum(F.exp(F.log(fx) * 2)).grad(fx), True),
        (lambda: F.sum(F.log(F.exp(fx) + F.exp(fy))).grad(fx), False),
        (lambda: F.sum(F.pow(fx, -1)).grad(fx), True),
    ],
    ids=["exp_log_chain", "log_of_exp_sum", "pow_-1_grad"],
)
def test_function_gradients(expr_fn, positive):
    run_case([expr_fn()], [], values=_fvals(positive=positive), dims=FDIMS)


def test_cross_entropy_value_and_grad():
    logits = Variable("l", fc)
    targ = Variable("t", fc)
    torch.manual_seed(0)
    vals = {
        logits: torch.randn(4, names=("fc",)),
        targ: torch.softmax(torch.randn(4), dim=0).rename("fc"),
    }
    ce = F.cross_entropy(logits, targ, dim="fc")
    run_case([ce, ce.grad(logits)], [], values=vals, dims=FDIMS)


def test_second_derivative_and_hvp():
    u = Variable("u", fi)
    torch.manual_seed(0)
    vals = {fv: torch.randn(3, names=("fi",)), u: torch.randn(3, names=("fi",))}
    expr = F.sum(F.pow(fv, 3))
    run_case([expr.grad(fv).grad(fv)], [], values=vals, dims=FDIMS)  # order-2 output
    run_case([(expr.grad(fv) @ u).grad(fv)], [], values=vals, dims=FDIMS)  # HVP


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda: F.exp(fv) * fx,  # Hadamard over shared fi
        lambda: F.exp(fv) @ fx,  # contract fi
        lambda: F.exp(fx) * fv,
        lambda: F.relu(fx) * F.exp(fv),
    ],
    ids=["expv_times_x", "expv_contract_x", "expx_times_v", "relu_times_exp"],
)
def test_function_broadcast(expr_fn):
    # Delta-broadcast paths through Function nodes
    run_case([expr_fn()], [], values=_fvals(), dims=FDIMS)


def test_dot_and_pairwise_distance():
    vals = _fvals()
    run_case([F.dot(fx, fy, dim="fj")], [], values=vals, dims=FDIMS)
    pd = F.pairwise_distance(fx, fy, dim="fj")
    run_case([pd, F.sum(pd).grad(fx)], [], values=vals, dims=FDIMS)


# ===========================================================================
# 3. Structured constants: Convolution and Reshape
# ===========================================================================

w_in_s, k_s, w_out_s = symbols("w_in ck w_out")
W_IN, K = 6, 3
W_OUT = W_IN - K + 1  # 4
CDIMS = {w_in_s: W_IN, k_s: K, w_out_s: W_OUT}


def _conv_setup():
    conv = F.Convolution(w_in_s, k=k_s, w_out=w_out_s)
    cx = Variable("cx", w_in=w_in_s)
    ker = Variable("ker", k=k_s)
    torch.manual_seed(0)
    vals = {cx: torch.randn(W_IN).rename("w_in"), ker: torch.randn(K).rename("k")}
    return conv, cx, ker, vals


def test_conv_unfold():
    conv, cx, ker, vals = _conv_setup()
    t = (cx @ conv).full_simplify()
    f = assert_compiles_like_evaluate([t], vals, CDIMS)
    # manual reference: unfold[j, o] = x[o + j] (cross-correlation, no flip)
    out = f(dict(vals), CDIMS)
    xv = vals[cx].rename(None)
    man = torch.zeros(K, W_OUT)
    for o in range(W_OUT):
        for jj in range(K):
            man[jj, o] = xv[o + jj]
    man = man.rename("k", "w_out").align_to(*out.names)
    torch.testing.assert_close(out.rename(None), man.rename(None), rtol=RTOL, atol=ATOL)


def test_conv1d_full_vs_torch():
    conv, cx, ker, vals = _conv_setup()
    t = (cx @ conv @ ker).full_simplify()
    f = assert_compiles_like_evaluate([t], vals, CDIMS)
    out = f(dict(vals), CDIMS)
    ref = torch.nn.functional.conv1d(
        vals[cx].rename(None).view(1, 1, -1), vals[ker].rename(None).view(1, 1, -1)
    ).view(-1)
    torch.testing.assert_close(out.rename(None), ref, rtol=RTOL, atol=ATOL)


def test_conv_grad_wrt_kernel_and_input():
    conv, cx, ker, vals = _conv_setup()
    loss = F.sum((cx @ conv @ ker) ** 2)
    gk = loss.grad(ker).full_simplify()
    gx = loss.grad(cx).full_simplify()  # transposed-conv pattern
    f = assert_compiles_like_evaluate([gk, gx], vals, CDIMS)
    # autograd cross-check for the kernel gradient
    gk_out, _ = f(dict(vals), CDIMS)
    kv = vals[ker].rename(None).clone().requires_grad_(True)
    yy = torch.nn.functional.conv1d(vals[cx].rename(None).view(1, 1, -1), kv.view(1, 1, -1))
    (yy**2).sum().backward()
    torch.testing.assert_close(gk_out.align_to("k").rename(None), kv.grad.view(-1), rtol=RTOL, atol=ATOL)


def test_conv_symmetric_sizes():
    # k_size == w_out triggers the symmetry set in the Convolution constructor
    s_in, s_k = symbols("s_in s_k")
    conv = F.Convolution(s_in, k=s_k, w_out=s_k)
    cx = Variable("cx", w_in=s_in)
    dims = {s_in: 5, s_k: 3}
    torch.manual_seed(0)
    vals = {cx: torch.randn(5).rename("w_in")}
    run_case([cx @ conv], [], values=vals, dims=dims)


def test_conv_kwarg_order_variant():
    # mnist argument order F.Convolution(h_in, h_out, hk=ks): kernel/output
    # roles swapped, valid by the j/k symmetry of the constant.
    h_in, h_out, ks = symbols("h_in h_out ks")
    conv = F.Convolution(h_in, h_out, hk=ks)
    cx = Variable("cx", w_in=h_in)
    dims = {h_in: 6, h_out: 4, ks: 3}
    torch.manual_seed(0)
    vals = {cx: torch.randn(6).rename("w_in")}
    run_case([cx @ conv], [], values=vals, dims=dims)


def test_reshape_flatten_and_unflatten():
    ri, rj, rl = symbols("ri rj rl")
    rdims = {ri: 4, rj: 2, rl: 2}
    resh = F.Reshape(ri=ri, rj=rj, rl=rl)
    M = Variable("M", rj=rj, rl=rl)
    v = Variable("v", ri=ri)
    torch.manual_seed(0)
    Mv = torch.randn(2, 2).rename("rj", "rl")
    vv = torch.randn(4).rename("ri")
    # flatten
    t = (M @ resh).full_simplify()
    f = assert_compiles_like_evaluate([t], {M: Mv}, rdims)
    out = f({M: Mv}, rdims)
    torch.testing.assert_close(
        out.align_to("ri").rename(None), Mv.rename(None).reshape(4), rtol=RTOL, atol=ATOL
    )
    # unflatten
    run_case([v @ resh], [], values={v: vv}, dims=rdims)


def test_mnist_style_conv2d_loss_and_grads():
    # 2D conv via two Convolution constants (as in examples/mnist.py), relu,
    # linear head, MSE loss; loss + grad(kernel) + grad(linear) share one DAG.
    batch, c0, w0, h0, out_s, ks = symbols("batch c0 w0 h0 out ks")
    c1, h1, w1 = symbols("c1 h1 w1")
    Bn, C0, W0n, H0n, OUT, KS = 2, 1, 6, 6, 3, 3
    C1, H1n, W1n = 2, H0n - KS + 1, W0n - KS + 1

    data = Variable("data", batch, c0, w0, h0)
    targets = Variable("targets", batch, out_s)
    h_conv = F.Convolution(h0, h1, hk=ks)
    w_conv = F.Convolution(w0, w1, wk=ks)
    kernel = Variable("kernel_0", c0, c1, hk=ks, wk=ks)
    linear = Variable("lin", c1, h1, w1, out_s)

    xx = F.relu(data @ (kernel @ h_conv @ w_conv)).simplify()
    logits = (xx @ linear).full_simplify(expand=False)
    loss = F.mean(F.mean((logits - targets) ** 2, dim="out"), dim="batch")
    gk = loss.grad(kernel).full_simplify(expand=False)
    gl = loss.grad(linear).full_simplify(expand=False)
    loss = loss.full_simplify(expand=False)

    shapes = {batch: Bn, c0: C0, w0: W0n, h0: H0n, ks: KS, out_s: OUT, c1: C1, h1: H1n, w1: W1n}
    torch.manual_seed(1)
    vals = {
        data: torch.randn(Bn, C0, W0n, H0n).rename("batch", "c0", "w0", "h0"),
        targets: torch.randn(Bn, OUT).rename("batch", "out"),
        kernel: torch.randn(C0, C1, KS, KS).rename("c0", "c1", "hk", "wk"),
        linear: torch.randn(C1, H1n, W1n, OUT).rename("c1", "h1", "w1", "out"),
    }
    f = assert_compiles_like_evaluate([loss, gk, gl], vals, shapes)
    # structure constants are hoisted once per shape signature: a second call
    # must not create a new specialization
    f(dict(vals), shapes)
    assert len(f._specializations) == 1


# ===========================================================================
# 4. ML workloads (examples/mlp.py shape) + specialization reuse
# ===========================================================================

batch, in_dim, hidden, out_dim = symbols("batch in_dim hidden out_dim")
mx = Variable("x", batch, in_dim)
my = Variable("y", batch, out_dim)
W1 = Variable("W1", in_dim, hidden)
b1 = Variable("b1", batch, hidden)
W2 = Variable("W2", hidden, out_dim)
b2 = Variable("b2", batch, out_dim)
MLP_PARAMS = [W1, b1, W2, b2]
mlp_h = F.relu(mx @ W1 + b1)
mlp_logits = mlp_h @ W2 + b2
mlp_loss = F.mean(F.cross_entropy(mlp_logits, my, dim="out_dim"))


def _onehot(bs, kk):
    lab = torch.randint(0, kk, (bs,))
    return torch.nn.functional.one_hot(lab, num_classes=kk).float().refine_names("batch", "out_dim")


@pytest.fixture(scope="module")
def mlp_program():
    """Loss + all 4 grads of the examples/mlp.py MLP, compiled once."""
    loss_s = mlp_loss.full_simplify()
    grads = [mlp_loss.grad(p).full_simplify() for p in MLP_PARAMS]
    f = compile_to_callable(loss_s, *grads)
    return [loss_s] + grads, f


def _mlp_values(dims):
    vals = {v: rand_named(v, dims) for v in [mx, W1, b1, W2, b2]}
    vals[my] = _onehot(dims[batch], dims[out_dim])
    return vals


def test_mlp_multi_signature_reuse(mlp_program):
    tensors, f = mlp_program
    sigs = [
        {batch: 4, in_dim: 5, hidden: 6, out_dim: 3},
        {batch: 7, in_dim: 5, hidden: 6, out_dim: 3},  # batch change only
        {batch: 3, in_dim: 4, hidden: 2, out_dim: 2},  # everything changes
    ]
    torch.manual_seed(0)
    for dims in sigs:
        vals = _mlp_values(dims)
        assert_compiles_like_evaluate(tensors, vals, dims, fn=f)
    assert len(f._specializations) == 3
    # warm call at a known signature must reuse the cached specialization
    f(_mlp_values(sigs[0]), sigs[0])
    assert len(f._specializations) == 3


def test_mlp_accuracy_expression():
    pred = F.argmax(mlp_logits, dim="out_dim")
    targ = F.argmax(my, dim="out_dim")
    acc = F.mean(F.equal(pred, targ)).full_simplify()
    dims = {batch: 6, in_dim: 5, hidden: 4, out_dim: 3}
    torch.manual_seed(1)
    vals = _mlp_values(dims)
    run_case([acc], [], values=vals, dims=dims, simplify=None)


def test_linear_regression_loss_and_grad():
    n, d = symbols("n d")
    X = Variable("X", n, d)
    w = Variable("w", d)
    yv = Variable("yv", n)
    lo = F.frobenius2(X @ w - yv)
    ls, g = lo.full_simplify(), lo.grad(w).full_simplify()
    dims = {n: 8, d: 5}
    torch.manual_seed(2)
    vals = {v: rand_named(v, dims) for v in [X, w, yv]}
    f = assert_compiles_like_evaluate([ls, g], vals, dims)
    # autograd sanity
    lo_o, go = f(dict(vals), dims)
    Xr = vals[X].rename(None)
    wr = vals[w].rename(None).clone().requires_grad_(True)
    yr = vals[yv].rename(None)
    lt = ((Xr @ wr - yr) ** 2).sum()
    lt.backward()
    torch.testing.assert_close(lo_o.rename(None), lt.detach(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(go.align_to("d").rename(None), wr.grad, rtol=RTOL, atol=ATOL)


def test_attention_block_value_and_grad():
    seq, seq2, dk, dv = symbols("seq seq2 dk dv")
    q = Variable("q", seq, dk)
    kk = Variable("k", seq2, dk)
    vv = Variable("v", seq2, dv)
    att = F.softmax(q @ kk, dim="seq2") @ vv
    att_s = att.full_simplify()
    gq = F.sum(att).grad(q).full_simplify()
    dims = {seq: 4, seq2: 5, dk: 3, dv: 2}
    torch.manual_seed(3)
    vals = {v: rand_named(v, dims) for v in [q, kk, vv]}
    f = assert_compiles_like_evaluate([att_s, gq], vals, dims)
    # autograd sanity
    ao, go = f(dict(vals), dims)
    qr = vals[q].rename(None).clone().requires_grad_(True)
    ref = torch.softmax(qr @ vals[kk].rename(None).T, dim=1) @ vals[vv].rename(None)
    torch.testing.assert_close(ao.align_to("seq", "dv").rename(None), ref.detach(), rtol=RTOL, atol=ATOL)
    ref.sum().backward()
    torch.testing.assert_close(go.align_to("seq", "dk").rename(None), qr.grad, rtol=RTOL, atol=1e-5)


def test_weight_tied_repeated_variable_grad():
    # X appears twice in the product; checks the symmetry factor in the grad.
    ti, tj = symbols("ti tj")
    X = Variable("X", ti, tj)
    expr = F.frobenius2(X @ X.rename(ti="i2") - Delta(ti, "ti", "i2"))
    es, g = expr.full_simplify(), expr.grad(X).full_simplify()
    dims = {ti: 4, tj: 6}
    torch.manual_seed(4)
    vals = {X: rand_named(X, dims)}
    f = assert_compiles_like_evaluate([es, g], vals, dims)
    lo, go = f(dict(vals), dims)
    Xr = vals[X].rename(None).clone().requires_grad_(True)
    lt = ((Xr @ Xr.T - torch.eye(4)) ** 2).sum()
    torch.testing.assert_close(lo.rename(None), lt.detach(), rtol=RTOL, atol=ATOL)
    lt.backward()
    torch.testing.assert_close(go.align_to("ti", "tj").rename(None), Xr.grad, rtol=RTOL, atol=1e-5)


def test_deep_chain_grad_first_layer():
    # 4-layer relu chain, long backward path to the FIRST weight.
    b, d0, d1, d2, d3, d4 = symbols("b d0 d1 d2 d3 d4")
    xv = Variable("xv", b, d0)
    yv = Variable("yv", b, d4)
    Ws = [Variable(f"L{n}", di, do) for n, (di, do) in enumerate([(d0, d1), (d1, d2), (d2, d3), (d3, d4)])]
    hcur = xv
    for W in Ws[:-1]:
        hcur = F.relu(hcur @ W)
    lo = F.frobenius2(hcur @ Ws[-1] - yv)
    ls, g = lo.full_simplify(), lo.grad(Ws[0]).full_simplify()
    dims = {b: 5, d0: 4, d1: 3, d2: 4, d3: 3, d4: 2}
    torch.manual_seed(5)
    vals = {v: rand_named(v, dims) for v in [xv, yv, *Ws]}
    assert_compiles_like_evaluate([ls, g], vals, dims)


def test_repeat_call_determinism(mlp_program):
    tensors, f = mlp_program
    dims = {batch: 4, in_dim: 5, hidden: 6, out_dim: 3}
    torch.manual_seed(0)
    vals = _mlp_values(dims)
    outs1 = f(dict(vals), dims)
    outs2 = f(dict(vals), dims)
    for o1, o2 in zip(outs1, outs2):
        assert torch.equal(o1.rename(None), o2.rename(None))  # bit-identical, no state leakage


def test_float64_propagation():
    # NOTE: compiled result is compared against a manual fp64 torch pipeline,
    # not against evaluate(): evaluate() itself loses precision at fp64
    # (materializes 1/n through float32 Delta constants -- oracle gap).
    n, d = symbols("n d")
    X = Variable("X", n, d)
    w = Variable("w", d)
    m = F.mean(F.relu(X @ w)).full_simplify()
    torch.manual_seed(0)
    Xt = torch.randn(6, 4, names=("n", "d"), dtype=torch.float64)
    wt = torch.randn(4, names=("d",), dtype=torch.float64)
    out = compile_to_callable(m)({X: Xt, w: wt}, {n: 6, d: 4})
    assert out.dtype == torch.float64
    manual = torch.relu(Xt.rename(None) @ wt.rename(None)).mean()
    torch.testing.assert_close(out.rename(None), manual, rtol=1e-12, atol=1e-14)

    # Hoisted-constant path: a diag embedding materializes an eye ('delta')
    # constant, which must be built in the inputs' dtype (fp64), not fp32.
    vv = Variable("vv", d)
    diag = Product([vv, Delta(d, "d", "r", "c")]).simplify()
    vt = torch.randn(4, names=("d",), dtype=torch.float64)
    outd = compile_to_callable(diag)({vv: vt}, {d: 4})
    assert outd.dtype == torch.float64
    torch.testing.assert_close(
        outd.align_to("r", "c").rename(None), torch.diag(vt.rename(None)), rtol=1e-12, atol=1e-14
    )


# ===========================================================================
# 5. Invariants on generated code (regression guards)
# ===========================================================================


def _specialized_fn(f, dims):
    f(_mlp_values(dims), dims)
    key = tuple(sorted((s.name, v) for s, v in dims.items()))
    return f._specializations[key]


def test_no_materialized_high_order_constants(mlp_program):
    # The MLP-gradient program must not materialize any order>=2 constant
    # tensor except genuine diagonal embeddings (delta consts): hyperedge
    # Deltas must be eliminated into einsum subscripts, not dense tensors.
    _, f = mlp_program
    torch.manual_seed(0)
    fn = _specialized_fn(f, {batch: 4, in_dim: 5, hidden: 6, out_dim: 3})
    assert hasattr(fn, "_source")
    for name, param in inspect.signature(fn).parameters.items():
        if isinstance(param.default, torch.Tensor) and param.default.dim() >= 2:
            assert "delta" in name, (
                f"materialized order-{param.default.dim()} constant {name!r} "
                f"is not a diagonal embedding:\n{fn._source}"
            )


def test_einsum_count_mlp_loss_and_grads(mlp_program):
    # Regression guard on step-level CSE for the wine-shaped MLP program
    # (examples/mlp.py loss + 4 grads; einsum count is independent of the
    # concrete sizes, so we specialize at small dims).
    _, f = mlp_program
    torch.manual_seed(0)
    fn = _specialized_fn(f, {batch: 8, in_dim: 8, hidden: 6, out_dim: 3})
    n_einsum = fn._source.count("torch.einsum(")
    assert n_einsum <= 35, f"einsum count regressed: {n_einsum} > 35\n{fn._source}"


def _check_del_invariants(src: str):
    """No name may be referenced after its `del`; return names are never
    deleted. Applies to any generated straight-line program."""
    import re

    body_lines = src.split("\n")
    dead: set[str] = set()
    for line in body_lines:
        stripped = line.strip()
        toks = set(re.findall(r"\w+", stripped))
        used_after_del = toks & dead
        assert not used_after_del, f"deleted name(s) {used_after_del} referenced in: {stripped}\n{src}"
        if stripped.startswith("del "):
            dead |= {n.strip() for n in stripped[4:].split(",")}


def test_liveness_del_emission(mlp_program):
    # Intermediates are freed at their last use (codegen_torch.EMIT_DEL):
    # the source must contain dels, obey the no-use-after-del invariant,
    # and produce values identical to the EMIT_DEL=False emission.
    from tensorgrad.compiler import codegen_torch

    _, f = mlp_program
    dims = {batch: 4, in_dim: 5, hidden: 6, out_dim: 3}
    torch.manual_seed(0)
    vals = _mlp_values(dims)
    outs = f(dict(vals), dims)
    fn = _specialized_fn(f, dims)
    assert "del " in fn._source, fn._source
    _check_del_invariants(fn._source)

    exprs, _ = mlp_program
    old = codegen_torch.EMIT_DEL
    codegen_torch.EMIT_DEL = False
    try:
        f2 = compile_to_callable(*exprs)
        outs2 = f2(dict(vals), dims)
        fn2 = _specialized_fn(f2, dims)
    finally:
        codegen_torch.EMIT_DEL = old
    assert "del " not in fn2._source
    for a, b in zip(outs, outs2):
        torch.testing.assert_close(a.rename(None), b.align_to(*a.names).rename(None))


def test_output_alignment_ordered_match_wins():
    # Two variables share an edge SET with different order (tied-embedding
    # shape: wte (vocab, d) vs lm_head (d, vocab)). A gradient output whose
    # declared edge order exactly matches one variable must keep that order —
    # not be silently align_to'd onto the other variable (grad(wte) once
    # adopted lm_head's axis order because it came first in the var scan).
    i, j = sympy.symbols("i j")
    A = Variable("A", i, j)
    B = Variable("B", j, i)
    loss = F.sum(A * B)
    prog = compile_to_callable(loss.full_simplify(), loss.grad(B).full_simplify())
    dims = {i: 3, j: 4}
    torch.manual_seed(0)
    va, vb = rand_named(A, dims), rand_named(B, dims)
    _, g = prog({A: va, B: vb}, dims)
    declared = prog.outputs[1][1]
    assert tuple(g.names) == tuple(declared), (
        f"output permuted away from its declared order {declared} -> {g.names}"
    )
    # dloss/dB = A; compare through names, which must survive unpermuted.
    torch.testing.assert_close(
        g.align_to("j", "i").rename(None), va.rename(None).T
    )


# ===========================================================================
# 6. Known failures (xfail, strict=False: they PASS automatically once fixed)
# ===========================================================================

NOINPUT_REASON = (
    "compiler-bug: const-only program (no Variable inputs) generates "
    "'def _compiled(, _c0_...=...)' -- leading comma in the signature "
    "(codegen_torch.py specialize, args/const_args join)"
)


# Failure mode: zero variable inputs + >=1 hoisted constant -> SyntaxError.
@pytest.mark.xfail(strict=False, reason=NOINPUT_REASON)
def test_xfail_standalone_delta():
    run_case([Delta(i, "a", "b")], [], simplify=None)


# Failure mode: same empty-args SyntaxError, via Zero as the sole output.
@pytest.mark.xfail(strict=False, reason=NOINPUT_REASON)
def test_xfail_standalone_zero():
    run_case([(x - x).full_simplify()], [x], simplify=None)


# Failure mode: same empty-args SyntaxError, via a standalone Convolution.
@pytest.mark.xfail(strict=False, reason=NOINPUT_REASON)
def test_xfail_standalone_convolution():
    conv = F.Convolution(w_in_s, k=k_s, w_out=w_out_s)
    run_case([conv], [], values={}, dims=CDIMS, simplify=None)


# Failure mode: same empty-args SyntaxError, via a standalone Reshape.
@pytest.mark.xfail(strict=False, reason=NOINPUT_REASON)
def test_xfail_standalone_reshape():
    ri, rj, rl = symbols("ri rj rl")
    run_case([F.Reshape(ri=ri, rj=rj, rl=rl)], [], values={}, dims={ri: 4, rj: 2, rl: 2}, simplify=None)


# Failure mode: codegen _emit_reduce only supports single-axis softmax;
# an unexpanded multi-axis _SoftmaxFunction raises NotImplementedError.
# (Workaround: full_simplify() expands it -- see test_softmax_two_axis_expanded.)
@pytest.mark.xfail(strict=False, reason="not-implemented: multi-axis softmax in codegen _emit_reduce")
def test_xfail_softmax_two_axis_unexpanded():
    sm = F.softmax(fx, dim=("fi", "fj"))
    run_case([sm], [], values=_fvals(), dims=FDIMS, simplify="noexpand")


# Failure mode: runtime.CompiledProgram._aligned calls t.permute(perm) on a
# tensor that still carries names; torch forbids permute on named tensors.
# Fix: t.align_to(*var.edges) (or rename(None) before permute).
@pytest.mark.xfail(strict=False, reason="compiler-bug: permuted named input hits .permute on a named tensor in runtime._aligned")
def test_xfail_permuted_named_input():
    n, d = symbols("n d")
    X = Variable("X", n, d)
    w = Variable("w", d)
    expr = F.frobenius2(X @ w).full_simplify()
    f = compile_to_callable(expr)
    torch.manual_seed(0)
    Xt = torch.randn(6, 4, names=("n", "d"))
    wt = torch.randn(4, names=("d",))
    dims = {n: 6, d: 4}
    want = f({X: Xt, w: wt}, dims)
    got = f({X: Xt.align_to("d", "n"), w: wt}, dims)  # permuted edge order
    torch.testing.assert_close(got.rename(None), want.rename(None), rtol=RTOL, atol=ATOL)


# Failure mode (pre-existing OLD-backend gap, not the new compiler): the old
# to_pytorch backend's generated Reshape code calls math.isqrt but 'math' is
# missing from its exec namespace -> NameError.
@pytest.mark.xfail(strict=False, reason="not-implemented: old to_pytorch backend Reshape codegen references math without importing it")
def test_xfail_old_backend_standalone_reshape():
    from tensorgrad.extras.to_pytorch import compile_to_callable as compile_old

    ri, rj, rl = symbols("ri rj rl")
    rdims = {ri: 4, rj: 2, rl: 2}
    resh = F.Reshape(ri=ri, rj=rj, rl=rl)
    f = compile_old(resh)
    out = f({}, rdims)
    ref = evaluate(resh, {}, dict(rdims))
    torch.testing.assert_close(out.rename(None), ref.align_to(*out.names).rename(None), rtol=RTOL, atol=ATOL)
