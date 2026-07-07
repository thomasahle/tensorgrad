"""Tests for the IR factoring pass (tensorgrad/compiler/factor.py).

Every transform is checked by randomized numeric equivalence: the factored
program is compared against the unfactored program (module toggle off) AND
against the evaluate() oracle, on random inputs at small dims.

Structural assertions pin the three behaviors the pass exists for:
  1. un-distribution — the shared softmax vector s = e/Z is factored out of
     expanded gradient terms (single exp, op count near the fused path);
  2. the distribution decision — the unexpanded MLP gradient must not fix an
     (out_dim x out_dim x batch) Sum into the einsum structure (68GB at
     large dims), while at tiny dims keeping it dense stays legal;
  3. Sum-boundary delta materialization — the LayerNorm-style centering
     Jacobian Delta_bb' Delta_ss' (Delta_dd' - J/d) must contract without a
     6-axis dense intermediate.
"""

import sys
from fractions import Fraction

import pytest
import torch
from sympy import symbols

import tensorgrad.compiler.factor as factor_mod
import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate
from tensorgrad.tensor import Delta, Ones, Product, Sum, Zero

torch.set_num_threads(2)

RTOL, ATOL = 1e-4, 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rand_named(var: Variable, dims: dict) -> torch.Tensor:
    if var.order == 0:
        return torch.randn([])
    edges, sizes = zip(*var.shape.items())
    return torch.randn([dims[s] for s in sizes]).rename(*edges)


def compile_factored(*tensors, on=True):
    """Compile with the factoring toggle in a known state. The toggle is read
    at specialization time, so we specialize eagerly on first call through a
    wrapper that pins it."""
    prog = compile_to_callable(*tensors)

    def call(values, dims):
        old = factor_mod.FACTOR
        factor_mod.FACTOR = on
        try:
            return prog(dict(values), dict(dims))
        finally:
            factor_mod.FACTOR = old

    call.program = prog
    return call


def source_of(prog) -> str:
    (spec,) = prog._specializations.values()
    return spec._source


def trace_intermediates(fn, values, dims):
    """Max element count over every tensor materialized inside the compiled
    function (inputs, constants, einsum partials, named intermediates)."""
    fn(values, dims)  # force specialization outside the trace
    max_numel, shapes = 0, set()

    def local(frame, event, arg):
        nonlocal max_numel
        for val in frame.f_locals.values():
            if isinstance(val, torch.Tensor):
                max_numel = max(max_numel, val.numel())
                shapes.add(tuple(val.shape))
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code.co_name == "_compiled" else None

    sys.settrace(tracer)
    try:
        fn(values, dims)
    finally:
        sys.settrace(None)
    return max_numel, shapes


def assert_equivalent(tensors, variables, dims, seeds=(0, 1, 2), rtol=RTOL, atol=ATOL):
    """Factored vs unfactored vs evaluate(), at several random inputs.
    Returns the factored callable (for structural inspection)."""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    f_on = compile_factored(*tensors, on=True)
    f_off = compile_factored(*tensors, on=False)
    for seed in seeds:
        torch.manual_seed(seed)
        values = {v: rand_named(v, dims) for v in variables}
        out_on = f_on(values, dims)
        out_off = f_off(values, dims)
        if len(tensors) == 1:
            out_on, out_off = (out_on,), (out_off,)
        for idx, (t, a, b) in enumerate(zip(tensors, out_on, out_off)):
            b = b.align_to(*a.names) if a.dim() > 0 else b
            torch.testing.assert_close(
                a.rename(None), b.rename(None), rtol=rtol, atol=atol,
                msg=lambda m, idx=idx, seed=seed: f"factored != unfactored (output {idx}, seed {seed}): {m}",
            )
            ref = evaluate(t, dict(values), dict(dims))
            if a.dim() > 0:
                ref = ref.align_to(*a.names)
            torch.testing.assert_close(
                a.rename(None), ref.rename(None).to(a.dtype), rtol=rtol, atol=atol,
                msg=lambda m, idx=idx, seed=seed: f"factored != evaluate (output {idx}, seed {seed}): {m}",
            )
    return f_on


# ---------------------------------------------------------------------------
# Shared expression builders
# ---------------------------------------------------------------------------

b, v = symbols("b v")


def primitive_softmax_loss():
    x = Variable("x", b, v)
    w = Variable("w", b, v)
    e = F.exp(x)
    s = e * F.pow(F.sum(e, dim="v", keepdims=True), -1)
    loss = F.sum(s * w)
    return x, w, loss


def fused_softmax_loss():
    x = Variable("x", b, v)
    w = Variable("w", b, v)
    loss = F.sum(F.softmax(x, dim=["v"]) * w)
    return x, w, loss


# ---------------------------------------------------------------------------
# 1. Un-distribution (hoisting): primitive softmax gradient
# ---------------------------------------------------------------------------


def test_softmax_grad_equivalent_and_shares_exp():
    """Full pipeline: expanded primitive-softmax grad is numerically exact and
    the factored program computes exp(x) exactly once (the shared e and the
    hoisted structure s * (w - <w, s>))."""
    x, w, loss = primitive_softmax_loss()
    g = loss.grad(x).full_simplify()
    fn = assert_equivalent([loss.full_simplify(), g], [x, w], {b: 4, v: 8})
    src = source_of(fn.program)
    # The stabilization pass (#16) normally supersedes the shared exp with a
    # single fused torch.softmax; with it disabled the factored exp must be
    # computed exactly once. Either way: no duplicated exponentials.
    n_exp, n_sm = src.count("torch.exp"), src.count("torch.softmax")
    assert (n_exp, n_sm) in {(0, 1), (1, 0)}, f"exp not shared:\n{src}"


def test_softmax_grad_ops_within_2x_of_fused():
    """Acceptance: the primitives-defined softmax gradient compiles to at most
    2x the torch calls of the hand-fused (F.softmax signature) path."""
    x, w, loss = primitive_softmax_loss()
    xf, wf, loss_f = fused_softmax_loss()
    dims = {b: 4, v: 8}
    torch.manual_seed(0)

    fn = compile_factored(loss.full_simplify(), loss.grad(x).full_simplify())
    fn({x: rand_named(x, dims), w: rand_named(w, dims)}, dims)
    fn_fused = compile_factored(loss_f.full_simplify(), loss_f.grad(xf).full_simplify())
    fn_fused({xf: rand_named(xf, dims), wf: rand_named(wf, dims)}, dims)

    n_prim = source_of(fn.program).count("torch.")
    n_fused = source_of(fn_fused.program).count("torch.")
    assert n_prim <= 2 * n_fused, f"{n_prim} torch calls vs fused {n_fused}"


def test_softmax_grad_light_simplify_compact():
    """The researcher path (light .simplify(), no expansion): the compiled
    gradient must not materialize anything larger than the input. Pre-pass,
    this program built (V,V,B,B) Jacobian intermediates."""
    x, w, loss = primitive_softmax_loss()
    g = loss.grad(x).simplify()
    fn = assert_equivalent([loss.simplify(), g], [x, w], {b: 5, v: 4})
    B, V = 16, 8
    torch.manual_seed(0)
    dims = {b: B, v: V}
    values = {x: rand_named(x, dims), w: rand_named(w, dims)}
    max_numel, shapes = trace_intermediates(fn, values, dims)
    assert max_numel <= B * V, f"non-compact: {max_numel} > {B * V}; shapes {sorted(shapes)}"


def test_hoist_shared_matrix_factor():
    """sum_t w_t * (A @ B_t) hoists A: the factored source contracts A once.
    A transposed use of A (different axis roles) must NOT join the group."""
    i = symbols("i")
    A = Variable("A", **{"i": i, "j": i})
    B1 = Variable("B1", **{"j": i, "k": i})
    B2 = Variable("B2", **{"j": i, "k": i})
    B3 = Variable("B3", **{"j": i, "k": i})
    expr = (A @ B1 + 3 * (A @ B2) - 2 * (A @ B3)).full_simplify()
    dims = {i: 6}
    fn = assert_equivalent(expr, [A, B1, B2, B3], dims)
    src = source_of(fn.program)
    # One contraction against A (the hoisted one) instead of three.
    n_contractions = sum("einsum" in line for line in src.splitlines())
    assert n_contractions <= 2, f"A was not hoisted:\n{src}"

    # Transposed decoy: A.T @ B is a different role signature — still correct.
    At = A.rename(i="j", j="i")
    expr2 = (A @ B1 + At @ B2).full_simplify()
    assert_equivalent(expr2, [A, B1, B2], dims)


def test_hoist_scalar_and_vector_factors():
    """Shared 0-dim and 1-dim factors across terms (the s = e/Z pattern)."""
    n = symbols("n")
    z = Variable("z")  # scalar
    u = Variable("u", n)
    p = Variable("p", n)
    q = Variable("q", n)
    expr = (z * u * p + 2 * (z * u * q)).full_simplify()
    assert_equivalent(expr, [z, u, p, q], {n: 7})


# ---------------------------------------------------------------------------
# 2. Distribution decision: the unexpanded-MLP hazard
# ---------------------------------------------------------------------------


def _mlp_grads(expand: bool):
    batch, in_dim, hidden, out_dim = symbols("batch in_dim hidden out_dim")
    x = Variable("x", batch, in_dim)
    y = Variable("y", batch, out_dim)
    W1 = Variable("W1", in_dim, hidden)
    b1 = Variable("b1", batch, hidden)
    W2 = Variable("W2", hidden, out_dim)
    b2 = Variable("b2", batch, out_dim)
    h = F.relu(x @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.mean(F.cross_entropy(logits, y, dim="out_dim"))
    params = [W1, b1, W2, b2]
    outs = [loss.full_simplify(expand=expand)] + [
        loss.grad(p).full_simplify(expand=expand) for p in params
    ]
    return (batch, in_dim, hidden, out_dim), (x, y, W1, b1, W2, b2), outs


def test_unexpanded_mlp_grad_max_intermediate():
    """Regression for the 68GB hazard: full_simplify(expand=False) leaves the
    (Delta_oo' - softmax) Sum as an einsum operand; unfactored, the backward
    materializes an (O,O,B) dense tensor. The factored program must stay
    within O(B * max(H, O, D)) at out_dim where O*O*B >> B*H.

    GEMM batching is disabled here: it deliberately stacks k small same-size
    tensors (a size-gated LAUNCH optimization that would not fire at scale),
    which breaks this small-dims proxy for FACTORING's scaling invariant."""
    from tensorgrad.compiler import gemm_batch

    gemm_batch.GEMM_BATCHING = False
    try:
        _check_mlp_grad_max_intermediate()
    finally:
        gemm_batch.GEMM_BATCHING = True


def _check_mlp_grad_max_intermediate():
    (batch, in_dim, hidden, out_dim), vars_, outs = _mlp_grads(expand=False)
    B, D, H, O = 64, 13, 32, 48
    dims = {batch: B, in_dim: D, hidden: H, out_dim: O}
    x, y, W1, b1, W2, b2 = vars_
    torch.manual_seed(0)
    values = {
        x: torch.randn(B, D).refine_names("batch", "in_dim"),
        y: torch.nn.functional.one_hot(torch.randint(0, O, (B,)), O)
        .float()
        .refine_names("batch", "out_dim"),
        W1: (torch.randn(D, H) / D**0.5).refine_names("in_dim", "hidden"),
        b1: torch.zeros(B, H).refine_names("batch", "hidden"),
        W2: (torch.randn(H, O) / H**0.5).refine_names("hidden", "out_dim"),
        b2: torch.zeros(B, O).refine_names("batch", "out_dim"),
    }
    fn = compile_factored(*outs, on=True)
    max_numel, shapes = trace_intermediates(fn, values, dims)
    bound = 2 * B * max(D, H, O)
    assert max_numel <= bound, (
        f"unexpanded MLP grad materializes {max_numel} elements (> {bound}); "
        f"largest shapes: {sorted(shapes, key=lambda s: -torch.Size(s).numel())[:4]}"
    )

    # Numeric equivalence against the fully-expanded compile at these dims.
    _, _, outs_exp = _mlp_grads(expand=True)
    fn_exp = compile_factored(*outs_exp, on=True)
    r1, r2 = fn(values, dims), fn_exp(values, dims)
    for a, c in zip(r1, r2):
        c = c.align_to(*a.names) if a.dim() > 0 else c
        torch.testing.assert_close(a.rename(None), c.rename(None), rtol=1e-3, atol=1e-5)


def test_unexpanded_mlp_grad_correct_small():
    """Same program, tiny dims, against evaluate(). At tiny dims the cost
    model may legitimately keep the Sum dense — semantics must not change."""
    (batch, in_dim, hidden, out_dim), vars_, outs = _mlp_grads(expand=False)
    x, y, W1, b1, W2, b2 = vars_
    dims = {batch: 5, in_dim: 3, hidden: 4, out_dim: 2}
    assert_equivalent(outs, list(vars_), dims, seeds=(0, 1))


# ---------------------------------------------------------------------------
# 3. Sum-boundary delta materialization (LN centering Jacobian pattern)
# ---------------------------------------------------------------------------

bb, ss, dd = symbols("bb ss dd")


def _centering_contraction():
    """v[b',s',d'] * (Delta_bb' Delta_ss' (Delta_dd' - J/d)) — the exact
    bridge-agent pattern: the Sum's terms carry all six edges, so lowering's
    delta elimination is blocked at the Sum boundary."""
    v6 = Variable("v", b1=bb, s1=ss, d1=dd)
    term1 = Product([Delta(bb, "b0", "b1"), Delta(ss, "s0", "s1"), Delta(dd, "d0", "d1")])
    term2 = Product(
        [Delta(bb, "b0", "b1"), Delta(ss, "s0", "s1"), Ones(d0=dd), Ones(d1=dd)]
    )
    S6 = Sum([term1, term2], [1, Fraction(-1, 4)])
    return v6, Product([v6, S6])


def test_ln_centering_jacobian_no_dense_blowup():
    """No (b,b',s,s',d,d') 6-axis dense intermediate: the factored program
    stays at O(B*S*D) where the unfactored one materializes (B*S*D)^2."""
    v6, expr = _centering_contraction()
    B, S, D = 6, 5, 4
    dims = {bb: B, ss: S, dd: D}
    fn = assert_equivalent(expr, [v6], dims)
    torch.manual_seed(0)
    values = {v6: rand_named(v6, dims)}
    max_numel, shapes = trace_intermediates(fn, values, dims)
    assert max_numel <= 2 * B * S * D, (
        f"centering Jacobian materialized {max_numel} elements; shapes {sorted(shapes)}"
    )
    assert all(len(s) < 6 for s in shapes), f"6-axis intermediate: {sorted(shapes)}"

    # Sanity: without factoring this really is the (B*S*D)^2 blowup.
    fn_off = compile_factored(expr, on=False)
    max_off, _ = trace_intermediates(fn_off, values, dims)
    assert max_off == (B * S * D) ** 2


def test_ln_vjp_grad_correct_and_compact():
    """End-to-end LayerNorm VJP (expanded): correct and O(B*S*D)."""
    xin = Variable("xin", batch=bb, seq=ss, d=dd)
    g = Variable("g", d=dd)
    gy = Variable("gy", batch=bb, seq=ss, d=dd)
    mu = F.mean(xin, dim="d", keepdims=True)
    dx = xin - mu
    var = F.mean(dx * dx, dim="d", keepdims=True)
    ln = dx * F.pow(var + 1e-5, Fraction(-1, 2)) * g
    gx = F.sum(ln * gy).grad(xin).full_simplify()
    B, S, D = 4, 5, 6
    dims = {bb: B, ss: S, dd: D}
    fn = assert_equivalent(gx, [xin, g, gy], dims, rtol=1e-3, atol=1e-5)
    torch.manual_seed(0)
    values = {t: rand_named(t, dims) for t in (xin, g, gy)}
    max_numel, shapes = trace_intermediates(fn, values, dims)
    assert max_numel <= B * S * D, f"LN VJP non-compact: {max_numel}; {sorted(shapes)}"


# ---------------------------------------------------------------------------
# 4. Trivial absorption unit cases (ones folding, orphan factors, zeros)
# ---------------------------------------------------------------------------


def test_ones_orphan_full_contraction():
    """Distributing a full contraction over the (delta - J/d) Sum leaves the
    ones-term with orphaned summed wires: their sizes must fold into the
    weight (sum over ones = dimension factor)."""
    u1 = Variable("u", d0=dd)
    u2 = Variable("w", d1=dd)
    S2 = Sum(
        [Delta(dd, "d0", "d1"), Product([Ones(d0=dd), Ones(d1=dd)])],
        [1, Fraction(-1, 4)],
    )
    expr = Product([u1, u2, S2])  # scalar: u^T (I - J/4) w
    assert_equivalent(expr, [u1, u2], {dd: 4})


def test_zero_sum_term_dropped():
    """A Zero term inside an un-simplified Sum operand must vanish cleanly."""
    u1 = Variable("u", d0=dd)
    u2 = Variable("w", d1=dd)
    S2 = Sum([Delta(dd, "d0", "d1"), Zero(d0=dd, d1=dd)])
    expr = Product([u1, u2, S2])
    assert_equivalent(expr, [u1, u2], {dd: 5})


def test_diag_output_delta_preserved():
    """A genuine diagonal embedding (two free output edges on one delta) must
    keep its eye constant — absorption only fires when it can alias wires."""
    u1 = Variable("u", d0=dd)
    expr = Product([u1, Delta(dd, "d0", "d1", "d2")])  # out: (d1, d2) diag
    assert_equivalent(expr, [u1], {dd: 5})


def test_multiple_shape_specializations():
    """Factoring runs per shape signature: different dims through the same
    program stay correct (and the cost model may decide differently)."""
    x, w, loss = primitive_softmax_loss()
    g = loss.grad(x).full_simplify()
    prog = compile_to_callable(loss.full_simplify(), g)
    for B, V in [(2, 3), (16, 8), (3, 17)]:
        dims = {b: B, v: V}
        torch.manual_seed(B * V)
        values = {x: rand_named(x, dims), w: rand_named(w, dims)}
        lo, gr = prog(dict(values), dict(dims))
        ref = evaluate(g, dict(values), dict(dims))
        torch.testing.assert_close(
            gr.rename(None), ref.align_to(*gr.names).rename(None), rtol=RTOL, atol=ATOL
        )


# ---------------------------------------------------------------------------
# 5. Randomized equivalence battery over all transforms
# ---------------------------------------------------------------------------

i, j, k = symbols("i j k")
A_ = Variable("A", i, j)
B_ = Variable("B", j, k)
C_ = Variable("C", j, k)
x_ = Variable("x", i)
y_ = Variable("y", j)

BATTERY = {
    # hoisting with permuted terms and weights
    "hoist_weighted": lambda: (
        (Product([A_, B_]) + 2 * Product([A_, C_])).full_simplify(),
        [A_, B_, C_],
        {i: 3, j: 4, k: 5},
    ),
    # distribution: einsum over an explicit Sum operand (not simplified away)
    "distribute_sum_operand": lambda: (
        Product([x_, Sum([Product([A_, y_]), Product([A_, Ones(j=j)])])]),
        [A_, x_, y_],
        {i: 6, j: 4},
    ),
    # nested linear flatten + hoist through it
    "nested_sums": lambda: (
        (A_ @ (B_ + C_) + A_ @ B_).full_simplify(),
        [A_, B_, C_],
        {i: 3, j: 4, k: 5},
    ),
    # elementwise map factors (exp shared through hoisting)
    "shared_exp_terms": lambda: (
        (F.exp(y_) * y_ + F.exp(y_) * F.pow(y_, 2)).full_simplify(),
        [y_],
        {j: 6},
    ),
    # softmax jacobian-vector against primitive definition, light simplify
    "softmax_jvp_light": lambda: (
        F.sum(
            (F.exp(x_) * F.pow(F.sum(F.exp(x_), dim="i", keepdims=True), -1)),
            dim="i",
        )
        .grad(x_)
        .simplify(),
        [x_],
        {i: 5},
    ),
}


@pytest.mark.parametrize("name", sorted(BATTERY))
def test_battery_equivalence(name):
    expr, variables, dims = BATTERY[name]()
    assert_equivalent(expr, variables, dims, seeds=(0, 1, 2))
