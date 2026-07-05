"""Layout assignment + matmul cell selection tests.

Every rewrite the layout pass and the mm/bmm/matmul/addmm cells introduce is
covered by a numeric-equivalence test: the same program compiled with the
optimizations ON and OFF (LAYOUT_ASSIGN / MATMUL_CELLS / ADDMM_FUSION) must
produce identical values on random inputs at small dims.
"""

import math
from fractions import Fraction

import pytest
import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Variable
import tensorgrad.compiler.codegen_torch as cg
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler.layout import assign_layouts, matmul_groups

RTOL, ATOL = 1e-5, 1e-6


@pytest.fixture
def flags():
    """Restore optimization flags after each test."""
    saved = (cg.LAYOUT_ASSIGN, cg.MATMUL_CELLS, cg.ADDMM_FUSION)
    yield
    cg.LAYOUT_ASSIGN, cg.MATMUL_CELLS, cg.ADDMM_FUSION = saved


def _values(variables, dims, seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = {}
    for v in variables:
        sizes = [int(dims[s]) for s in v.shape.values()]
        vals[v] = torch.randn(sizes, generator=g).rename(*v.shape.keys())
    return vals


def _run(tensors, vals, dims):
    fn = compile_to_callable(*tensors)
    outs = fn(vals, dims)
    outs = outs if isinstance(outs, tuple) else (outs,)
    srcs = [s._source for s in fn._specializations.values()]
    return outs, srcs[-1]


def check_equivalent(tensors, variables, dims, seed=0):
    """Compile with optimizations on vs off; outputs must match."""
    vals = _values(variables, dims, seed)
    on, src_on = _run(tensors, vals, dims)
    cg.LAYOUT_ASSIGN, cg.MATMUL_CELLS, cg.ADDMM_FUSION = False, False, False
    try:
        off, _ = _run(tensors, vals, dims)
    finally:
        cg.LAYOUT_ASSIGN, cg.MATMUL_CELLS, cg.ADDMM_FUSION = True, True, True
    assert len(on) == len(off)
    for a, b in zip(on, off):
        an = a.align_to(*b.names).rename(None) if a.names != b.names else a.rename(None)
        torch.testing.assert_close(an, b.rename(None), rtol=RTOL, atol=ATOL)
    return src_on


# ---------------------------------------------------------------------------
# matmul cells: every mm/bmm/matmul shape
# ---------------------------------------------------------------------------


def test_plain_mm(flags):
    i, j, k = symbols("i j k")
    A, B = Variable("A", i, j), Variable("B", j, k)
    src = check_equivalent([A @ B], [A, B], {i: 3, j: 4, k: 5})
    assert "torch.mm(" in src and "einsum" not in src


def test_mm_transposed_operands(flags):
    # contraction on the FIRST axis of both operands: free op(A) transposes
    i, j, k = symbols("i j k")
    A, B = Variable("A", j, i), Variable("B", j, k)
    prod = F.dot(A, B, dim=("j", "j"))
    src = check_equivalent([prod], [A, B], {i: 3, j: 4, k: 5})
    assert "torch.mm(" in src and "einsum" not in src


def test_batched_bmm(flags):
    b, i, j, k = symbols("b i j k")
    A, B = Variable("A", b, i, j), Variable("B", b, j, k)
    prod = F.dot(A, B, dim=("j", "j"))
    src = check_equivalent([prod], [A, B], {b: 2, i: 3, j: 4, k: 5})
    assert "torch.bmm(" in src or "torch.matmul(" in src
    assert "einsum" not in src


def test_multi_batch_matmul(flags):
    b, h, i, j, k = symbols("b h i j k")
    A, B = Variable("A", b, h, i, j), Variable("B", b, h, j, k)
    prod = F.dot(A, B, dim=("j", "j"))
    src = check_equivalent([prod], [A, B], {b: 2, h: 3, i: 4, j: 5, k: 6})
    assert "torch.matmul(" in src and "einsum" not in src


def test_merged_m_group(flags):
    # x (b,s,d) @ W (d,h): M-block {b,s} merges through a reshape view
    b, s, d, h = symbols("b s d h")
    x, W = Variable("x", b, s, d), Variable("W", d, h)
    src = check_equivalent([x @ W], [x, W], {b: 2, s: 3, d: 4, h: 5})
    assert "torch.mm(" in src and "einsum" not in src


def test_merged_k_group(flags):
    # weight gradient shape: contract over BOTH leading dims at once
    b, s, d, h = symbols("b s d h")
    x, gy = Variable("x", b, s, d), Variable("gy", b, s, h)
    prod = x @ gy  # Product contracts all shared edges (b, s)
    src = check_equivalent([prod], [x, gy], {b: 2, s: 3, d: 4, h: 5})
    assert "torch.mm(" in src and "einsum" not in src


def test_matvec_empty_m(flags):
    i, j = symbols("i j")
    A, v = Variable("A", i, j), Variable("v", i)
    prod = F.dot(A, v, dim=("i", "i"))
    src = check_equivalent([prod], [A, v], {i: 3, j: 4})
    assert "einsum" not in src


def test_batched_dot_stays_hadamard(flags):
    # m and n empty: (A*B).sum beats a degenerate bmm — must NOT become a cell
    b, v = symbols("b v")
    A, B = Variable("A", b, v), Variable("B", b, v)
    prod = F.sum(A * B, dim="v")
    src = check_equivalent([prod], [A, B], {b: 3, v: 4})
    assert "torch.mm" not in src and "torch.bmm" not in src


def test_hadamard_no_contraction_stays_mul(flags):
    b, v = symbols("b v")
    A, B = Variable("A", b, v), Variable("B", b, v)
    src = check_equivalent([A * B], [A, B], {b: 3, v: 4})
    assert "torch.mm" not in src and "einsum" not in src


def test_three_operand_chain(flags):
    i, j, k, l = symbols("i j k l")
    A, B, C = Variable("A", i, j), Variable("B", j, k), Variable("C", k, l)
    src = check_equivalent([A @ B @ C], [A, B, C], {i: 2, j: 3, k: 4, l: 5})
    assert "einsum" not in src  # both pairwise steps are mm-shaped


# ---------------------------------------------------------------------------
# addmm fusion
# ---------------------------------------------------------------------------


def test_addmm_bias_n_side(flags):
    b, s, d, h = symbols("b s d h")
    x, W, bias = Variable("x", b, s, d), Variable("W", d, h), Variable("bias", h)
    src = check_equivalent([x @ W + bias], [x, W, bias], {b: 2, s: 3, d: 4, h: 5})
    assert "torch.addmm(" in src


def test_addmm_weighted(flags):
    b, s, d, h = symbols("b s d h")
    x, W, bias = Variable("x", b, s, d), Variable("W", d, h), Variable("bias", h)
    src = check_equivalent([2 * (x @ W) - 3 * bias], [x, W, bias], {b: 2, s: 3, d: 4, h: 5})
    assert "torch.addmm(" in src and "alpha=" in src and "beta=" in src


def test_addmm_shared_mm_not_fused(flags):
    # the mm term has TWO structural consumers (the bias add and an
    # elementwise pow): it must stay a standalone mm
    b, d, h = symbols("b d h")
    x, W, bias = Variable("x", b, d), Variable("W", d, h), Variable("bias", h)
    y = x @ W
    z = y + bias
    src = check_equivalent([z, F.pow(y, 3)], [x, W, bias], {b: 2, d: 3, h: 4})
    assert "torch.addmm(" not in src and "torch.mm(" in src


def test_mlp_gradients_equivalence(flags):
    b, s, d, h = symbols("b s d h")
    x = Variable("x", batch=b, seq=s, d=d)
    w1, b1 = Variable("w1", d=d, d_mlp=h), Variable("b1", d_mlp=h)
    w2, b2 = Variable("w2", d_mlp=h, d=d), Variable("b2", d=d)
    c = math.sqrt(2.0 / math.pi)
    hpre = x @ w1 + b1
    act = 0.5 * hpre * (1 + F.tanh(c * (hpre + 0.044715 * F.pow(hpre, 3))))
    z = act @ w2 + b2
    loss = F.sum(z * z)
    grads = [loss.grad(v) for v in (w1, b1, w2, b2, x)]
    check_equivalent([loss, *grads], [x, w1, b1, w2, b2], {b: 2, s: 3, d: 4, h: 5})


# ---------------------------------------------------------------------------
# layout assignment: consumers that must adapt to assigned layouts
# ---------------------------------------------------------------------------


def test_linear_permuted_terms(flags):
    # B's logical axis order is transposed relative to A; the Sum aligns by
    # edge name, so one term needs a (view) permute
    i, j = symbols("i j")
    A, B = Variable("A", i, j), Variable("B", j, i)
    check_equivalent([A + B], [A, B], {i: 3, j: 4})


def test_softmax_layout(flags):
    b, s, v = symbols("b s v")
    x = Variable("x", b, s, v)
    sm = F.softmax(x, dim="s")  # non-last logical axis
    loss = F.sum(sm * sm)
    check_equivalent([sm, loss, loss.grad(x)], [x], {b: 2, s: 3, v: 4})


def test_log_softmax_layout(flags):
    b, v = symbols("b v")
    x, y = Variable("x", b, v), Variable("y", b, v)
    ls = F.log_softmax(x, dim="v")
    loss = -F.sum(ls * y)
    check_equivalent([loss, loss.grad(x)], [x, y], {b: 3, v: 5})


def test_max_and_argmax_layout(flags):
    b, s, v = symbols("b s v")
    x = Variable("x", b, s, v)
    check_equivalent([F.max(x, dim="s")], [x], {b: 2, s: 3, v: 4})


def test_sum_reduction_layout(flags):
    b, s, v = symbols("b s v")
    x = Variable("x", b, s, v)
    check_equivalent([F.sum(x, dim="s")], [x], {b: 2, s: 3, v: 4})


def test_broadcast_expand_layout(flags):
    b, s, v = symbols("b s v")
    x, y = Variable("x", v), Variable("y", b, s, v)
    check_equivalent([x + y, y + x], [x, y], {b: 2, s: 3, v: 4})


def test_outputs_returned_in_logical_order(flags):
    # a program whose internal layout differs from the logical output order
    b, s, d, h = symbols("b s d h")
    x, W = Variable("x", b, s, d), Variable("W", d, h)
    y = x @ W
    vals = _values([x, W], {b: 2, s: 3, d: 4, h: 5})
    fn = compile_to_callable(y)
    out = fn(vals, {b: 2, s: 3, d: 4, h: 5})
    ref = vals[x].rename(None) @ vals[W].rename(None)
    torch.testing.assert_close(out.align_to("b", "s", "h").rename(None), ref)


def test_gradient_lands_in_variable_order(flags):
    # gradients vote the variable's edge order; must still be correct
    b, v = symbols("b v")
    x, w = Variable("x", b, v), Variable("w", b, v)
    mu = F.mean(x, dim="v", keepdims=True)
    dx = x - mu
    var = F.mean(dx * dx, dim="v", keepdims=True)
    ln = dx * F.pow(var + 1e-5, Fraction(-1, 2))
    loss = F.sum(ln * w)
    g = loss.grad(x)
    check_equivalent([loss, g], [x, w], {b: 4, v: 6})


def test_attention_pattern(flags):
    # multi-head attention: batch wires force einsum fallbacks + views;
    # numeric equivalence is the contract
    b, s, sk, hd, e, dv = symbols("b s sk hd e dv")
    q = Variable("q", batch=b, seq=s, head=hd, hs=e)
    k = Variable("k", batch=b, seq_k=sk, head=hd, hs=e)
    v = Variable("v", batch=b, seq_k=sk, head=hd, dv=dv)
    att = F.softmax(F.dot(q, k, dim=("hs", "hs")) / 2.0, dim="seq_k")
    out = F.dot(att, v, dim=("seq_k", "seq_k"))
    loss = F.sum(out * out)
    dims = {b: 2, s: 3, sk: 3, hd: 2, e: 4, dv: 4}
    check_equivalent([out, loss, loss.grad(q)], [q, k, v], dims)


def test_gather_and_scatter_layout(flags):
    # embedding lookup fwd + scatter bwd with layout assignment on
    b, s, vv, d = symbols("b s v d")
    idx = Variable("idx", batch=b, seq=s)
    table = Variable("table", v=vv, d=d)
    emb = F.gather(table, idx, dim="v")
    loss = F.sum(emb * emb)
    g = loss.grad(table)
    dims = {b: 2, s: 3, vv: 5, d: 4}
    gt = torch.randint(0, 5, (2, 3)).float().rename("batch", "seq")
    tt = torch.randn(5, 4).rename("v", "d")
    fn = compile_to_callable(emb, loss, g)
    e1, l1, g1 = fn({idx: gt, table: tt}, dims)
    cg.LAYOUT_ASSIGN, cg.MATMUL_CELLS, cg.ADDMM_FUSION = False, False, False
    try:
        fn2 = compile_to_callable(emb, loss, g)
        e2, l2, g2 = fn2({idx: gt, table: tt}, dims)
    finally:
        cg.LAYOUT_ASSIGN, cg.MATMUL_CELLS, cg.ADDMM_FUSION = True, True, True
    torch.testing.assert_close(e1.rename(None), e2.align_to(*e1.names).rename(None))
    torch.testing.assert_close(l1.rename(None), l2.rename(None))
    torch.testing.assert_close(g1.rename(None), g2.align_to(*g1.names).rename(None))


def test_random_contractions_fuzz(flags):
    """Random 2-operand contractions: one contracted edge, the remaining
    shared edges become batch dims; kept edges form m/n groups."""
    import random

    rng = random.Random(7)
    names = ["a", "b", "c", "d", "e"]
    syms = dict(zip(names, symbols("a b c d e")))
    done = 0
    for trial in range(40):
        n1 = rng.sample(names, rng.randint(1, 4))
        n2 = rng.sample(names, rng.randint(1, 4))
        shared = sorted(set(n1) & set(n2))
        if not shared:
            continue
        A = Variable("A", **{n: syms[n] for n in n1})
        B = Variable("B", **{n: syms[n] for n in n2})
        con = rng.choice(shared)
        prod = F.dot(A, B, dim=(con, con))
        dims = {syms[n]: rng.randint(2, 4) for n in names}
        check_equivalent([prod], [A, B], dims, seed=trial)
        done += 1
        if done >= 12:
            break
    assert done >= 8


# ---------------------------------------------------------------------------
# layout pass unit checks
# ---------------------------------------------------------------------------


def test_matmul_groups_classification():
    from tensorgrad.compiler.lower import lower_program

    i, j, k = symbols("i j k")
    A, B = Variable("A", i, j), Variable("B", j, k)
    _, [(node, _)] = lower_program([(A @ B).simplify()])
    assert matmul_groups(node) is not None
    C = Variable("C", i, j)
    _, [(h, _)] = lower_program([(A * C).simplify()])
    assert matmul_groups(h) is None  # no contracted wire


def test_assign_layouts_identity_for_pinned():
    from tensorgrad.compiler.lower import lower_program
    from tensorgrad.compiler.ir import toposort, InputNode

    i, j, k = symbols("i j k")
    A, B = Variable("A", i, j), Variable("B", j, k)
    builder, outputs = lower_program([(A @ B).simplify()])
    order = toposort([n for n, _ in outputs])
    phys = assign_layouts(order, outputs)
    for n in order:
        if isinstance(n, InputNode):
            assert phys[id(n)] == tuple(range(n.order))
        assert sorted(phys[id(n)]) == list(range(n.order))  # a permutation


# ---------------------------------------------------------------------------
# static strides + torch.compile fullgraph
# ---------------------------------------------------------------------------


def test_static_strides_modes_agree():
    """STATIC_STRIDES=True (spec-time integer strides) and =False (runtime
    .stride() queries) must produce identical values; only the True form is
    fullgraph-traceable (no torch calls returning python ints)."""
    from tensorgrad.compiler import affine_convolution, affine_shift

    w_in, kk, w_out, i, o = symbols("w_in k w_out i o")
    x, ker = Variable("x", w_in), Variable("ker", kk)
    conv = (x @ affine_convolution(w_in, kk, w_out, stride=2) @ ker).full_simplify()
    x2 = Variable("x2", i)
    shifted = (x2 @ affine_shift(2, i=i, o=o)).simplify()  # boundary: pad path
    dims = {w_in: 11, kk: 3, w_out: 5, i: 8, o: 8}
    g = torch.Generator().manual_seed(0)
    vals = {
        x: torch.randn(11, generator=g).rename("w_in"),
        ker: torch.randn(3, generator=g).rename("k"),
        x2: torch.randn(8, generator=g).rename("i"),
    }
    on, src_on = _run([conv, shifted], vals, dims)
    assert ".stride(" not in src_on and ".storage_offset(" not in src_on
    saved = cg.STATIC_STRIDES
    cg.STATIC_STRIDES = False
    try:
        off, src_off = _run([conv, shifted], vals, dims)
    finally:
        cg.STATIC_STRIDES = saved
    assert ".stride(" in src_off  # the escape hatch really queries at runtime
    for a, b in zip(on, off):
        torch.testing.assert_close(a.rename(None), b.rename(None), rtol=RTOL, atol=ATOL)


def test_torch_compile_fullgraph_smoke():
    """torch_compile=True: the generated program (matmul cells, layout
    permute views, static-stride affine views) must trace as ONE dynamo
    graph — no fullgraph fallback — and match the eager specialization."""
    i, j, k = symbols("i j k")
    A, B, bias = Variable("A", i, j), Variable("B", j, k), Variable("bias", k)
    y = F.softmax(A @ B + bias, dim="k")
    dims = {i: 4, j: 5, k: 6}
    vals = _values([A, B, bias], dims)
    r1 = compile_to_callable(y)(vals, dims)
    compiled = compile_to_callable(y, torch_compile=True)
    r2 = compiled(vals, dims)
    torch.testing.assert_close(
        r2.align_to(*r1.names).rename(None), r1.rename(None), rtol=RTOL, atol=ATOL
    )
    assert compiled.used_fullgraph_fallback is False


def test_flags_off_is_identity():
    """LAYOUT_ASSIGN=False must reproduce pre-pass behavior: no permutes
    added by layout, no cells."""
    saved = (cg.LAYOUT_ASSIGN, cg.MATMUL_CELLS, cg.ADDMM_FUSION)
    cg.LAYOUT_ASSIGN, cg.MATMUL_CELLS, cg.ADDMM_FUSION = False, False, False
    try:
        i, j, k = symbols("i j k")
        A, B = Variable("A", i, j), Variable("B", j, k)
        fn = compile_to_callable(A @ B)
        vals = _values([A, B], {i: 3, j: 4, k: 5})
        fn(vals, {i: 3, j: 4, k: 5})
        src = list(fn._specializations.values())[-1]._source
        assert "torch.einsum(" in src
        assert "torch.mm(" not in src
    finally:
        cg.LAYOUT_ASSIGN, cg.MATMUL_CELLS, cg.ADDMM_FUSION = saved
