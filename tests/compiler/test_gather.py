"""Tests for the gather/scatter tier (integer embedding lookup).

The language has no gather primitive: indexing `table` along `dim` with the
integer-valued (float-carried) tensor `idx` is the contraction with the
F.one_hot indicator (the local `gather` helper below). The compiler emits
torch.index_select for the forward and a zeros().index_add_ scatter for the
table-gradient, instead of the dense one-hot matmul both are equal to.

Correctness is checked against the evaluate() oracle, torch.autograd, and the
explicit one-hot matmul formulation; a microbench asserts the compiled paths
actually beat the one-hot formulation at embedding-scale sizes.
"""

import time

import pytest
import sympy
import torch

import tensorgrad.functions as F
from tensorgrad import Variable, Zero
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

from tests.compiler.test_compiler import assert_compiles_like_evaluate, rand_named

torch.set_num_threads(2)

RTOL, ATOL = 1e-4, 1e-6


def gather(table: Variable, idx, dim: str):
    """Embedding lookup spelled as the indicator contraction (the deleted
    F.gather sugar): sum_v table[v, ...] * [idx == v]."""
    return table @ F.one_hot(idx, table.shape[dim], dim)


def rand_idx(var: Variable, dims: dict, num_classes: int) -> torch.Tensor:
    """Random integral-valued float tensor for an index Variable."""
    sizes = [dims[s] for s in var.shape.values()]
    t = torch.randint(0, num_classes, sizes).float()
    return t.rename(*var.shape.keys()) if var.order else t


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_gather_forward_matches_evaluate():
    V, D, B, S = sympy.symbols("V D B S")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B, S=S)
    dims = {V: 7, D: 3, B: 2, S: 5}
    g = gather(table, idx, "V").full_simplify()
    values = {table: rand_named(table, dims), idx: rand_idx(idx, dims, 7)}
    assert_compiles_like_evaluate(g, values, dims)


def test_gather_forward_matches_onehot_matmul():
    V, D, B = sympy.symbols("V D B")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B)
    dims = {V: 7, D: 3, B: 11}
    tval, ival = rand_named(table, dims), rand_idx(idx, dims, 7)
    g = gather(table, idx, "V").full_simplify()
    out = compile_to_callable(g)({table: tval, idx: ival}, dims)
    onehot = (ival.rename(None).long().unsqueeze(1) == torch.arange(7)).float()
    ref = onehot @ tval.rename(None)  # (B, D)
    torch.testing.assert_close(out.align_to("B", "D").rename(None), ref, rtol=RTOL, atol=ATOL)


def test_gather_middle_axis_and_multi_table_edges():
    # The vocab edge is not the first table axis, and the table has two
    # feature edges; idx has two edges as well.
    V, D1, D2, B, S = sympy.symbols("V D1 D2 B S")
    table = Variable("table", D1=D1, V=V, D2=D2)
    idx = Variable("idx", B=B, S=S)
    dims = {V: 6, D1: 3, D2: 4, B: 2, S: 3}
    g = gather(table, idx, "V").full_simplify()
    values = {table: rand_named(table, dims), idx: rand_idx(idx, dims, 6)}
    f = assert_compiles_like_evaluate(g, values, dims)
    # Direct torch cross-check
    out = f(dict(values), dict(dims))
    tval = values[table].align_to("V", "D1", "D2").rename(None)
    ref = tval[values[idx].rename(None).long()]  # (B, S, D1, D2)
    torch.testing.assert_close(
        out.align_to("B", "S", "D1", "D2").rename(None), ref, rtol=RTOL, atol=ATOL
    )


def test_gather_scalar_idx():
    V, D = sympy.symbols("V D")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx")  # order-0 index
    dims = {V: 5, D: 4}
    g = gather(table, idx, "V").full_simplify()
    values = {table: rand_named(table, dims), idx: torch.tensor(3.0)}
    assert_compiles_like_evaluate(g, values, dims)


def test_gather_rename():
    V, D, B = sympy.symbols("V D B")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B)
    dims = {V: 5, D: 4, B: 3}
    g = gather(table, idx, "V").rename(D="feat", B="pos").full_simplify()
    assert g.edges == {"feat", "pos"}
    values = {table: rand_named(table, dims), idx: rand_idx(idx, dims, 5)}
    assert_compiles_like_evaluate(g, values, dims)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_gather_grad_matches_evaluate_and_autograd():
    V, D, B, S = sympy.symbols("V D B S")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B, S=S)
    dims = {V: 7, D: 3, B: 2, S: 5}
    tval, ival = rand_named(table, dims), rand_idx(idx, dims, 7)
    values = {table: tval, idx: ival}

    g = gather(table, idx, "V")
    loss = F.sum(F.pow(g, 2)) / 2
    grad = loss.grad(table).full_simplify()
    f = assert_compiles_like_evaluate([loss.full_simplify(), grad], values, dims)

    # torch.autograd cross-check
    t2 = tval.rename(None).clone().requires_grad_(True)
    (t2[ival.rename(None).long()] ** 2).sum().div(2).backward()
    _, gout = f(dict(values), dict(dims))
    torch.testing.assert_close(
        gout.align_to("V", "D").rename(None), t2.grad, rtol=RTOL, atol=ATOL
    )


def test_gather_grad_matches_onehot_matmul_formulation():
    V, D, B = sympy.symbols("V D B")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B)
    dims = {V: 8, D: 3, B: 6}
    tval, ival = rand_named(table, dims), rand_idx(idx, dims, 8)

    g = gather(table, idx, "V")
    grad = (F.sum(F.pow(g, 2)) / 2).grad(table).full_simplify()
    gout = compile_to_callable(grad)({table: tval, idx: ival}, dims)

    onehot = (ival.rename(None).long().unsqueeze(1) == torch.arange(8)).float()  # (B, V)
    y = onehot @ tval.rename(None)
    ref = onehot.T @ y
    torch.testing.assert_close(gout.align_to("V", "D").rename(None), ref, rtol=RTOL, atol=ATOL)


def test_gather_grad_wrt_idx_is_zero():
    V, D, B = sympy.symbols("V D B")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B)
    g = gather(table, idx, "V")
    grad = F.sum(F.pow(g, 2)).grad(idx).full_simplify()
    assert isinstance(grad, Zero), f"Expected Zero, got {grad}"


def test_gather_full_jacobian_dense_onehot_fallback():
    # The uncontracted jacobian d gather/d table keeps free idx wires on the
    # one-hot, so the scatter peephole must NOT fire; this exercises the
    # dense (deferred) one-hot materialization.
    V, D, B = sympy.symbols("V D B")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B)
    dims = {V: 5, D: 3, B: 4}
    jac = gather(table, idx, "V").grad(table).full_simplify()
    values = {table: rand_named(table, dims), idx: rand_idx(idx, dims, 5)}
    f = assert_compiles_like_evaluate(jac, values, dims)
    assert "arange" in f.codegen.specialize({s: dims[s] for s in dims})._source


def test_gather_second_grad_is_zero():
    V, D, B = sympy.symbols("V D B")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B)
    g = gather(table, idx, "V")
    hess = F.sum(g).grad(table).grad(table).full_simplify()
    assert isinstance(hess, Zero), f"Expected Zero, got {hess}"


# ---------------------------------------------------------------------------
# Composition: embedding -> MLP -> loss
# ---------------------------------------------------------------------------


def test_gather_in_mlp_loss_and_grads():
    V, D, C, B = sympy.symbols("V D C B")
    emb = Variable("emb", V=V, D=D)
    w = Variable("w", D=D, C=C)
    idx = Variable("idx", B=B)
    dims = {V: 7, D: 4, C: 3, B: 5}
    values = {emb: rand_named(emb, dims), w: rand_named(w, dims), idx: rand_idx(idx, dims, 7)}

    h = gather(emb, idx, "V")  # (B, D)
    y = F.relu(h @ w)  # (B, C)
    loss = F.sum(F.pow(y, 2)) / 2

    outputs = [loss.full_simplify(), loss.grad(emb).full_simplify(), loss.grad(w).full_simplify()]
    f = assert_compiles_like_evaluate(outputs, values, dims)

    # torch.autograd cross-check of the whole pipeline
    e2 = values[emb].rename(None).clone().requires_grad_(True)
    w2 = values[w].rename(None).clone().requires_grad_(True)
    yy = torch.relu(e2[values[idx].rename(None).long()] @ w2)
    (yy**2).sum().div(2).backward()
    lval, gemb, gw = f(dict(values), dict(dims))
    torch.testing.assert_close(gemb.align_to("V", "D").rename(None), e2.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(gw.align_to("D", "C").rename(None), w2.grad, rtol=RTOL, atol=ATOL)


def test_gather_cross_entropy_grads():
    # Embedding + linear + fused log_softmax cross-entropy, grads vs evaluate.
    V, D, C, B = sympy.symbols("V D C B")
    emb = Variable("emb", V=V, D=D)
    w = Variable("w", D=D, C=C)
    idx = Variable("idx", B=B)
    tgt = Variable("tgt", B=B, C=C)
    dims = {V: 6, D: 3, C: 4, B: 5}
    tgt_val = torch.softmax(torch.randn(dims[B], dims[C]), dim=1).rename("B", "C")
    values = {
        emb: rand_named(emb, dims),
        w: rand_named(w, dims),
        idx: rand_idx(idx, dims, 6),
        tgt: tgt_val,
    }
    logits = gather(emb, idx, "V") @ w  # (B, C)
    loss = F.sum(F.cross_entropy(logits, tgt, dim="C")) / dims[B]
    outputs = [loss.full_simplify(), loss.grad(emb).full_simplify(), loss.grad(w).full_simplify()]
    assert_compiles_like_evaluate(outputs, values, dims)


# ---------------------------------------------------------------------------
# Performance: must beat the one-hot matmul formulation
# ---------------------------------------------------------------------------


def _best_of(fn, n=5):
    fn()  # warmup
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_gather_microbench_beats_onehot():
    V, D, B = sympy.symbols("V D B")
    NV, ND, NB = 10000, 256, 2048
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B)
    dims = {V: NV, D: ND, B: NB}
    tval = torch.randn(NV, ND).rename("V", "D")
    ival = torch.randint(0, NV, (NB,)).float().rename("B")
    values = {table: tval, idx: ival}

    g = gather(table, idx, "V")
    f_fwd = compile_to_callable(g.full_simplify())
    f_bwd = compile_to_callable((F.sum(F.pow(g, 2)) / 2).grad(table).full_simplify())

    def onehot_fwd():
        oh = (ival.rename(None).long().unsqueeze(1) == torch.arange(NV)).float()
        return oh @ tval.rename(None)

    def onehot_bwd():
        oh = (ival.rename(None).long().unsqueeze(1) == torch.arange(NV)).float()
        return oh.T @ (oh @ tval.rename(None))

    # Correctness at benchmark scale
    out = f_fwd(dict(values), dict(dims))
    gout = f_bwd(dict(values), dict(dims))
    torch.testing.assert_close(out.align_to("B", "D").rename(None), onehot_fwd(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        gout.align_to("V", "D").rename(None), onehot_bwd(), rtol=1e-3, atol=1e-4
    )

    t_cf = _best_of(lambda: f_fwd(dict(values), dict(dims)))
    t_cb = _best_of(lambda: f_bwd(dict(values), dict(dims)))
    t_of = _best_of(onehot_fwd)
    t_ob = _best_of(onehot_bwd)
    print(
        f"\ngather fwd: {t_cf * 1e3:.3f}ms vs onehot {t_of * 1e3:.3f}ms ({t_of / t_cf:.1f}x); "
        f"bwd: {t_cb * 1e3:.3f}ms vs onehot {t_ob * 1e3:.3f}ms ({t_ob / t_cb:.1f}x)"
    )
    # Measured ~76x fwd / ~25x bwd; assert conservatively to avoid CI flakes.
    assert t_of / t_cf > 5, f"fwd speedup only {t_of / t_cf:.1f}x"
    assert t_ob / t_cb > 3, f"bwd speedup only {t_ob / t_cb:.1f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


def test_grad_scatter_never_materializes_delta_jacobian():
    """The table-gradient must compile to zeros().index_add_, NOT an
    index_select into the dense Delta(v,v')xDelta(d,d') jacobian (vocab^2*d^2
    elements — regression: lowering's index_select peephole once fused into
    the purely structural delta einsum before factoring could dissolve it)."""
    V, D, B = sympy.symbols("V D B")
    table = Variable("table", V=V, D=D)
    idx = Variable("idx", B=B)
    tgt = Variable("tgt", B=B, D=D)
    loss = F.sum(gather(table, idx, "V") * tgt)
    f = compile_to_callable(loss, loss.grad(table))
    NV, ND, NB = 64, 8, 4
    values = {
        table: torch.randn(NV, ND).rename("V", "D"),
        idx: rand_idx(idx, {B: NB}, NV),
        tgt: torch.randn(NB, ND).rename("B", "D"),
    }
    outs = f(values, {V: NV, D: ND, B: NB})
    src = next(iter(f._specializations.values()))._source
    assert "index_add_" in src, f"scatter peephole did not fire:\n{src}"
    assert f"({NV}, {NV}" not in src, f"dense vocab^2 intermediate:\n{src}"
    w = values[table].rename(None).clone().requires_grad_(True)
    ref = (w[values[idx].rename(None).long()] * values[tgt].rename(None)).sum()
    ref.backward()
    torch.testing.assert_close(outs[0].rename(None), ref.detach(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(outs[1].rename(None), w.grad, rtol=RTOL, atol=ATOL)
