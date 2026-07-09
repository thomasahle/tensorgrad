"""Kernel-level niceties for transformer workloads (task: SDPA peephole,
native tanh/erf/gelu, dtype threading, specialization LRU).

Each feature is compared against the evaluate() oracle and/or a manual torch
reference, and the generated source is inspected for the expected kernels.
"""

import pytest
import torch
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Delta, Product, Variable
import tensorgrad.compiler.codegen_torch as cg
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)

RTOL, ATOL = 1e-4, 1e-6


def _source(f, dims, dtype=None):
    key = tuple(sorted((s.name, v) for s, v in dims.items()))
    if dtype is not None and dtype != torch.get_default_dtype():
        key = key + (str(dtype),)
    return f._specializations[key]._source


# ===========================================================================
# 1. Native tanh / erf / gelu
# ===========================================================================

i_s = symbols("i")


def test_tanh_native_and_no_overflow():
    # The old exp-composition tanh = (e^x - e^-x)/(e^x + e^-x) NaNs for
    # |x| >~ 89 in fp32 (exp overflow). The native function must not.
    x = Variable("x", i_s)
    xt = torch.tensor([-120.0, -30.0, -0.5, 0.5, 30.0, 120.0]).rename("i")
    dims = {i_s: 6}
    expr = F.tanh(x).simplify()
    f = compile_to_callable(expr)
    out = f({x: xt}, dims)
    assert torch.isfinite(out.rename(None)).all()
    torch.testing.assert_close(out.rename(None), torch.tanh(xt.rename(None)), rtol=RTOL, atol=ATOL)
    assert "torch.tanh(" in _source(f, dims)
    # evaluate() oracle agrees
    ref = evaluate(expr, {x: xt}, dims)
    torch.testing.assert_close(out.rename(None), ref.rename(None), rtol=RTOL, atol=ATOL)


def test_tanh_grad_and_second_derivative():
    x = Variable("x", i_s)
    torch.manual_seed(0)
    xt = (torch.randn(8) * 30).rename("i")  # 30x scale: overflowed pre-fix
    dims = {i_s: 8}
    g = F.sum(F.tanh(x)).grad(x).full_simplify()
    gv = compile_to_callable(g)({x: xt}, dims)
    xr = xt.rename(None).clone().requires_grad_(True)
    torch.tanh(xr).sum().backward()
    torch.testing.assert_close(gv.rename(None), xr.grad, rtol=RTOL, atol=ATOL)
    # second derivative through the D_tanh -> D2_tanh chain
    h = F.sum(F.tanh(x)).grad(x).grad(x).full_simplify()
    hv = compile_to_callable(h)({x: xt}, dims)
    xr2 = xt.rename(None).clone().requires_grad_(True)
    hess = torch.autograd.functional.hessian(lambda t: torch.tanh(t).sum(), xr2.detach())
    torch.testing.assert_close(hv.align_to("i", "i_").rename(None), hess, rtol=RTOL, atol=1e-5)


def test_erf_value_and_grad():
    x = Variable("x", i_s)
    torch.manual_seed(1)
    xt = torch.randn(16).rename("i")
    dims = {i_s: 16}
    expr = F.erf(x).simplify()
    f = compile_to_callable(expr)
    out = f({x: xt}, dims)
    torch.testing.assert_close(out.rename(None), torch.erf(xt.rename(None)), rtol=RTOL, atol=ATOL)
    assert "torch.erf(" in _source(f, dims)
    g = F.sum(F.erf(x)).grad(x).full_simplify()
    gv = compile_to_callable(g)({x: xt}, dims)
    xr = xt.rename(None).clone().requires_grad_(True)
    torch.erf(xr).sum().backward()
    torch.testing.assert_close(gv.rename(None), xr.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("mode,tmode", [("exact", "none"), ("tanh", "tanh")])
def test_gelu(mode, tmode):
    x = Variable("x", i_s)
    torch.manual_seed(2)
    xt = torch.randn(16).rename("i")
    dims = {i_s: 16}
    out = compile_to_callable(F.gelu(x, approximate=mode).full_simplify())({x: xt}, dims)
    ref = torch.nn.functional.gelu(xt.rename(None), approximate=tmode)
    torch.testing.assert_close(out.rename(None), ref, rtol=RTOL, atol=1e-5)
    g = F.sum(F.gelu(x, approximate=mode)).grad(x).full_simplify()
    gv = compile_to_callable(g)({x: xt}, dims)
    xr = xt.rename(None).clone().requires_grad_(True)
    torch.nn.functional.gelu(xr, approximate=tmode).sum().backward()
    torch.testing.assert_close(gv.rename(None), xr.grad, rtol=RTOL, atol=1e-5)


# ===========================================================================
# 2. SDPA peephole
# ===========================================================================

b_s, h_s, s_s, s2_s, d_s, dv_s = symbols("b h s s2 d dv")
SDPA_DIMS = {b_s: 2, h_s: 3, s_s: 4, s2_s: 5, d_s: 6, dv_s: 7}


def _sdpa_vars():
    q = Variable("q", b_s, h_s, s_s, d_s)
    k = Variable("k", b_s, h_s, s2_s, d_s)
    v = Variable("v", b_s, h_s, s2_s, dv_s)
    m = Variable("m", b_s, h_s, s_s, s2_s)
    return q, k, v, m


def _sdpa_vals(q, k, v, m):
    torch.manual_seed(3)
    return {
        q: torch.randn(2, 3, 4, 6, names=("b", "h", "s", "d")),
        k: torch.randn(2, 3, 5, 6, names=("b", "h", "s2", "d")),
        v: torch.randn(2, 3, 5, 7, names=("b", "h", "s2", "dv")),
        m: torch.randn(2, 3, 4, 5, names=("b", "h", "s", "s2")),
    }


def _attention(q, k, v, m=None, scale=0.125):
    scores = F.graph("q -d- k", q=q, k=k) * scale
    if m is not None:
        scores = scores + m
    att = F.softmax(scores, dim="s2")
    return F.graph("att -s2- v", att=att, v=v).simplify({"expand_functions": False})


@pytest.mark.parametrize("with_mask", [False, True], ids=["nomask", "mask"])
def test_sdpa_fused_matches_unfused_and_torch(with_mask):
    q, k, v, m = _sdpa_vars()
    vals = _sdpa_vals(q, k, v, m)
    expr = _attention(q, k, v, m if with_mask else None)

    f_fused = compile_to_callable(expr)
    out = f_fused(dict(vals), dict(SDPA_DIMS))
    assert "scaled_dot_product_attention" in _source(f_fused, SDPA_DIMS)

    old = cg.SDPA_FUSION
    try:
        cg.SDPA_FUSION = False
        f_unfused = compile_to_callable(expr)
        ref = f_unfused(dict(vals), dict(SDPA_DIMS))
        assert "scaled_dot_product_attention" not in _source(f_unfused, SDPA_DIMS)
    finally:
        cg.SDPA_FUSION = old

    torch.testing.assert_close(
        out.rename(None), ref.align_to(*out.names).rename(None), rtol=RTOL, atol=1e-5
    )
    tref = torch.nn.functional.scaled_dot_product_attention(
        vals[q].rename(None),
        vals[k].rename(None),
        vals[v].rename(None),
        attn_mask=vals[m].rename(None) if with_mask else None,
        scale=0.125,
    )
    torch.testing.assert_close(
        out.align_to("b", "h", "s", "dv").rename(None), tref, rtol=RTOL, atol=1e-5
    )


def test_sdpa_permuted_layouts():
    # Variables declared in scrambled edge orders still fuse (via permutes).
    q = Variable("q", d_s, s_s, b_s, h_s)
    k = Variable("k", h_s, s2_s, d_s, b_s)
    v = Variable("v", dv_s, b_s, s2_s, h_s)
    torch.manual_seed(4)
    vals = {
        q: torch.randn(6, 4, 2, 3, names=("d", "s", "b", "h")),
        k: torch.randn(3, 5, 6, 2, names=("h", "s2", "d", "b")),
        v: torch.randn(7, 2, 5, 3, names=("dv", "b", "s2", "h")),
    }
    expr = _attention(q, k, v)
    f = compile_to_callable(expr)
    out = f(dict(vals), dict(SDPA_DIMS))
    assert "scaled_dot_product_attention" in _source(f, SDPA_DIMS)
    ref = evaluate(expr, dict(vals), dict(SDPA_DIMS))
    torch.testing.assert_close(
        out.rename(None), ref.align_to(*out.names).rename(None), rtol=RTOL, atol=1e-5
    )


def test_sdpa_no_batch_2d():
    q = Variable("q", s_s, d_s)
    k = Variable("k", s2_s, d_s)
    v = Variable("v", s2_s, dv_s)
    torch.manual_seed(5)
    vals = {
        q: torch.randn(4, 6, names=("s", "d")),
        k: torch.randn(5, 6, names=("s2", "d")),
        v: torch.randn(5, 7, names=("s2", "dv")),
    }
    expr = _attention(q, k, v)
    f = compile_to_callable(expr)
    out = f(dict(vals), dict(SDPA_DIMS))
    assert "scaled_dot_product_attention" in _source(f, SDPA_DIMS)
    ref = evaluate(expr, dict(vals), dict(SDPA_DIMS))
    torch.testing.assert_close(
        out.rename(None), ref.align_to(*out.names).rename(None), rtol=RTOL, atol=1e-5
    )


def test_sdpa_declines_when_softmax_shared():
    # The attention weights are also a program output: the softmax has a
    # second consumer, so the chain cannot be suppressed. Fusing anyway
    # would make the SDPA call RECOMPUTE the qk GEMM + softmax the program
    # still emits — measured as a pure loss where sharing is pervasive
    # (second-order programs: 11 shared-site fires cost +40% eager on the
    # 12-block softmax-stack HVP, task #63). The cost gate declines; both
    # outputs come off the plain einsum path, still correct.
    q, k, v, m = _sdpa_vars()
    vals = _sdpa_vals(q, k, v, m)
    att = F.softmax(F.graph("q -d- k", q=q, k=k) * 0.125, dim="s2")
    expr = F.graph("att -s2- v", att=att, v=v).simplify({"expand_functions": False})
    att_s = att.simplify({"expand_functions": False})
    f = compile_to_callable(expr, att_s)
    out, att_out = f(dict(vals), dict(SDPA_DIMS))
    src = _source(f, SDPA_DIMS)
    assert "scaled_dot_product_attention" not in src
    assert "softmax" in src  # the shared softmax is emitted once, reused
    for t, o in [(expr, out), (att_s, att_out)]:
        ref = evaluate(t, dict(vals), dict(SDPA_DIMS))
        torch.testing.assert_close(
            o.rename(None), ref.align_to(*o.names).rename(None), rtol=RTOL, atol=1e-5
        )


def test_sdpa_folded_output_projection():
    # The output projection contracted straight onto the AV product (one
    # 3-operand einsum, the shape every transformer forward pass lowers to):
    # SDPA fires, followed by the leftover contraction with W_O.
    q, k, v, m = _sdpa_vars()
    wo = Variable("wo", h_s, dv_s, d_s)
    vals = _sdpa_vals(q, k, v, m)
    torch.manual_seed(7)
    vals[wo] = torch.randn(3, 7, 6, names=("h", "dv", "d"))
    expr = (_attention(q, k, v, m) @ wo).simplify({"expand_functions": False})
    f = compile_to_callable(expr)
    out = f(dict(vals), dict(SDPA_DIMS))
    assert "scaled_dot_product_attention" in _source(f, SDPA_DIMS)
    ref = evaluate(expr, dict(vals), dict(SDPA_DIMS))
    torch.testing.assert_close(
        out.rename(None), ref.align_to(*out.names).rename(None), rtol=RTOL, atol=1e-5
    )


def test_sdpa_fallback_v_missing_batch():
    # V without the batch edges is not a clean (B, S, E) layout: fallback.
    q, k, _, _ = _sdpa_vars()
    v = Variable("v", s2_s, dv_s)
    torch.manual_seed(6)
    vals = {
        q: torch.randn(2, 3, 4, 6, names=("b", "h", "s", "d")),
        k: torch.randn(2, 3, 5, 6, names=("b", "h", "s2", "d")),
        v: torch.randn(5, 7, names=("s2", "dv")),
    }
    expr = _attention(q, k, v)
    f = compile_to_callable(expr)
    out = f(dict(vals), dict(SDPA_DIMS))
    assert "scaled_dot_product_attention" not in _source(f, SDPA_DIMS)
    ref = evaluate(expr, dict(vals), dict(SDPA_DIMS))
    torch.testing.assert_close(
        out.rename(None), ref.align_to(*out.names).rename(None), rtol=RTOL, atol=1e-5
    )


# ===========================================================================
# 3. dtype threading
# ===========================================================================


def test_dtype_threading_fp64_hoisted_delta():
    # diag-embedding output materializes a hoisted eye ('delta') constant;
    # with an fp64 input, the constant must be built in fp64 (mixed-dtype
    # einsum would otherwise throw or truncate).
    d = symbols("dd")
    v = Variable("v", d)
    diag = Product([v, Delta(d, "dd", "r", "c")]).simplify()
    vt = torch.randn(4, dtype=torch.float64, names=("dd",))
    f = compile_to_callable(diag)
    out = f({v: vt}, {d: 4})
    assert out.dtype == torch.float64
    torch.testing.assert_close(
        out.align_to("r", "c").rename(None), torch.diag(vt.rename(None)), rtol=1e-12, atol=1e-14
    )
    # same program still runs in fp32 (separate specialization per dtype)
    out32 = f({v: vt.to(torch.float32)}, {d: 4})
    assert out32.dtype == torch.float32


def test_dtype_threading_fp64_full_constant():
    # A pure-broadcast term emits torch.full(...): must carry the dtype too.
    n = symbols("n")
    x = Variable("x", n)
    expr = (x + 1).simplify()  # Ones-broadcast constant + input
    xt = torch.randn(5, dtype=torch.float64, names=("n",))
    out = compile_to_callable(expr)({x: xt}, {n: 5})
    assert out.dtype == torch.float64
    torch.testing.assert_close(out.rename(None), xt.rename(None) + 1, rtol=1e-12, atol=1e-14)


def test_dtype_threading_bf16():
    n, dd = symbols("n dd")
    X = Variable("X", n, dd)
    w = Variable("w", dd)
    m = F.mean(F.relu(X @ w)).full_simplify()
    torch.manual_seed(7)
    Xt = torch.randn(6, 4, names=("n", "dd"), dtype=torch.bfloat16)
    wt = torch.randn(4, names=("dd",), dtype=torch.bfloat16)
    out = compile_to_callable(m)({X: Xt, w: wt}, {n: 6, dd: 4})
    assert out.dtype == torch.bfloat16
    manual = torch.relu(Xt.rename(None).float() @ wt.rename(None).float()).mean()
    torch.testing.assert_close(out.rename(None).float(), manual, rtol=0.05, atol=0.05)


# ===========================================================================
# 4. Specialization LRU cap
# ===========================================================================


def test_specialization_lru_cap():
    from tensorgrad.compiler.runtime import SPECIALIZATION_CACHE_SIZE

    n = symbols("n")
    x = Variable("x", n)
    f = compile_to_callable(F.sum(x).simplify())
    for size in range(1, SPECIALIZATION_CACHE_SIZE + 6):
        xt = torch.randn(size, names=("n",))
        out = f({x: xt}, {n: size})
        torch.testing.assert_close(out.rename(None), xt.rename(None).sum())
    assert len(f._specializations) == SPECIALIZATION_CACHE_SIZE
    # Oldest entries were evicted, newest kept; evicted sizes recompile fine.
    assert (("n", 1),) not in f._specializations
    assert (("n", SPECIALIZATION_CACHE_SIZE + 5),) in f._specializations
    out = f({x: torch.randn(1, names=("n",))}, {n: 1})
    assert out.rename(None).shape == ()
