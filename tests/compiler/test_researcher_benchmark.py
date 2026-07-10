"""THE NORTH-STAR RESEARCHER BENCHMARK.

Tensorgrad's founding promise: researchers define functions from PRIMITIVES and
get optimized, stable derivatives automatically — no hand-written Jacobians.
Origin: a researcher telling the author how hard the (cross-entropy ∘) softmax
Hessian was to derive by hand.

Every expression here uses ONLY leaf primitives (exp, log, pow, sum) — no
F.softmax, no fused signatures. Hand-fused Jacobian signatures no longer exist:
F.softmax's own derivative is derived from this same primitive definition.

Tier 1 — gradients of primitives-defined softmax / gelu / layernorm:
    loss = sum(f(x) * w);  g = loss.grad(x).simplify()   [zero hand rules]
    compile(loss, g) graded on:
      DERIVATIVE-FREE  a light .simplify() removes all Derivative nodes  [passes]
      CORRECT          matches torch autograd at moderate |x| (fp64)     [passes]
      STABLE           fp32, |x| <= 200: finite and matches autograd    [layernorm
                       passes; softmax/gelu need IR stability re-fusion #16]
      COMPACT          no intermediate larger than max(input, output)   [needs IR
                       factoring pass #17]
      FAST             per-call <= 1.5x a hand-fused torch equivalent   [needs #17]

Tier 2 — the founding problem: the CE∘softmax Hessian wrt logits, batched.
    L = -sum(y * log(primitive_softmax(x, 'v'))) / B;  H = L.grad(x).grad(x)
      TERM COUNT   simplified H is a 2-term Sum: diag(s) - s s^T shape  [passes]
      CORRECT      matches torch.autograd.functional.hessian, tiny dims [passes]
      Y-FREE       with y declared simplex (sum_v y[b,v] = 1, true for one-hot
                   targets, via Variable.with_eq_constraint), the compiled H has no
                   InputNode for y — with a *general* y the Hessian provably
                   retains the factor sum_v y[b,v]                      [passes]
      HVP FAST     compiled HVP beats torch double-backward, B=64 V=256
                   [passes via nested grads + reverse-over-reverse, #49]
      BLOCK-DIAG   HVP program has no batch x batch intermediate        [passes]
    Secondary: bare softmax 3-index Hessian H_ijk correctness           [passes]

Every stage runs in a spawned subprocess with a HARD 120s timeout: a hang or an
OOM-kill is a FAIL, never a hang. Stage results are cached per process, so the
per-function stages run once. Quick run (skips the timing stages):
    pytest tests/compiler/test_researcher_benchmark.py -k "not perf"
"""

import multiprocessing as mp
import queue as queue_mod
import sys
import time
import traceback
from fractions import Fraction

import pytest
import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.tensor import Delta, Derivative, Sum, Tensor

HARD_TIMEOUT_S = 120  # any stage exceeding this = fail, not hang
RTOL = 1e-4

XFAIL_STABILITY = pytest.mark.xfail(
    strict=False,
    reason="needs IR-level stability re-fusion (#16: recognize exp/Σexp and logΣexp): "
    "the expanded exp overflows fp32 at |x|=200",
)
XFAIL_FACTORING = pytest.mark.xfail(
    strict=False,
    reason="needs the IR factoring pass (#17: automatic optimized backwards): the "
    "expanded Sum-of-Products gradient materializes (input x input)-sized intermediates",
)


# ---------------------------------------------------------------------------
# Primitives-only definitions — as a researcher would write them
# ---------------------------------------------------------------------------


def primitive_softmax(x, dim):
    """softmax from primitives only."""
    e = F.exp(x)
    return e * F.pow(F.sum(e, dim=dim, keepdims=True), -1)


def primitive_gelu(x):
    """tanh-approximation gelu from primitives only (tanh spelled with exp)."""
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * F.pow(x, 3))
    e, em = F.exp(inner), F.exp(-inner)
    return 0.5 * x * (1 + (e - em) * F.pow(e + em, -1))


def primitive_layernorm(x, dim, eps=1e-5):
    mu = F.mean(x, dim=dim, keepdims=True)
    d = x - mu
    var = F.mean(d * d, dim=dim, keepdims=True)
    return d * F.pow(var + eps, Fraction(-1, 2))


TIER1 = {"softmax": None, "gelu": None, "layernorm": None}  # names only; built in stages


def _build_tier1(fname, x):
    if fname == "softmax":
        return primitive_softmax(x, "v")
    if fname == "gelu":
        return primitive_gelu(x)
    if fname == "layernorm":
        return primitive_layernorm(x, "v")
    raise KeyError(fname)


def _fused_torch(fname, xr, wr):
    """Hand-fused torch loss+grad — the bar tensorgrad must be within 1.5x of."""
    c, a = 0.7978845608028654, 0.044715
    if fname == "softmax":
        s = torch.softmax(xr, dim=1)
        lo = (s * wr).sum()
        gr = s * (wr - (s * wr).sum(dim=1, keepdim=True))
    elif fname == "gelu":
        t = torch.tanh(c * (xr + a * xr**3))
        lo = (0.5 * xr * (1 + t) * wr).sum()
        gr = wr * (0.5 * (1 + t) + 0.5 * xr * (1 - t * t) * c * (1 + 3 * a * xr * xr))
    else:  # layernorm
        mu = xr.mean(dim=1, keepdim=True)
        d = xr - mu
        rstd = ((d * d).mean(dim=1, keepdim=True) + 1e-5).rsqrt()
        xh = d * rstd
        lo = (xh * wr).sum()
        gr = rstd * (wr - wr.mean(dim=1, keepdim=True) - xh * (wr * xh).mean(dim=1, keepdim=True))
    return lo, gr


# ---------------------------------------------------------------------------
# Harness: every stage in a spawned subprocess with a hard timeout
# ---------------------------------------------------------------------------


def _child(q, fn_name, args):
    try:
        q.put(("ok", globals()[fn_name](*args)))
    except Exception:
        q.put(("err", traceback.format_exc()))


def _run_staged(fn_name, *args):
    """Run a module-level stage function in a subprocess with a hard timeout.

    A hang, crash, or OOM-kill is a FAIL, never a hang of the suite.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, fn_name, args), daemon=True)
    p.start()
    deadline = time.monotonic() + HARD_TIMEOUT_S
    status = payload = None
    try:
        while True:
            try:
                status, payload = q.get(timeout=1.0)
                break
            except queue_mod.Empty:
                if not p.is_alive():
                    pytest.fail(f"stage {fn_name}{args} died without a result (OOM-kill?)")
                if time.monotonic() > deadline:
                    pytest.fail(
                        f"stage {fn_name}{args} exceeded {HARD_TIMEOUT_S}s (a hang is a FAIL)"
                    )
    finally:
        if p.is_alive():
            p.kill()
        p.join(10)
    if status == "err":
        pytest.fail(f"stage {fn_name}{args} crashed:\n{payload}")
    return payload


_CACHE: dict = {}


def _staged_cached(fn_name, *args):
    key = (fn_name, args)
    if key not in _CACHE:
        _CACHE[key] = _run_staged(fn_name, *args)
    return _CACHE[key]


def _contains_derivative(t: Tensor) -> bool:
    seen, stack = set(), [t]
    while stack:
        u = stack.pop()
        if id(u) in seen:
            continue
        seen.add(id(u))
        if isinstance(u, Derivative):
            return True
        for val in vars(u).values():
            if isinstance(val, Tensor):
                stack.append(val)
            elif isinstance(val, (list, tuple)):
                stack.extend(z for z in val if isinstance(z, Tensor))
    return False


def _trace_intermediates(fn, values, dims):
    """Max element count (and shape set) over every tensor materialized by the
    compiled specialization — inputs, hoisted constants, einsum partials, all
    named intermediates. sys.settrace on the exec'd `_compiled` frame."""
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


def _median_ms(f, n=30):
    for _ in range(3):
        f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[n // 2]


# ---------------------------------------------------------------------------
# Tier 1 stages (run in subprocesses)
# ---------------------------------------------------------------------------


def _stage_tier1(fname):
    b, v = symbols("b v")
    x, w = Variable("x", b, v), Variable("w", b, v)
    loss = F.sum(_build_tier1(fname, x) * w)

    t0 = time.monotonic()
    g = loss.grad(x)
    gs = g.simplify()  # the LIGHTEST simplify — no full_simplify, no hand rules
    ls = loss.simplify()
    simplify_s = time.monotonic() - t0

    res = {"simplify_s": simplify_s, "derivative_free": not _contains_derivative(gs)}
    fn = compile_to_callable(ls, gs)

    # CORRECT: moderate inputs, fp64, vs torch autograd
    B, V = 4, 8
    torch.manual_seed(0)
    xt = torch.randn(B, V, dtype=torch.float64).rename("b", "v")
    wt = torch.randn(B, V, dtype=torch.float64).rename("b", "v")
    lv, gv = fn({x: xt, w: wt}, {b: B, v: V})
    xr = xt.rename(None).clone().requires_grad_(True)
    lo, gr = _fused_torch(fname, xr, wt.rename(None))
    res["correct"] = bool(
        torch.allclose(lv.rename(None), lo.detach(), rtol=RTOL, atol=1e-8)
        and torch.allclose(gv.rename(None), gr, rtol=RTOL, atol=1e-8)
    )

    # STABLE: fp32, |x| <= 200 (where the expanded exp form overflows but the
    # fused torch kernels are exact)
    torch.manual_seed(1)
    xt2 = (torch.rand(B, V) * 400 - 200).rename("b", "v")
    wt2 = torch.randn(B, V).rename("b", "v")
    lv2, gv2 = fn({x: xt2, w: wt2}, {b: B, v: V})
    _, gr2 = _fused_torch(fname, xt2.rename(None), wt2.rename(None))
    res["stable_finite"] = bool(
        torch.isfinite(lv2.rename(None)).all() and torch.isfinite(gv2.rename(None)).all()
    )
    res["stable_correct"] = res["stable_finite"] and bool(
        torch.allclose(gv2.rename(None), gr2, rtol=RTOL, atol=1e-5)
    )

    # COMPACT: no intermediate larger than max(input, output) element count.
    # B >= V so that even a legitimate eye(V) stays within the bound.
    B, V = 16, 8
    xt3 = torch.randn(B, V).rename("b", "v")
    wt3 = torch.randn(B, V).rename("b", "v")
    bound = max(B * V, 1)  # inputs are (B,V); outputs are scalar + (B,V)
    max_numel, shapes = _trace_intermediates(fn, {x: xt3, w: wt3}, {b: B, v: V})
    res["compact_bound"] = bound
    res["compact_max_numel"] = max_numel
    res["compact"] = max_numel <= bound
    res["shapes"] = sorted(shapes)
    return res


def _stage_tier1_perf(fname):
    """Per-call time vs the hand-fused torch equivalent at (256, 1024).

    Guard: refuse to run a non-compact program at scale (a (B·V)^2 intermediate
    at these dims is ~275 GB) — non-compact is reported as ratio=None.
    """
    b, v = symbols("b v")
    x, w = Variable("x", b, v), Variable("w", b, v)
    loss = F.sum(_build_tier1(fname, x) * w)
    gs = loss.grad(x).simplify()
    fn = compile_to_callable(loss.simplify(), gs)

    Bs, Vs = 16, 8  # small probe dims for the compactness guard
    probe = {
        x: torch.randn(Bs, Vs).rename("b", "v"),
        w: torch.randn(Bs, Vs).rename("b", "v"),
    }
    max_numel, _ = _trace_intermediates(fn, probe, {b: Bs, v: Vs})
    if max_numel > Bs * Vs:
        return {"compact": False, "probe_max_numel": max_numel, "ratio": None}

    B, V = 256, 1024
    torch.manual_seed(0)
    xt = torch.randn(B, V).rename("b", "v")
    wt = torch.randn(B, V).rename("b", "v")
    xr, wr = xt.rename(None), wt.rename(None)
    t_ours = _median_ms(lambda: fn({x: xt, w: wt}, {b: B, v: V}))
    t_fused = _median_ms(lambda: _fused_torch(fname, xr, wr))
    return {"compact": True, "ratio": t_ours / t_fused, "ours_ms": t_ours, "fused_ms": t_fused}


# ---------------------------------------------------------------------------
# Tier 2 stages: the CE∘softmax Hessian (the founding problem)
# ---------------------------------------------------------------------------


def _ce_hessian():
    b, v = symbols("b v")
    x, y = Variable("x", b, v), Variable("y", b, v)
    s = primitive_softmax(x, "v")
    L = -F.sum(y * F.log(s)) / Delta(b)  # mean over the batch
    H = L.grad(x).grad(x, new_names={"b": "b2", "v": "v2"})
    return b, v, x, y, H.simplify()


def _one_hot_targets(B, V, dtype=torch.float32):
    torch.manual_seed(2)
    yt = torch.zeros(B, V, dtype=dtype)
    yt[torch.arange(B), torch.randint(V, (B,))] = 1.0
    return yt


def _stage_tier2_hessian():
    b, v, x, y, Hs = _ce_hessian()
    res = {"n_terms": len(Hs.terms) if isinstance(Hs, Sum) else 1}

    fn = compile_to_callable(Hs)
    res["input_names"] = list(fn.input_names)  # y-freeness = 'y' not needed

    # CORRECT vs torch.autograd.functional.hessian at tiny dims (fp64)
    B, V = 3, 5
    torch.manual_seed(0)
    xt = torch.randn(B, V, dtype=torch.float64).rename("b", "v")
    yt = _one_hot_targets(B, V, torch.float64)
    Hv = fn({x: xt, y: yt.rename("b", "v")}, {b: B, v: V})
    Hv = Hv.align_to("b", "v", "b2", "v2").rename(None)

    def torch_L(xf):
        return -(yt * torch.log_softmax(xf, dim=1)).sum() / B

    Href = torch.autograd.functional.hessian(torch_L, xt.rename(None))
    res["correct"] = bool(torch.allclose(Hv, Href, rtol=RTOL, atol=1e-10))
    res["max_err"] = float((Hv - Href).abs().max())
    return res


def _stage_tier2_hessian_simplex():
    """The founding cancellation. Targets y are one-hot (rows on the simplex), so the
    researcher declares sum_v y[b,v] = 1 via Variable.with_eq_constraint. The Hessian
    H = (sum_v y[b,v]) * delta_{b,b2} (diag(s) - s s^T) / B must then simplify to a
    y-FREE 2-term form, and the compiled program must not consume y at all."""
    b, v = symbols("b v")
    x = Variable("x", b, v)
    y0 = Variable("y", b, v)
    y = y0.with_eq_constraint(F.sum(y0, dim=("v",)), 1)
    s = primitive_softmax(x, "v")
    L = -F.sum(y * F.log(s)) / Delta(b)  # mean over the batch
    H = L.grad(x).grad(x, new_names={"b": "b2", "v": "v2"})
    # expand=True: the sum_v y factor only becomes a flat ones-contraction (which the
    # simplex declaration cancels) after the rational layer is distributed.
    Hs = H.simplify({"expand": True})

    res = {
        "n_terms": len(Hs.terms) if isinstance(Hs, Sum) else 1,
        "depends_on_y": Hs.depends_on(y),
    }
    fn = compile_to_callable(Hs)
    res["input_names"] = list(fn.input_names)  # y-freeness = no y InputNode

    # CORRECT vs torch.autograd.functional.hessian with an ACTUAL one-hot y (fp64)
    B, V = 3, 5
    torch.manual_seed(0)
    xt = torch.randn(B, V, dtype=torch.float64).rename("b", "v")
    yt = _one_hot_targets(B, V, torch.float64)
    values = {x: xt}
    if "y" in res["input_names"]:  # pragma: no cover - only on regression
        values[y] = yt.rename("b", "v")
    Hv = fn(values, {b: B, v: V}).align_to("b", "v", "b2", "v2").rename(None)

    def torch_L(xf):
        return -(yt * torch.log_softmax(xf, dim=1)).sum() / B

    Href = torch.autograd.functional.hessian(torch_L, xt.rename(None))
    res["correct"] = bool(torch.allclose(Hv, Href, rtol=RTOL, atol=1e-10))
    res["max_err"] = float((Hv - Href).abs().max())
    return res


def _stage_tier2_hvp():
    """Hessian-vector product: correctness, block-diagonal structure, speed.

    Built the way the torch reference computes it -- nested gradients,
    grad of <grad(L), u> (reverse-over-reverse, task #49) -- NOT by
    contracting the explicit Hessian, which materialized a 6-index
    (B,B,B,V,V,V) intermediate and measured 50x slower even at tiny dims.
    Structure is probed on the eager build (intermediates visible; the
    Inductor build fuses everything into one opaque region); the perf gate
    times the Inductor tier -- the shipped fast configuration -- against
    eager double-backward, which is torch's best (torch.compile cannot
    trace create_graph autograd)."""
    b, v = symbols("b v")
    x, y = Variable("x", b, v), Variable("y", b, v)
    s_ = primitive_softmax(x, "v")
    L = -F.sum(y * F.log(s_)) / Delta(b)  # mean over the batch
    u = Variable("u", b, v)
    hvp = F.sum(L.grad(x) * u).grad(x)
    fn = compile_to_callable(hvp)

    def vals(B, V, dtype=torch.float32):
        torch.manual_seed(0)
        return {
            x: torch.randn(B, V, dtype=dtype).rename("b", "v"),
            y: _one_hot_targets(B, V, dtype).rename("b", "v"),
            u: torch.randn(B, V, dtype=dtype).rename("b", "v"),
        }

    def torch_hvp(vv, B):
        xr = vv[x].rename(None).clone().requires_grad_(True)
        yr, ur = vv[y].rename(None), vv[u].rename(None)
        loss = -(yr * torch.log_softmax(xr, dim=1)).sum() / B
        (gr,) = torch.autograd.grad(loss, xr, create_graph=True)
        (hv,) = torch.autograd.grad((gr * ur).sum(), xr)
        return hv

    # Correctness + structure probe at tiny dims (B != V so batch dims are
    # unambiguous in shapes).
    B, V = 3, 5
    vv = vals(B, V, torch.float64)
    out = fn(vv, {b: B, v: V}).align_to("b", "v").rename(None)
    res = {"correct": bool(torch.allclose(out, torch_hvp(vv, B), rtol=RTOL, atol=1e-10))}
    max_numel, shapes = _trace_intermediates(fn, vv, {b: B, v: V})
    res["probe_max_numel"] = max_numel
    res["shapes"] = sorted(shapes)
    # BLOCK-DIAG: no batch x batch tensor anywhere (not even an eye(B) helper)
    res["batch_sq_shapes"] = sorted(
        shp for shp in shapes if sum(1 for d in shp if d == B) >= 2
    )
    # An efficient HVP materializes only O(B*V) tensors. Refuse to time a
    # program that scales worse (the old Hessian-contraction form's 6-index
    # intermediate would be (B*V)^3 ~ 4e12 elements at B=64,V=256).
    if max_numel > 8 * B * V:
        res["ratio"] = None
        return res

    B, V = 64, 256
    vv = vals(B, V)
    fn_fast = compile_to_callable(hvp, torch_compile=True)
    fn_fast(vv, {b: B, v: V})  # Inductor warmup outside the timed region
    t_ours = _median_ms(lambda: fn_fast(vv, {b: B, v: V}))
    t_torch = _median_ms(lambda: torch_hvp(vv, B))
    res.update(ratio=t_ours / t_torch, ours_ms=t_ours, torch_ms=t_torch)
    return res


def _stage_tier2_bare_softmax():
    """Secondary: 3-index Hessian H_ijk of bare primitive softmax (order-1)."""
    n = symbols("n")
    x = Variable("x", i=n)
    s = primitive_softmax(x, "i")
    Hs = s.grad(x, new_names={"i": "i2"}).grad(x, new_names={"i": "i3"}).simplify()
    fn = compile_to_callable(Hs)
    N = 5
    torch.manual_seed(0)
    xt = torch.randn(N, dtype=torch.float64).rename("i")
    out = fn({x: xt}, {n: N}).align_to("i", "i2", "i3").rename(None)
    from torch.func import jacrev

    ref = jacrev(jacrev(lambda z: torch.softmax(z, dim=0)))(xt.rename(None))
    return {
        "correct": bool(torch.allclose(out, ref, rtol=RTOL, atol=1e-10)),
        "max_err": float((out - ref).abs().max()),
    }


# ---------------------------------------------------------------------------
# Tier 1 tests
# ---------------------------------------------------------------------------

TIER1_NAMES = ["softmax", "gelu", "layernorm"]


@pytest.mark.parametrize("fname", TIER1_NAMES)
def test_tier1_light_simplify_eliminates_derivatives(fname):
    """loss.grad(x).simplify() must be Derivative-free (and compile) — no
    full_simplify, no hand-written Jacobians."""
    res = _staged_cached("_stage_tier1", fname)
    assert res["derivative_free"], f"Derivative nodes survive .simplify() for {fname}"
    assert res["simplify_s"] < 30, f"simplify took {res['simplify_s']:.1f}s (time cap)"


@pytest.mark.parametrize("fname", TIER1_NAMES)
def test_tier1_grad_matches_autograd(fname):
    """Compiled loss+grad match the fused torch reference at moderate |x|."""
    res = _staged_cached("_stage_tier1", fname)
    assert res["correct"], f"compiled loss/grad mismatch vs torch for {fname}"


@pytest.mark.parametrize(
    "fname",
    [
        pytest.param("softmax", marks=XFAIL_STABILITY),
        pytest.param("gelu", marks=XFAIL_STABILITY),
        "layernorm",
    ],
)
def test_tier1_grad_stable_at_extreme_inputs(fname):
    """fp32, |x| <= 200: finite and matches torch's fused kernels (which are)."""
    res = _staged_cached("_stage_tier1", fname)
    assert res["stable_finite"], f"{fname}: nan/inf at |x|<=200 in fp32"
    assert res["stable_correct"], f"{fname}: wrong gradient at |x|<=200 in fp32"


@pytest.mark.parametrize(
    "fname", [pytest.param(f, marks=XFAIL_FACTORING) for f in TIER1_NAMES]
)
def test_tier1_grad_compact(fname):
    """No materialized intermediate larger than max(input, output) elements."""
    res = _staged_cached("_stage_tier1", fname)
    assert res["compact"], (
        f"{fname}: largest intermediate has {res['compact_max_numel']} elements "
        f"(bound {res['compact_bound']}); shapes seen: {res['shapes']}"
    )


@pytest.mark.parametrize(
    "fname", [pytest.param(f, marks=XFAIL_FACTORING) for f in TIER1_NAMES]
)
def test_tier1_grad_perf_vs_hand_fused(fname):
    """Per-call within 1.5x of a hand-fused torch loss+grad at (256, 1024)."""
    res = _staged_cached("_stage_tier1_perf", fname)
    assert res["compact"], (
        f"{fname}: non-compact program (probe intermediate of "
        f"{res['probe_max_numel']} elements) — refusing to time it at (256, 1024)"
    )
    assert res["ratio"] <= 1.5, (
        f"{fname}: {res['ours_ms']:.3f}ms vs hand-fused {res['fused_ms']:.3f}ms "
        f"(ratio {res['ratio']:.1f}x > 1.5x)"
    )


# ---------------------------------------------------------------------------
# Tier 2 tests
# ---------------------------------------------------------------------------


def test_tier2_ce_hessian_two_terms():
    """The simplified CE∘softmax Hessian collapses to the classical 2-term
    diag(s) - s s^T structure (per batch block)."""
    res = _staged_cached("_stage_tier2_hessian")
    assert res["n_terms"] == 2, f"expected 2 terms, got {res['n_terms']}"


def test_tier2_ce_hessian_correct():
    """Matches torch.autograd.functional.hessian at tiny dims (fp64)."""
    res = _staged_cached("_stage_tier2_hessian")
    assert res["correct"], f"Hessian mismatch, max err {res['max_err']:.2e}"


def test_tier2_ce_hessian_y_free():
    """The founding cancellation: with y declared on the simplex (sum_v y[b,v] = 1,
    which one-hot targets satisfy), the sum_v y factor cancels, the Hessian keeps the
    classical 2-term diag(s) - s s^T form, and the compiled program must not need y."""
    res = _staged_cached("_stage_tier2_hessian_simplex")
    assert not res["depends_on_y"], "simplified Hessian still depends on simplex-declared y"
    assert "y" not in res["input_names"], (
        f"compiled Hessian still consumes y: inputs={res['input_names']}"
    )
    assert res["n_terms"] == 2, f"expected 2 terms, got {res['n_terms']}"
    assert res["correct"], (
        f"y-free Hessian mismatch vs autograd with one-hot y, max err {res['max_err']:.2e}"
    )


def test_tier2_ce_hvp_correct():
    """Compiled Hessian-vector product matches torch double-backward (tiny dims)."""
    res = _staged_cached("_stage_tier2_hvp")
    assert res["correct"]


def test_tier2_ce_hvp_block_diagonal():
    """H is block-diagonal over the batch: the compiled HVP program must not
    materialize any batch x batch tensor (not even an eye(B) helper)."""
    res = _staged_cached("_stage_tier2_hvp")
    assert not res["batch_sq_shapes"], (
        f"batch x batch intermediates: {res['batch_sq_shapes']} "
        f"(all shapes: {res['shapes']})"
    )


def test_tier2_ce_hvp_perf_vs_double_backward():
    """Compiled HVP must beat torch double-backward at (B=64, V=256) on CPU."""
    res = _staged_cached("_stage_tier2_hvp")
    assert res["ratio"] is not None, (
        f"HVP program scales worse than O(B*V) (probe intermediate of "
        f"{res['probe_max_numel']} elements) — refusing to run it at (64, 256)"
    )
    assert res["ratio"] < 1.0, (
        f"compiled HVP {res['ours_ms']:.3f}ms vs double-backward "
        f"{res['torch_ms']:.3f}ms (ratio {res['ratio']:.1f}x)"
    )


def test_tier2_bare_softmax_hessian_3index():
    """Secondary: d2 softmax_i / dx_j dx_k — 3-index bookkeeping is exact."""
    res = _staged_cached("_stage_tier2_bare_softmax")
    assert res["correct"], f"3-index Hessian mismatch, max err {res['max_err']:.2e}"
