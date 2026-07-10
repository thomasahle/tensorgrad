"""The equality-saturation stage (compiler/egraph.py, task #66): IR bridge,
saturate-and-extract, szfp self-gate. EGRAPH is default-off; these tests
force it on and pin (a) the stage rewrites what it recognizes (the adjoint
valley on real IR), (b) factoring-directed collapse fires, (c) unrecognized
programs pass through untouched, (d) every rewrite is exactly value-
preserving on randomized orientation fuzz (the decode axis bookkeeping)."""

from __future__ import annotations

import random

import pytest
import torch
from sympy import symbols

torch.set_num_threads(2)

egglog = pytest.importorskip("egglog")

import tensorgrad.compiler.egraph as eg  # noqa: E402
import tensorgrad.functions as F  # noqa: E402
from tensorgrad import Variable  # noqa: E402
from tensorgrad.compiler import szfp  # noqa: E402
from tensorgrad.compiler.ir import EinsumNode  # noqa: E402
from tensorgrad.compiler.lower import lower_program  # noqa: E402


@pytest.fixture(autouse=True)
def _egraph_on(monkeypatch):
    monkeypatch.setattr(eg, "EGRAPH", True)


def _chain_expr(depth, n):
    """Left-associated matrix chain ending in a vector: the forward-mode
    shape whose reverse-mode reassociation greedy provably misses."""
    ms = [Variable(f"m{i}", **{f"a{i}": n, f"a{i+1}": n}) for i in range(depth)]
    g = Variable("g", **{f"a{depth}": n})
    expr = ms[0]
    for i in range(1, depth):
        expr = (expr @ ms[i])
    return (expr @ g), ms, g


def _flops(outputs, dims):
    """Crude flop count over einsum nodes (product of wire dims per node)."""
    from tensorgrad.compiler.ir import toposort
    import sympy as sp

    total = 0
    for nd in toposort([n for n, _ in outputs]):
        if isinstance(nd, EinsumNode) and nd.ops:
            p = 1
            for d in nd.wire_dims:
                p *= int(sp.sympify(d).subs(dims))
            total += p
    return total


def test_reassociates_the_chain_on_real_ir():
    n = symbols("n")
    expr, ms, g = _chain_expr(6, n)
    b, outs = lower_program([expr.full_simplify()])
    dims = {n: 32}
    before = _flops(outs, dims)
    rewritten = eg.egraph_outputs(b, outs, dims)
    assert rewritten is not outs, "stage did not fire on a pure matrix chain"
    assert szfp.outputs_equal(outs, rewritten)
    after = _flops(rewritten, dims)
    assert after < before / 5, f"no reverse-mode collapse: {before} -> {after}"


def test_factoring_collapse_fires():
    """x@w1 + x@w2 -> x@(w1+w2): one GEMM instead of two."""
    n, m = symbols("n m")
    x = Variable("x", b=n, d=m)
    w1 = Variable("w1", d=m, o=m)
    w2 = Variable("w2", d=m, o=m)
    expr = (x @ w1) + (x @ w2)
    b, outs = lower_program([expr.full_simplify()])
    dims = {n: 8, m: 16}
    rewritten = eg.egraph_outputs(b, outs, dims)
    assert rewritten is not outs
    assert szfp.outputs_equal(outs, rewritten)
    assert _flops(rewritten, dims) < _flops(outs, dims)


def test_unrecognized_program_passes_through():
    n = symbols("n")
    x = Variable("x", i=n)
    expr = F.sum(F.exp(x) * F.exp(x))  # maps + hadamard: no matmul fragment
    b, outs = lower_program([expr.full_simplify()])
    assert eg.egraph_outputs(b, outs, {n: 8}) is outs


def test_orientation_fuzz_exact():
    """Random chains with random axis-declaration orders (exercising the Tr
    bookkeeping) and random integer weights: whenever the stage rewrites,
    values must be EXACTLY preserved (szfp, mod-P) and dims reproduced; and
    it must fire on a healthy fraction (not vacuously sound)."""
    rng = random.Random(0)
    n, m, k = symbols("n m k")
    sizes = {n: 5, m: 7, k: 3}
    syms = [n, m, k]
    names = iter(f"v{i}" for i in range(10_000))
    fired = 0
    for trial in range(25):
        hops = rng.randrange(2, 6)
        seq = [rng.choice(syms) for _ in range(hops + 1)]
        expr = None
        for i in range(hops):
            din, dout = seq[i], seq[i + 1]
            ein, eout = f"h{i}", f"h{i+1}"
            if rng.random() < 0.5:
                w = Variable(next(names), **{ein: din, eout: dout})
            else:  # swapped declaration order: same edges, transposed dims
                w = Variable(next(names), **{eout: dout, ein: din})
            if rng.random() < 0.3:
                w = w * rng.choice([2, -1, 3])
            expr = w if expr is None else F.dot(expr, w, dim=ein)
        if rng.random() < 0.3:  # sometimes contract the tail to a vector
            v = Variable(next(names), **{f"h{hops}": seq[-1]})
            expr = F.dot(expr, v, dim=f"h{hops}")
        b, outs = lower_program([expr.full_simplify()])
        rewritten = eg.egraph_outputs(b, outs, dict(sizes))
        if rewritten is not outs:
            fired += 1
            assert szfp.outputs_equal(outs, rewritten), f"trial {trial} value break"
            assert tuple(rewritten[0][0].dims) == tuple(outs[0][0].dims)
    assert fired >= 5, f"stage fired on only {fired}/25 fuzz programs"


def test_compiled_end_to_end_matches():
    """Full compile path with the stage on: values equal to the stage off."""
    from tensorgrad.compiler import compile_to_callable

    n = symbols("n")
    expr, ms, g = _chain_expr(4, n)
    N = 16
    torch.manual_seed(0)
    vals = {v: torch.randn(N, N, dtype=torch.float64).rename(*v.edges) for v in ms}
    vals[g] = torch.randn(N, dtype=torch.float64).rename(*g.edges)
    prog_on = compile_to_callable(expr)
    out_on = prog_on(dict(vals), {n: N})
    eg_flag = eg.EGRAPH
    try:
        eg.EGRAPH = False
        prog_off = compile_to_callable(expr * 1)  # distinct tree, same math
        out_off = prog_off(dict(vals), {n: N})
    finally:
        eg.EGRAPH = eg_flag
    torch.testing.assert_close(
        out_on.rename(None), out_off.align_to(*out_on.names).rename(None),
        rtol=1e-12, atol=1e-12,
    )
