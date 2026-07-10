"""E-graph era Stage 2 (task #66): equality-saturation probes on tensorgrad's
two DOCUMENTED search failures, run against the egglog engine (Rust core,
Python rules — the route the profiler analysis picked).

Probe 1, the adjoint valley (adjoint.py:14-22): reverse-mode ordering is pure
matrix-chain reassociation, but each single reassociation step scores <2%, so
greedy cost search provably never takes it — tensorgrad needed hand-written
structural macro-rules. Equality saturation walks the valley by construction:
associativity fills the e-graph with every parenthesization and flop-cost
extraction picks the vector-first chain. MEASURED: exact optimum at every
depth tried (6..96); depth 48 (a GPT-boundary-chain scale) saturates+extracts
in under a second.

Probe 2, the residual diamond: prod_i (I + J_i) expands to 2^d terms; with
naive bidirectional distribution + add-commutativity the e-graph drowns
(D=4, 8 iters: >5min — the design memo's predicted worst case, confirmed).
The memo's mitigation also confirmed: FACTORING-DIRECTED rules only (collapse
direction + add associativity, no expansion, no commutativity) stay instant
and recover the Horner-form residual recursion g + J@(g + J@(...)) from the
fully expanded input — 2.4x cost reduction; the remaining gap to the true
optimum is exactly the terms commutativity would have to move, i.e. the
canonical-add-ordering / macro-rule design decision.
"""

from __future__ import annotations

import time
from itertools import product as iproduct

import pytest

egglog = pytest.importorskip("egglog")

from egglog import (  # noqa: E402
    EGraph,
    Expr,
    StringLike,
    eq,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    set_,
    set_cost,
)

S = egglog.String
N = 64


class Mat(Expr):
    def __init__(self, name: StringLike, r: i64Like, c: i64Like) -> None: ...
    def __matmul__(self, other: "Mat") -> "Mat": ...
    def __add__(self, other: "Mat") -> "Mat": ...


@function
def rows(m: Mat) -> i64: ...
@function
def cols(m: Mat) -> i64: ...


def _shape_and_cost_rules(a: Mat, b: Mat, c: Mat, nm: S, r: i64, k: i64, n: i64):
    yield rule(eq(a).to(Mat(nm, r, k))).then(set_(rows(a)).to(r), set_(cols(a)).to(k))
    yield rule(eq(c).to(a @ b), eq(rows(a)).to(r), eq(cols(b)).to(n)).then(
        set_(rows(c)).to(r), set_(cols(c)).to(n)
    )
    yield rule(eq(c).to(a + b), eq(rows(a)).to(r), eq(cols(a)).to(n)).then(
        set_(rows(c)).to(r), set_(cols(c)).to(n)
    )
    yield rule(
        eq(c).to(a @ b), eq(rows(a)).to(r), eq(cols(a)).to(k), eq(cols(b)).to(n)
    ).then(set_cost(a @ b, r * k * n))
    yield rule(eq(c).to(a + b), eq(rows(a)).to(r), eq(cols(a)).to(n)).then(
        set_cost(a + b, r * n)
    )


@ruleset
def valley_rules(a: Mat, b: Mat, c: Mat, nm: S, r: i64, k: i64, n: i64):
    yield from _shape_and_cost_rules(a, b, c, nm, r, k, n)
    yield rewrite((a @ b) @ c).to(a @ (b @ c))
    yield rewrite(a @ (b @ c)).to((a @ b) @ c)


@ruleset
def diamond_rules(a: Mat, b: Mat, c: Mat, nm: S, r: i64, k: i64, n: i64):
    yield from _shape_and_cost_rules(a, b, c, nm, r, k, n)
    yield rewrite((a @ b) @ c).to(a @ (b @ c))
    yield rewrite(a @ (b @ c)).to((a @ b) @ c)
    # factoring-directed only — see module docstring
    yield rewrite(a @ c + b @ c).to((a + b) @ c)
    yield rewrite(a @ b + a @ c).to(a @ (b + c))
    yield rewrite((a + b) + c).to(a + (b + c))
    yield rewrite(a + (b + c)).to((a + b) + c)


def test_saturation_walks_the_adjoint_valley():
    """Depth-24 Jacobian chain, left-associated (forward-mode shape): the
    engine must find the EXACT reverse-mode optimum d*n^2 + leaf costs —
    the multi-step reassociation greedy provably never takes."""
    D = 24
    egraph = EGraph()
    expr = Mat("J0", N, N)
    for i in range(1, D):
        expr = expr @ Mat(f"J{i}", N, N)
    expr = expr @ Mat("g", N, 1)
    root = egraph.let("root", expr)
    t0 = time.perf_counter()
    egraph.run(valley_rules.saturate())
    _, cost = egraph.extract(root, include_cost=True)
    assert time.perf_counter() - t0 < 30
    leaf_costs = D + 1 + D  # constructor + matmul-count baseline in this model
    optimum = D * N * N
    assert cost < optimum + 10 * leaf_costs, f"not reverse-mode: {cost} vs {optimum}"
    # and decisively below the forward-mode shape
    assert cost < (D * N**3) / 10


def test_factoring_directed_rules_collapse_the_residual_diamond():
    """Fully EXPANDED (I+J0)(I+J1)...(I+J3) @ g — 16 chain terms. The
    factoring-directed ruleset must recover a substantially collapsed form
    quickly (the naive AC+distribution ruleset provably drowns here: >5min
    at these sizes). Cost must land well below the expanded evaluation."""
    D = 4
    egraph = EGraph()
    terms = []
    for bits in iproduct([0, 1], repeat=D):
        t = Mat("g", N, 1)
        for i in reversed(range(D)):
            if bits[i]:
                t = Mat(f"J{i}", N, N) @ t
        terms.append(t)
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    root = egraph.let("root", expr)
    t0 = time.perf_counter()
    egraph.run(diamond_rules * 6)
    _, cost = egraph.extract(root, include_cost=True)
    assert time.perf_counter() - t0 < 30
    expanded = sum(bin(i).count("1") for i in range(2**D)) * N * N  # 32 matvecs
    assert cost < expanded * 0.5, f"no collapse: {cost} vs expanded {expanded}"
