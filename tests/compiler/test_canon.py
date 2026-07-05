"""Validation of tensorgrad.compiler.canon (compositional structural hashing).

Contracts under test:
  SOUNDNESS   structural_fingerprint(a) == structural_fingerprint(b)
              => a.is_isomorphic(b)                (zero violations allowed)
  INVARIANCE  a.is_isomorphic(b)
              => structural_hash(a) == structural_hash(b)
  CONSISTENCY refined fingerprints equal => coarse hashes equal
  EFFECTIVENESS  isomorphism-preserving rewrites (free-edge renames,
              inner-edge renames, term/factor permutations) should mostly
              preserve the fingerprint (reported; loose lower bound asserted).
"""

import random
from fractions import Fraction

import pytest
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Delta, Product, Sum, Variable, Zero
from tensorgrad.tensor import Derivative, Rename, Tensor, function
from tensorgrad.compiler.canon import (
    canon_info,
    structural_fingerprint,
    structural_hash,
)

n, m = symbols("n m")


# ---------------------------------------------------------------------------
# Random expression corpus.  A single size symbol everywhere maximizes
# symmetry (hardest case for soundness: lots of nearly-isomorphic pairs).
# ---------------------------------------------------------------------------

EDGE_NAMES = list("abcdef")


def _rand_leaf(rng: random.Random) -> Tensor:
    kind = rng.randrange(4)
    edges = rng.sample(EDGE_NAMES, rng.randint(1, 3))
    if kind == 0:
        return Variable(rng.choice("uvw"), **{e: n for e in edges})
    if kind == 1:
        return Delta(n, *edges)
    if kind == 2:
        return Zero(**{e: n for e in edges})
    v = Variable(rng.choice("uvw"), **{e: n for e in edges})
    if len(edges) >= 2 and rng.random() < 0.5:
        v = v.with_symmetries({frozenset(edges[:2])} | {frozenset({e}) for e in edges[2:]})
    return v


def _rand_expr(rng: random.Random, depth: int) -> Tensor:
    if depth == 0:
        return _rand_leaf(rng)
    op = rng.randrange(7)
    t = _rand_expr(rng, depth - 1)
    if op == 0:  # weighted sum (broadcasts mismatched edges)
        u = _rand_expr(rng, depth - 1)
        w1, w2 = rng.choice([1, 2, -1, Fraction(1, 2)]), rng.choice([1, 3, -2])
        return Sum([t, u], [w1, w2])
    if op == 1:  # product (any shared edge has multiplicity exactly 2)
        u = _rand_expr(rng, depth - 1)
        return Product([t, u])
    if op == 2:  # rename one free edge to a fresh name
        if t.edges:
            e = rng.choice(sorted(t.edges))
            fresh = e + "9"
            while fresh in t.edges:
                fresh += "9"
            return t.rename(**{e: fresh})
        return t
    if op == 3:  # contract an edge away
        if t.edges:
            return F.sum(t, [rng.choice(sorted(t.edges))])
        return t
    if op == 4:  # a Function node
        if t.edges:
            return F.softmax(t, dim=rng.choice(sorted(t.edges)))
        return F.pow(t, 2)
    if op == 5:  # an unexpanded Derivative
        x = Variable("w0", g=n)
        return Derivative(t, x)
    return t  # depth padding


def _permute_free_edges(rng: random.Random, t: Tensor) -> Tensor:
    edges = sorted(t.edges)
    if len(edges) < 2:
        return Rename(t, {})
    perm = edges[:]
    rng.shuffle(perm)
    # two-step rename through fresh names so the permutation is valid
    tmp = {e: e + "_tmp" for e in edges}
    back = {e + "_tmp": p for e, p in zip(edges, perm)}
    return t.rename(**tmp).rename(**back)


def _shuffle_children(rng: random.Random, t: Tensor) -> Tensor:
    """An isomorphic rebuild: permute Sum terms / Product factors, and rename
    the inner (contracted) edges of a Product."""
    if isinstance(t, Sum):
        idx = list(range(len(t.terms)))
        rng.shuffle(idx)
        return Sum([t.terms[i] for i in idx], [t.weights[i] for i in idx])
    if isinstance(t, Product):
        inner = {e for f in t.factors for e in f.edges} - set(t.edges)
        used = {e for f in t.factors for e in f.edges}
        ren = {}
        for e in inner:
            fresh = e + "8"
            while fresh in used:
                fresh += "8"
            ren[e] = fresh
            used.add(fresh)
        factors = [f.rename(**{e: v for e, v in ren.items() if e in f.edges}) for f in t.factors]
        rng.shuffle(factors)
        return Product(factors)
    return Rename(t, {})


def _corpus():
    rng = random.Random(0xC0FFEE)
    exprs = [_rand_expr(rng, rng.randint(1, 3)) for _ in range(110)]
    variants = []
    for t in exprs[:60]:
        variants.append((t, _permute_free_edges(rng, t)))
        variants.append((t, _shuffle_children(rng, t)))
    return exprs, variants


EXPRS, VARIANT_PAIRS = _corpus()


# ---------------------------------------------------------------------------
# 1. Soundness: fingerprint equality must imply isomorphism.
# ---------------------------------------------------------------------------


def test_soundness_on_corpus():
    pool = EXPRS + [b for _, b in VARIANT_PAIRS] + _targeted_pairs_flat()
    fps = [structural_fingerprint(t) for t in pool]
    violations = []
    checked = 0
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if fps[i] == fps[j]:
                checked += 1
                if not pool[i].is_isomorphic(pool[j]):
                    violations.append((i, j, pool[i], pool[j]))
    assert not violations, f"{len(violations)} soundness violations, e.g. {violations[0]}"
    assert checked > 50, "corpus too easy: almost no fingerprint-equal pairs were exercised"


# ---------------------------------------------------------------------------
# 2. Invariance: isomorphic tensors must get the same coarse hash.
# ---------------------------------------------------------------------------


def test_hash_invariance_on_variants():
    for a, b in VARIANT_PAIRS:
        assert a.is_isomorphic(b), "variant construction should be isomorphism-preserving"
        assert structural_hash(a) == structural_hash(b), f"hash not invariant for {a!r}"


def test_hash_invariance_on_random_iso_pairs():
    # Cross-check on organically isomorphic pairs (not built as variants),
    # using the exact nx isomorphism test as ground truth.
    import networkx as nx

    pool = EXPRS
    graphs = [t.edge_structural_graph(match_edges=False)[0] for t in pool]
    found = 0
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            G1, G2 = graphs[i], graphs[j]
            if G1.number_of_nodes() != G2.number_of_nodes():
                continue
            if nx.is_isomorphic(G1, G2, node_match=lambda n1, n2: n1.get("name") == n2.get("name")):
                found += 1
                assert structural_hash(pool[i]) == structural_hash(pool[j])
    assert found > 0, "corpus produced no organic isomorphic pairs"


# ---------------------------------------------------------------------------
# 3. Consistency: refined equality must never split what coarse joins.
# ---------------------------------------------------------------------------


def test_refined_equal_implies_coarse_equal():
    pool = EXPRS + [b for _, b in VARIANT_PAIRS]
    seen = {}
    for t in pool:
        info = canon_info(t)
        if info.refined_fp in seen:
            assert seen[info.refined_fp] == info.coarse_fp
        else:
            seen[info.refined_fp] = info.coarse_fp


# ---------------------------------------------------------------------------
# 4. Effectiveness: iso-preserving rewrites should mostly keep the fingerprint.
# ---------------------------------------------------------------------------


def test_effectiveness_on_variants():
    hits = sum(
        structural_fingerprint(a) == structural_fingerprint(b) for a, b in VARIANT_PAIRS
    )
    frac = hits / len(VARIANT_PAIRS)
    # Partial automorphism invariance is allowed (tie-breaks), but the common
    # rewrites (renames / permutations) should essentially always match.
    assert frac >= 0.9, f"effectiveness too low: {frac:.2%} ({hits}/{len(VARIANT_PAIRS)})"


# ---------------------------------------------------------------------------
# 5. Targeted cases.
# ---------------------------------------------------------------------------


def _targeted_pairs_flat():
    out = []
    for a, b, _ in _targeted_cases():
        out += [a, b]
    return out


def _targeted_cases():
    """(a, b, expected_isomorphic) triples with known-tricky structure."""
    cases = []

    # Cross-wiring: P,Q attached to the same copy of a duplicated square
    # variable vs. to different copies.  Isomorphic child multisets and
    # identical per-child orbit colors -- only the joint wiring differs.
    sq = Variable("A", i=n, j=n)
    P, Q = Variable("p", i=n), Variable("q", i=n)
    A1, A2 = sq.rename(i="i1", j="j1"), sq.rename(i="i2", j="j2")
    cases.append(
        (
            Product([A1, A2, P.rename(i="i1"), Q.rename(i="j1")]),
            Product([A1, A2, P.rename(i="i1"), Q.rename(i="j2")]),
            False,
        )
    )

    # Softmax-Hessian pattern from tensor.py:1633: (o-i o-<jk) vs (o-j o-<ik)
    # are isomorphic with different outer labels; __hash__/__eq__ (and hence
    # the fingerprint) must identify them, while match_edges=True would not.
    cases.append(
        (
            Product([Delta(n, "o", "i"), Delta(n, "p", "j", "k")]),
            Product([Delta(n, "o", "j"), Delta(n, "p", "i", "k")]),
            True,
        )
    )

    # Declared symmetries are part of the structure.
    cases.append(
        (
            Variable("X", i=n, j=n).with_symmetries("i j"),
            Variable("X", i=n, j=n),
            False,
        )
    )

    # Variables are edge-name sensitive; Rename wrappers are transparent.
    v = Variable("x", i=n, j=n)
    cases.append((v, Variable("x", a=n, b=n), False))
    cases.append((v, Rename(v, {"i": "a", "j": "b"}), True))

    # Weights: nx labels use str(w), so 2 vs 2.0 differ; Fraction(2,1) == "2".
    cases.append((Sum([v], [2]), Sum([v], [2.0]), False))
    cases.append((Sum([v], [2]), Sum([v], [Fraction(2, 1)]), True))

    # Sum combining is edge-structure sensitive.
    cases.append((Sum([v, v]), Sum([v, v.rename(i="j", j="i")]), False))

    # Function output names matter; broadcast edge names do not. (Uses the
    # generic function() helper: F.softmax is a plain composition since task
    # #34, so its "output" edge is an ordinary free edge and renaming it IS
    # an isomorphism — a fused signature is needed to pin output names.)
    t = Variable("t", i=n, j=n)
    f1 = function("f", {"j": n}, (t, "j"))
    cases.append((f1, function("f", {"j": n}, (t.rename(i="z"), "j")), True))
    cases.append((f1, function("f", {"z": n}, (t.rename(j="z"), "z")), False))

    # Deltas: name-insensitive, size- and order-sensitive.
    cases.append((Delta(n, "a", "b"), Delta(n, "x", "y"), True))
    cases.append((Delta(n, "a", "b"), Delta(m, "a", "b"), False))
    cases.append((Delta(n, "a", "b"), Delta(n, "a", "b", "c"), False))

    # Derivative: sensitive to wrt-variable, transparent to new_names choice.
    w0 = Variable("w0", g=n)
    w1 = Variable("w1", g=n)
    expr = Product([v, Variable("y", j=n)])
    cases.append((Derivative(expr, w0), Derivative(expr, w0, {"g": "h"}), True))
    cases.append((Derivative(expr, w0), Derivative(expr, w1), False))

    return cases


@pytest.mark.parametrize("idx", range(len(_targeted_cases())))
def test_targeted_case(idx):
    a, b, expect_iso = _targeted_cases()[idx]
    assert a.is_isomorphic(b) == expect_iso, "test-case premise failed"
    fa, fb = structural_fingerprint(a), structural_fingerprint(b)
    ha, hb = structural_hash(a), structural_hash(b)
    if expect_iso:
        # These particular iso pairs are ones the design promises to identify.
        assert fa == fb, f"fingerprint failed to identify isomorphic pair {a!r} ~ {b!r}"
        assert ha == hb
    else:
        assert fa != fb, f"SOUNDNESS: fingerprint merged non-isomorphic pair {a!r} != {b!r}"


def test_affine_cases():
    from tensorgrad.compiler.affine import affine_convolution, affine_delta, affine_shift

    c1 = affine_convolution(x=n, k=m, y=n)
    c2 = affine_convolution(x=n, k=m, y=n)
    c3 = affine_convolution(x=n, k=m, y=n, stride=2)
    assert structural_fingerprint(c1) == structural_fingerprint(c2)
    assert structural_fingerprint(c1) != structural_fingerprint(c3)
    assert not c1.is_isomorphic(c3)
    d1, d2 = affine_delta(n, "a", "b"), affine_delta(n, "a", "b")
    assert structural_fingerprint(d1) == structural_fingerprint(d2)
    s1, s2 = affine_shift(1, i=n, o=n), affine_shift(2, i=n, o=n)
    assert structural_fingerprint(s1) != structural_fingerprint(s2)
    assert not s1.is_isomorphic(s2)


def test_expectation_case():
    from tensorgrad.extras.expectation import Expectation

    x = Variable("x", i=n)
    e1 = Expectation(Product([x, x.rename(i="i2")]), x)
    e2 = Expectation(Product([x, x.rename(i="i3")]), x)
    assert e1.is_isomorphic(e2)
    assert structural_fingerprint(e1) == structural_fingerprint(e2)
    assert structural_hash(e1) == structural_hash(e2)


# ---------------------------------------------------------------------------
# 6. Mechanics: memoization, no recursion limit, hash() compatibility.
# ---------------------------------------------------------------------------


def test_deep_chain_no_recursion_error():
    A = Variable("A", i=n)
    x = A
    for _ in range(1500):
        x = x + A
    assert isinstance(structural_fingerprint(x), int)  # would RecursionError if recursive


def test_memoized_on_object():
    A = Variable("A", i=n, j=n)
    t = Product([A, A.rename(i="j", j="i")])
    info1 = canon_info(t)
    assert canon_info(t) is info1  # cached on the object


def test_matches_builtin_hash_contract():
    # For every ==-equal (isomorphic) pair we exercise, structural_hash agrees,
    # i.e. it is a valid drop-in for Tensor.__hash__.
    for a, b in VARIANT_PAIRS[:30]:
        assert (a == b) and structural_hash(a) == structural_hash(b)
