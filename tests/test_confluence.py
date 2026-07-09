"""Local-confluence fuzzing over the simplify rule catalog (task G).

Not a proof: empirical evidence, on random small expressions, that the
destructive rewrite system has well-defined NORMAL FORMS -- the same
expression must normalize to the same form regardless of how it was
presented (term/factor order), how simplification was scheduled
(subterm-first vs whole), and what its edges are called -- and the normal
form must stay numerically sound against ground truth. Each property that
fails prints its seed for a one-line repro.

These are exactly the obligations a rule CATALOG owes its users (the
paper's confluence box): the engine applies rules in one fixed order, so
any order-sensitivity shows up here as a normal-form mismatch.
"""

import random

import torch

torch.set_num_threads(2)

from tensorgrad.extras.evaluate import evaluate
from tensorgrad.structure import is_isomorphic
from tensorgrad.tensor import Derivative, Function, Product, Rename, Sum
from tensorgrad.testutils import assert_close, random_tensor_expr

N_EXPRS = 40
DEPTHS = (2, 3)
DIMS = (2, 3)


def _shuffle(t, rng: random.Random):
    """A structurally different but semantically identical presentation:
    recursively permute Sum terms and Product factors (the AC freedoms a
    canonical form must quotient away)."""
    if isinstance(t, Sum):
        idx = list(range(len(t.terms)))
        rng.shuffle(idx)
        return Sum([_shuffle(t.terms[k], rng) for k in idx], [t.weights[k] for k in idx])
    if isinstance(t, Product):
        factors = [_shuffle(f, rng) for f in t.factors]
        rng.shuffle(factors)
        return Product(factors)
    return t


def _children_simplified(t):
    """Rebuild t with each child simplified first (subterm-first schedule).
    Returns None for node types the rebuild does not know."""
    if isinstance(t, Sum):
        return Sum([c.simplify() for c in t.terms], list(t.weights))
    if isinstance(t, Product):
        return Product([c.simplify() for c in t.factors])
    if isinstance(t, Rename):
        return Rename(t.tensor.simplify(), dict(t.mapping))
    if isinstance(t, Function):
        return Function(t.signature, [c.simplify() for c in t.inputs], dict(t.shape_out))
    if isinstance(t, Derivative):
        return Derivative(t.tensor.simplify(), t.x, t.new_names)
    return None


def _nf_equal(a, b) -> bool:
    """Normal forms must agree as NAMED graphs (free edges matched), not
    merely up to an arbitrary isomorphism."""
    return is_isomorphic(a, b, match_edges=True)


def _cases():
    it = 0
    for depth in DEPTHS:
        for dim in DIMS:
            for _ in range(N_EXPRS // (len(DEPTHS) * len(DIMS))):
                it += 1
                random.seed(it)
                torch.manual_seed(it)
                expr, expected, variables = random_tensor_expr(max_depth=depth, max_dim=dim)
                yield it, depth, dim, expr, expected, variables


def test_normal_forms_idempotent():
    """fs(e) is a fixpoint: simplifying a normal form changes nothing."""
    bad = []
    for it, depth, dim, expr, _, _ in _cases():
        nf = expr.full_simplify()
        if not _nf_equal(nf, nf.full_simplify()):
            bad.append((it, depth, dim))
    assert not bad, f"non-idempotent normal forms at (seed, depth, dim): {bad}"


def test_normal_forms_presentation_invariant():
    """AC presentation must not matter: shuffled Sum/Product orders
    normalize to the same named graph."""
    bad = []
    for it, depth, dim, expr, _, _ in _cases():
        nf = expr.full_simplify()
        rng = random.Random(it * 7919)
        for _ in range(2):
            nf2 = _shuffle(expr, rng).full_simplify()
            if not _nf_equal(nf, nf2):
                bad.append((it, depth, dim))
                break
    assert not bad, f"presentation-dependent normal forms at (seed, depth, dim): {bad}"


def test_normal_forms_schedule_invariant():
    """Subterm-first scheduling must agree with whole-expression
    simplification (the documented single-pass gap class: a rewrite left
    behind a boundary that a different schedule would have crossed)."""
    bad = []
    for it, depth, dim, expr, _, _ in _cases():
        pre = _children_simplified(expr)
        if pre is None:
            continue
        if not _nf_equal(expr.full_simplify(), pre.full_simplify()):
            bad.append((it, depth, dim))
    assert not bad, f"schedule-dependent normal forms at (seed, depth, dim): {bad}"


def test_normal_forms_sound():
    """The normal form evaluates to the ground truth the generator computed
    alongside the expression."""
    for it, depth, dim, expr, expected, variables in _cases():
        nf = expr.full_simplify()
        assert_close(evaluate(nf, variables), expected, atol=1e-2, rtol=1e-2)
