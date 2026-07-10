"""Cached Schwartz-Zippel fingerprints on Tensor objects.

`szfp(t)` is the tensor-level face of the compiler's value fingerprints
(compiler/szfp.py): exact evaluation mod P at k seeded points, keyed by
variable NAMES and edge NAMES — program-independent, so two independently
built expressions over the same variables agree exactly when they are equal
as rational functions of the atoms. The result is cached on the node
(tensors are immutable, so it never invalidates), which is what makes the
semantic tier cheap: consolidation-style questions ("equal?", "zero?")
become dict lookups after first touch.

The fingerprint is NAME-SENSITIVE by design: `x_ij` and its transpose have
different fingerprints, because the consumers (Sum cancellation) compare
terms under the Sum's name alignment. Structural identity (`t == u`,
isomorphism) and value identity (`szfp`) are deliberately separate notions:
value identity must never feed `__eq__` — mod-P equality identifies
float-DIFFERENT groupings (the stabilized-form class) and distinguishes
analytically-equal atoms, so it is only safe in gated/propose-verify
contexts (see the design notes in compiler/szfp.py).

v1 computes by lowering the subtree (the existing `numeric_fingerprint`),
returning None for anything unlowerable (unresolved Derivatives,
Expectations, exotic signatures) — a None simply opts the node out of the
semantic tier. v2 is the incremental per-node evaluator sharing the IR
atom keys, which makes fingerprints a construction-time invariant; the
consumers below are written against this module so that upgrade is
invisible to them.
"""

from typing import Optional

from tensorgrad.tensor import Tensor

_FP_ATTR = "_szfp_v1"
_SIZE_ATTR = "_tsize_v1"
K_TRIALS = 3

# v1 fingerprints lower the whole subtree per (new) node, so an unbounded
# semantic tier would make deep-model normalization quadratic (measured:
# minutes on the minGPT monolith). Terms above this budget keep the
# structural path; the batched zerograd pass (compiler/zerograd.py) remains
# the tier for whole-cotangent-chain proofs until the v2 incremental
# evaluator makes size-independence real.
MAX_NODES = 320


def tree_size(t: Tensor) -> int:
    """Cached tree-size (shared subtrees counted per occurrence — a
    conservative overestimate of DAG size, which is the right direction for
    a work budget). Iterative: gradient chains are deeper than the Python
    recursion limit."""
    from tensorgrad.structure import _children

    hit = t.__dict__.get(_SIZE_ATTR)
    if hit is not None:
        return hit
    stack: list = [(t, False)]
    while stack:
        node, expanded = stack.pop()
        if node.__dict__.get(_SIZE_ATTR) is not None:
            continue
        kids = list(_children(node))
        if expanded:
            node.__dict__[_SIZE_ATTR] = 1 + sum(c.__dict__.get(_SIZE_ATTR, 1) for c in kids)
        else:
            stack.append((node, True))
            stack.extend((c, False) for c in kids if c.__dict__.get(_SIZE_ATTR) is None)
    return t.__dict__[_SIZE_ATTR]


def szfp(t: Tensor) -> Optional[tuple]:
    """Cached value fingerprint of `t` — (weight-symbol support, digest,
    is-zero) — or None when `t` is outside the lowerable fragment or over
    the size budget (which opts it out of semantic rewrites). The support
    component makes terms that scale by different dim polynomials (n*x vs
    m*x) key-unequal regardless of the random dim draws; the is-zero bit
    rides along from the same evaluation."""
    hit = t.__dict__.get(_FP_ATTR, _FP_ATTR)
    if hit is not _FP_ATTR:
        return hit
    try:
        if tree_size(t) > MAX_NODES:
            fp = None
        else:
            from tensorgrad.compiler import szfp as _sz

            (fp,) = _sz.fingerprint_with_support([t], k=K_TRIALS)
    except Exception:
        fp = None
    t.__dict__[_FP_ATTR] = fp
    return fp


def is_zero(t: Tensor) -> Optional[bool]:
    """Cached exact-mod-P zero test; None when undecidable (unlowerable or
    over the size budget — the zerograd pass covers oversized chains).
    Shares szfp's single evaluation: the zero bit is part of the bundle."""
    fp = szfp(t)
    return fp[2] if fp is not None else None
