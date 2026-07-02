"""Compositional (Merkle-style) structural hashing for tensorgrad expressions.

Backs ``Tensor.__hash__`` and the ``is_isomorphic`` fast paths (see
tensor.py), replacing the networkx ``weisfeiler_lehman`` bottleneck for
hashing/equality.  The nx path rebuilds
the FULL structural graph (tree-expanded, no DAG sharing) on every hash and
every isomorphism test; on a 1-block gpt-nano loss that is ~13.5M nx edge
inserts and dominates simplify().  This module computes hashes bottom-up
with an O(local) combiner per node and a per-object memo, so shared
subexpressions (residual streams!) are hashed once.

Two hashes are computed in one traversal, with different contracts:

``structural_hash(t)`` -- the *invariant* (coarse) hash.
    Contract: ``a.is_isomorphic(b)  =>  structural_hash(a) == structural_hash(b)``.
    This is the direction ``Tensor.__hash__`` needs (hash must agree on
    ==-equal objects; collisions are resolved by __eq__).  It is built from
    pure multiset/WL-style encodings with NO tie-breaking, so it is
    invariant under free-edge renaming, inner-edge renaming, term/factor
    reordering and graph automorphisms.

``structural_fingerprint(t)`` -- the *sound* (refined) fingerprint.
    Contract: ``structural_fingerprint(a) == structural_fingerprint(b)
    => a.is_isomorphic(b)`` (up to 64-bit hash collisions).  This is the
    direction a cache/interning layer needs to SKIP the expensive
    nx.is_isomorphic call.  It is a complete encoding of the structural
    graph: where WL refinement cannot canonically order symmetric parts,
    deterministic tie-breaks (edge names, list position) are used.  A
    tie-break can make two isomorphic tensors get DIFFERENT fingerprints
    (an effectiveness loss, never a soundness loss).

Invariances achieved by the fingerprint (documented per the task):
  * free-edge-name insensitivity: free edges are encoded positionally
    ("de Bruijn"-style port colors), never by name -- EXCEPT where
    tensorgrad itself bakes names into the structural graph: Variable orbit
    names, Function output-edge names, and Affine row signatures (there,
    is_isomorphic is name-sensitive too, so we mirror it).
  * term/factor order insensitivity: Sum/Product encode children as
    multisets.
  * inner-edge-name insensitivity: Product wiring is encoded via
    (canonical factor id, port color) pairs, not via edge names.  Names
    only enter as tie-breaks between WL-indistinguishable factors/edges.
  * automorphism invariance is PARTIAL: configurations that 1-WL with
    child-fingerprint seeds cannot distinguish (e.g. two isomorphic factors
    wired fully symmetrically) fall back to name/position tie-breaks and
    may distinguish some isomorphic pairs.  This is sound for caching.

Soundness architecture (why fingerprint equality implies isomorphism):
every combiner returns, besides the fingerprint, a *color* per free edge
with two invariants:
  (I1) equal fingerprints imply there is an isomorphism that preserves
       edge colors, and
  (I2) any color-preserving permutation of one tensor's free edges is
       realized by an automorphism (colors only merge edges that are
       independently interchangeable).
(I2) is what lets a parent encode wiring "up to color" without losing
information; whenever full interchangeability cannot be cheaply proven,
edges get distinct colors (sound, possibly less effective).

Caveats:
  * Hash values are SEED-STABLE (blake2b over a canonical repr): equal
    expressions get equal fingerprints across processes and PYTHONHASHSEED
    values.  This lets simplify() use fingerprints as deterministic sort
    keys.  (Exception: unknown Tensor subclasses fall back to ``id``.)
  * Size keys use ``(name, assumptions)`` of a Symbol, mirroring tensor.py's
    ``id(size)`` graph labels (sympy interns symbols by (name, assumptions),
    so the two coincide in-process, except after ``clear_cache()``).
  * Unknown Tensor subclasses: the fingerprint falls back to per-object
    identity (trivially sound); the coarse hash falls back to
    (type-name, order) (trivially invariant, maximally colliding).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from numbers import Number
from typing import Callable, Optional

from sympy import Symbol

from tensorgrad.tensor import (
    Constant,
    Delta,
    Derivative,
    Function,
    Product,
    Rename,
    Sum,
    Tensor,
    Variable,
)

_ATTR = "_canon_info_v2"


@dataclass(frozen=True)
class CanonInfo:
    """Result of canonicalizing one tensor node."""

    coarse_fp: int  # invariant hash (iso => equal); for __hash__
    refined_fp: int  # sound fingerprint (equal => iso); for caching
    coarse_colors: dict  # edge name -> invariant color
    refined_colors: dict  # edge name -> (I1)/(I2) color


def _H(*args) -> int:
    """Stable 64-bit combiner: independent of PYTHONHASHSEED and process.

    All arguments are (nested) tuples of str/int/bool, whose ``repr`` is a
    canonical encoding, so blake2b over it is collision-resistant.  Stability
    is required because fingerprints double as deterministic sort keys for
    simplify() output ordering (see refined_sort_key)."""
    return int.from_bytes(hashlib.blake2b(repr(args).encode(), digest_size=8).digest(), "big")


@lru_cache(maxsize=None)
def _size_key(s) -> tuple:
    """Mirror tensor.py's size-node labels, which are (size.name, id(size))
    for Symbols.  Since sympy interns Symbols by (name, assumptions), the
    assumptions dict is a process-stable stand-in for id()."""
    if isinstance(s, Symbol):
        return ("sym", s.name, tuple(sorted((k, str(v)) for k, v in s.assumptions0.items())))
    return ("num", repr(s))


def structural_hash(t: Tensor) -> int:
    """Invariant hash: a.is_isomorphic(b) implies equal hashes.  Use for __hash__."""
    return canon_info(t).coarse_fp


def structural_fingerprint(t: Tensor) -> int:
    """Sound fingerprint: equal fingerprints imply a.is_isomorphic(b)."""
    return canon_info(t).refined_fp


def refined_sort_key(t: Tensor) -> tuple:
    """Deterministic (seed- and process-stable) total-order key on tensors.

    Name-sensitive: two tensors get the same key only if they are isomorphic
    *with matching edge names* (fingerprint equality plus identical
    name->color maps realizes a name-preserving isomorphism, by (I1)+(I2)).
    Used by simplify() to order Sum terms / Product factors independently of
    construction order and PYTHONHASHSEED."""
    info = canon_info(t)
    return (info.refined_fp, tuple(sorted((e, info.refined_colors[e]) for e in t.edges)))


def canon_info(t: Tensor) -> CanonInfo:
    """Compute (memoized, iteratively -- no recursion limit) the CanonInfo of t."""
    info = t.__dict__.get(_ATTR)
    if info is not None:
        return info
    # Iterative post-order over the expression DAG (id-sharing respected via memo).
    stack = [t]
    while stack:
        node = stack[-1]
        if _ATTR in node.__dict__:
            stack.pop()
            continue
        pending = [k for k in _children(node) if _ATTR not in k.__dict__]
        if pending:
            stack.extend(pending)
            continue
        node.__dict__[_ATTR] = _compute(node)
        stack.pop()
    return t.__dict__[_ATTR]


def _children(t: Tensor) -> list[Tensor]:
    if isinstance(t, Sum):
        return t.terms
    if isinstance(t, Product):
        return t.factors
    if isinstance(t, Function):
        return t.inputs
    if isinstance(t, Derivative):
        return [t.x, t.tensor]
    if isinstance(t, Rename):
        return [t.tensor]
    if _Expectation is not None and isinstance(t, _Expectation):
        return [t.tensor, t.wrt, t.mu, t.covar]
    return []


try:  # Optional: extras may not be importable in minimal environments.
    from tensorgrad.extras.expectation import Expectation as _Expectation
except Exception:  # pragma: no cover
    _Expectation = None

try:
    from tensorgrad.compiler.affine import Affine as _Affine
except Exception:  # pragma: no cover
    _Affine = None


# ---------------------------------------------------------------------------
# Per-type combiners.  Each returns a CanonInfo given the children's infos.
# ---------------------------------------------------------------------------


def _compute(t: Tensor) -> CanonInfo:
    if isinstance(t, Variable):
        return _canon_variable(t)
    if isinstance(t, Delta):
        return _canon_delta(t)
    if _Affine is not None and isinstance(t, _Affine):
        return _canon_affine(t)
    if isinstance(t, Constant):
        return _canon_constant(t)
    if isinstance(t, Rename):
        return _canon_rename(t)
    if isinstance(t, Sum):
        return _canon_sum(t)
    if isinstance(t, Product):
        return _canon_product(t)
    if isinstance(t, Function):
        return _canon_function(t)
    if isinstance(t, Derivative):
        return _canon_derivative(t)
    if _Expectation is not None and isinstance(t, _Expectation):
        return _canon_expectation(t)
    return _canon_opaque(t)


def _canon_variable(v: Variable) -> CanonInfo:
    # tensor.py:565: node ("Variable", name); size nodes (f"size={name}", id);
    # orbit nodes ("Orbit Node", " ".join(sorted(orbit))).  Orbit NAMES are in
    # the labels, so variables are name-sensitive (mirrors is_isomorphic).
    orbits = []
    for orbit in v._symmetries:
        e0 = next(iter(orbit))
        orbits.append((_size_key(v._shape[e0]), " ".join(sorted(orbit))))
    fp = _H("Variable", v.name, tuple(sorted(orbits)))
    colors = {}
    for size_key, orbit_name in orbits:
        c = _H("var-orbit", fp, size_key, orbit_name)
        for e in orbit_name.split(" ") if orbit_name else []:
            colors[e] = c
    # Degenerate: scalar variable has no orbits/edges.
    return CanonInfo(fp, fp, colors, dict(colors))


def _canon_delta(t: Delta) -> CanonInfo:
    # tensor.py:807: root "Delta" + one size node; ALL edges attach to the
    # size node, so they are fully interchangeable: one shared color.
    fp = _H("Delta", _size_key(t.size), t.order)
    c = _H("delta-edge", fp)
    colors = {e: c for e in t.edges}
    return CanonInfo(fp, fp, colors, dict(colors))


def _canon_constant(t: Constant) -> CanonInfo:
    # tensor.py:732: root = type name; orbit nodes are UNLABELED, so orbits
    # of equal (size, cardinality) are interchangeable as blocks.
    groups = sorted(
        (_size_key(t._shape[next(iter(orbit))]), len(orbit)) for orbit in t._symmetries
    )
    fp = _H("Constant", type(t).__name__, tuple(groups))
    coarse_colors, refined_colors = {}, {}
    # Refined: singleton orbits of the same size share a color (independently
    # swappable); orbits with >= 2 edges each get a distinct color, tie-broken
    # by sorted edge names (rank only -- names never enter the hash).
    by_group = defaultdict(list)
    for orbit in t._symmetries:
        e0 = next(iter(orbit))
        by_group[(_size_key(t._shape[e0]), len(orbit))].append(sorted(orbit))
    for (size_key, card), orbs in by_group.items():
        coarse_c = _H("const-orbit-c", fp, size_key, card)
        for rank, orbit_edges in enumerate(sorted(orbs)):
            refined_c = (
                _H("const-orbit-r", fp, size_key, 1)
                if card == 1
                else _H("const-orbit-r", fp, size_key, card, rank)
            )
            for e in orbit_edges:
                coarse_colors[e] = coarse_c
                refined_colors[e] = refined_c
    return CanonInfo(fp, fp, coarse_colors, refined_colors)


def _canon_affine(t) -> CanonInfo:
    # affine.py:70: the root label carries the canonical row signature, which
    # INCLUDES edge names, but the edge->size mapping is NOT in the graph.
    # Coarse must therefore not depend on the edge->size map (isomorphic
    # Affines have identical row strings but may attach sizes differently);
    # refined pins everything by name (finer than nx-iso: always sound).
    rows_sig = t._canonical_rows()
    sizes = tuple(sorted(_size_key(s) for s in t._shape.values()))
    coarse_fp = _H("Affine-c", rows_sig, sizes)
    refined_fp = _H("Affine-r", rows_sig, tuple(sorted((e, _size_key(s)) for e, s in t._shape.items())))
    coarse_colors = {e: _H("affine-e-c", coarse_fp, _size_key(t._shape[e])) for e in t.edges}
    refined_colors = {e: _H("affine-e-r", refined_fp, e) for e in t.edges}
    return CanonInfo(coarse_fp, refined_fp, coarse_colors, refined_colors)


def _canon_rename(t: Rename) -> CanonInfo:
    # tensor.py:695: Rename does not create a node; identical graph, edge dict
    # remapped.  So: same fingerprints, colors relocated to the new names.
    k = canon_info(t.tensor)
    coarse = {t.mapping.get(e, e): c for e, c in k.coarse_colors.items()}
    refined = {t.mapping.get(e, e): c for e, c in k.refined_colors.items()}
    return CanonInfo(k.coarse_fp, k.refined_fp, coarse, refined)


def _stable_partition_rounds(n: int, sigs: list[int], step: Callable[[list[int]], list[int]]) -> list[int]:
    """Iterate `step` until the induced partition of range(n) stabilizes.

    Round count is partition-driven, hence identical on isomorphic inputs.
    """

    def partition(s):
        groups = defaultdict(list)
        for i, v in enumerate(s):
            groups[v].append(i)
        return sorted(tuple(g) for g in groups.values())

    part = partition(sigs)
    for _ in range(n + 1):
        new = step(sigs)
        new_part = partition(new)
        if new_part == part:
            return new
        sigs, part = new, new_part
    return sigs


def _canon_sum(t: Sum) -> CanonInfo:
    # tensor.py:1660: root "Sum"; one unlabeled Plus Node per free edge; each
    # term hangs off a labeled f"Weight {w}" node and wires its edge nodes to
    # the Plus Nodes BY NAME (Sum.__init__ broadcasts: all terms carry all
    # edges).  Encoding: the terms x edges color matrix.
    kids = [canon_info(x) for x in t.terms]
    ws = [str(w) for w in t.weights]  # nx uses f"Weight {w}" -> str, not numeric equality
    edges = list(t.edges)
    n = len(t.terms)

    def build(coarse: bool):
        colors_k = [k.coarse_colors if coarse else k.refined_colors for k in kids]
        fps_k = [k.coarse_fp if coarse else k.refined_fp for k in kids]

        # Joint WL refinement of edge classes and term signatures.
        eclass = {e: 0 for e in edges}
        tsig = [0] * n

        def step(esigs):
            ec = dict(zip(edges, esigs))
            for i in range(n):
                tsig[i] = _H("st", ws[i], fps_k[i], tuple(sorted((ec[e], colors_k[i][e]) for e in edges)))
            return [_H("se", tuple(sorted((tsig[i], colors_k[i][e]) for i in range(n)))) for e in edges]

        esigs = _stable_partition_rounds(len(edges), [eclass[e] for e in edges], step)
        eclass = dict(zip(edges, esigs))
        # One final tsig computation against the stable edge classes.
        step(esigs)

        if coarse:
            fp = _H("Sum", len(edges), tuple(sorted(tsig)))
            colors = {e: _H("sumedge-c", fp, eclass[e]) for e in edges}
            return fp, colors

        # Refined: canonical edge order (WL class, then name as tie-break --
        # the name only picks the ORDER, it never enters any hash), then the
        # full color matrix row per term.
        order = sorted(edges, key=lambda e: (eclass[e], e))
        tsig_full = [
            _H("stf", ws[i], fps_k[i], tuple(colors_k[i][e] for e in order)) for i in range(n)
        ]
        fp = _H("Sum", len(edges), tuple(sorted(tsig_full)))
        # Output color of edge e = its column, listed over sig-sorted terms.
        # Ties between identical full sigs are harmless: identical sigs imply
        # identical rows, hence identical column entries.
        torder = sorted(range(n), key=lambda i: tsig_full[i])
        colors = {e: _H("sumedge-r", fp, tuple(colors_k[i][e] for i in torder)) for e in edges}
        return fp, colors

    coarse_fp, coarse_colors = build(coarse=True)
    refined_fp, refined_colors = build(coarse=False)
    return CanonInfo(coarse_fp, refined_fp, coarse_colors, refined_colors)


def _canon_product(t: Product) -> CanonInfo:
    # tensor.py:1497: root "Product"; factors attach unlabeled; an inner edge
    # (a name shared by exactly two factors) becomes a bidirectional edge pair
    # between the two factors' edge nodes; single-occurrence edges are free.
    kids = [canon_info(x) for x in t.factors]
    n = len(t.factors)
    owners = defaultdict(list)
    for i, f in enumerate(t.factors):
        for e in f.edges:
            owners[e].append(i)
    inner = {e: os for e, os in owners.items() if len(os) == 2}
    pendant = {e: os[0] for e, os in owners.items() if len(os) == 1}
    # Neighbor lists: factor i -> [(edge, other factor)], and pendant edges.
    nbrs = [[] for _ in range(n)]
    pends = [[] for _ in range(n)]
    for e, (a, b) in inner.items():
        nbrs[a].append((e, b))
        nbrs[b].append((e, a))
    for e, i in pendant.items():
        pends[i].append(e)

    def build(coarse: bool):
        colors_k = [k.coarse_colors if coarse else k.refined_colors for k in kids]
        fps_k = [k.coarse_fp if coarse else k.refined_fp for k in kids]

        def step(sigs):
            return [
                _H(
                    "pwl",
                    sigs[i],
                    tuple(sorted((colors_k[i][e], sigs[j], colors_k[j][e]) for e, j in nbrs[i])),
                    tuple(sorted(colors_k[i][e] for e in pends[i])),
                )
                for i in range(n)
            ]

        sigs = _stable_partition_rounds(n, [_H("pf", fp) for fp in fps_k], step)

        if coarse:
            wires = tuple(
                sorted(
                    tuple(sorted([(sigs[a], colors_k[a][e]), (sigs[b], colors_k[b][e])]))
                    for e, (a, b) in inner.items()
                )
            )
            pend_items = tuple(sorted((sigs[i], colors_k[i][e]) for e, i in pendant.items()))
            fp = _H("Product", tuple(sorted(sigs)), wires, pend_items)
            colors = {e: _H("pedge-c", fp, sigs[i], colors_k[i][e]) for e, i in pendant.items()}
            return fp, colors

        # Refined: total canonical order over factors.  Tie-breaks (factor
        # edge names, then list position) only pick the order; the encoding
        # below is a COMPLETE description of the wiring, so equal encodings
        # imply isomorphism regardless of how ties were broken.
        order = sorted(range(n), key=lambda i: (sigs[i], tuple(sorted(t.factors[i].edges)), i))
        rank = {i: r for r, i in enumerate(order)}
        nodes = tuple(fps_k[i] for i in order)
        wires = tuple(
            sorted(
                tuple(sorted([(rank[a], colors_k[a][e]), (rank[b], colors_k[b][e])]))
                for e, (a, b) in inner.items()
            )
        )
        pend_items = tuple(sorted((rank[i], colors_k[i][e]) for e, i in pendant.items()))
        fp = _H("Product", n, nodes, wires, pend_items)
        # Pendant colors: rank-based in general (edges on distinct factors are
        # not independently swappable -- see the cross-wiring test).  EXCEPT:
        # a factor with exactly one edge, which is free (no wires), can be
        # block-swapped with any WL-identical sibling independently of
        # everything else, so its pendant color may be sig-based.  This is
        # the broadcast `x @ Ones(...)` pattern, so it matters a lot for
        # matching Sum terms.
        singleton = [len(t.factors[i].edges) == 1 and not nbrs[i] for i in range(n)]
        colors = {
            e: (
                _H("pedge-r1", fp, sigs[i], colors_k[i][e])
                if singleton[i]
                else _H("pedge-r", fp, rank[i], colors_k[i][e])
            )
            for e, i in pendant.items()
        }
        return fp, colors

    coarse_fp, coarse_colors = build(coarse=True)
    refined_fp, refined_colors = build(coarse=False)
    return CanonInfo(coarse_fp, refined_fp, coarse_colors, refined_colors)


def _canon_function(t: Function) -> CanonInfo:
    # tensor.py:1201: root ("Function", sig.name); output edges are labeled
    # ("Edge Out", e) -- output NAMES matter; inputs are ordered (labeled
    # f"{i}"); consumed input edges wire into the function node; the rest of
    # each input's edges broadcast through as free edges.
    kids = [canon_info(x) for x in t.inputs]
    sig = t.signature
    out_names = tuple(sorted(t.shape_out.keys()))

    def build(coarse: bool):
        colors_k = [k.coarse_colors if coarse else k.refined_colors for k in kids]
        fps_k = [k.coarse_fp if coarse else k.refined_fp for k in kids]
        enc = tuple(
            (fps_k[i], tuple(sorted(colors_k[i][e] for e in sig.inputs[i])))
            for i in range(len(t.inputs))
        )
        fp = _H("Function", sig.name, out_names, enc)
        colors = {e: _H("fout", fp, e) for e in t.shape_out}
        for i, (x, es) in enumerate(zip(t.inputs, sig.inputs)):
            for e in x.edges:
                if e not in es:
                    colors[e] = _H("fbc", fp, i, colors_k[i][e])
        return fp, colors

    coarse_fp, coarse_colors = build(coarse=True)
    refined_fp, refined_colors = build(coarse=False)
    return CanonInfo(coarse_fp, refined_fp, coarse_colors, refined_colors)


def _canon_derivative(t: Derivative) -> CanonInfo:
    # tensor.py:1293: root "Derivative"; x under label "self.x", tensor under
    # "self.tensor".  Each new_names edge is a pendant on the corresponding
    # orbit node of x, so its color is derived from x's edge color.
    kx, kt = canon_info(t.x), canon_info(t.tensor)

    def build(coarse: bool):
        cx = kx.coarse_colors if coarse else kx.refined_colors
        ct = kt.coarse_colors if coarse else kt.refined_colors
        fp = _H(
            "Derivative",
            kx.coarse_fp if coarse else kx.refined_fp,
            kt.coarse_fp if coarse else kt.refined_fp,
        )
        colors = {e: _H("dt", fp, ct[e]) for e in t.tensor.edges}
        for oe, ne in t.new_names.items():
            colors[ne] = _H("dx", fp, cx[oe])
        return fp, colors

    coarse_fp, coarse_colors = build(coarse=True)
    refined_fp, refined_colors = build(coarse=False)
    return CanonInfo(coarse_fp, refined_fp, coarse_colors, refined_colors)


def _canon_expectation(t) -> CanonInfo:
    # extras/expectation.py:227: children under labels self.tensor / self.wrt
    # / self.mu / self.covar; free edges are the tensor's (covar_names is NOT
    # in the nx graph -- mirrored here).
    ks = [canon_info(x) for x in (t.tensor, t.wrt, t.mu, t.covar)]
    coarse_fp = _H("Expectation", tuple(k.coarse_fp for k in ks))
    refined_fp = _H("Expectation", tuple(k.refined_fp for k in ks))
    coarse_colors = {e: _H("exp-c", coarse_fp, ks[0].coarse_colors[e]) for e in t.tensor.edges}
    refined_colors = {e: _H("exp-r", refined_fp, ks[0].refined_colors[e]) for e in t.tensor.edges}
    return CanonInfo(coarse_fp, refined_fp, coarse_colors, refined_colors)


def _canon_opaque(t: Tensor) -> CanonInfo:
    # Unknown subclass.  Refined: unique per live object (equal only to
    # itself -> trivially sound).  Coarse: type name + order (trivially
    # invariant under isomorphism -> safe for __hash__, just colliding).
    coarse_fp = _H("Opaque-c", type(t).__name__, t.order)
    refined_fp = _H("Opaque-r", id(t))
    coarse_c = _H("opaque-e-c", coarse_fp)
    coarse_colors = {e: coarse_c for e in t.edges}
    refined_colors = {e: _H("opaque-e-r", refined_fp, e) for e in t.edges}
    return CanonInfo(coarse_fp, refined_fp, coarse_colors, refined_colors)
