"""Declarative structural identity for tensorgrad expressions.

Every node class implements a single ``structure()`` method returning a
:class:`Structure` — the ONLY source of structural identity.  Two generic
folds consume it:

  * :func:`build_graph` — the networkx multigraph used for exact isomorphism
    verification (VF2), ``isomorphisms()`` edge mappings, symmetry-orbit
    discovery and drawing.
  * :func:`canon_info` — compositional Merkle-style fingerprints (an
    invariant coarse hash for ``__hash__``, a sound refined fingerprint for
    the ``is_isomorphic`` fast paths, and per-edge colors) — see the contract
    documentation below.

Having both derived from one declaration removes the drift hazard of the old
design, where each class hand-built its nx graph (``structural_graph``) while
canon.py re-encoded the same facts (symmetries, constraints, weights, ...)
with its own conventions.  A real instance of that drift: the old canon
encoder sorted raw ``(Tensor, Tensor)`` constraint pairs, so ``hash()``
crashed with a TypeError on any Variable carrying two or more value
constraints, while the graph path used the sortable ``_constraint_keys``.

The Structure schema
--------------------

Endpoints (plain tuples)::

    ("child", i, edge)   # `edge` of children[i]
    ("free", name)       # a free edge of this node
    ("port", role)       # an anchor on the node itself (orbit, function slot)

``Structure.junctions`` is a frozenset of junctions; a junction (frozenset of
endpoints) means "these endpoints are the same wire".  Inner edge names are
used only to GROUP endpoints into junctions and are then discarded, so
alpha-invariance holds by construction.  Free names appear only inside
``("free", name)`` endpoints, and only the match_edges variants of the folds
consume them.  ``child_roles``: children with equal roles form an
interchangeable multiset (Product factors, equal-weight Sum terms); children
with distinct roles are slotted (Function inputs, Derivative wrt/tensor).

Fingerprint contracts (unchanged from the old canon module)
-----------------------------------------------------------

``structural_hash(t)`` — the *invariant* (coarse) hash.
    ``a.is_isomorphic(b)  =>  structural_hash(a) == structural_hash(b)``.
    Built from multiset/WL-style encodings with no tie-breaking, so it is
    invariant under free-edge renaming, inner-edge renaming, term/factor
    reordering and graph automorphisms.  Backs ``Tensor.__hash__``.

``structural_fingerprint(t)`` — the *sound* (refined) fingerprint.
    ``structural_fingerprint(a) == structural_fingerprint(b) =>
    a.is_isomorphic(b)`` (up to 64-bit collisions).  Where WL refinement
    cannot canonically order symmetric parts, deterministic tie-breaks (edge
    names, positions) pick an order; ties can make two isomorphic tensors get
    DIFFERENT fingerprints (an effectiveness loss, never a soundness loss).

Every combiner also returns a *color* per free edge with two invariants:
  (I1) equal fingerprints imply there is an isomorphism that preserves edge
       colors, and
  (I2) any color-preserving permutation of one tensor's free edges is
       realized by an automorphism (colors only merge edges that are
       independently interchangeable).
(I2) lets a parent encode wiring "up to color" without losing information.

Caveats:
  * Hash values are SEED-STABLE (blake2b over a canonical repr): equal
    expressions get equal fingerprints across processes and PYTHONHASHSEED
    values, so simplify() can use them as deterministic sort keys.
  * Size keys use ``(name, assumptions)`` of a Symbol (sympy interns symbols
    by these, so it matches object identity in-process).
  * Unknown Tensor subclasses (no ``structure()``): the refined fingerprint
    falls back to per-object identity (trivially sound); the coarse hash to
    (type-name, order) (trivially invariant, maximally colliding).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Generator, cast

import networkx as nx
from sympy import Symbol

__all__ = [
    "Structure",
    "structure_of",
    "build_graph",
    "CanonInfo",
    "canon_info",
    "structural_hash",
    "structural_fingerprint",
    "refined_sort_key",
    "edge_structural_graph",
    "is_isomorphic",
    "isomorphisms",
    "symmetry_orbits",
    "graph_to_string",
]


@dataclass(frozen=True)
class Structure:
    """Complete structural identity of one node (see module docstring).

    ``label`` must be a (nested) tuple of str/int/bool — all non-child
    identity: class tag, sizes (via :func:`size_key`), signature name,
    constraint keys, Affine rows, ...  ``transparent`` marks a node that does
    not exist structurally (Rename): it must have exactly one child and only
    ``{("child", 0, old), ("free", new)}`` junctions; the folds splice the
    child through and just relocate its free edges.
    """

    label: tuple
    children: tuple = ()
    child_roles: tuple = ()
    junctions: frozenset = frozenset()
    transparent: bool = False


def structure_of(t: Any) -> Structure:
    """Memoized ``t.structure()`` (tensors are immutable once constructed)."""
    # (cast: pyright resolves instance __dict__ through the metaclass as a
    # read-only mappingproxy; instances have a plain mutable dict)
    d = cast(dict, t.__dict__)
    s = d.get("_structure_v1")
    if s is None:
        d["_structure_v1"] = s = t.structure()
    return s


def _H(*args: Any) -> int:
    """Stable 64-bit combiner: independent of PYTHONHASHSEED and process.

    All arguments are (nested) tuples of str/int/bool, whose ``repr`` is a
    canonical encoding, so blake2b over it is collision-resistant."""
    return int.from_bytes(hashlib.blake2b(repr(args).encode(), digest_size=8).digest(), "big")


@lru_cache(maxsize=None)
def size_key(s: Any) -> tuple:
    """Process-stable identity of an edge size (sympy Symbol or number).

    Sympy interns Symbols by (name, assumptions), so this coincides with
    object identity in-process while staying stable across processes."""
    if isinstance(s, Symbol):
        return ("sym", s.name, tuple(sorted((k, str(v)) for k, v in s.assumptions0.items())))
    return ("num", repr(s))


# ---------------------------------------------------------------------------
# Junction preprocessing shared by both folds.
# ---------------------------------------------------------------------------

# A junction, canonicalized: (sorted port-role reprs, sorted (i, edge) child
# endpoints, sorted free names).
_Junc = tuple[tuple[str, ...], tuple[tuple[int, str], ...], tuple[str, ...]]


def _prepared_junctions(s: Structure) -> list[_Junc]:
    """Canonical (deterministic, seed-independent) list of junctions."""
    out: list[_Junc] = []
    for j in s.junctions:
        childs: list[tuple[int, str]] = []
        frees: list[str] = []
        ports: list[str] = []
        for ep in j:
            kind = ep[0]
            if kind == "child":
                childs.append((ep[1], ep[2]))
            elif kind == "free":
                frees.append(ep[1])
            else:
                ports.append(repr(ep[1]))
        out.append((tuple(sorted(ports)), tuple(sorted(childs)), tuple(sorted(frees))))
    out.sort()
    return out


def _rename_pairs(juncs: list[_Junc]) -> list[tuple[str, str]]:
    """(old, new) pairs of a transparent node's junctions."""
    pairs = []
    for ports, childs, frees in juncs:
        assert not ports and len(childs) == 1 and len(frees) == 1, "bad transparent junction"
        pairs.append((childs[0][1], frees[0]))
    return pairs


# ---------------------------------------------------------------------------
# Fold 1: the structural graph (for VF2 verification, mappings, drawing).
# ---------------------------------------------------------------------------


def build_graph(t: Any) -> tuple[nx.MultiDiGraph, dict[str, int]]:
    """Build the structural graph of ``t``: one mutable graph threaded
    through the whole fold (no per-level union/relabel copies).

    Returns (G, edges) where node 0 is the root and ``edges`` maps each free
    edge name to the node it attaches to.  Shape: root(label) -> role node ->
    child root per child; one node per junction, receiving an edge from every
    member child edge node, attached to the root iff it carries a port; free
    edges resolve to their junction's node.
    """
    G = nx.MultiDiGraph()
    root, edges = _emit(G, t)
    assert root == 0
    return G, edges


def _emit(G: nx.MultiDiGraph, t: Any) -> tuple[int, dict[str, int]]:
    s = structure_of(t)
    juncs = _prepared_junctions(s)
    if s.transparent:
        root, inner = _emit(G, s.children[0])
        return root, {new: inner[old] for old, new in _rename_pairs(juncs)}
    root = G.number_of_nodes()
    G.add_node(root, name=("N", s.label), tensor=t)
    child_edges: list[dict[str, int]] = []
    for child, role in zip(s.children, s.child_roles):
        rn = G.number_of_nodes()
        G.add_node(rn, name=("Role", repr(role)))
        G.add_edge(root, rn)
        c_root, c_edges = _emit(G, child)
        G.add_edge(rn, c_root)
        child_edges.append(c_edges)
    edges: dict[str, int] = {}
    for ports, childs, frees in juncs:
        jn = G.number_of_nodes()
        G.add_node(jn, name=("J", ports))
        if ports:
            G.add_edge(root, jn)
        for i, e in childs:
            G.add_edge(child_edges[i][e], jn)
        for name in frees:
            edges[name] = jn
    return root, edges


# ---------------------------------------------------------------------------
# Fold 2: compositional fingerprints (hashing / equality fast paths).
# ---------------------------------------------------------------------------

_ATTR = "_canon_info_v3"


@dataclass(frozen=True)
class CanonInfo:
    """Result of canonicalizing one tensor node."""

    coarse_fp: int  # invariant hash (iso => equal); for __hash__
    refined_fp: int  # sound fingerprint (equal => iso); for caching
    coarse_colors: dict  # edge name -> invariant color
    refined_colors: dict  # edge name -> (I1)/(I2) color


def structural_hash(t: Any) -> int:
    """Invariant hash: a.is_isomorphic(b) implies equal hashes.  Use for __hash__."""
    return canon_info(t).coarse_fp


def structural_fingerprint(t: Any) -> int:
    """Sound fingerprint: equal fingerprints imply a.is_isomorphic(b)."""
    return canon_info(t).refined_fp


def refined_sort_key(t: Any) -> tuple:
    """Deterministic (seed- and process-stable) total-order key on tensors.

    Name-sensitive: two tensors get the same key only if they are isomorphic
    *with matching edge names* (fingerprint equality plus identical
    name->color maps realizes a name-preserving isomorphism, by (I1)+(I2)).
    Used by simplify() to order Sum terms / Product factors independently of
    construction order and PYTHONHASHSEED."""
    info = canon_info(t)
    return (info.refined_fp, tuple(sorted((e, info.refined_colors[e]) for e in t.edges)))


def canon_info(t: Any) -> CanonInfo:
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
        cast(dict, node.__dict__)[_ATTR] = _compute(node)
        stack.pop()
    return t.__dict__[_ATTR]


def _children(t: Any) -> tuple:
    try:
        return structure_of(t).children
    except NotImplementedError:
        return ()


def _stable_partition_rounds(n: int, sigs: list[int], step: Callable[[list[int]], list[int]]) -> list[int]:
    """Iterate `step` until the induced partition of range(n) stabilizes.

    Round count is partition-driven, hence identical on isomorphic inputs.
    """

    def partition(s: list[int]) -> list[tuple[int, ...]]:
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


def _compute(t: Any) -> CanonInfo:
    try:
        s = structure_of(t)
    except NotImplementedError:
        return _canon_opaque(t)
    juncs = _prepared_junctions(s)

    if s.transparent:
        k = canon_info(s.children[0])
        pairs = _rename_pairs(juncs)
        coarse = {new: k.coarse_colors[old] for old, new in pairs}
        refined = {new: k.refined_colors[old] for old, new in pairs}
        return CanonInfo(k.coarse_fp, k.refined_fp, coarse, refined)

    kids = [canon_info(c) for c in s.children]
    nc, nj = len(kids), len(juncs)
    by_child: list[list[tuple[int, str]]] = [[] for _ in range(nc)]
    for jidx, (_ports, childs, _frees) in enumerate(juncs):
        for i, e in childs:
            by_child[i].append((jidx, e))
    roles = tuple(repr(r) for r in s.child_roles)

    def build(coarse: bool) -> tuple[int, dict[str, int]]:
        fps = [k.coarse_fp if coarse else k.refined_fp for k in kids]
        cols = [k.coarse_colors if coarse else k.refined_colors for k in kids]

        # Joint WL refinement over children and junctions, seeded by child
        # fingerprints (+roles) and junction ports (+free counts).  Free and
        # inner edge NAMES never enter any hash.
        seeds_c = [_H("cs", roles[i], fps[i]) for i in range(nc)]

        def jstep(jsig: list[int], csig: list[int]) -> list[int]:
            return [
                _H(
                    "jw",
                    jsig[j],
                    juncs[j][0],
                    len(juncs[j][2]),
                    tuple(sorted((csig[i], cols[i][e]) for i, e in juncs[j][1])),
                )
                for j in range(nj)
            ]

        seeds_j = [_H("js", ports, len(frees)) for ports, _childs, frees in juncs]
        if len(set(seeds_c)) == nc:
            # Fast path: the children are already discrete (always true for
            # slotted roles and leaves), so no WL round can change any
            # partition — one junction refinement is maximally informative.
            # Iso-invariant: discreteness of the child seed partition is
            # preserved by isomorphism, so isomorphic tensors agree on the
            # path taken.
            csig, jsig = seeds_c, jstep(seeds_j, seeds_c)
        else:

            def step(sigs: list[int]) -> list[int]:
                csig = sigs[:nc]
                new_j = jstep(sigs[nc:], csig)
                new_c = [
                    _H("cw", csig[i], tuple(sorted((new_j[j], cols[i][e]) for j, e in by_child[i])))
                    for i in range(nc)
                ]
                return new_c + new_j

            sigs = _stable_partition_rounds(nc + nj, seeds_c + seeds_j, step)
            csig, jsig = sigs[:nc], sigs[nc:]

        if coarse:
            fp = _H("node-c", s.label, tuple(sorted(csig)), tuple(sorted(jsig)))
            colors: dict[str, int] = {}
            for j in range(nj):
                frees = juncs[j][2]
                if frees:
                    c = _H("ce", fp, jsig[j])
                    for name in frees:
                        colors[name] = c
            return fp, colors

        # Refined: total canonical order over children.  Tie-breaks (child
        # edge names, then list position) only pick the ORDER; the encoding
        # below is a COMPLETE description of the wiring, so equal encodings
        # imply isomorphism regardless of how ties were broken.
        order = sorted(range(nc), key=lambda i: (csig[i], tuple(sorted(s.children[i].edges)), i))
        rank = {i: r for r, i in enumerate(order)}
        encs: list[tuple] = []
        color_encs: list[tuple] = []
        for ports, childs, frees in juncs:
            enc = (ports, len(frees), tuple(sorted((rank[i], cols[i][e]) for i, e in childs)))
            encs.append(enc)
            # Singleton exception: a child whose ONLY edge ends in this
            # port-free junction can be block-swapped with any WL-identical
            # sibling independently of everything else, so its edge color may
            # be sig-based instead of rank-based.  This is the broadcast
            # `x @ Ones(...)` pattern, which matters for matching Sum terms.
            if len(childs) == 1 and not ports and s.children[childs[0][0]].order == 1:
                i, e = childs[0]
                color_encs.append((ports, len(frees), (("sig", csig[i], cols[i][e]),)))
            else:
                color_encs.append(enc)
        fp = _H(
            "node-r",
            s.label,
            tuple((roles[i], fps[i]) for i in order),
            tuple(sorted(encs)),
        )
        # Colors: junctions with identical encodings are interchangeable as
        # blocks; sharing one color across a group is (I2)-sound when each
        # junction contributes a single free edge (a block swap is then just
        # a transposition of edges).  Groups of multi-free junctions (equal
        # orbits of a Constant) get one color PER junction, tie-broken by
        # sorted free names (the rank enters the hash, names never do).
        groups: dict[tuple, list[int]] = defaultdict(list)
        for j in range(nj):
            if juncs[j][2]:
                groups[color_encs[j]].append(j)
        colors = {}
        for enc, js in groups.items():
            if len(js) == 1 or all(len(juncs[j][2]) == 1 for j in js):
                c = _H("re", fp, enc)
                for j in js:
                    for name in juncs[j][2]:
                        colors[name] = c
            else:
                for r, j in enumerate(sorted(js, key=lambda j: juncs[j][2])):
                    c = _H("re2", fp, enc, r)
                    for name in juncs[j][2]:
                        colors[name] = c
        return fp, colors

    coarse_fp, coarse_colors = build(coarse=True)
    refined_fp, refined_colors = build(coarse=False)
    return CanonInfo(coarse_fp, refined_fp, coarse_colors, refined_colors)


def _canon_opaque(t: Any) -> CanonInfo:
    # Unknown subclass (no structure()).  Refined: unique per live object
    # (equal only to itself -> trivially sound).  Coarse: type name + order
    # (trivially invariant under isomorphism -> safe for __hash__).
    coarse_fp = _H("Opaque-c", type(t).__name__, t.order)
    refined_fp = _H("Opaque-r", id(t))
    coarse_c = _H("opaque-e-c", coarse_fp)
    coarse_colors = {e: coarse_c for e in t.edges}
    refined_colors = {e: _H("opaque-e-r", refined_fp, e) for e in t.edges}
    return CanonInfo(coarse_fp, refined_fp, coarse_colors, refined_colors)


################################################################################
# Isomorphism queries. The bodies behind Tensor.is_isomorphic /
# Tensor.isomorphisms / Tensor.symmetries (thin delegates in tensor.py); they
# live here with the canon machinery they consume. edge_structural_graph and
# graph_to_string have no Tensor delegates — call them here directly. Like the
# rest of this module, tensors are duck-typed (t: Any) to keep the import edge
# pointing tensor.py -> here.
################################################################################


def edge_structural_graph(
    t: Any, match_edges: bool = True, edge_names: dict[str, Any] | None = None
) -> tuple[nx.MultiDiGraph, list[str]]:
    """Build a structural graph of the tensor with dummy nodes for outer edges.

    Args:
        match_edges: If True the names are used to match edges.
        edge_names: An optional mapping of edge names.

    Returns:
        A tuple (G, edge_list) where G is the graph and edge_list is the list
        of edge names.
    """
    if edge_names is None:
        edge_names = {}

    G, edges = build_graph(t)

    for e in edges.keys():
        if e not in edge_names:
            edge_names[e] = ("Outer Edge", e) if match_edges else ""

    for e, node in edges.items():
        n = G.number_of_nodes()
        G.add_node(n, name=edge_names[e])
        G.add_edge(node, n)
    return G, list(edges.keys())


def is_isomorphic(
    t: Any, other: Any, match_edges: bool = False, edge_names: dict[str, Any] | None = None
) -> bool:
    """Test whether two tensors are isomorphic.

    Args:
        match_edges: Whether to require a matching of edge names.
        edge_names: Optional mapping to use for edge renaming.
    """
    a, b = canon_info(t), canon_info(other)
    # Sound reject: any isomorphism (with or without edge matching)
    # implies equal invariant hashes.
    if a.coarse_fp != b.coarse_fp:
        return False
    if a.refined_fp == b.refined_fp:
        if not match_edges and not edge_names:
            # Sound accept: equal fingerprints imply isomorphic.
            return True

        # Name-sensitive accept: by (I1) equal fingerprints yield a
        # color-preserving isomorphism, and by (I2) any
        # color-preserving edge permutation is an automorphism, so a
        # label-preserving isomorphism exists iff the (color, label)
        # multisets agree. (Labels mirror edge_structural_graph.)
        def label(e: str) -> Any:
            default = ("Outer Edge", e) if match_edges else ""
            return default if edge_names is None else edge_names.get(e, default)

        if Counter((a.refined_colors[e], label(e)) for e in t.edges) == Counter(
            (b.refined_colors[e], label(e)) for e in other.edges
        ):
            return True
    # Ambiguous (hash-equal but fingerprint/label-distinct): only now
    # pay for the exact nx isomorphism test.
    G1, _ = edge_structural_graph(t, match_edges=match_edges, edge_names=edge_names)
    G2, _ = edge_structural_graph(other, match_edges=match_edges, edge_names=edge_names)
    return nx.is_isomorphic(G1, G2, node_match=lambda n1, n2: n1.get("name") == n2.get("name"))


def isomorphisms(t: Any, other: Any) -> Generator[dict[str, str], None, None]:
    """Yield all isomorphisms (edge renamings) between t and other."""
    G1, edges1 = edge_structural_graph(t, match_edges=False)
    G2, edges2 = edge_structural_graph(other, match_edges=False)
    for matching in nx.algorithms.isomorphism.MultiDiGraphMatcher(
        G1, G2, node_match=lambda n1, n2: n1.get("name") == n2.get("name")
    ).isomorphisms_iter():
        # Matching is a dict {i: j} where i is the node in G1 and j is the node in G2
        # We are only interested in then `len(t.edges)` last nodes, which correspond to the outer edges
        start_i = G1.number_of_nodes() - len(t.edges)
        start_j = G2.number_of_nodes() - len(t.edges)
        yield {
            edges1[i - start_i]: edges2[j - start_j]
            for i, j in matching.items()
            if i >= start_i and j >= start_j
        }


def symmetry_orbits(t: Any) -> set[frozenset[str]]:
    """The orbits of the automorphism group of the tensor: the sets of edges
    that ever get mapped to each other."""
    G = nx.Graph([(i, j) for mapping in isomorphisms(t, t) for i, j in mapping.items()])
    return set(map(frozenset, nx.connected_components(G)))


def graph_to_string(t: Any) -> str:
    """An ASCII tree-like representation of the structural graph."""
    G, _ = edge_structural_graph(t, match_edges=True)
    # (networkx accepts an attribute name for with_labels; the stub says bool)
    return "\n".join(
        nx.generate_network_text(G, with_labels="name", sources=[0])  # pyright: ignore[reportArgumentType]
    )
