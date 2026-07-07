r"""Book-grammar layout for tensor diagrams.

Turns a tensorgrad ``Tensor`` into a diagram laid out the way the Tensor
Cookbook draws them by hand.  This is deliberately NOT a general graph-drawing
algorithm: the book's figures follow a small layout grammar, and this module
implements that grammar directly.

The grammar (see also paper/animations/SKILL.md):

1.  **Spine**: each connected component has a horizontal backbone -- the path
    between the two principal free edges (row enters left, column exits
    right).  Nodes sit ON the line.
2.  **Closures**: wires not on the spine become arcs ABOVE it, nested like a
    chord diagram (a one-page book embedding).  A crossing that cannot be
    nested away is drawn as a crossing -- in this notation that is meaningful.
3.  **Branches**: off-spine pendant structure hangs BELOW its attachment node.
4.  **Free edges**: horizontal stubs at the spine ends; extra free edges on
    interior nodes leave as short labeled stubs upward.
5.  **Sums**: terms side by side with +/- (weights as prefixes).
6.  **Scalar factors** (closed components) are juxtaposed before the open
    component.

Orientation convention: a ``Variable``'s declared edge order is its port
order -- the FIRST edge prefers to exit left, the LAST to exit right.  This
is what makes covariance automatic.  Callers can override with
``left=``/``right=``.

The output is renderer-neutral geometry (``BookLayout``) plus a TikZ emitter
(``to_book_tikz``) that speaks the book's own style vocabulary
(paper/chapters/tikz-styles.tex: ``copydot`` etc.), so generated code can be
pasted into the book.

The whole tensorgrad language is supported:

* **Variable / Zero** -- labelled nodes (Zero is a ``0`` node).
* **Delta** (any order) -- order 2 is an identity wire, order 0 a scalar
  ``|n|``, order >= 3 a copy-dot; a closed ring of identities is a circle.
* **Product** -- contraction; shared edges become wires, the rest stubs.
* **Rename** -- re-keys free edges (including swaps).
* **Sum** -- top level AND as parenthesized product factors (drawn as a
  bracketed group with wires passing through).
* **Function** -- application arrows: solid for consumed edges, densely
  dotted for elementwise; broadcast edges stay on the argument.
* **Derivative** (unevaluated) -- a Penrose loop: a fit-ellipse around the
  differentiated subexpression with a boundary dot and labelled whiskers
  for the new edges (one bends left; a pair bends left/right, like the
  book's \dloop/\dwhiskers). Nested derivatives nest their ellipses. A
  derivative whose new edge is contracted onward raises NotImplementedError
  -- call simplify() first.
* **Expectation** -- an ``E[.]`` fit-box around the inner subexpression.

A 2-port variable forced into reversed port order (a transpose, e.g. the
A^T term of a gradient) is drawn rotated 180 degrees, but only when it has
a free edge -- internal contraction nodes stay upright.

Entry points (all equivalent): ``tensorgrad.to_book_tikz(t, **opts)``,
``t.to_book_tikz(**opts)``, or this module's ``to_book_tikz``.  Options:
``left=``/``right=`` fix a free edge's side; ``max_width=`` scales a
too-wide diagram down to fit; ``scale=`` sets an explicit factor.

Known limitations (both narrow):

* A densely-connected, genuinely 2-D contraction with no clean linear
  spine -- e.g. an *unsimplified* softmax over a matrix axis times V
  (attention) -- flattens onto the spine with hanging pendants and can
  look cramped. Simplify first, or draw such states by hand.
* An elementwise function of a *closed-ring scalar* (e.g. ``exp(Tr(I))``,
  a contraction with no atom and no free edge) draws the function glyph
  separately from the ring circle. Real expressions don't produce this.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from numbers import Number
from typing import Optional

from tensorgrad.tensor import (
    Delta, Derivative, Function, Product, Rename, Sum, Tensor, Variable, Zero,
)

try:
    from tensorgrad.extras.expectation import Expectation as _Expectation
except Exception:  # pragma: no cover
    _Expectation = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Metric constants (calibrated against the book's hand figures)
# ---------------------------------------------------------------------------

PITCH = 0.62  # horizontal distance between adjacent spine node centers
CHAR_W = 0.14  # extra pitch per label character beyond the first
STUB = 0.32  # length of a free-edge stub
PENDANT_DY = 0.55  # vertical drop of a hanging branch level
MIDSTUB = 0.4  # length of an interior free-edge stub (drawn upward)
TERM_GAP = 0.55  # gap between a sum term and its +/- sign
COMPONENT_GAP = 0.7  # gap between juxtaposed components


# ---------------------------------------------------------------------------
# Open-graph extraction
# ---------------------------------------------------------------------------


@dataclass
class AtomSpec:
    """One drawable node: a variable, a copy-dot, or a sum-terminating dot."""

    id: int
    kind: str  # 'var' | 'copydot' | 'scalar' | 'func' | 'group'
    label: str  # tex for 'var'/'scalar'/'func', '' for dots
    ports: list[str] = field(default_factory=list)  # wire ids, declared order
    sub: Optional[Tensor] = None  # 'group': the parenthesized inner Sum
    port_names: dict[str, str] = field(default_factory=dict)  # wire id -> edge


@dataclass
class OpenGraph:
    atoms: list[AtomSpec] = field(default_factory=list)
    # wire id -> list of atom ids (len 2 = internal, len 1 = free,
    # len 0 = bare identity wire between two absorbed free ends)
    wires: dict[str, list[int]] = field(default_factory=dict)
    free_wires: dict[str, list[str]] = field(default_factory=dict)  # wid -> names
    rings: int = 0  # closed loops of pure identities (each = a circle)
    bare_wires: list[tuple[str, str]] = field(default_factory=list)
    # wire id -> 'solid' | 'dotted': drawn with an arrowhead INTO the
    # function atom (application arrows; dotted = elementwise)
    arrows: dict[str, str] = field(default_factory=dict)
    arrow_heads: dict[str, int] = field(default_factory=dict)  # wire -> fn atom
    # Penrose derivative loops: (atom ids enclosed, new edge names)
    derivs: list[tuple[list[int], list[str]]] = field(default_factory=list)
    deriv_tokens: set = field(default_factory=set)  # wires owned by loops
    # E[.] expectation boxes: (enclosed atom ids, label tex)
    boxes: list[tuple[list[int], str]] = field(default_factory=list)


def _tex_edge(name: str) -> str:
    """Edge names to math: i_ -> i'; i_0_0 -> i_{0,0}; plain stays plain."""
    if "_" not in name:
        return name
    base, *subs = name.split("_")
    primes = "'" * sum(1 for p in subs if p == "")
    subs = [p for p in subs if p]
    out = base + primes
    if subs:
        out += "_{" + ",".join(subs) + "}"
    return out


def _pretty_fn(name: str) -> str:
    """'pow(k=-1)' -> pow_{-1}; 'pow(k=Fraction(1,2))' -> pow_{1/2}; 'exp' -> exp."""
    if "(" not in name:
        return name
    base, _, arg = name.partition("(")
    # drop exactly the matching outer ')'
    if arg.endswith(")"):
        arg = arg[:-1]
    if "=" in arg:
        arg = arg.split("=", 1)[1]
    # Fraction(a,b) -> a/b (sympy powers use rational exponents)
    m = re.fullmatch(r"Fraction\((-?\d+),\s*(-?\d+)\)", arg)
    if m:
        arg = rf"{m.group(1)}/{m.group(2)}"
    return rf"{base}_{{{arg}}}"


class _UnionFind(dict):
    def find(self, x: str) -> str:
        while self.setdefault(x, x) != x:
            self[x] = self[self[x]]
            x = self[x]
        return x

    def union(self, a: str, b: str) -> None:
        self[self.find(a)] = self.find(b)


def extract_graph(tensor: Tensor) -> OpenGraph:
    """Flatten a (product of) tensor(s) into atoms + wires + free edges."""
    g = OpenGraph()
    uf = _UnionFind()
    counter = itertools.count()
    identity_pairs: list[tuple[str, str]] = []

    def fresh(prefix: str) -> str:
        return f"__{prefix}{next(counter)}"

    def walk(t: Tensor) -> dict[str, str]:
        """Return a map from t's free edge names to internal wire tokens."""
        if isinstance(t, Rename):
            inner = walk(t.tensor)
            # mapping: old free name -> new free name.  Re-key the exposed map.
            return {t.mapping.get(old, old): tok for old, tok in inner.items()}
        if isinstance(t, Variable):
            aid = len(g.atoms)
            toks = [fresh("w") for _ in t.edges]
            label = t.name
            g.atoms.append(AtomSpec(aid, "var", label, toks))
            return dict(zip(list(t.edges), toks))
        if isinstance(t, Zero):
            edges = list(t.edges)
            aid = len(g.atoms)
            if not edges:
                g.atoms.append(AtomSpec(aid, "scalar", "0", []))
                return {}
            toks = [fresh("w") for _ in edges]
            g.atoms.append(AtomSpec(aid, "var", "0", toks))
            return dict(zip(edges, toks))
        if isinstance(t, Delta):
            edges = list(t.edges)
            if len(edges) == 2:
                # identity: the two ends are the same wire
                ta, tb = fresh("w"), fresh("w")
                identity_pairs.append((ta, tb))
                return {edges[0]: ta, edges[1]: tb}
            if len(edges) == 0:
                aid = len(g.atoms)
                g.atoms.append(AtomSpec(aid, "scalar", str(t.size), []))
                return {}
            # order 1 (sum dot) and order >= 3 (copy dot) are both drawn as
            # a filled dot; order is implied by the wire count.
            aid = len(g.atoms)
            toks = [fresh("w") for _ in edges]
            g.atoms.append(AtomSpec(aid, "copydot", "", toks))
            return dict(zip(edges, toks))
        if isinstance(t, Product):
            exposed: dict[str, list[str]] = {}
            for f in t.factors:
                for name, tok in walk(f).items():
                    exposed.setdefault(name, []).append(tok)
            out: dict[str, str] = {}
            for name, toks in exposed.items():
                if len(toks) == 2:  # contraction: same wire
                    uf.union(toks[0], toks[1])
                elif len(toks) == 1:
                    out[name] = toks[0]
                else:  # pragma: no cover - Product.__init__ rejects this
                    raise ValueError(f"edge {name} has multiplicity {len(toks)}")
            return out
        if isinstance(t, Function):
            aid = len(g.atoms)
            fn = AtomSpec(aid, "func", _pretty_fn(t.signature.name), [])
            g.atoms.append(fn)
            fout: dict[str, str] = {}
            for inp, consumed in zip(t.inputs, t.signature.inputs):
                first_atom = len(g.atoms)
                sub = walk(inp)
                elementwise = not consumed
                for e in consumed:
                    # consumed edge: a real wire into the function, drawn
                    # with a (solid) application arrowhead
                    tok = sub.pop(e)
                    port = fresh("w")
                    fn.ports.append(port)
                    uf.union(tok, port)
                    g.arrows[port] = "solid"
                    g.arrow_heads[port] = fn.id
                if elementwise:
                    anchor: Optional[int] = (
                        first_atom if first_atom < len(g.atoms) else None
                    )
                    if anchor is None and sub:
                        # the argument produced no atom (a bare identity /
                        # pass-through wire, e.g. exp(I)): materialize a copy-dot
                        # anchor carrying its edges, so the function isn't left
                        # orphaned and disconnected from its data
                        anchor = len(g.atoms)
                        anchor_toks: list[str] = []
                        for e in list(sub):
                            atok = fresh("w")
                            uf.union(sub[e], atok)
                            anchor_toks.append(atok)
                        g.atoms.append(AtomSpec(anchor, "copydot", "", anchor_toks))
                    if anchor is not None:
                        # synthesize the dotted application arrow from the
                        # input's principal atom into the function
                        ta, tb = fresh("w"), fresh("w")
                        fn.ports.append(ta)
                        g.atoms[anchor].ports.append(tb)
                        uf.union(ta, tb)
                        g.arrows[ta] = "dotted"
                        g.arrow_heads[ta] = fn.id
                fout.update(sub)  # broadcast edges pass through by name
            for e in t.shape_out:
                port = fresh("w")
                fn.ports.append(port)
                fout[e] = port
            return fout
        if isinstance(t, Derivative):
            first_atom = len(g.atoms)
            sub = walk(t.tensor)
            enclosed = list(range(first_atom, len(g.atoms)))
            new_edges = [e for e in t.edges if e not in t.tensor.edges]
            dout = dict(sub)
            names = []
            for e in new_edges:
                tok = fresh("w")
                g.deriv_tokens.add(tok)
                dout[e] = tok
                names.append(e)
            g.derivs.append((enclosed, names))
            return dout
        if _Expectation is not None and isinstance(t, _Expectation):
            first_atom = len(g.atoms)
            sub = walk(t.tensor)
            enclosed = list(range(first_atom, len(g.atoms)))
            g.boxes.append((enclosed, "E"))
            return sub
        if isinstance(t, Sum):
            # a Sum factor becomes a parenthesized group atom; its inner
            # terms are laid out recursively at layout time
            aid = len(g.atoms)
            toks = [fresh("w") for _ in t.edges]
            atom = AtomSpec(aid, "group", "", toks, sub=t)
            atom.port_names = dict(zip(toks, list(t.edges)))
            g.atoms.append(atom)
            return dict(zip(list(t.edges), toks))
        raise NotImplementedError(f"book_layout does not support {type(t).__name__} yet")

    exposed = walk(tensor)
    for ta, tb in identity_pairs:
        uf.union(ta, tb)
    g.arrows = {uf.find(k): v for k, v in g.arrows.items()}
    g.arrow_heads = {uf.find(k): v for k, v in g.arrow_heads.items()}
    for atom in g.atoms:
        atom.port_names = {uf.find(k): v for k, v in atom.port_names.items()}

    # Build wires from union-find classes.
    endpoints: dict[str, list[int]] = {}
    for atom in g.atoms:
        atom.ports = [uf.find(p) for p in atom.ports]
        for p in atom.ports:
            endpoints.setdefault(p, []).append(atom.id)
    free_names: dict[str, str] = {name: uf.find(tok) for name, tok in exposed.items()}

    for wid, ends in endpoints.items():
        if len(ends) > 2:
            raise ValueError("wire with more than two endpoints (use Delta)")
        g.wires[wid] = ends
    for name, wid in free_names.items():
        if wid in endpoints:
            g.free_wires.setdefault(wid, []).append(name)
        # a wire with no atoms at all: bare identity between two free ends
    # bare identity wires: group free names by wire id with no endpoints
    deriv_reps = {uf.find(tok) for tok in g.deriv_tokens}
    bare: dict[str, list[str]] = {}
    for name, wid in free_names.items():
        if wid not in endpoints and wid not in deriv_reps:
            bare.setdefault(wid, []).append(name)
    for wid, names in bare.items():
        if len(names) == 2:
            g.bare_wires.append((names[0], names[1]))
        else:  # a dangling free edge with no atom: nothing to draw it from
            raise ValueError(f"free edge {names} attached to nothing")
    for tok in g.deriv_tokens:
        if uf.find(tok) in endpoints:
            raise NotImplementedError(
                "a derivative's new edge is contracted onward; simplify() "
                "first (drawing wires from the loop boundary is not "
                "supported yet)"
            )
    # closed rings of pure order-2 identities: no atoms, no free names --
    # invisible to the loops above, but each class is a scalar circle
    ring_classes = set()
    for ta, tb in identity_pairs:
        rep = uf.find(ta)
        if rep not in endpoints and rep not in set(free_names.values()):
            ring_classes.add(rep)
    g.rings = len(ring_classes)
    return g


# ---------------------------------------------------------------------------
# Renderer-neutral layout
# ---------------------------------------------------------------------------


@dataclass
class LNode:
    id: int
    kind: str  # 'var' | 'copydot' | 'scalar' | 'func' | 'group' | 'sign'
    label: str
    x: float
    y: float
    sub: Optional["BookLayout"] = None  # 'group': inner sum layout
    width: float = 0.0  # 'group': inner width incl. parens
    rotated: bool = False  # var drawn upside down = transpose (book style)


@dataclass
class LWire:
    kind: str  # 'segment' | 'arc' | 'loop' | 'stub' | 'pendant' | 'bare'
    a: Optional[int] = None  # node id
    b: Optional[int] = None  # node id (arcs, segments, pendants)
    direction: str = ""  # stubs: 'left' | 'right' | 'up' | 'down'
    label: str = ""  # optional tiny edge label
    arrow: str = ""  # '' | 'solid' | 'dotted': arrowhead into node `b`
    span: int = 0  # arcs: number of spine steps covered
    lane: int = 0  # arcs: nesting level (1 = innermost)
    x: float = 0.0  # bare wires: start x
    y: float = 0.0


@dataclass
class BookLayout:
    nodes: list[LNode] = field(default_factory=list)
    wires: list[LWire] = field(default_factory=list)
    xmin: float = 0.0
    xmax: float = 0.0
    left_edge: Optional[str] = None  # free edge that exits leftmost
    right_edge: Optional[str] = None
    derivs: list[tuple[list[int], list[str]]] = field(default_factory=list)
    boxes: list[tuple[list[int], str]] = field(default_factory=list)
    # exact vertical extent when known (the vertical stacker sets these; the
    # estimator then uses them instead of re-adding margins, which compounded
    # per nesting level into oversized brackets)
    ytop: Optional[float] = None
    ybot: Optional[float] = None


def _label_halfwidth(label: str) -> float:
    # A label may carry a TeX subscript (pow_{-1}): the base word sets most
    # of the width, and the subscript adds a bit more (rendered smaller).
    # Dropping the subscript entirely made pow_{-k} nodes far too narrow, so
    # their application arrows had no room and jammed against the glyph.
    base, sep, sub = label.partition("_{")
    sub = sub.rstrip("}") if sep else ""
    width = CHAR_W * max(0, len(base) - 1)
    width += 0.6 * CHAR_W * len(sub.replace("{", "").replace("}", ""))
    return 0.12 + width / 2


def _adjacency(g: OpenGraph) -> dict[int, dict[int, list[str]]]:
    adj: dict[int, dict[int, list[str]]] = {a.id: {} for a in g.atoms}
    for wid, ends in g.wires.items():
        if len(ends) == 2 and ends[0] != ends[1]:
            a, b = ends
            adj[a].setdefault(b, []).append(wid)
            adj[b].setdefault(a, []).append(wid)
    return adj


def _components(g: OpenGraph) -> list[list[int]]:
    adj = _adjacency(g)
    seen: set[int] = set()
    comps = []
    for a in g.atoms:
        if a.id in seen:
            continue
        stack, comp = [a.id], []
        seen.add(a.id)
        while stack:
            v = stack.pop()
            comp.append(v)
            for u in adj[v]:
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        comps.append(comp)
    return comps


_PATH_CAP = 20000


def _simple_paths(adj: dict[int, dict[int, list[str]]], comp: list[int]) -> list[list[int]]:
    """Simple vertex paths within a component, deduped by orientation and
    capped (dense components would otherwise blow up exponentially)."""
    paths: list[list[int]] = []

    def dfs(path: list[int], seen: set[int]) -> None:
        if len(paths) >= _PATH_CAP:
            return
        if path[0] <= path[-1]:  # each path once; scoring tries both directions
            paths.append(list(path))
        for u in adj[path[-1]]:
            if u in seen or u not in compset:
                continue
            seen.add(u)
            path.append(u)
            dfs(path, seen)
            path.pop()
            seen.remove(u)

    compset = set(comp)
    for v in comp:
        dfs([v], {v})
    return paths


def _atom_free_names(g: OpenGraph, aid: int) -> list[tuple[int, str]]:
    """Free wires on atom `aid` as (port_index, edge_name)."""
    out = []
    atom = g.atoms[aid]
    for idx, wid in enumerate(atom.ports):
        if len(g.wires[wid]) == 1 and wid in g.free_wires:
            for name in g.free_wires[wid]:
                out.append((idx, name))
    return out


def _crossings(chords: list[tuple[int, int]]) -> int:
    n = 0
    for (a1, b1), (a2, b2) in itertools.combinations(chords, 2):
        if a1 < a2 < b1 < b2 or a2 < a1 < b2 < b1:
            n += 1
    return n


def _spine_chords(
    g: OpenGraph, path: list[int], adj: dict[int, dict[int, list[str]]]
) -> list[tuple[int, int, str]]:
    """Wires drawn as arcs, as (i, j, wire_id) with spine indices i <= j."""
    pos = {v: i for i, v in enumerate(path)}
    used_link: set[str] = set()
    # mark one connecting wire per adjacent pair as the spine segment
    for i in range(len(path) - 1):
        wids = adj[path[i]].get(path[i + 1], [])
        if wids:
            used_link.add(wids[0])
    chords = []
    for wid, ends in g.wires.items():
        if wid in used_link or len(ends) != 2:
            continue
        a, b = ends
        if a in pos and b in pos:
            i, j = sorted((pos[a], pos[b]))
            chords.append((i, j, wid))
    return chords


def _choose_spine(
    g: OpenGraph,
    comp: list[int],
    left: Optional[str],
    right: Optional[str],
) -> list[int]:
    adj = _adjacency(g)
    best, best_score = None, None
    for path in _simple_paths(adj, comp):
        for cand in (path, path[::-1]):
            score = 0.0
            head, tail = cand[0], cand[-1]
            head_frees = _atom_free_names(g, head)
            tail_frees = _atom_free_names(g, tail)
            if left is not None and any(n == left for _, n in head_frees):
                score += 1000
            if right is not None and any(n == right for _, n in tail_frees):
                score += 1000
            if head_frees:
                score += 100
                # the head's free edge prefers to be the FIRST declared port
                if any(idx == 0 for idx, _ in head_frees):
                    score += 5
            if tail_frees and tail != head:
                score += 100
                nports = len(g.atoms[tail].ports)
                if any(idx == nports - 1 for idx, _ in tail_frees):
                    score += 5
            elif tail == head and len(head_frees) >= 2:
                score += 100  # one node, two horizontal frees: both ends served
            score += 10 * len(cand)
            chords3 = _spine_chords(g, cand, adj)
            chords = [(i, j) for i, j, _ in chords3]
            score -= 3 * _crossings(chords)
            # prefer arcs that span far (trace closing over everything)
            score += 0.1 * sum(j - i for i, j in chords)
            # a closed component's spine should be part of its cycle: bonus
            # when a leftover wire joins the two spine endpoints (trace dome)
            if any(i == 0 and j == len(cand) - 1 for i, j in chords):
                score += 50
            # read in expression order (A B C D, not A D C B)
            score += 0.5 * sum(
                1 for u, w in zip(cand, cand[1:]) if u < w
            )
            # a function sits LEFT of its argument (its application arrow
            # then points leftward into it, as the book draws it)
            for u, w in zip(cand, cand[1:]):
                for wid in adj[u].get(w, []):
                    if g.arrows.get(wid):
                        head_atom = g.arrow_heads.get(wid)
                        score += 6 if head_atom == u else -6
            # keep each derivative region's atoms CONTIGUOUS on the spine, so
            # its Penrose ellipse wraps only its own atoms instead of
            # stretching across foreign ones and crossing other loops
            pos = {v: k for k, v in enumerate(cand)}
            for enclosed, _ in g.derivs:
                idxs = sorted(pos[a] for a in enclosed if a in pos)
                if len(idxs) >= 2:
                    span = idxs[-1] - idxs[0]
                    score -= 8 * (span - (len(idxs) - 1))
            score -= 0.01 * cand[0]
            if best_score is None or score > best_score:
                best, best_score = list(cand), score
    assert best is not None
    return best


def _assign_lanes(chords: list[tuple[int, int, str]]) -> dict[str, int]:
    """Nest arcs: smaller intervals get lower lanes; conflicts stack up."""
    lanes: dict[str, int] = {}
    placed: list[tuple[int, int, int]] = []  # (i, j, lane)
    for i, j, wid in sorted(chords, key=lambda c: (c[1] - c[0], c[0])):
        lane = 1
        while True:
            ok = True
            for pi, pj, pl in placed:
                if pl != lane:
                    continue
                # same-lane conflict unless strictly disjoint; identical
                # intervals (double traces) always conflict
                if (i, j) == (pi, pj) or not (j <= pi or pj <= i):
                    ok = False
                    break
            # also: an arc strictly containing a placed arc must be above it
            for pi, pj, pl in placed:
                if i <= pi and pj <= j and lane <= pl and (i, j) != (pi, pj):
                    ok = False
                    break
            if ok:
                break
            lane += 1
        lanes[wid] = lane
        placed.append((i, j, lane))
    return lanes


def _layout_component(
    g: OpenGraph,
    comp: list[int],
    layout: BookLayout,
    x0: float,
    left: Optional[str],
    right: Optional[str],
    solo_side: str = "left",
) -> float:
    """Lay out one component starting at x0; return its rightmost extent."""
    spine = _choose_spine(g, comp, left, right)
    pos_index = {v: i for i, v in enumerate(spine)}
    adj = _adjacency(g)

    # -- group atoms: lay out their inner Sum first (width matters) --
    subs: dict[int, "BookLayout"] = {}
    for k, v in enumerate(spine):
        atom = g.atoms[v]
        if atom.kind != "group" or atom.sub is None:
            continue
        le = re = None
        if k > 0:
            wids = adj[spine[k - 1]].get(v, [])
            if wids:
                le = atom.port_names.get(wids[0])
        if k < len(spine) - 1:
            wids = adj[v].get(spine[k + 1], [])
            if wids:
                re = atom.port_names.get(wids[0])
        # The OUTSIDE decides which side each of the group's edges exits;
        # the inner layout follows. Otherwise an edge exits LEFT inside the
        # bracket but RIGHT outside it (reads like a spurious transpose).
        # This mirrors the outer stub logic exactly, including the
        # single-node solo_side case (a lone group component whose one free
        # edge points to solo_side).
        frees_v = [nm for _, nm in _atom_free_names(g, v)]
        if len(spine) == 1 and frees_v:
            if len(frees_v) == 1:
                if solo_side == "right":
                    re = frees_v[0]
                else:
                    le = frees_v[0]
            else:
                le = left if left in frees_v else frees_v[0]
                cand_r = (right if right in frees_v and right != le
                          else frees_v[-1])
                if cand_r != le:
                    re = cand_r
        else:
            if k == 0 and le is None and frees_v:
                le = left if left in frees_v else frees_v[0]
            if k == len(spine) - 1 and re is None and frees_v:
                cand_r = right if right in frees_v else frees_v[-1]
                if cand_r != le:
                    re = cand_r
        subs[v] = layout_any(atom.sub, left=le, right=re)
    # group atoms that land OFF the spine (as pendants) still need their
    # inner layout computed, or emission crashes on `assert n.sub is not None`
    for v in comp:
        atom = g.atoms[v]
        if atom.kind == "group" and atom.sub is not None and v not in subs:
            subs[v] = layout_any(atom.sub)

    def halfwidth(v: int) -> float:
        atom = g.atoms[v]
        if atom.kind == "group":
            return subs[v].xmax / 2 + 0.5  # parens + symmetric padding
        if atom.kind == "copydot":
            return 0.06
        return _label_halfwidth(atom.label)

    # -- build the below-spine pendant forest FIRST, so its subtree widths
    # can widen the spine: two adjacent spine nodes that each carry pendant
    # structure must be far enough apart that their forests don't overlap --
    PEND_GAP = 0.42
    placed = set(spine)
    children: dict[int, list[int]] = {}
    parent_of: dict[int, int] = {}
    parent_wire: dict[int, str] = {}
    _frontier = list(spine)
    while _frontier:
        u = _frontier.pop(0)
        for v in sorted(adj[u]):
            if v in placed:
                continue
            placed.add(v)
            children.setdefault(u, []).append(v)
            parent_of[v] = u
            parent_wire[v] = adj[u][v][0]
            _frontier.append(v)

    def _pw(v: int) -> float:
        atom = g.atoms[v]
        if atom.kind == "group":
            return 2 * halfwidth(v)
        return 0.12 if atom.kind == "copydot" else 2 * _label_halfwidth(atom.label)

    subtree_w: dict[int, float] = {}

    def subtree_width(v: int) -> float:
        if v in subtree_w:
            return subtree_w[v]
        kids = children.get(v, [])
        w = _pw(v)
        if kids:
            kids_w = sum(subtree_width(k) for k in kids) + PEND_GAP * (len(kids) - 1)
            w = max(w, kids_w)
        subtree_w[v] = w
        return w

    def forest_half(u: int) -> float:
        kids = children.get(u, [])
        if not kids:
            return 0.0
        total = sum(subtree_width(k) for k in kids) + PEND_GAP * (len(kids) - 1)
        return total / 2

    # A single-node component with ONE free edge points it to solo_side only,
    # so it reserves stub space on just that side (an outer product's two
    # vectors were each reserving both sides -> a big empty gap between them).
    single = len(spine) == 1
    n_head_frees = len(_atom_free_names(g, spine[0]))
    reserve_left = bool(n_head_frees) and (
        not single or n_head_frees > 1 or solo_side == "left"
    )
    reserve_right = bool(_atom_free_names(g, spine[-1])) and (
        not single or n_head_frees > 1 or solo_side == "right"
    )

    # -- x positions along the spine (label-width aware, pendant-aware) --
    xs: list[float] = []
    x = x0
    if reserve_left:
        x += STUB + 0.05
    x += max(0.0, halfwidth(spine[0]) - 0.12)  # wide heads (groups) start inside
    prev_half = 0.0
    for k, v in enumerate(spine):
        half = halfwidth(v)
        if k > 0:
            # an application arrow between two glyphs needs a wider clear
            # zone than a plain wire: room for the arrowhead AND its dots
            wids = adj[spine[k - 1]].get(v, [])
            clearance = 0.58 if any(g.arrows.get(w) for w in wids) else 0.38
            gap = max(PITCH, prev_half + half + clearance)
            # widen so neighbouring pendant forests don't interleave
            gap = max(gap, forest_half(spine[k - 1]) + forest_half(v) + 0.3)
            x += gap
        xs.append(x)
        prev_half = half
    for v, xv in zip(spine, xs):
        atom = g.atoms[v]
        layout.nodes.append(
            LNode(v, atom.kind, atom.label, xv, 0.0,
                  sub=subs.get(v), width=2 * halfwidth(v))
        )

    drawn: list[str] = []  # every wid rendered; audited for conservation below

    def _group_edge_name(wid: str, *atoms_: int) -> str:
        # a wire into a group is otherwise unidentifiable (the group is an
        # opaque box with several ports) -- recover its edge name so the
        # emission can label it
        for aid in atoms_:
            atom_ = g.atoms[aid]
            if atom_.kind == "group":
                nm = atom_.port_names.get(wid)
                if nm:
                    return nm
        return ""

    # -- spine segments (application arrows point into their function) --
    for k in range(len(spine) - 1):
        wids = adj[spine[k]].get(spine[k + 1], [])
        if wids:
            wid = wids[0]
            drawn.append(wid)
            arrow = g.arrows.get(wid, "")
            a, b = spine[k], spine[k + 1]
            if arrow and g.arrow_heads.get(wid) == a:
                a, b = b, a
            layout.wires.append(
                LWire("segment", a, b, arrow=arrow,
                      label=_group_edge_name(wid, a, b)))

    # -- arcs (keep application-arrow decorations; tip at the function) --
    chords = _spine_chords(g, spine, adj)
    lanes = _assign_lanes(chords)
    drawn.extend(lanes)
    for i, j, wid in chords:
        arrow = g.arrows.get(wid, "")
        tip = ""
        if arrow:
            tip = "->" if g.arrow_heads.get(wid) == spine[j] else "<-"
        if i == j:
            layout.wires.append(LWire("loop", spine[i], spine[i], lane=lanes[wid]))
        else:
            layout.wires.append(
                LWire("arc", spine[i], spine[j], direction=tip, arrow=arrow,
                      span=j - i, lane=lanes[wid],
                      label=_group_edge_name(wid, spine[i], spine[j]))
            )

    # -- pendants (off-spine structure hangs below its attachment) --
    # The forest (children/parent_of/subtree_width) was built above so the
    # spine could reserve room for it; now lay each subtree out recursively:
    # a subtree reserves its full width so siblings never overlap, and every
    # parent is centered over its children.
    node_x = {n.id: n.x for n in layout.nodes}

    def place_subtree(v: int, cx: float, y: float) -> None:
        atom = g.atoms[v]
        layout.nodes.append(
            LNode(v, atom.kind, atom.label, cx, y,
                  sub=subs.get(v),
                  width=2 * halfwidth(v) if atom.kind == "group" else 0.0)
        )
        wid = parent_wire[v]
        drawn.append(wid)
        arrow = g.arrows.get(wid, "")
        pa, pb = parent_of[v], v
        if arrow and g.arrow_heads.get(wid) == pa:
            pa, pb = pb, pa
        layout.wires.append(
            LWire("pendant", pa, pb, arrow=arrow,
                  label=_group_edge_name(wid, pa, pb)))
        node_x[v] = cx
        kids = children.get(v, [])
        if not kids:
            return
        total = sum(subtree_width(k) for k in kids) + PEND_GAP * (len(kids) - 1)
        x = cx - total / 2
        for k in kids:
            w = subtree_width(k)
            place_subtree(k, x + w / 2, y - PENDANT_DY)
            x += w + PEND_GAP

    for u in spine:
        kids = children.get(u, [])
        if not kids:
            continue
        ux = node_x[u]
        if g.atoms[u].kind == "group":
            # a node contracted with a group sits just OUTSIDE its bracket
            # with a short labeled wire to the paren -- not under the group's
            # centre (which drew a long wire crossing the group's content)
            x = ux + halfwidth(u) + 0.4
            for k in kids:
                w = subtree_width(k)
                place_subtree(k, x + w / 2, -0.75)
                x += w + PEND_GAP
            continue
        total = sum(subtree_width(k) for k in kids) + PEND_GAP * (len(kids) - 1)
        x = ux - total / 2
        for k in kids:
            w = subtree_width(k)
            place_subtree(k, x + w / 2, -PENDANT_DY)
            x += w + PEND_GAP

    # -- wire conservation: anything not yet drawn becomes an extra
    # connector (below the spine) or a loop -- NEVER silently dropped --
    from collections import Counter

    want = Counter(
        wid for wid, ends in g.wires.items()
        if len(ends) == 2 and ends[0] in placed and ends[1] in placed
        for _ in range(g.atoms[ends[0]].ports.count(wid) // 2
                       if ends[0] == ends[1] else 1)
    )
    have = Counter(drawn)
    for wid, cnt in want.items():
        missing = cnt - have.get(wid, 0)
        for extra_i in range(missing):
            a, b = g.wires[wid]
            arrow = g.arrows.get(wid, "")
            tip = ""
            if arrow:
                tip = "->" if g.arrow_heads.get(wid) == b else "<-"
            if a == b:
                layout.wires.append(LWire("loop", a, a, lane=2 + extra_i))
            else:
                layout.wires.append(
                    LWire("extra", a, b, direction=tip, arrow=arrow,
                          label=_group_edge_name(wid, a, b))
                )

    # -- free-edge stubs --
    stub_ports: dict[int, dict[str, str]] = {}
    for k, v in enumerate(spine):
        frees = _atom_free_names(g, v)
        if not frees:
            continue
        frees.sort()  # port order
        assigned: list[tuple[str, str]] = []  # (direction, name)
        if k == 0 and k == len(spine) - 1:  # single-node spine
            if len(frees) == 1:
                assigned.append((solo_side, frees[0][1]))
            else:
                names = [nm for _, nm in frees]
                lpick = names.index(left) if left in names else 0
                rest = [m for m in range(len(names)) if m != lpick]
                rpick = (names.index(right)
                         if right in names and names.index(right) != lpick
                         else rest[-1])
                assigned.append(("left", names[lpick]))
                assigned.append(("right", names[rpick]))
                for m, nm in enumerate(names):
                    if m not in (lpick, rpick):
                        assigned.append(("up", nm))
        elif k == 0:
            pick = 0
            if left is not None:
                for m, (_, nm) in enumerate(frees):
                    if nm == left:
                        pick = m
            assigned.append(("left", frees[pick][1]))
            for m, (_, name) in enumerate(frees):
                if m != pick:
                    assigned.append(("up", name))
        elif k == len(spine) - 1:
            pick = len(frees) - 1
            if right is not None:
                for m, (_, nm) in enumerate(frees):
                    if nm == right:
                        pick = m
            assigned.append(("right", frees[pick][1]))
            for m, (_, name) in enumerate(frees):
                if m != pick:
                    assigned.append(("up", name))
        else:
            for _, name in frees:
                assigned.append(("up", name))
        ups = [nm for d, nm in assigned if d == "up"]
        ui = 0
        atom_v = g.atoms[v]
        inner_of = {}
        if atom_v.kind == "group":
            # a group port whose inner name differs from the outer free name
            # is a RENAMING edge: label it with both, inner::outer
            for idx_, nm_ in _atom_free_names(g, v):
                inner = atom_v.port_names.get(atom_v.ports[idx_])
                if inner and inner != nm_:
                    inner_of[nm_] = inner
        for direction, name in assigned:
            # store the edge name on every stub; left/right labels are only
            # DRAWN when edge_labels is on (up/down always draw)
            off = 0.0
            if direction == "up" and len(ups) > 1:
                off = (ui - (len(ups) - 1) / 2) * 0.26
                ui += 1
            lbl = f"{inner_of[name]}::{name}" if name in inner_of else name
            layout.wires.append(
                LWire("stub", v, direction=direction, label=lbl, x=off))
            if direction == "left" and layout.left_edge is None:
                layout.left_edge = name
            if direction == "right":
                layout.right_edge = name
            stub_ports.setdefault(v, {})[direction] = name

    # -- transpose detection: a 2-port var whose left-facing port comes
    # AFTER its right-facing port in declared order is drawn rotated 180
    # degrees (the book's transpose convention) --
    name_to_wid = {}
    for v in comp:
        for wid in g.atoms[v].ports:
            for nm in g.free_wires.get(wid, []):
                name_to_wid[nm] = wid
    for k, v in enumerate(spine):
        atom = g.atoms[v]
        real_ports = [p for p in atom.ports if p not in g.arrows]
        if atom.kind != "var" or len(real_ports) != 2:
            continue
        left_wid = right_wid = None
        if k > 0:
            wids = adj[spine[k - 1]].get(v, [])
            left_wid = wids[0] if wids else None
        elif stub_ports.get(v, {}).get("left") is not None:
            left_wid = name_to_wid.get(stub_ports[v]["left"])
        if k < len(spine) - 1:
            wids = adj[v].get(spine[k + 1], [])
            right_wid = wids[0] if wids else None
        elif stub_ports.get(v, {}).get("right") is not None:
            right_wid = name_to_wid.get(stub_ports[v]["right"])
        if left_wid is None or right_wid is None or left_wid == right_wid:
            continue
        try:
            li = real_ports.index(left_wid)
            ri = real_ports.index(right_wid)
        except ValueError:
            continue
        # Rotation marks a transpose, which is a boundary/covariance concept:
        # only meaningful when at least one port is a FREE edge whose side we
        # are tracking (e.g. A^T in a gradient term). A matrix contracted on
        # both sides has an arbitrary internal orientation -- rotating it just
        # produces a confusing upside-down glyph (a chain's internal K read as
        # a backwards R), so leave those upright.
        has_free = any(
            len(g.wires.get(p, [])) == 1 for p in real_ports
        )
        if li > ri and has_free:
            node = next(n for n in layout.nodes if n.id == v)
            node.rotated = True

    # -- double-edge transpose: two adjacent 2-port matrices that share BOTH
    # their edges are either Tr(AB) (ports pair CROSSED: A's inner index meets
    # B's inner index) or Tr(AB^T) / ||A||_F (ports pair PARALLEL: same index
    # on both). A parallel pairing must rotate the right matrix, else the two
    # genuinely different tensors would draw byte-identically (a segment + arc
    # with no crossing) -- the semantic-fidelity bug. --
    rotated_ids = {n.id for n in layout.nodes if n.rotated}
    for k in range(len(spine) - 1):
        u, v = spine[k], spine[k + 1]
        shared = adj[u].get(v, [])
        au, av = g.atoms[u], g.atoms[v]
        if (len(shared) != 2 or au.kind != "var" or av.kind != "var"
                or len(au.ports) != 2 or len(av.ports) != 2):
            continue
        try:
            parallel = all(au.ports.index(w) == av.ports.index(w) for w in shared)
        except ValueError:
            continue
        if parallel and v not in rotated_ids:
            node = next(n for n in layout.nodes if n.id == v)
            node.rotated = True
            rotated_ids.add(v)

    # free edges on pendants leave downward
    for v in comp:
        if v in pos_index:
            continue
        for idx, name in _atom_free_names(g, v):
            layout.wires.append(LWire("stub", v, direction="down", label=name))

    xmax = xs[-1] + halfwidth(spine[-1])
    if reserve_right:
        xmax += STUB + 0.05
    return xmax


def layout_tensor(
    tensor: Tensor, left: Optional[str] = None, right: Optional[str] = None
) -> BookLayout:
    """Renderer-neutral book-grammar layout of a single (non-Sum) tensor."""
    g = extract_graph(tensor)
    layout = BookLayout()
    x = 0.0
    # bare identity wires and identity rings first
    for _ in g.bare_wires:
        layout.wires.append(LWire("bare", x=x, y=0.0))
        x += 1.0 + COMPONENT_GAP
    for _ in range(g.rings):
        layout.wires.append(LWire("ring", x=x + 0.28, y=0.0))
        x += 0.56 + COMPONENT_GAP
    comps = _components(g)

    def has_frees(comp: list[int]) -> bool:
        return any(_atom_free_names(g, v) for v in comp)

    def comp_order(comp: list[int]) -> tuple:
        if not has_frees(comp):
            return (0, 0)  # closed scalar factors first
        names = {nm for v in comp for _, nm in _atom_free_names(g, v)}
        if left is not None and left in names:
            return (1, 0)
        if right is not None and right in names:
            return (1, 2)
        return (1, 1)

    comps.sort(key=comp_order)
    open_comps = [c for c in comps if has_frees(c)]
    for comp in comps:
        is_open = has_frees(comp)
        l = left if is_open else None
        r = right if is_open else None
        # first open component points its lone stub left, the last right
        solo = "left"
        if is_open and open_comps and comp is open_comps[-1] and len(open_comps) > 1:
            solo = "right"
        x_end = _layout_component(g, comp, layout, x, l, r, solo_side=solo)
        x = x_end + COMPONENT_GAP
    layout.xmax = x - COMPONENT_GAP
    layout.derivs = list(g.derivs)
    layout.boxes = list(g.boxes)
    _separate_deriv_loops(layout)
    _reserve_loop_margins(layout)
    return layout


def _reserve_loop_margins(layout: "BookLayout") -> None:
    """Widen the layout's footprint by the horizontal reach of its Penrose
    ellipses (and E-boxes): a fit-ellipse extends well past its atoms, so a
    sum places terms by xmax and their ellipses would otherwise overlap the
    neighbouring term (and swallow the +/- signs)."""
    if not layout.derivs and not layout.boxes:
        return
    nb = {n.id: n for n in layout.nodes}
    lo, hi = 0.0, layout.xmax
    regions = [(enc, 0.5 + 0.28 * sum(
        1 for e2, _ in layout.derivs if set(e2) < set(enc)))
        for enc, _ in layout.derivs]
    regions += [(enc, 0.35) for enc, _ in layout.boxes]
    for enc, m in regions:
        xs = [nb[a].x for a in enc if a in nb]
        if not xs:
            continue
        lo = min(lo, min(xs) - m - 0.1)
        hi = max(hi, max(xs) + m + 0.1)
    if lo < 0:  # shift everything right so the leftmost ellipse starts at 0
        for n in layout.nodes:
            n.x -= lo
        for w in layout.wires:
            if w.kind in ("bare", "ring"):
                w.x -= lo
        hi -= lo
    layout.xmax = max(layout.xmax, hi)


# a top-level sum wider than this (cm) is stacked vertically (one term per
# row) instead of laid out horizontally, so its terms stay full-size and
# legible instead of shrinking to an illegible thread
STACK_WIDTH = 13.0


def _separate_deriv_loops(layout: "BookLayout") -> None:
    """Push apart derivative-loop groups whose fit-ellipses would intersect.

    A Penrose ellipse extends well beyond the atoms it wraps, so two loops
    around vertically-adjacent atom groups (e.g. a spine loop and the loop
    around the pendant hanging below it, as in a Frobenius-of-2nd-derivative)
    cross even though their atoms don't. This respects each loop's bounding
    box: concentric loops (same atom set) are left to the emission's nesting
    padding, but distinct groups that overlap in x are separated vertically by
    moving the lower group (and everything below it in that column) down.
    """
    if len(layout.derivs) < 2:
        return
    nb = {n.id: n for n in layout.nodes}

    # concentric loops share an atom set -> one "group"; depth = loops nested
    # strictly inside it (each adds ellipse padding, hence a bigger box)
    depth: dict[frozenset, int] = {}
    for enc, _ in layout.derivs:
        key = frozenset(a for a in enc if a in nb)
        if key:
            depth[key] = sum(
                1 for e2, _ in layout.derivs if frozenset(e2) < key
            )
    keys = list(depth)

    def ybox(key: frozenset) -> tuple[float, float, float, float]:
        xs = [nb[a].x for a in key]
        ys = [nb[a].y for a in key]
        m = 0.5 + 0.28 * depth[key]  # ellipse reaches ~this far past the atoms
        return min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m

    for _ in range(30):
        moved = False
        for a in keys:
            for c in keys:
                if a is c or a <= c or c <= a:
                    continue  # same or nested (concentric): emission handles it
                ax0, ax1, ay0, ay1 = ybox(a)
                cx0, cx1, cy0, cy1 = ybox(c)
                if ax1 <= cx0 or cx1 <= ax0:
                    continue  # no horizontal overlap -> ellipses miss anyway
                if ay1 <= cy0 or cy1 <= ay0:
                    continue  # already vertically clear
                # push the lower group (and its column below) further down
                low, hi = (c, a) if _cy(c, nb) < _cy(a, nb) else (a, c)
                lb = ybox(low)
                hb = ybox(hi)
                # move `low` down until its box TOP drops below `hi`'s box
                # BOTTOM, plus a small gap
                delta = (lb[3] - hb[2]) + 0.2
                if delta <= 0:
                    continue
                x0, x1 = lb[0], lb[1]
                lowmin = min(nb[i].y for i in low)
                for n in layout.nodes:
                    if n.id in low or (n.y <= lowmin and x0 <= n.x <= x1):
                        n.y -= delta
                moved = True
        if not moved:
            break


def _cy(key: frozenset, nb: dict) -> float:
    ys = [nb[a].y for a in key]
    return (min(ys) + max(ys)) / 2


def _term_extent(sub: "BookLayout") -> tuple[float, float]:
    """Approximate (top, bottom) y-extent of a laid-out term, including the
    room its arcs/loops rise above the spine, pendants drop below it, and any
    nested group's own inner content extends (recursively)."""
    if sub.ytop is not None and sub.ybot is not None:
        return sub.ytop, sub.ybot
    tops = [0.0]
    bots = [0.0]
    for nd in sub.nodes:
        tops.append(nd.y)
        bots.append(nd.y)
        if nd.kind == "group" and nd.sub is not None:
            # emission CENTRES the group's inner content on nd.y, so it
            # occupies nd.y +- half its height (not nd.y+gb..nd.y+gt)
            gt, gb = _term_extent(nd.sub)
            half = (gt - gb) / 2
            tops.append(nd.y + half)
            bots.append(nd.y - half)
    top, bot = max(tops), min(bots)
    if sub.derivs:
        # a Penrose ellipse extends well above AND below the nodes it wraps,
        # and nested ones grow with depth (capped -- a pathologically nested
        # term like a Frobenius-of-2nd-derivative can't be spaced clean anyway)
        grow = 0.55 + 0.3 * min(len(sub.derivs), 3)
        top += grow
        bot -= grow
    elif any(w.kind in ("arc", "loop") for w in sub.wires):
        top += 0.75
    else:
        top += 0.2
    if any(w.kind == "stub" and w.direction == "up" for w in sub.wires):
        top += 0.35
    bot -= 0.2
    if any(w.kind == "stub" and w.direction == "down" for w in sub.wires):
        bot -= 0.35
    return top, bot


def layout_any(
    tensor: Tensor, left: Optional[str] = None, right: Optional[str] = None,
    stack: Optional[bool] = None,
) -> BookLayout:
    """Layout for any supported tensor, including a top-level Sum.

    A wide top-level Sum is stacked vertically (one term per row) so its
    terms stay legible; `stack` forces this on/off (None = auto by width).
    """
    if not isinstance(tensor, Sum):
        return layout_tensor(tensor, left, right)
    pairs = _flatten_sum(tensor)
    horizontal = _layout_sum_horizontal(pairs, left, right)
    if stack is False or (stack is None and horizontal.xmax <= STACK_WIDTH):
        return horizontal
    return _layout_sum_vertical(pairs, left, right)


def _flatten_sum(tensor: Sum) -> list[tuple[Tensor, object]]:
    """Flatten nested Sums into (term, weight) pairs, multiplying weights
    through. Addition is associative, so ((a + 2b) + c) -- the shape that
    repeated '+=' builds -- needs no brackets: it IS a + 2b + c. Brackets
    remain only where they mean something (a Sum inside a Product)."""
    pairs: list[tuple[Tensor, object]] = []

    def rec(t: Tensor, w: object) -> None:
        if isinstance(t, Sum):
            for tt, ww in zip(t.terms, t.weights):
                rec(tt, w * ww if (w != 1 or ww != 1) else 1)  # type: ignore[operator]
        else:
            pairs.append((t, w))

    rec(tensor, 1)
    return pairs


def _layout_sum_vertical(
    pairs: list[tuple[Tensor, object]], left: Optional[str], right: Optional[str]
) -> BookLayout:
    """Stack a sum's terms vertically: term k on its own row, the operator in
    a left column, rows spaced by each term's true vertical extent."""
    out = BookLayout()
    sign_id = -1
    offset = 0
    ROW_GAP = 0.45
    SIGN_COL = 0.0     # operators sit here
    TERM_COL = 0.75    # terms start here (room for the sign)
    cursor = 0.0       # y of the top of the next row
    max_x = 0.0
    for idx, (term, weight) in enumerate(pairs):
        sub = layout_tensor(term, left, right)
        if left is None and sub.left_edge is not None:
            left = sub.left_edge
        if right is None and sub.right_edge is not None:
            right = sub.right_edge
        top, bot = _term_extent(sub)
        row_y = cursor - top
        sign, coeff = _fmt_weight(weight)
        if idx > 0 or sign == "-" or coeff:
            sgn = sign if (idx > 0 or sign == "-") else ""
            tex = f"{sgn}{coeff}".strip()
            if tex:
                out.nodes.append(LNode(sign_id, "sign", tex, SIGN_COL, row_y))
                sign_id -= 1
        for nd in sub.nodes:
            out.nodes.append(
                LNode(nd.id + offset, nd.kind, nd.label,
                      nd.x + TERM_COL, nd.y + row_y,
                      sub=nd.sub, width=nd.width, rotated=nd.rotated)
            )
        for w in sub.wires:
            wx = w.x + TERM_COL if w.kind in ("bare", "ring") else w.x
            out.wires.append(
                LWire(w.kind,
                      None if w.a is None else w.a + offset,
                      None if w.b is None else w.b + offset,
                      w.direction, w.label, w.arrow, w.span, w.lane,
                      wx, w.y + (row_y if w.kind in ("bare", "ring") else 0.0))
            )
        for enclosed, names in sub.derivs:
            out.derivs.append(([a + offset for a in enclosed], names))
        for enclosed, lbl in sub.boxes:
            out.boxes.append(([a + offset for a in enclosed], lbl))
        max_local = max((nd.id for nd in sub.nodes), default=-1)
        offset += max_local + 1
        max_x = max(max_x, TERM_COL + sub.xmax)
        cursor = row_y + bot - ROW_GAP
    out.xmax = max_x
    out.ytop = 0.15
    out.ybot = cursor + ROW_GAP - 0.15
    return out


def _layout_sum_horizontal(
    pairs: list[tuple[Tensor, object]], left: Optional[str], right: Optional[str]
) -> BookLayout:
    """Lay a sum's terms side by side on one line."""
    out = BookLayout()
    x = 0.0
    sign_id = -1  # negative ids never collide with term/atom ids
    # each term's node ids are shifted by a running offset past the previous
    # term's largest id -- collision-proof for any term size (a fixed
    # multiplier would cap the atoms-per-term it can separate)
    offset = 0
    for idx, (term, weight) in enumerate(pairs):
        sign, coeff = _fmt_weight(weight)
        show_sign = idx > 0 or sign == "-" or coeff
        if show_sign:
            sgn = sign if (idx > 0 or sign == "-") else ""
            tex = f"{sgn}{coeff}".strip()
            if tex:
                pad = 0.58 + 0.1 * len(tex)  # isolate the operator with space
                out.nodes.append(LNode(sign_id, "sign", tex, x + pad - TERM_GAP, 0.0))
                sign_id -= 1
                x += 2 * pad - TERM_GAP
        sub = layout_tensor(term, left, right)
        if left is None and sub.left_edge is not None:
            left = sub.left_edge
        if right is None and sub.right_edge is not None:
            right = sub.right_edge
        for nd in sub.nodes:
            out.nodes.append(
                LNode(nd.id + offset, nd.kind, nd.label, nd.x + x, nd.y,
                      sub=nd.sub, width=nd.width, rotated=nd.rotated)
            )
        for w in sub.wires:
            wx = w.x + x if w.kind in ("bare", "ring") else w.x
            out.wires.append(
                LWire(w.kind,
                      None if w.a is None else w.a + offset,
                      None if w.b is None else w.b + offset,
                      w.direction, w.label, w.arrow, w.span, w.lane,
                      wx, w.y)
            )
        for enclosed, names in sub.derivs:
            out.derivs.append(([a + offset for a in enclosed], names))
        for enclosed, lbl in sub.boxes:
            out.boxes.append(([a + offset for a in enclosed], lbl))
        max_local = max((nd.id for nd in sub.nodes), default=-1)
        offset += max_local + 1
        x += sub.xmax + TERM_GAP
    out.xmax = x - TERM_GAP
    return out


# ---------------------------------------------------------------------------
# TikZ emission (the book's vocabulary)
# ---------------------------------------------------------------------------


def _tex_label(label: str, kind: str = "var") -> str:
    # variables render in math italic (cursive), functions in roman -- so
    # 'target' (a tensor) is visually distinct from 'exp' (a function)
    if kind == "var":
        return rf"$\mathit{{{label}}}$" if len(label) > 1 else f"${label}$"
    if len(label) > 2:
        return rf"$\mathrm{{{label}}}$"
    return f"${label}$"


def _fmt_weight(w: object) -> tuple[str, str]:
    """Return (sign, coefficient-tex) for a sum weight."""
    if isinstance(w, Number):
        # numbers.Number has no static __float__ in typeshed, but every
        # weight tensorgrad produces (int / sympy Integer|Rational|Float) does
        wf = float(w)  # type: ignore[arg-type]
        if wf == 1:
            return "+", ""
        if wf == -1:
            return "-", ""
        num: float | int = int(wf) if wf == int(wf) else wf
        return ("-", f"{abs(num)}\\,") if num < 0 else ("+", f"{num}\\,")
    return "+", f"{w}\\,"


# uniform text metrics so subscripted glyphs align on a common axis
_AXIS = "inner sep=1pt, text height=1.55ex, text depth=0.35ex"


def _emit_layout(layout: BookLayout, lines: list[str], prefix: str, dx: float,
                 edge_labels: bool = False, dy: float = 0.0) -> None:
    name = {n.id: f"{prefix}n{n.id}" for n in layout.nodes}
    nodes = {n.id: n for n in layout.nodes}
    group_side: dict[int, tuple[str, str]] = {}
    for n in layout.nodes:
        x = n.x + dx
        y = n.y + dy
        if n.kind == "copydot":
            lines.append(rf"\node[copydot] ({name[n.id]}) at ({x:.2f},{y:.2f}) {{}};")
        elif n.kind == "group":
            # parenthesized inner sum: two paren nodes + recursive emission.
            # Parens are sized to the inner content's true height (so a tall
            # or vertically-stacked inner sum gets big brackets). The inner
            # content is CENTRED on the group's spine line (dy shift), so
            # external wires attach at the paren horizontally -- no diagonal
            # into a tall stacked group.
            hw = n.width / 2
            assert n.sub is not None
            _gt, _gb = _term_extent(n.sub)
            sub_dy = y - (_gt + _gb) / 2  # sub midline -> group spine level
            hex_ = max(1.6, (_gt - _gb + 0.2) * 6.3)  # bracket height in ex
            pl, pr = f"{name[n.id]}L", f"{name[n.id]}R"
            lines.append(
                rf"\node[inner sep=0.5pt] ({pl}) at ({x - hw + 0.12:.2f},{y:.2f})"
                rf" {{$\left(\vphantom{{\rule{{0pt}}{{{hex_:.1f}ex}}}}\right.$}};"
            )
            lines.append(
                rf"\node[inner sep=0.5pt] ({pr}) at ({x + hw - 0.12:.2f},{y:.2f})"
                rf" {{$\left.\vphantom{{\rule{{0pt}}{{{hex_:.1f}ex}}}}\right)$}};"
            )
            group_side[n.id] = (pl, pr)
            # extra breathing room after the '(' before the first term (which
            # may be a wide node like pow_{-1}, else the paren jams into it)
            _emit_layout(n.sub, lines, prefix=f"{name[n.id]}i", dx=x - hw + 0.48,
                         edge_labels=edge_labels, dy=sub_dy)
        elif n.kind == "sign":
            # a sum operator is enlarged, spaced out (term-gap widened at
            # layout time) AND bold so it reads clearly as an operator, not as
            # another short horizontal wire -- a free-edge stub and a thin
            # minus are otherwise the same stroke at the same height. Only bare
            # +/- are bolded; numeric coefficients (2, -3) don't look like
            # wires and stay in the normal weight.
            body = (rf"\boldsymbol{{{n.label}}}" if n.label in ("+", "-")
                    else n.label)
            lines.append(
                rf"\node[{_AXIS}, scale=1.4] ({name[n.id]}) at"
                rf" ({x:.2f},{y:.2f}) {{${body}$}};"
            )
        else:
            # a transpose is shown with a ^T superscript (clearer than an
            # upside-down glyph). _AXIS pins a uniform text height/depth so a
            # subscripted label centers on the same axis as its neighbours.
            label = _tex_label(n.label, n.kind)
            if n.rotated:  # insert the transpose superscript inside the $...$
                label = label[:-1] + r"^{\top}$"
            lines.append(
                rf"\node[{_AXIS}] ({name[n.id]}) at ({x:.2f},{y:.2f})"
                rf" {{{label}}};"
            )

    def endpoint(nid: Optional[int], other: Optional[int]) -> str:
        # segments touching a group attach to the NEARER paren (by actual
        # paren position -- comparing against the group centre sent a wire
        # from a node under the middle all the way to the far paren)
        assert nid is not None  # node-to-node wires always have both ends
        if nid in group_side and other is not None:
            pl, pr = group_side[nid]
            hw = nodes[nid].width / 2
            lx, rx = nodes[nid].x - hw, nodes[nid].x + hw
            ox = nodes[other].x
            return pl if abs(ox - lx) <= abs(ox - rx) else pr
        return name[nid]

    # application arrows keep clearance at both ends: the head stops short of
    # the function glyph (>= side; extra room so it clears a subscript like
    # pow_{-2}), the tail leaves the argument glyph (<= side).
    _clear = "shorten >=2pt, shorten <=1pt"
    style = {
        "": "",
        "solid": f"[->, {_clear}]",
        "dotted": f"[densely dotted, ->, {_clear}]",
    }
    def _wire_label(w: LWire) -> str:
        if not w.label:
            return ""
        return (rf" node[midway, above, font=\scriptsize, inner sep=1.5pt]"
                rf" {{${_tex_edge(w.label)}$}}")

    for w in layout.wires:
        if w.kind == "segment" or w.kind == "pendant":
            lines.append(
                rf"\draw{style[w.arrow].strip()} ({endpoint(w.a, w.b)})"
                rf" -- ({endpoint(w.b, w.a)}){_wire_label(w)};"
            )
        elif w.kind == "loop":
            assert w.a is not None
            if w.a in group_side:
                pl, pr = group_side[w.a]
                lines.append(
                    rf"\path ({pl}) edge [out=160, in=20, looseness=1.6] ({pr});"
                )
            else:
                dist = f", min distance={4 + 5 * (w.lane - 1)}mm, looseness=5.5"
                lines.append(
                    rf"\path ({name[w.a]}) edge [out=160, in=20, loop{dist}]"
                    rf" ({name[w.a]});"
                )
        elif w.kind == "arc":
            assert w.a is not None and w.b is not None
            tip = f", {w.direction}" if w.direction else ""
            dot = "densely dotted, " if w.arrow == "dotted" else ""
            mid = ""
            if w.label:
                mid = (rf" node[pos=0.5, above, font=\scriptsize,"
                       rf" inner sep=1.5pt] {{${_tex_edge(w.label)}$}}")
            if w.a in group_side or w.b in group_side:
                # a closure between groups (e.g. a Frobenius trace) domes from
                # the FAR parens over everything, labeled with its edge
                def _far(nid: int, other: int) -> str:
                    if nid in group_side:
                        pl, pr = group_side[nid]
                        return pl if nodes[other].x >= nodes[nid].x else pr
                    return name[nid]

                fa, fb = _far(w.a, w.b), _far(w.b, w.a)
                span_x = abs(nodes[w.a].x - nodes[w.b].x) + nodes[w.a].width
                h = 0.45 + 0.05 * span_x + 0.25 * (w.lane - 1)
                lines.append(
                    rf"\draw ({fa}.north) .. controls +(0.2,{h:.2f}) and"
                    rf" +(-0.2,{h:.2f}) .. ({fb}.north){mid};"
                )
            elif w.span <= 3:
                na, nb = endpoint(w.a, w.b), endpoint(w.b, w.a)
                loose = 0.55 + 0.47 * w.span + 0.5 * (w.lane - 1)
                lines.append(
                    rf"\path[{dot.rstrip(', ')}] ({na}) edge [out=160, in=20{tip}, "
                    rf"looseness={loose:.2f}]{mid} ({nb});"
                )
            else:
                na, nb = endpoint(w.a, w.b), endpoint(w.b, w.a)
                # long spans: flat north-anchored dome hugging the chain
                h = 0.35 + 0.08 * w.span + 0.28 * (w.lane - 1)
                arrowopt = f"[{dot}{w.direction}] " if (w.direction or dot) else ""
                lines.append(
                    rf"\draw{arrowopt.strip()} ({na}.north west) .. controls"
                    rf" +(-0.15,{h:.2f}) and +(0.15,{h:.2f}) .."
                    rf" ({nb}.north east){mid};"
                )
        elif w.kind == "extra":
            a, b = w.a, w.b
            assert a is not None and b is not None
            direction = w.direction  # '->'=head at b, '<-'=head at a (orig order)
            # start from the lower endpoint; bend so the curve bows DOWN,
            # away from the spine and its arcs
            if nodes[a].y > nodes[b].y or (
                nodes[a].y == nodes[b].y and nodes[a].x > nodes[b].x
            ):
                a, b = b, a
                # the endpoints swapped, so the arrowhead token must flip too,
                # or an application arrow points into the argument not the func
                direction = {"->": "<-", "<-": "->"}.get(direction, direction)
            bend_dir = "right" if nodes[a].x < nodes[b].x else "left"
            # long connectors sag deeper so they route BELOW tall content
            # (e.g. a stacked group) instead of striking through it
            span_x = abs(nodes[a].x - nodes[b].x)
            bend = min(80, int(30 + 6 * span_x))
            na, nb = endpoint(a, b), endpoint(b, a)
            tip = f"{direction}, " if direction else ""
            dot = "densely dotted, " if w.arrow == "dotted" else ""
            lines.append(
                rf"\draw[{tip}{dot}] ({na}) to[bend {bend_dir}={bend}]"
                rf" ({nb}){_wire_label(w)};"
            )
        elif w.kind == "ring":
            lines.append(
                rf"\draw ({w.x + dx:.2f},{w.y + dy:.2f}) circle (0.28);"
            )
        elif w.kind == "stub":
            assert w.a is not None
            d, anch = {
                "left": (f"({-STUB:.2f},0)", "west"),
                "right": (f"({STUB:.2f},0)", "east"),
                "up": (f"(0,{MIDSTUB:.2f})", "north"),
                "down": (f"(0,{-MIDSTUB:.2f})", "south"),
            }[w.direction]
            lab = ""
            # up/down stubs always label their edge; left/right only when
            # edge_labels is on (keeps simple chains like -A-B- uncluttered)
            show = w.label and (w.direction in ("up", "down") or edge_labels)
            if show:
                # a left/right edge label sits just ABOVE the free END of its
                # wire, centred over the tip (like the cookbook) -- above the
                # wire so it has no inline footprint (no cross-component
                # collision), and hugging the end symmetrically for both sides
                anchor = {"up": "south", "down": "north",
                          "left": "south", "right": "south"}[w.direction]
                outer_lbl = w.label
                pre = ""
                if "::" in w.label:
                    # a RENAMING edge (group port renamed): inner name near
                    # the bracket, outer name at the free end -- like
                    # to_tikz's double-labeled edges
                    inner_lbl, outer_lbl = w.label.split("::", 1)
                    pre = (rf" node[pos=0.1, anchor={anchor},"
                           rf" font=\scriptsize, inner sep=1.5pt]"
                           rf" {{${_tex_edge(inner_lbl)}$}}")
                lab = pre + (rf" node[anchor={anchor}, font=\scriptsize,"
                             rf" inner sep=1.5pt] {{${_tex_edge(outer_lbl)}$}}")
            src = name[w.a]
            if w.a in group_side:
                src = group_side[w.a][0 if w.direction == "left" else 1]
            shift = f"[xshift={w.x:.2f}cm]" if w.x else ""
            lines.append(rf"\draw ({shift}{src}.{anch}) -- ++{d}{lab};")
        elif w.kind == "bare":
            lines.append(
                rf"\draw ({w.x + dx:.2f},{w.y + dy:.2f}) -- ++({1.0:.2f},0);"
            )
    _emit_boxes(layout, lines, prefix, name, group_side)
    _emit_derivs(layout, lines, prefix, name, group_side, dx, dy)


def _emit_boxes(layout: BookLayout, lines: list[str], prefix: str,
                name: dict[int, str], group_side: dict[int, tuple[str, str]]) -> None:
    """Expectation E[.] boxes: a fit-rectangle with an E label on the left."""
    for bi, (enclosed, lbl) in enumerate(layout.boxes):
        parts: list[str] = []
        for aid in enclosed:
            if aid in group_side:
                parts.extend(f"({p})" for p in group_side[aid])
            elif aid in name:
                parts.append(f"({name[aid]})")
        if not parts:
            continue
        inside = sum(1 for e2, _ in layout.boxes if set(e2) < set(enclosed))
        sep = 3.5 + 4.0 * inside
        bn = f"{prefix}bE{bi}"
        # an INVISIBLE fit node sizes the brackets; we draw actual [ ] shapes
        # (a vertical edge with short top/bottom serifs) exactly around the
        # content -- no full rectangle (redundant), no fixed-size text bracket
        lines.append(
            rf"\node[inner sep={sep:.1f}pt, fit={{{''.join(parts)}}}] ({bn}) {{}};"
        )
        lines.append(
            rf"\draw ([xshift=3pt]{bn}.north west) -- ({bn}.north west)"
            rf" -- ({bn}.south west) -- ([xshift=3pt]{bn}.south west);"
        )
        lines.append(
            rf"\draw ([xshift=-3pt]{bn}.north east) -- ({bn}.north east)"
            rf" -- ({bn}.south east) -- ([xshift=-3pt]{bn}.south east);"
        )
        lines.append(
            rf"\node[anchor=east, inner sep=3pt] at ({bn}.west) {{$\mathbb{{{lbl}}}$}};"
        )


def _emit_derivs(layout: BookLayout, lines: list[str], prefix: str,
                 name: dict[int, str], group_side: dict[int, tuple[str, str]],
                 dx: float = 0.0, dy: float = 0.0) -> None:
    """Penrose derivative loops: fit-ellipse + boundary dot + whiskers."""
    nodes = {n.id: n for n in layout.nodes}
    for di, (enclosed, new_names) in enumerate(layout.derivs):
        eset = set(enclosed)
        parts: list[str] = []
        for aid in enclosed:
            if aid in group_side:
                parts.extend(f"({p})" for p in group_side[aid])
            elif aid in name:
                parts.append(f"({name[aid]})")
        if not parts:
            continue
        # arcs/loops enclosed by the region must fit inside the ellipse:
        # include their topmost point in the fit spec
        for w in layout.wires:
            if w.kind == "arc" and w.a in eset and w.b in eset:
                na_, nb_ = nodes[w.a], nodes[w.b]
                if na_.kind == "group" or nb_.kind == "group":
                    # mirror the group-dome emission: it rises from the paren
                    # TOPS, so the ellipse must reach above tall brackets too
                    span_x = abs(na_.x - nb_.x) + na_.width
                    h = 0.45 + 0.05 * span_x + 0.25 * (w.lane - 1)
                    base = 0.0
                    for nd_ in (na_, nb_):
                        half = 0.35
                        if nd_.kind == "group" and nd_.sub is not None:
                            gt_, gb_ = _term_extent(nd_.sub)
                            half = (gt_ - gb_) / 2 + 0.15
                        base = max(base, nd_.y + half)
                    my = base + h + 0.15 + dy
                else:
                    h = (0.32 + 0.16 * w.span + 0.24 * (w.lane - 1)
                         if w.span <= 3 else 0.35 + 0.08 * w.span)
                    my = max(na_.y, nb_.y) + h + 0.12 + dy
                mx = (na_.x + nb_.x) / 2 + dx
                parts.append(f"({mx:.2f},{my:.2f})")
            elif w.kind == "loop" and w.a in eset:
                lx = nodes[w.a].x + dx
                ly = nodes[w.a].y + 0.42 + 0.2 * w.lane + dy
                parts.append(f"({lx:.2f},{ly:.2f})")
        # nesting: an ellipse enclosing another loop's region (equal
        # regions nest by creation order: inner derivatives walk first)
        inside = sum(
            1 for dj, (e2, _) in enumerate(layout.derivs)
            if dj != di and set(e2) <= eset and (set(e2) < eset or dj < di)
        )
        sep = 2.5 + 4.0 * inside
        en = f"{prefix}dE{di}"
        lines.append(
            rf"\node[ellipse, draw, inner sep={sep:.1f}pt, "
            rf"fit={{{''.join(parts)}}}] ({en}) {{}};"
        )
        # whisker labels ride BESIDE the wire (midway, offset to the side)
        # -- a label AT the free tip reads as if it were a tensor node
        if len(new_names) == 1:
            ang = 125 + 14 * inside  # fan nested whiskers apart
            lines.append(rf"\fill ({en}.{ang}) circle (1.4pt);")
            lines.append(
                rf"\draw ({en}.{ang}) .. controls +({ang - 25}:.12) .."
                rf" ++(-.24,.26)"
                rf" node[pos=0.75, above right, font=\scriptsize,"
                rf" inner sep=0.5pt] {{${_tex_edge(new_names[0])}$}};"
            )
        else:
            ang = 55
            lines.append(rf"\fill ({en}.{ang}) circle (1.4pt);")
            whisk = [("80:.12", "++(-.2,.28)", "above left"),
                     ("45:.12", "++(.28,.18)", "above right")]
            for nm, (ctrl, end, anch) in zip(new_names, whisk):
                lines.append(
                    rf"\draw ({en}.{ang}) .. controls +({ctrl}) .. {end}"
                    rf" node[pos=0.85, {anch}, font=\scriptsize,"
                    rf" inner sep=0.5pt] {{${_tex_edge(nm)}$}};"
                )


def to_book_tikz(
    tensor: Tensor,
    left: Optional[str] = None,
    right: Optional[str] = None,
    baseline: str = "-.25em",
    scale: Optional[float] = None,
    max_width: Optional[float] = None,
    edge_labels: bool = False,
    stack: Optional[bool] = None,
) -> str:
    """Render a tensorgrad Tensor as book-style TikZ (uses tikz-styles.tex).

    Args:
        left/right: force the named free edge to exit that side (covariance).
        baseline: TikZ baseline anchor for inline use.
        scale: explicit TikZ scale factor for the whole picture.
        max_width: if the laid-out diagram is wider than this (in cm), scale
            it down to fit -- wide gradients/products then stay on the page
            instead of overflowing. Ignored if `scale` is given.
        edge_labels: also label the left/right free-edge stubs with their
            index names (mid-spine stubs are always labelled). Useful when
            the contraction structure/symmetries would otherwise be
            ambiguous -- e.g. an Isserlis expansion of separate covariance
            factors. Off by default to keep simple chains uncluttered.
        stack: stack a top-level Sum's terms vertically (one per row) instead
            of side by side. None (default) = automatic: stack when the
            horizontal layout would be wider than STACK_WIDTH cm, so long
            sums (Taylor series, moment expansions) stay legible.
    """
    layout = layout_any(tensor, left, right, stack=stack)
    if scale is None and max_width is not None and layout.xmax > max_width > 0:
        scale = max_width / layout.xmax
    # wires a touch lighter than a font glyph's stroke, so a free-edge stub
    # reads as a thin line while a sum operator (enlarged) reads as an operator
    opts = f"baseline={baseline}, inner sep=1pt, line width=0.45pt"
    if scale is not None and abs(scale - 1.0) > 1e-6:
        # `transform shape` scales node glyphs too, so the whole diagram
        # shrinks uniformly instead of nodes overlapping at moved coordinates
        opts += f", scale={scale:.3f}, transform shape"
    lines: list[str] = [rf"\begin{{tikzpicture}}[{opts}]"]
    _emit_layout(layout, lines, prefix="", dx=0.0, edge_labels=edge_labels)
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)
