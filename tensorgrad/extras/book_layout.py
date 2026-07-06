"""Book-grammar layout for tensor diagrams.

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

Supported today: Variable, Delta (any order), Product, Rename, Sum (top
level AND as parenthesized product factors), and Function (application
arrows: solid for consumed edges, densely dotted for elementwise; broadcast
edges stay on the argument, as the book draws them).  Transposed 2-port
variables (spine forces reversed port order) are drawn rotated 180 degrees,
the book's transpose convention.  Unevaluated Derivative nodes render as
Penrose derivative loops: a fit-ellipse around the differentiated
subexpression with a dot on the boundary and labeled whiskers for the new
edges (one whisker bends left; a pair bends left/right, like the book's
\dloop/\dwhiskers).  Nested derivatives nest their ellipses.  A derivative
whose new edge is contracted onward still raises NotImplementedError --
simplify() first.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from numbers import Number
from typing import Optional

from tensorgrad.tensor import Delta, Derivative, Function, Product, Rename, Sum, Tensor, Variable

# ---------------------------------------------------------------------------
# Metric constants (calibrated against the book's hand figures)
# ---------------------------------------------------------------------------

PITCH = 0.62  # horizontal distance between adjacent spine node centers
CHAR_W = 0.11  # extra pitch per label character beyond the first
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
    """'pow(k=-1)' -> pow_{-1}; 'exp' -> exp."""
    if "(" in name:
        base, _, arg = name.partition("(")
        arg = arg.rstrip(")")
        if "=" in arg:
            arg = arg.split("=", 1)[1]
        return rf"{base}_{{{arg}}}"
    return name


class _UnionFind(dict):
    def find(self, x):
        while self.setdefault(x, x) != x:
            self[x] = self[self[x]]
            x = self[x]
        return x

    def union(self, a, b):
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
            out: dict[str, str] = {}
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
                if elementwise and first_atom < len(g.atoms):
                    # no consumed edges: synthesize the dotted application
                    # arrow from the input's principal atom
                    ta, tb = fresh("w"), fresh("w")
                    fn.ports.append(ta)
                    g.atoms[first_atom].ports.append(tb)
                    uf.union(ta, tb)
                    g.arrows[ta] = "dotted"
                    g.arrow_heads[ta] = fn.id
                out.update(sub)  # broadcast edges pass through by name
            for e in t.shape_out:
                port = fresh("w")
                fn.ports.append(port)
                out[e] = port
            return out
        if isinstance(t, Derivative):
            first_atom = len(g.atoms)
            sub = walk(t.tensor)
            enclosed = list(range(first_atom, len(g.atoms)))
            new_edges = [e for e in t.edges if e not in t.tensor.edges]
            out = dict(sub)
            names = []
            for e in new_edges:
                tok = fresh("w")
                g.deriv_tokens.add(tok)
                out[e] = tok
                names.append(e)
            g.derivs.append((enclosed, names))
            return out
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
        if atom.kind != "group":
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
        subs[v] = layout_any(atom.sub, left=le, right=re)

    def halfwidth(v: int) -> float:
        atom = g.atoms[v]
        if atom.kind == "group":
            return subs[v].xmax / 2 + 0.27  # parens + padding
        if atom.kind == "copydot":
            return 0.06
        return _label_halfwidth(atom.label)

    # -- x positions along the spine (label-width aware) --
    xs: list[float] = []
    x = x0
    head_frees = _atom_free_names(g, spine[0])
    if head_frees:
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
            x += max(PITCH, prev_half + half + clearance)
        xs.append(x)
        prev_half = half
    for v, xv in zip(spine, xs):
        atom = g.atoms[v]
        layout.nodes.append(
            LNode(v, atom.kind, atom.label, xv, 0.0,
                  sub=subs.get(v), width=2 * halfwidth(v))
        )

    drawn: list[str] = []  # every wid rendered; audited for conservation below

    # -- spine segments (application arrows point into their function) --
    for k in range(len(spine) - 1):
        wids = adj[spine[k]].get(spine[k + 1])
        if wids:
            wid = wids[0]
            drawn.append(wid)
            arrow = g.arrows.get(wid, "")
            a, b = spine[k], spine[k + 1]
            if arrow and g.arrow_heads.get(wid) == a:
                a, b = b, a
            layout.wires.append(LWire("segment", a, b, arrow=arrow))

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
                      span=j - i, lane=lanes[wid])
            )

    # -- pendants (off-spine vertices hang below their attachment) --
    hang_x: dict[int, float] = {}
    placed = set(spine)
    frontier = list(spine)
    while frontier:
        u = frontier.pop(0)
        for v, wids in adj[u].items():
            if v in placed:
                continue
            placed.add(v)
            ux = next(n.x for n in layout.nodes if n.id == u)
            uy = next(n.y for n in layout.nodes if n.id == u)
            atom = g.atoms[v]
            half = _label_halfwidth(atom.label)
            off = hang_x.get(u, 0.0)
            if off:
                off += half  # previous sibling's right edge + our halfwidth
            hang_x[u] = off + half + 0.25
            layout.nodes.append(
                LNode(v, atom.kind, atom.label, ux + off, uy - PENDANT_DY)
            )
            wid = wids[0]
            drawn.append(wid)
            arrow = g.arrows.get(wid, "")
            pa, pb = u, v
            if arrow and g.arrow_heads.get(wid) == pa:
                pa, pb = pb, pa
            layout.wires.append(LWire("pendant", pa, pb, arrow=arrow))
            frontier.append(v)

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
                    LWire("extra", a, b, direction=tip, arrow=arrow)
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
        for direction, name in assigned:
            lab = name if direction in ("up", "down") else ""
            off = 0.0
            if direction == "up" and len(ups) > 1:
                off = (ui - (len(ups) - 1) / 2) * 0.26
                ui += 1
            layout.wires.append(
                LWire("stub", v, direction=direction, label=lab, x=off))
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
        if li > ri:
            node = next(n for n in layout.nodes if n.id == v)
            node.rotated = True
    # free edges on pendants leave downward
    for v in comp:
        if v in pos_index:
            continue
        for idx, name in _atom_free_names(g, v):
            layout.wires.append(LWire("stub", v, direction="down", label=name))

    xmax = xs[-1] + halfwidth(spine[-1])
    if _atom_free_names(g, spine[-1]):
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
    return layout


def layout_any(
    tensor: Tensor, left: Optional[str] = None, right: Optional[str] = None
) -> BookLayout:
    """Layout for any supported tensor, including a top-level Sum."""
    if not isinstance(tensor, Sum):
        return layout_tensor(tensor, left, right)
    out = BookLayout()
    x = 0.0
    sign_id = 100000
    for idx, (term, weight) in enumerate(zip(tensor.terms, tensor.weights)):
        sign, coeff = _fmt_weight(weight)
        show_sign = idx > 0 or sign == "-" or coeff
        if show_sign:
            sgn = sign if (idx > 0 or sign == "-") else ""
            tex = f"{sgn}{coeff}".strip()
            if tex:
                pad = 0.42 + 0.1 * len(tex)
                out.nodes.append(LNode(sign_id, "sign", tex, x + pad - TERM_GAP, 0.0))
                sign_id += 1
                x += 2 * pad - TERM_GAP
        sub = layout_tensor(term, left, right)
        if left is None and sub.left_edge is not None:
            left = sub.left_edge
        if right is None and sub.right_edge is not None:
            right = sub.right_edge
        for nd in sub.nodes:
            out.nodes.append(
                LNode(nd.id + idx * 1000, nd.kind, nd.label, nd.x + x, nd.y,
                      sub=nd.sub, width=nd.width, rotated=nd.rotated)
            )
        for w in sub.wires:
            wx = w.x + x if w.kind in ("bare", "ring") else w.x
            out.wires.append(
                LWire(w.kind,
                      None if w.a is None else w.a + idx * 1000,
                      None if w.b is None else w.b + idx * 1000,
                      w.direction, w.label, w.arrow, w.span, w.lane,
                      wx, w.y)
            )
        for enclosed, names in sub.derivs:
            out.derivs.append(([a + idx * 1000 for a in enclosed], names))
        x += sub.xmax + TERM_GAP
    out.xmax = x - TERM_GAP
    return out


# ---------------------------------------------------------------------------
# TikZ emission (the book's vocabulary)
# ---------------------------------------------------------------------------


def _tex_label(label: str) -> str:
    if len(label) > 2:
        return rf"$\mathrm{{{label}}}$"
    return f"${label}$"


def _fmt_weight(w) -> tuple[str, str]:
    """Return (sign, coefficient-tex) for a sum weight."""
    if isinstance(w, Number):
        if w == 1:
            return "+", ""
        if w == -1:
            return "-", ""
        if w == int(w):
            w = int(w)
        return ("-", f"{abs(w)}\\,") if w < 0 else ("+", f"{w}\\,")
    return "+", f"{w}\\,"


def _emit_layout(layout: BookLayout, lines: list[str], prefix: str, dx: float) -> None:
    name = {n.id: f"{prefix}n{n.id}" for n in layout.nodes}
    nodes = {n.id: n for n in layout.nodes}
    group_side: dict[int, tuple[str, str]] = {}
    for n in layout.nodes:
        x = n.x + dx
        if n.kind == "copydot":
            lines.append(rf"\node[copydot] ({name[n.id]}) at ({x:.2f},{n.y:.2f}) {{}};")
        elif n.kind == "group":
            # parenthesized inner sum: two paren nodes + recursive emission
            hw = n.width / 2
            pl, pr = f"{name[n.id]}L", f"{name[n.id]}R"
            lines.append(
                rf"\node[scale=1.45, inner sep=0.5pt] ({pl}) at ({x - hw + 0.1:.2f},{n.y:.2f}) {{$($}};"
            )
            lines.append(
                rf"\node[scale=1.45, inner sep=0.5pt] ({pr}) at ({x + hw - 0.1:.2f},{n.y:.2f}) {{$)$}};"
            )
            group_side[n.id] = (pl, pr)
            assert n.sub is not None
            _emit_layout(n.sub, lines, prefix=f"{name[n.id]}i", dx=x - hw + 0.17)
        elif n.kind == "sign":
            lines.append(
                rf"\node ({name[n.id]}) at ({x:.2f},{n.y:.2f}) {{${n.label}$}};"
            )
        else:
            rot = "rotate=180, " if n.rotated else ""
            lines.append(
                rf"\node[{rot}inner sep=1pt] ({name[n.id]}) at ({x:.2f},{n.y:.2f})"
                rf" {{{_tex_label(n.label)}}};"
            )

    def endpoint(nid: int, other: Optional[int]) -> str:
        # segments touching a group attach to the nearer paren
        if nid in group_side and other is not None:
            pl, pr = group_side[nid]
            return pl if nodes[other].x < nodes[nid].x else pr
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
    for w in layout.wires:
        if w.kind == "segment" or w.kind == "pendant":
            lines.append(
                rf"\draw{style[w.arrow].strip()} ({endpoint(w.a, w.b)})"
                rf" -- ({endpoint(w.b, w.a)});"
            )
        elif w.kind == "loop":
            if w.a in group_side:
                pl, pr = group_side[w.a]
                lines.append(
                    rf"\path ({pl}) edge [out=160, in=20, looseness=1.6] ({pr});"
                )
            else:
                dist = ""
                if w.lane > 1:
                    dist = f", min distance={6 + 5 * (w.lane - 1)}mm"
                lines.append(
                    rf"\path ({name[w.a]}) edge [out=160, in=20, loop{dist}]"
                    rf" ({name[w.a]});"
                )
        elif w.kind == "arc":
            na, nb = endpoint(w.a, w.b), endpoint(w.b, w.a)
            tip = f", {w.direction}" if w.direction else ""
            dot = "densely dotted, " if w.arrow == "dotted" else ""
            if w.span <= 3:
                loose = 0.55 + 0.47 * w.span + 0.5 * (w.lane - 1)
                lines.append(
                    rf"\path[{dot.rstrip(', ')}] ({na}) edge [out=160, in=20{tip}, "
                    rf"looseness={loose:.2f}] ({nb});"
                )
            else:
                # long spans: flat north-anchored dome hugging the chain
                h = 0.35 + 0.08 * w.span + 0.28 * (w.lane - 1)
                arrowopt = f"[{dot}{w.direction}] " if (w.direction or dot) else ""
                lines.append(
                    rf"\draw{arrowopt.strip()} ({na}.north west) .. controls"
                    rf" +(-0.15,{h:.2f}) and +(0.15,{h:.2f}) .."
                    rf" ({nb}.north east);"
                )
        elif w.kind == "extra":
            a, b = w.a, w.b
            # start from the lower endpoint; bend so the curve bows DOWN,
            # away from the spine and its arcs
            if nodes[a].y > nodes[b].y or (
                nodes[a].y == nodes[b].y and nodes[a].x > nodes[b].x
            ):
                a, b = b, a
            bend = "right" if nodes[a].x < nodes[b].x else "left"
            na, nb = endpoint(a, b), endpoint(b, a)
            tip = f"{w.direction}, " if w.direction else ""
            dot = "densely dotted, " if w.arrow == "dotted" else ""
            lines.append(
                rf"\draw[{tip}{dot}] ({na}) to[bend {bend}=45] ({nb});"
            )
        elif w.kind == "ring":
            lines.append(
                rf"\draw ({w.x + dx:.2f},{w.y:.2f}) circle (0.28);"
            )
        elif w.kind == "stub":
            d, anch = {
                "left": (f"({-STUB:.2f},0)", "west"),
                "right": (f"({STUB:.2f},0)", "east"),
                "up": (f"(0,{MIDSTUB:.2f})", "north"),
                "down": (f"(0,{-MIDSTUB:.2f})", "south"),
            }[w.direction]
            if nodes[w.a].rotated:  # 180-degree node: anchors are mirrored
                anch = {"west": "east", "east": "west",
                        "north": "south", "south": "north"}[anch]
            lab = ""
            if w.label:
                anchor = {"up": "south", "down": "north"}.get(w.direction, "west")
                lab = rf" node[anchor={anchor}, font=\scriptsize, inner sep=1.5pt] {{${w.label}$}}"
            src = name[w.a]
            if w.a in group_side:
                src = group_side[w.a][0 if w.direction == "left" else 1]
            shift = f"[xshift={w.x:.2f}cm]" if w.x else ""
            lines.append(rf"\draw ({shift}{src}.{anch}) -- ++{d}{lab};")
        elif w.kind == "bare":
            lines.append(
                rf"\draw ({w.x + dx:.2f},{w.y:.2f}) -- ++({1.0:.2f},0);"
            )
    _emit_derivs(layout, lines, prefix, name, group_side)


def _emit_derivs(layout: BookLayout, lines: list[str], prefix: str,
                 name: dict, group_side: dict) -> None:
    """Penrose derivative loops: fit-ellipse + boundary dot + whiskers."""
    nodes = {n.id: n for n in layout.nodes}
    for di, (enclosed, new_names) in enumerate(layout.derivs):
        eset = set(enclosed)
        parts = []
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
                h = (0.32 + 0.16 * w.span + 0.24 * (w.lane - 1)
                     if w.span <= 3 else 0.35 + 0.08 * w.span)
                mx = (nodes[w.a].x + nodes[w.b].x) / 2
                parts.append(f"({mx:.2f},{h + 0.12:.2f})")
            elif w.kind == "loop" and w.a in eset:
                parts.append(f"({nodes[w.a].x:.2f},{0.42 + 0.2 * w.lane:.2f})")
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
        if len(new_names) == 1:
            ang = 125 + 14 * inside  # fan nested whiskers apart
            lines.append(rf"\fill ({en}.{ang}) circle (1.4pt);")
            lines.append(
                rf"\draw ({en}.{ang}) .. controls +({ang - 25}:.12) .."
                rf" ++(-.24,.26)"
                rf" node[anchor=south east, font=\scriptsize, inner sep=1pt]"
                rf" {{${_tex_edge(new_names[0])}$}};"
            )
        else:
            ang = 55
            lines.append(rf"\fill ({en}.{ang}) circle (1.4pt);")
            whisk = [("80:.12", "++(-.2,.28)", "south east"),
                     ("45:.12", "++(.28,.18)", "south west")]
            for nm, (ctrl, end, anch) in zip(new_names, whisk):
                lines.append(
                    rf"\draw ({en}.{ang}) .. controls +({ctrl}) .. {end}"
                    rf" node[anchor={anch}, font=\scriptsize, inner sep=1pt]"
                    rf" {{${_tex_edge(nm)}$}};"
                )


def to_book_tikz(
    tensor: Tensor,
    left: Optional[str] = None,
    right: Optional[str] = None,
    baseline: str = "-.25em",
) -> str:
    """Render a tensorgrad Tensor as book-style TikZ (uses tikz-styles.tex)."""
    lines: list[str] = [
        rf"\begin{{tikzpicture}}[baseline={baseline}, inner sep=1pt]"
    ]
    _emit_layout(layout_any(tensor, left, right), lines, prefix="", dx=0.0)
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)
