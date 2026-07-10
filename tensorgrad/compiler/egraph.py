"""Equality-saturation stage: the IR's reassociation/factoring SEARCH as an
e-graph, replacing greedy sweeps with saturate-and-extract.

The hand-sequenced factor pipeline makes its cost decisions greedily, with
hysteresis margins (DIST_MARGIN), best-seen patches and orbit detection
bolted on to keep it from cycling (factor.py). This stage hands the SEARCH
to egglog: rules add equivalent forms non-destructively, and a flop+memory
cost extraction picks the best DAG — no phase ordering, no margins, no
orbiting. Measured foundations (tests/compiler/test_egglog_stage2.py):
saturation walks the adjoint valley greedy provably misses (exact
reverse-mode optimum, depths 6..96, <1s at GPT boundary-chain scale), and
the residual-diamond blowup is avoided by keeping rules FACTORING-DIRECTED.

Scope (v1, the matrix fragment): the bridge recognizes the IR subset where
reassociation is classical — 2-operand einsums in matmul normal form
(single contracted wire, operands of order <= 2, no constraints, rational
weights) and LinearNodes with identity perms — and treats EVERYTHING ELSE
as opaque leaves the rules carry but never open. The leaf boundary makes
partial coverage sound; the vocabulary grows by earning it.

Design laws (each measured, see the simplify-as-search memo):
- fp-grouping is semantics: stabilized/fused forms are unrecognized ->
  atomic leaves; the e-graph cannot regroup through them.
- No free AC: Add chains are encoded in the IR's canonical (already
  sorted) LinearNode order; rules use associativity and factoring-directed
  collapse only. Free commutativity measured >5min vs 0.01s on the D=4
  residual diamond.
- Transposes are free bookkeeping: Tr is algebra on the decode-side axis
  permutation, never a materialized node.
- The whole stage self-gates with szfp exact equality (a wrong extraction
  is a missed optimization, never a miscompile) and every decoded root
  must reproduce the original dims exactly or the stage aborts.

Cost model mirrors factor._program_score: matmul = flops + MEM_WEIGHT *
output elements; adds pay per-term element traffic. (Known approximation:
egglog scores Add chains binarily while the decoder rebuilds them as one
n-ary LinearNode; relative ranking of alternatives is unaffected.)

EGRAPH gates the stage (default False until the benchmark A/B says
otherwise; flipping it is a measured decision).
"""

from fractions import Fraction
from typing import Any, Optional

import sympy

from tensorgrad.compiler.ir import Builder, EinsumNode, LinearNode, Node, toposort

EGRAPH = False
MEM_WEIGHT = 4
MAX_ENCODED = 4000  # composite-term cap: past this, skip (saturation cost)


def egraph_outputs(builder: Builder, outputs: list, dims: Optional[dict] = None) -> list:
    """Saturate-and-extract over the program. Returns rewritten outputs, or
    `outputs` unchanged when the stage is disabled, egglog is unavailable,
    nothing in the program is mappable, or the szfp gate refuses."""
    if not EGRAPH or not dims:
        return outputs
    try:
        import egglog  # noqa: F401
    except ImportError:
        return outputs
    from tensorgrad.compiler import szfp

    try:
        rewritten = _saturate_extract(builder, outputs, dims)
    except Exception:
        return outputs  # a failed rewrite is a missed optimization
    if rewritten is None:
        return outputs
    try:
        ok = szfp.outputs_equal(outputs, rewritten)
    except Exception:
        return outputs  # unverifiable -> keep the proven form
    return rewritten if ok else outputs


# --------------------------------------------------------------------------
# The egglog program (vocabulary + rules), built lazily on first use so the
# egglog import cost is only paid when the stage is enabled.
# --------------------------------------------------------------------------

_LANG: dict[str, Any] = {}


def _lang() -> dict[str, Any]:
    if _LANG:
        return _LANG
    from egglog import (
        Expr,
        function,
        i64,
        i64Like,
        rewrite,
        rule,
        ruleset,
        set_,
        set_cost,
        eq,
    )

    class T(Expr):
        def __matmul__(self, other: "T") -> "T": ...  # type: ignore[empty-body]
        def __add__(self, other: "T") -> "T": ...  # type: ignore[empty-body]

    @function
    def Leaf(idx: i64Like) -> T: ...  # type: ignore[empty-body]
    @function
    def BLeaf(idx: i64Like, batch_pos: i64Like) -> T: ...  # type: ignore[empty-body]
    @function
    def Tr(x: T) -> T: ...  # type: ignore[empty-body]
    @function
    def Scale(num: i64Like, den: i64Like, x: T) -> T: ...  # type: ignore[empty-body]
    @function
    def rows(x: T) -> i64: ...  # type: ignore[empty-body]
    @function
    def cols(x: T) -> i64: ...  # type: ignore[empty-body]
    @function
    def bat(x: T) -> i64: ...  # type: ignore[empty-body]

    @ruleset
    def rules(a: T, b: T, c: T, r: i64, k: i64, n: i64, p: i64, q: i64, s: i64, t: i64):  # type: ignore[no-untyped-def]
        # ---- shape propagation (leaves seeded by the encoder) ----
        yield rule(eq(c).to(a @ b), eq(rows(a)).to(r), eq(cols(b)).to(n)).then(
            set_(rows(c)).to(r), set_(cols(c)).to(n)
        )
        yield rule(eq(c).to(a + b), eq(rows(a)).to(r), eq(cols(a)).to(n)).then(
            set_(rows(c)).to(r), set_(cols(c)).to(n)
        )
        yield rule(eq(c).to(Tr(a)), eq(rows(a)).to(r), eq(cols(a)).to(n)).then(
            set_(rows(c)).to(n), set_(cols(c)).to(r)
        )
        yield rule(eq(c).to(Scale(p, q, a)), eq(rows(a)).to(r), eq(cols(a)).to(n)).then(
            set_(rows(c)).to(r), set_(cols(c)).to(n)
        )
        # ---- batch propagation: the batch of a product/sum is the larger
        #      side (unbatched operands broadcast, encoded as bat = 1) ----
        yield rule(eq(c).to(a @ b), eq(bat(a)).to(p), eq(bat(b)).to(i64(1))).then(set_(bat(c)).to(p))
        yield rule(eq(c).to(a @ b), eq(bat(a)).to(i64(1)), eq(bat(b)).to(p)).then(set_(bat(c)).to(p))
        yield rule(eq(c).to(a @ b), eq(bat(a)).to(p), eq(bat(b)).to(p)).then(set_(bat(c)).to(p))
        yield rule(eq(c).to(a + b), eq(bat(a)).to(p)).then(set_(bat(c)).to(p))
        yield rule(eq(c).to(Tr(a)), eq(bat(a)).to(p)).then(set_(bat(c)).to(p))
        yield rule(eq(c).to(Scale(p, q, a)), eq(bat(a)).to(s)).then(set_(bat(c)).to(s))
        # ---- costs (mirror factor._program_score's flops + MEM model) ----
        yield rule(
            eq(c).to(a @ b), eq(rows(a)).to(r), eq(cols(a)).to(k), eq(cols(b)).to(n),
            eq(bat(a @ b)).to(p),
        ).then(set_cost(a @ b, p * (r * k * n + MEM_WEIGHT * r * n)))
        yield rule(
            eq(c).to(a + b), eq(rows(a)).to(r), eq(cols(a)).to(n), eq(bat(a + b)).to(p)
        ).then(set_cost(a + b, p * (2 + MEM_WEIGHT) * r * n))
        yield rule(
            eq(c).to(Scale(p, q, a)), eq(rows(a)).to(r), eq(cols(a)).to(n), eq(bat(a)).to(s)
        ).then(set_cost(Scale(p, q, a), s * (1 + MEM_WEIGHT) * r * n))
        yield rule(eq(c).to(Tr(a))).then(set_cost(Tr(a), 0))
        yield rule(eq(c).to(Leaf(p))).then(set_cost(Leaf(p), 0))
        yield rule(eq(c).to(BLeaf(p, q))).then(set_cost(BLeaf(p, q), 0))
        # ---- matmul associativity (the adjoint valley) ----
        yield rewrite((a @ b) @ c).to(a @ (b @ c))
        yield rewrite(a @ (b @ c)).to((a @ b) @ c)
        # ---- factoring-DIRECTED collapse (never the expansion direction) ----
        yield rewrite(a @ c + b @ c).to((a + b) @ c)
        yield rewrite(a @ b + a @ c).to(a @ (b + c))
        # ---- add associativity (brings factorable pairs adjacent; no
        #      commutativity — the encoder's canonical order stands in) ----
        yield rewrite((a + b) + c).to(a + (b + c))
        yield rewrite(a + (b + c)).to((a + b) + c)
        # ---- transpose algebra (free views), DIRECTED toward leaves:
        #      bidirectional Tr-through-matmul measured a saturation blowup
        #      between chain depths 20 and 24 (Tr-mirrored e-classes multiply
        #      against reassociation); pushing Tr down leaves the matmul
        #      spine Tr-free, which is all associativity needs ----
        yield rewrite(Tr(Tr(a))).to(a)
        yield rewrite(Tr(a @ b)).to(Tr(b) @ Tr(a))
        yield rewrite(Tr(a) + Tr(b)).to(Tr(a + b))
        # ---- scale movement (weights ride to where they fold away) ----
        yield rewrite(Scale(p, q, a) @ b).to(Scale(p, q, a @ b))
        yield rewrite(a @ Scale(p, q, b)).to(Scale(p, q, a @ b))
        yield rewrite(Scale(p, q, Scale(s, t, a))).to(Scale(p * s, q * t, a))
        yield rewrite(Scale(p, q, a) + Scale(p, q, b)).to(Scale(p, q, a + b))
        yield rewrite(Tr(Scale(p, q, a))).to(Scale(p, q, Tr(a)))

    _LANG.update(T=T, Leaf=Leaf, BLeaf=BLeaf, Tr=Tr, Scale=Scale, rows=rows, cols=cols, bat=bat, rules=rules)
    return _LANG


# --------------------------------------------------------------------------
# Encode: IR DAG -> egglog terms. Decoded values carry (node, axes) where
# axes maps MNF roles (row, col) to node axis indices (None = absent role);
# Tr is a swap of that mapping, never a materialized node.
# --------------------------------------------------------------------------


def _as_rational(w: Any) -> Optional[Fraction]:
    e = sympy.sympify(w)
    if e.is_Integer:
        return Fraction(int(e))
    if e.is_Rational:
        return Fraction(int(e.p), int(e.q))
    return None


class _Encoder:
    """IR -> egglog. Decode-side values carry role axes (batch, row, col)
    into node.dims; leaves used with a batch role encode as BLeaf(idx,
    batch_axis) — the same node under different views is legitimately a
    different term. Tr is the (row, col) swap; batch is untouched by Tr."""

    def __init__(self, lang: dict[str, Any], dims: dict):
        self.lang = lang
        self.dims = dims
        self.leaves: list[Node] = []
        self.leaf_ids: dict[int, int] = {}
        self.views: set[tuple[int, int]] = set()  # (leaf_idx, batch_pos or -1)
        self.memo: dict[int, Any] = {}
        self.composites = 0

    def _dim(self, d: Any) -> int:
        e = sympy.sympify(d)
        v = e.subs(self.dims) if e.free_symbols else e
        if not v.is_Integer:
            raise ValueError(f"dim {d} unresolved")
        return int(v)

    def _leaf_idx(self, node: Node) -> int:
        i = self.leaf_ids.get(id(node))
        if i is None:
            i = len(self.leaves)
            self.leaves.append(node)
            self.leaf_ids[id(node)] = i
        return i

    def leaf(self, node: Node, batch_axis: Optional[int] = None) -> Any:
        i = self._leaf_idx(node)
        if batch_axis is None:
            self.views.add((i, -1))
            return self.lang["Leaf"](i)
        self.views.add((i, batch_axis))
        return self.lang["BLeaf"](i, batch_axis)

    def view_shape(self, node: Node, batch_pos: int) -> tuple[int, int, int]:
        """(bat, rows, cols) for a leaf view. batch_pos = -1 means no batch
        role: order-2 = (1, d0, d1); order-1 = column (1, d0, 1); order-0 =
        (1, 1, 1); order>2 without a batch role = (1, numel, 1) (Add-only).
        With a batch role, that axis is the batch and the remaining axes
        (ascending) are (row, col)."""
        ds = [self._dim(d) for d in node.dims]
        if batch_pos >= 0:
            b = ds[batch_pos]
            rest = [d for a, d in enumerate(ds) if a != batch_pos]
            if len(rest) == 2:
                return b, rest[0], rest[1]
            if len(rest) == 1:
                return b, rest[0], 1
            return b, 1, 1
        if len(ds) == 2:
            return 1, ds[0], ds[1]
        if len(ds) == 1:
            return 1, ds[0], 1
        if not ds:
            return 1, 1, 1
        n = 1
        for d in ds:
            n *= d
        return 1, n, 1

    def encode(self, node: Node) -> Any:
        return self._enc(node)[0]

    def _enc(self, node: Node) -> tuple[Any, tuple]:
        """(term, roles): roles = (batch_axis, row_axis, col_axis) into the
        ORIGINAL node's axes for the tensor the term denotes. Leaves use the
        canonical identity view; composites carry their true arrangement so
        parents (and root restoration) never have to guess."""
        hit = self.memo.get(id(node))
        if hit is not None:
            return hit
        res = self._encode_matmul(node) if isinstance(node, EinsumNode) else None
        if res is None and isinstance(node, LinearNode):
            res = self._encode_linear(node)
        if res is None:
            term = self.leaf(node)
            if node.order == 2:
                roles: tuple = (None, 0, 1)
            elif node.order == 1:
                roles = (None, 0, None)
            else:
                roles = (None, None, None)
            res = (term, roles)
        else:
            self.composites += 1
        self.memo[id(node)] = res
        return res

    # ---- einsum chains (batched or not) --------------------------------

    def _encode_matmul(self, n: EinsumNode) -> Optional[Any]:
        """Path-shaped einsums (n-ary chains included: lowering flattens
        whole @-chains into ONE einsum) decompose into binary MNF hops. At
        most ONE wire may ride as a broadcast batch role: it must surface in
        the output and every operand carrying it treats it as its batch
        axis. Anything non-path (stars, cycles, extra hyperedge wires,
        diagonals, constraints) stays a leaf."""
        if len(n.ops) < 2 or n.constraints:
            return None
        w = _as_rational(n.weight)
        if w is None:
            return None
        deg: dict[int, int] = {}
        for subs in n.in_subs:
            if len(set(subs)) != len(subs):
                return None  # diagonal
            for wr in subs:
                deg[wr] = deg.get(wr, 0) + 1
        outw = set(n.out_subs)
        if len(set(n.out_subs)) != len(n.out_subs):
            return None
        hyper = [wr for wr, d in deg.items() if d > 2]
        wb: Optional[int] = None
        if hyper:
            if len(hyper) > 1 or hyper[0] not in outw:
                return None
            wb = hyper[0]
        elif len(n.ops) == 2:
            # a 2-operand einsum can carry a degree-2 batch wire (shared and
            # surfacing): detect it so bmm-shaped einsums encode
            shared_out = [wr for wr, d in deg.items() if d == 2 and wr in outw]
            if len(shared_out) == 1:
                wb = shared_out[0]
            elif len(shared_out) > 1:
                return None
        # non-batch structure
        def nb(subs: tuple) -> list:
            return [wr for wr in subs if wr != wb]

        for subs in n.in_subs:
            if len(nb(subs)) > 2:
                return None
        free_out = [wr for wr in n.out_subs if wr != wb]
        if len(free_out) > 2:
            return None
        for wr, d in deg.items():
            if wr == wb:
                continue
            if (d == 1) != (wr in outw):
                return None
        if any(wr not in deg for wr in outw):
            return None  # broadcast-only output wire
        # operand adjacency via shared non-batch wires: one simple path
        m = len(n.ops)
        adj: dict[int, list[int]] = {i: [] for i in range(m)}
        holders: dict[int, list[int]] = {}
        for i, subs in enumerate(n.in_subs):
            for wr in nb(subs):
                holders.setdefault(wr, []).append(i)
        for wr, hs in holders.items():
            if len(hs) == 2:
                a, b = hs
                if a == b:
                    return None
                adj[a].append(b)
                adj[b].append(a)
        ends = [i for i in range(m) if len(adj[i]) == 1]
        if any(len(adj[i]) not in (1, 2) for i in range(m)) or len(ends) != 2:
            return None
        order = [ends[0]]
        prev = -1
        while len(order) < m:
            nxt = [j for j in adj[order[-1]] if j != prev]
            if len(nxt) != 1:
                return None
            prev = order[-1]
            order.append(nxt[0])
        if len(set(order)) != m:
            return None

        def hop_wire(i: int, j: int) -> int:
            (wr,) = set(nb(n.in_subs[i])) & set(nb(n.in_subs[j]))
            return wr

        term = None
        for pos in range(m - 1):
            i, j = order[pos], order[pos + 1]
            kw = hop_wire(i, j)
            if term is None:
                left = self._oriented(n.ops[i], n.in_subs[i], kw, wb, role_first=True)
                if left is None:
                    return None
                term = left
            right = self._oriented(n.ops[j], n.in_subs[j], kw, wb, role_first=False)
            if right is None:
                return None
            term = term @ right
        free_first = [wr for wr in nb(n.in_subs[order[0]]) if deg[wr] == 1]
        free_last = [wr for wr in nb(n.in_subs[order[-1]]) if deg[wr] == 1]
        mnf_out = free_first + free_last
        if free_out != mnf_out:
            if len(free_out) == 2 and free_out == list(reversed(mnf_out)):
                term = self.lang["Tr"](term)
                mnf_out = free_out
            else:
                return None
        if w != 1:
            term = self.lang["Scale"](w.numerator, w.denominator, term)
        # roles into n's own output axes
        out_list = list(n.out_subs)
        b_ax = out_list.index(wb) if wb is not None and wb in out_list else None
        r_ax = out_list.index(mnf_out[0]) if len(mnf_out) >= 1 else None
        c_ax = out_list.index(mnf_out[1]) if len(mnf_out) >= 2 else None
        return term, (b_ax, r_ax, c_ax)

    def _oriented(self, op: Node, subs: tuple, kw: int, wb: Optional[int], role_first: bool) -> Optional[Any]:
        """Operand oriented for MNF: left operands want (row, k), right want
        (k, col); the batch axis (if the operand carries wb) rides outside
        the orientation. Batched operands take a leaf VIEW (BLeaf); plain
        operands encode recursively, and the child's own role arrangement is
        consulted — a composite whose term is role-swapped relative to its
        node axes orients through its roles, never through axis positions."""
        if wb is not None and wb in subs:
            batch_axis = subs.index(wb)
            t = self.leaf(op, batch_axis)
            rest = [a for a in range(len(subs)) if a != batch_axis]
            roles = (batch_axis, rest[0] if rest else None, rest[1] if len(rest) > 1 else None)
        else:
            t, roles = self._enc(op)
        nbsubs = [wr for wr in subs if wr != wb]
        if len(nbsubs) == 1:
            return self.lang["Tr"](t) if role_first else t
        # which of the operand's axes carries the contraction wire?
        k_axis = subs.index(kw)
        if roles[1] == k_axis:
            k_role_is_row = True
        elif roles[2] == k_axis:
            k_role_is_row = False
        else:
            return None  # k rides on the batch axis: not a contraction
        # left wants k in the col role; right wants k in the row role
        good = (not k_role_is_row) if role_first else k_role_is_row
        return t if good else self.lang["Tr"](t)

    # ---- linear combinations -------------------------------------------

    def _encode_linear(self, n: LinearNode) -> Optional[tuple[Any, tuple]]:
        idperm = tuple(range(len(n.dims)))
        term_out = None
        out_roles: Optional[tuple] = None
        for t, pm, w in zip(n.terms, n.perms, n.weights):
            wr = _as_rational(w)
            if wr is None:
                return None
            enc, roles = self._enc(t)
            if tuple(pm) == (1, 0):
                enc = self.lang["Tr"](enc)
                roles = (roles[0], roles[2], roles[1])
            elif tuple(pm) != idperm:
                return None
            if wr != 1:
                enc = self.lang["Scale"](wr.numerator, wr.denominator, enc)
            if term_out is None:
                term_out = enc
                # the linear's output axis j is the base term's axis pm[j];
                # with identity/Tr-normalized perms that is axis j, so the
                # base roles ARE the linear's roles
                out_roles = roles
            else:
                term_out = term_out + enc
        if term_out is None or out_roles is None:
            return None
        return term_out, out_roles

# --------------------------------------------------------------------------
# Decode: extracted term -> IR through the Builder. Values are
# (node, roles, scale) with roles = (batch_axis, row_axis, col_axis) into
# node.dims (None = absent role) and a pending rational scale.
# --------------------------------------------------------------------------


def _canon_roles(t: tuple) -> tuple:
    """With a single non-batch role, row vs col is pure matrix-space
    bookkeeping over the SAME node axis: normalize it to the row slot."""
    if (t[1] is None) != (t[2] is None):
        return (t[0], t[1] if t[1] is not None else t[2], None)
    return t


class _Decoder:
    def __init__(self, builder: Builder, leaves: list[Node]):
        self.b = builder
        self.leaves = leaves

    def _name(self, call: Any) -> str:
        c = call.callable
        n = getattr(c, "method_name", None) or getattr(getattr(c, "ident", None), "name", None)
        return n if n is not None else str(c)

    def _walk(self, call: Any) -> tuple[Node, tuple, Fraction]:
        name = self._name(call)
        args = call.args
        if name == "Leaf":
            node = self.leaves[args[0].expr.value]
            if node.order == 2:
                roles = (None, 0, 1)
            elif node.order == 1:
                roles = (None, 0, None)
            else:
                roles = (None, None, None)
            return node, roles, Fraction(1)
        if name == "BLeaf":
            node = self.leaves[args[0].expr.value]
            bp = args[1].expr.value
            rest = [a for a in range(node.order) if a != bp]
            roles = (bp, rest[0] if rest else None, rest[1] if len(rest) > 1 else None)
            return node, roles, Fraction(1)
        if name == "Tr":
            node, roles, s = self._walk(args[0].expr)
            return node, (roles[0], roles[2], roles[1]), s
        if name == "Scale":
            pv = args[0].expr.value
            qv = args[1].expr.value
            node, roles, s = self._walk(args[2].expr)
            return node, roles, s * Fraction(pv, qv)
        if name == "__matmul__":
            return self._matmul(args[0].expr, args[1].expr)
        if name == "__add__":
            return self._add(call)
        raise ValueError(f"egraph decode: unknown constructor {name}")

    def _matmul(self, ca: Any, cb: Any) -> tuple[Node, tuple, Fraction]:
        A, ra_, sa = self._walk(ca)
        B, rb_, sb = self._walk(cb)
        ab, ar, ak = ra_  # A roles: batch, row, k(=A's col)
        bb, bk, bc = rb_  # B roles: batch, k(=B's row), col
        # Vector operands: with a single non-batch axis, row-vs-col naming
        # is bookkeeping (the third place this lesson bites); the CONTRACT
        # position decides — a left vector contracts its only axis, ditto a
        # right vector.
        if ak is None and ar is not None:
            ak, ar = ar, None
        if bk is None and bc is not None:
            bk, bc = bc, None
        if ak is None or bk is None:
            raise ValueError("matmul on scalar-role operand")
        # wires: batch=3, row=0, k=1, col=2
        wa: list = [None] * A.order
        wb_: list = [None] * B.order
        wa[ak] = 1
        wb_[bk] = 1
        if ar is not None:
            wa[ar] = 0
        if ab is not None:
            wa[ab] = 3
        if bc is not None:
            wb_[bc] = 2
        if bb is not None:
            wb_[bb] = 3
        out_subs = []
        if ab is not None or bb is not None:
            out_subs.append(3)
        if ar is not None:
            out_subs.append(0)
        if bc is not None:
            out_subs.append(2)
        wire_dims = {1: A.dims[ak]}
        if ar is not None:
            wire_dims[0] = A.dims[ar]
        if bc is not None:
            wire_dims[2] = B.dims[bc]
        if ab is not None:
            wire_dims[3] = A.dims[ab]
        elif bb is not None:
            wire_dims[3] = B.dims[bb]
        node = self.b.einsum(
            [A, B],
            [tuple(w for w in wa if w is not None), tuple(w for w in wb_ if w is not None)],
            tuple(out_subs),
            wire_dims,
            sympy.Rational((sa * sb).numerator, (sa * sb).denominator),
        )
        # result axes: canonical output order is (batch?, row?, col?)
        pos = 0
        roles: list = [None, None, None]
        if 3 in out_subs:
            roles[0] = pos
            pos += 1
        if 0 in out_subs:
            roles[1] = pos
            pos += 1
        if 2 in out_subs:
            roles[2] = pos
        return node, tuple(roles), Fraction(1)

    def _add(self, call: Any) -> tuple[Node, tuple, Fraction]:
        """Flatten the +-chain and rebuild ONE n-ary LinearNode; per-term
        perms map through the ROLE correspondence, so transposed or
        batch-permuted twins combine without materialized views."""
        terms: list[tuple[Node, tuple, Fraction]] = []

        def flat(c: Any) -> None:
            if self._name(c) == "__add__":
                flat(c.args[0].expr)
                flat(c.args[1].expr)
            else:
                terms.append(self._walk(c))

        flat(call)
        terms = [(nd, _canon_roles(rl), sc) for nd, rl, sc in terms]
        base_node, base_roles, _ = terms[0]
        order = base_node.order
        # output axis j corresponds to the role that the BASE term carries on
        # its j-th axis (base perm = identity)
        role_of_axis = {a: r for r, a in enumerate(base_roles) if a is not None}
        if len(role_of_axis) != order:
            raise ValueError("add with unassigned axes")
        out_role_seq = [role_of_axis[a] for a in range(order)]
        nodes, perms, weights = [], [], []
        for node, roles, s in terms:
            if node.order != order:
                raise ValueError("add of mismatched orders")
            pm = []
            for role in out_role_seq:
                ax = roles[role]
                if ax is None:
                    raise ValueError("add of mismatched roles")
                pm.append(ax)
            nodes.append(node)
            perms.append(tuple(pm))
            weights.append(sympy.Rational(s.numerator, s.denominator))
        out = self.b.linear(nodes, perms, weights)
        return out, base_roles, Fraction(1)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def _saturate_extract(builder: Builder, outputs: list, dims: dict) -> Optional[list]:
    """Encode -> saturate -> extract -> decode. Returns None for 'no change'
    (nothing recognized, caps hit, or extraction reproduced the input)."""
    from egglog import EGraph, i64, set_

    lang = _lang()
    enc = _Encoder(lang, dims)
    encoded = [enc._enc(node) for node, order in outputs]
    roots = [t for t, _ in encoded]
    expected_roles = [r for _, r in encoded]
    if enc.composites == 0 or enc.composites > MAX_ENCODED:
        return None

    egraph = EGraph()
    let_roots = [egraph.let(f"r{i}", t) for i, t in enumerate(roots)]
    for i, bp in sorted(enc.views):
        node = enc.leaves[i]
        b_, r_, c_ = enc.view_shape(node, bp)
        leaf = lang["Leaf"](i) if bp < 0 else lang["BLeaf"](i, bp)
        egraph.register(set_(lang["rows"](leaf)).to(i64(r_)))
        egraph.register(set_(lang["cols"](leaf)).to(i64(c_)))
        egraph.register(set_(lang["bat"](leaf)).to(i64(b_)))
    # Bounded schedule, NOT .saturate(): egglog's saturation detector
    # thrashes on the shape/cost analysis actions (measured: depth-24 chain
    # >120s under saturate, 0.06s and already at fixpoint under rules*30).
    # Reassociation closure needs ~O(spine length) rounds; leaves bound it.
    iters = min(64, 2 * len(enc.leaves) + 8)
    egraph.run(lang["rules"] * iters)

    dec = _Decoder(builder, enc.leaves)
    new_outputs = []
    changed = False
    for (node, order), lroot, want in zip(outputs, let_roots, expected_roles):
        best = egraph.extract(lroot)
        out_node, roles, scale = dec._walk(_expr_decl(best))
        out_node = _restore_axis_order(builder, out_node, roles, want)
        if out_node is None or tuple(out_node.dims) != tuple(node.dims):
            return None  # decode must reproduce dims exactly
        if scale != 1:
            out_node = builder.linear(
                [out_node], [tuple(range(out_node.order))],
                [sympy.Rational(scale.numerator, scale.denominator)],
            )
        changed |= out_node is not node
        new_outputs.append((out_node, order))
    return new_outputs if changed else None


def _expr_decl(expr: Any) -> Any:
    from egglog import expr_parts

    return expr_parts(expr).expr


def _restore_axis_order(builder: Builder, node: Node, roles: tuple, want: tuple) -> Optional[Node]:
    """Permute the decoded node so each ROLE lands on the axis the original
    root carried it on (`want`, recorded by the encoder). Exact by
    construction — no dims-based guessing, so repeated sizes (square
    matrices) can never silently accept a transposed arrangement."""
    roles = _canon_roles(roles)
    want = _canon_roles(want)
    present = [i for i in range(3) if want[i] is not None]
    if [i for i in range(3) if roles[i] is not None] != present:
        return None
    order = len(present)
    if node.order != order:
        return None
    perm = [0] * order  # perm[j] = decoded axis providing original axis j
    for i in present:
        perm[want[i]] = roles[i]
    if perm == list(range(order)):
        return node
    return builder.einsum(
        [node],
        [tuple(perm.index(a) for a in range(order))],
        tuple(range(order)),
        {j: node.dims[perm[j]] for j in range(order)},
    )
