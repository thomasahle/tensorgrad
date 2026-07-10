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
    def Tr(x: T) -> T: ...  # type: ignore[empty-body]
    @function
    def Scale(num: i64Like, den: i64Like, x: T) -> T: ...  # type: ignore[empty-body]
    @function
    def rows(x: T) -> i64: ...  # type: ignore[empty-body]
    @function
    def cols(x: T) -> i64: ...  # type: ignore[empty-body]

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
        # ---- costs (mirror factor._program_score's flops + MEM model) ----
        yield rule(
            eq(c).to(a @ b), eq(rows(a)).to(r), eq(cols(a)).to(k), eq(cols(b)).to(n)
        ).then(set_cost(a @ b, r * k * n + MEM_WEIGHT * r * n))
        yield rule(eq(c).to(a + b), eq(rows(a)).to(r), eq(cols(a)).to(n)).then(
            set_cost(a + b, (2 + MEM_WEIGHT) * r * n)
        )
        yield rule(eq(c).to(Scale(p, q, a)), eq(rows(a)).to(r), eq(cols(a)).to(n)).then(
            set_cost(Scale(p, q, a), (1 + MEM_WEIGHT) * r * n)
        )
        yield rule(eq(c).to(Tr(a))).then(set_cost(Tr(a), 0))
        yield rule(eq(c).to(Leaf(p))).then(set_cost(Leaf(p), 0))
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
        # ---- transpose algebra (free views) ----
        yield rewrite(Tr(Tr(a))).to(a)
        yield rewrite(Tr(a @ b)).to(Tr(b) @ Tr(a))
        yield rewrite(Tr(b) @ Tr(a)).to(Tr(a @ b))
        yield rewrite(Tr(a) + Tr(b)).to(Tr(a + b))
        # ---- scale movement (weights ride to where they fold away) ----
        yield rewrite(Scale(p, q, a) @ b).to(Scale(p, q, a @ b))
        yield rewrite(a @ Scale(p, q, b)).to(Scale(p, q, a @ b))
        yield rewrite(Scale(p, q, Scale(s, t, a))).to(Scale(p * s, q * t, a))
        yield rewrite(Scale(p, q, a) + Scale(p, q, b)).to(Scale(p, q, a + b))
        yield rewrite(Tr(Scale(p, q, a))).to(Scale(p, q, Tr(a)))

    _LANG.update(T=T, Leaf=Leaf, Tr=Tr, Scale=Scale, rows=rows, cols=cols, rules=rules)
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
    def __init__(self, lang: dict[str, Any], dims: dict):
        self.lang = lang
        self.dims = dims
        self.leaves: list[Node] = []
        self.leaf_ids: dict[int, int] = {}
        self.memo: dict[int, Any] = {}
        self.composites = 0

    def _dim(self, d: Any) -> int:
        e = sympy.sympify(d)
        v = e.subs(self.dims) if e.free_symbols else e
        if not v.is_Integer:
            raise ValueError(f"dim {d} unresolved")
        return int(v)

    def leaf(self, node: Node) -> Any:
        i = self.leaf_ids.get(id(node))
        if i is None:
            i = len(self.leaves)
            self.leaves.append(node)
            self.leaf_ids[id(node)] = i
        return self.lang["Leaf"](i)

    def leaf_shape(self, node: Node) -> tuple[int, int]:
        """(rows, cols) in the MNF convention: order-2 = (d0, d1); order-1 =
        column (d0, 1); order-0 = (1, 1); order>2 = (numel, 1) — such leaves
        never enter matmuls (the recognizer rejects order>2 operands) but may
        ride in Add chains, where only the element count matters."""
        ds = [self._dim(d) for d in node.dims]
        if len(ds) == 2:
            return ds[0], ds[1]
        if len(ds) == 1:
            return ds[0], 1
        if not ds:
            return 1, 1
        n = 1
        for d in ds:
            n *= d
        return n, 1

    def encode(self, node: Node) -> Any:
        """Egglog term for `node`, treating unrecognized structure as leaves.
        Every RECOGNIZED composite increments self.composites (coverage)."""
        hit = self.memo.get(id(node))
        if hit is not None:
            return hit
        term = self._encode_matmul(node) if isinstance(node, EinsumNode) else None
        if term is None and isinstance(node, LinearNode):
            term = self._encode_linear(node)
        if term is None:
            term = self.leaf(node)
        else:
            self.composites += 1
        self.memo[id(node)] = term
        return term

    def _encode_matmul(self, n: EinsumNode) -> Optional[Any]:
        """Path-shaped einsums (n-ary chains included: lowering flattens
        whole @-chains into ONE einsum) decompose into binary MNF hops; the
        rules then own the reassociation. Anything non-path (stars, cycles,
        hyperedge wires, diagonals, constraints) stays a leaf."""
        if len(n.ops) < 2 or n.constraints:
            return None
        w = _as_rational(n.weight)
        if w is None:
            return None
        for subs, op in zip(n.in_subs, n.ops):
            if len(set(subs)) != len(subs) or len(subs) not in (1, 2):
                return None
        if len(set(n.out_subs)) != len(n.out_subs) or len(n.out_subs) > 2:
            return None
        # wire degrees over operands
        deg: dict[int, int] = {}
        for subs in n.in_subs:
            for wr in subs:
                deg[wr] = deg.get(wr, 0) + 1
        if any(d > 2 for d in deg.values()):
            return None  # hyperedge wire
        outw = set(n.out_subs)
        for wr, d in deg.items():
            if (d == 1) != (wr in outw):
                return None  # free wires must surface; shared wires must not
        if any(wr not in deg for wr in outw):
            return None  # broadcast-only output wire
        # operand adjacency via shared wires: must form one simple path
        m = len(n.ops)
        adj: dict[int, list[int]] = {i: [] for i in range(m)}
        holders: dict[int, list[int]] = {}
        for i, subs in enumerate(n.in_subs):
            for wr in subs:
                holders.setdefault(wr, []).append(i)
        for wr, hs in holders.items():
            if len(hs) == 2:
                a, b = hs
                if a == b:
                    return None  # self-contraction (trace)
                adj[a].append(b)
                adj[b].append(a)
        ends = [i for i in range(m) if len(adj[i]) == 1]
        if any(len(adj[i]) not in (1, 2) for i in range(m)) or len(ends) != 2:
            return None  # not a single simple path
        # walk the path from one end
        order = [ends[0]]
        prev = -1
        while len(order) < m:
            nxt = [j for j in adj[order[-1]] if j != prev]
            if len(nxt) != 1:
                return None
            prev = order[-1]
            order.append(nxt[0])
        if len(set(order)) != m:
            return None  # cycle
        # contracted wire per hop
        def hop_wire(i: int, j: int) -> int:
            (wr,) = set(n.in_subs[i]) & set(n.in_subs[j])
            return wr

        term = None
        for pos in range(m - 1):
            i, j = order[pos], order[pos + 1]
            kw = hop_wire(i, j)
            if term is None:
                left = self._oriented(n.ops[i], n.in_subs[i], kw, want_role_first=True)
                if left is None:
                    return None
                term = left
            right = self._oriented(n.ops[j], n.in_subs[j], kw, want_role_first=False)
            if right is None:
                return None
            term = term @ right
        # output order: MNF yields (first end's free wire, last end's free wire)
        free_first = [wr for wr in n.in_subs[order[0]] if deg[wr] == 1]
        free_last = [wr for wr in n.in_subs[order[-1]] if deg[wr] == 1]
        mnf_out = free_first + free_last
        out = list(n.out_subs)
        if out != mnf_out:
            if len(out) == 2 and out == list(reversed(mnf_out)):
                term = self.lang["Tr"](term)
            else:
                return None
        if w != 1:
            term = self.lang["Scale"](w.numerator, w.denominator, term)
        return term

    def _oriented(self, op: Node, subs: tuple, kw: int, want_role_first: bool) -> Optional[Any]:
        """Encode operand `op` oriented for MNF: left operands want (row, k),
        right operands want (k, col). A leaf in the wrong axis order is
        wrapped in Tr (free); vectors orient by construction."""
        t = self.encode(op)
        if len(subs) == 1:
            # pure-k vector: row vector on the left (Tr of the canonical
            # column), plain column on the right
            return self.lang["Tr"](t) if want_role_first else t
        k_pos = subs.index(kw)
        good = (k_pos == 1) if want_role_first else (k_pos == 0)
        return t if good else self.lang["Tr"](t)

    def _encode_linear(self, n: LinearNode) -> Optional[Any]:
        idperm = tuple(range(len(n.dims)))
        term_out = None
        for t, pm, w in zip(n.terms, n.perms, n.weights):
            wr = _as_rational(w)
            if wr is None:
                return None
            enc = self.encode(t)
            if tuple(pm) == (1, 0):
                enc = self.lang["Tr"](enc)  # transposed term: free bookkeeping
            elif tuple(pm) != idperm:
                return None
            if wr != 1:
                enc = self.lang["Scale"](wr.numerator, wr.denominator, enc)
            term_out = enc if term_out is None else term_out + enc
        return term_out


# --------------------------------------------------------------------------
# Decode: extracted egglog term -> IR nodes via the Builder (inheriting its
# smart-constructor algebra). Values are (node, axes) with axes = (row_axis,
# col_axis) into node.dims, None for absent roles.
# --------------------------------------------------------------------------


class _Decoder:
    def __init__(self, builder: Builder, leaves: list[Node]):
        self.b = builder
        self.leaves = leaves

    def decode(self, expr: Any) -> tuple[Node, tuple, Fraction]:
        """Returns (node, axes, scale): the IR node, the MNF role->axis map,
        and a pending rational scale (folded into the enclosing einsum or
        linear rather than materialized)."""
        from egglog import expr_parts

        return self._walk(expr_parts(expr).expr)

    def _name(self, call: Any) -> str:
        c = call.callable
        n = getattr(c, "method_name", None) or getattr(getattr(c, "ident", None), "name", None)
        if n is None:
            n = str(c)
        return n

    def _walk(self, call: Any) -> tuple[Node, tuple, Fraction]:
        name = self._name(call)
        args = call.args
        if name == "Leaf":
            node = self.leaves[args[0].expr.value]
            axes = (0, 1) if node.order == 2 else ((0, None) if node.order == 1 else (None, None))
            if node.order > 2:
                axes = (None, None)  # rides in Adds only; roles unused
            return node, axes, Fraction(1)
        if name == "Tr":
            node, axes, s = self._walk(args[0].expr)
            return node, (axes[1], axes[0]), s
        if name == "Scale":
            p = args[0].expr.value
            q = args[1].expr.value
            node, axes, s = self._walk(args[2].expr)
            return node, axes, s * Fraction(p, q)
        if name == "__matmul__":
            return self._matmul(args[0].expr, args[1].expr)
        if name == "__add__":
            return self._add(call)
        raise ValueError(f"egraph decode: unknown constructor {name}")

    def _matmul(self, ca: Any, cb: Any) -> tuple[Node, tuple, Fraction]:
        A, aax, sa = self._walk(ca)
        B, bax, sb = self._walk(cb)
        ra, ka = aax  # row role axis, contract role axis (A's col IS k)
        kb, cbx = bax
        if ka is None or kb is None:
            raise ValueError("matmul on scalar-role operand")
        wires_a: list[Optional[int]] = [None] * A.order
        wires_b: list[Optional[int]] = [None] * B.order
        wires_a[ka] = 1
        wires_b[kb] = 1
        out_subs = []
        out_axes: list[Optional[int]] = [None, None]
        if ra is not None:
            wires_a[ra] = 0
            out_subs.append(0)
        if cbx is not None:
            wires_b[cbx] = 2
            out_subs.append(2)
        wire_dims = {1: A.dims[ka]}
        if ra is not None:
            wire_dims[0] = A.dims[ra]
        if cbx is not None:
            wire_dims[2] = B.dims[cbx]
        node = self.b.einsum(
            [A, B],
            [tuple(w for w in wires_a if w is not None), tuple(w for w in wires_b if w is not None)],
            tuple(out_subs),
            wire_dims,
            sympy.Rational((sa * sb).numerator, (sa * sb).denominator),
        )
        n_out = len(out_subs)
        axes = (0, 1) if n_out == 2 else ((0, None) if ra is not None else (None, 0) if cbx is not None else (None, None))
        return node, axes, Fraction(1)

    def _add(self, call: Any) -> tuple[Node, tuple, Fraction]:
        """Flatten the whole +-chain and rebuild ONE n-ary LinearNode (the
        IR-canonical form, avoiding the binary-chain cost skew)."""
        terms: list[tuple[Node, tuple, Fraction]] = []

        def flat(c: Any) -> None:
            if self._name(c) == "__add__":
                flat(c.args[0].expr)
                flat(c.args[1].expr)
            else:
                terms.append(self._walk(c))
        flat(call)
        base_node, base_axes, _ = terms[0]
        base_order = base_node.order
        base_roles = [a for a in base_axes if a is not None]
        nodes, perms, weights = [], [], []
        for node, axes, s in terms:
            if node.order != base_order:
                raise ValueError("add of mismatched orders")
            # perm[j] = term axis providing output axis j; output axis order
            # follows the FIRST term's own axis order.
            if base_order == 2:
                # base output axes in base's natural order; map through roles
                role_of_base = {base_axes[0]: 0, base_axes[1]: 1}  # axis -> role
                pm = [0, 1]
                for out_pos, base_axis in enumerate(sorted(role_of_base)):
                    role = role_of_base[base_axis]
                    pm[out_pos] = axes[role] if axes[role] is not None else out_pos
                perms.append(tuple(pm))
            else:
                perms.append(tuple(range(base_order)))
            nodes.append(node)
            weights.append(sympy.Rational(s.numerator, s.denominator))
        out = self.b.linear(nodes, perms, weights)
        # output axes follow the base term's natural (sorted-axis) order
        if base_order == 2:
            first, second = sorted([base_axes[0], base_axes[1]])
            role_first = 0 if base_axes[0] == first else 1
            axes = (0, 1) if role_first == 0 else (1, 0)
        elif base_order == 1:
            axes = ((0, None) if base_axes[0] is not None else (None, 0))
        else:
            axes = (None, None)
        return out, axes, Fraction(1)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def _saturate_extract(builder: Builder, outputs: list, dims: dict) -> Optional[list]:
    """Encode -> saturate -> extract -> decode. Returns None for 'no change'
    (nothing recognized, caps hit, or extraction reproduced the input)."""
    from egglog import EGraph, set_cost as _set_cost  # noqa: F401

    lang = _lang()
    enc = _Encoder(lang, dims)
    roots = []
    for node, order in outputs:
        roots.append(enc.encode(node))
    if enc.composites == 0 or enc.composites > MAX_ENCODED:
        return None

    egraph = EGraph()
    let_roots = [egraph.let(f"r{i}", t) for i, t in enumerate(roots)]
    # seed leaf shapes and zero leaf costs
    from egglog import set_, i64  # noqa: F401

    for i, node in enumerate(enc.leaves):
        r, c = enc.leaf_shape(node)
        leaf = lang["Leaf"](i)
        egraph.register(set_(lang["rows"](leaf)).to(i64(r)))
        egraph.register(set_(lang["cols"](leaf)).to(i64(c)))
    egraph.run(lang["rules"].saturate())

    dec = _Decoder(builder, enc.leaves)
    new_outputs = []
    changed = False
    for (node, order), lroot in zip(outputs, let_roots):
        best = egraph.extract(lroot)
        out_node, axes, scale = dec._walk(_expr_decl(best))
        if scale != 1:
            out_node = builder.linear(
                [out_node], [tuple(range(out_node.order))],
                [sympy.Rational(scale.numerator, scale.denominator)],
            )
        out_node = _restore_axis_order(builder, out_node, axes, node)
        if tuple(out_node.dims) != tuple(node.dims):
            return None  # decode must reproduce dims exactly
        changed |= out_node is not node
        new_outputs.append((out_node, order))
    return new_outputs if changed else None


def _expr_decl(expr: Any) -> Any:
    from egglog import expr_parts

    return expr_parts(expr).expr


def _restore_axis_order(builder: Builder, node: Node, axes: tuple, original: Node) -> Node:
    """The decoded node's axis order may be role-permuted relative to the
    original root; restore with a free view einsum when needed."""
    if node.order != original.order or node.order != 2:
        return node
    if tuple(node.dims) == tuple(original.dims) and axes == (0, 1):
        return node
    if axes == (1, 0):
        return builder.einsum(
            [node], [(1, 0)], (0, 1), {0: node.dims[1], 1: node.dims[0]}
        )
    return node
