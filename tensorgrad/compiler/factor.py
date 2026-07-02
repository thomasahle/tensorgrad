"""IR factoring passes: recover hand-factored derivative structure automatically.

Researchers compose functions from primitives (exp, pow, sum, ...); symbolic
differentiation plus expansion then yields sums of contractions that (a) fail
to share common factors across terms and (b) can fix enormous dense
intermediates into the einsum structure: a Sum appearing as an einsum operand
is materialized *before* the contraction is planned, so the planner never gets
a chance to avoid it (measured: the unexpanded MLP gradient bakes an
out_dim x out_dim x batch operand into the backward, 68GB at large dims).

This module rewrites the IR DAG between lowering and codegen, once per shape
specialization (the cost model wants concrete dims). Three local rewrites are
iterated to a fixpoint:

1. UN-DISTRIBUTE (on LinearNode):  sum_t w_t * einsum(A, B_t)  with a common
   operand A playing identical *roles* in every term (same axis-to-output
   wiring) becomes  einsum(A, sum_t w_t * B_t).  This is what recovers the
   shared softmax vector s = e/Z across expanded-softmax gradient terms and
   shrinks the CE-softmax Hessian toward its diag(s) - s s^T form.

2. DISTRIBUTE (on EinsumNode with a LinearNode operand):
   einsum(..., sum_t T_t) becomes sum_t einsum(..., T_t) when the contraction
   cost model (same opt_einsum machinery codegen plans with: flops plus a
   memory-write penalty) says the distributed form is cheaper. Distributed
   terms are flattened into the new einsums when that duplicates no work,
   which hands the intermediate-size decision back to the path planner.

3. TRIVIAL ABSORPTION (on EinsumNode): operands that are explicit delta
   constants alias their wires instead of materializing (Sum boundaries block
   lowering's delta elimination; after pass 2 the deltas surface here), pure
   `ones` operands fold into the scalar weight (summed-out ones contribute a
   factor of their dimension), zero operands annihilate the node. This keeps
   e.g. the LayerNorm centering Jacobian Delta_bb' Delta_ss' (Delta_dd' - J/d)
   from ever materializing a 6-axis dense tensor.

Every rewrite preserves semantics exactly (verified by randomized numeric
equivalence tests in tests/compiler/test_factor.py). The distribute /
un-distribute pair cannot oscillate: distribution requires a strictly better
score (DIST_MARGIN) while un-distribution accepts ties, and both judge with
the same cost function.
"""

import string

import opt_einsum as oe
import sympy

from tensorgrad.compiler.ir import (
    Builder,
    ConstNode,
    EinsumNode,
    GatherNode,
    LinearNode,
    MapNode,
    Node,
    ReduceNode,
    toposort,
)

# Master toggle so tests can compare factored vs unfactored programs.
FACTOR = True

# Cost model: score = flops + MEM_WEIGHT * elements written to intermediates.
MEM_WEIGHT = 4.0
# Distribution must strictly beat keeping the Sum operand dense; hoisting
# accepts ties. This asymmetry (hysteresis) makes the rewrite pair acyclic.
DIST_MARGIN = 0.98
# A shared einsum operand is only flattened (= recomputed inside a consumer)
# when the consumer gets strictly, clearly cheaper than reading it dense.
SHARED_FLAT_MARGIN = 0.5
MAX_SWEEPS = 20  # sweeps early-exit at the fixpoint; deep module graphs need ~10
MAX_DIST_TERMS = 8  # never distribute an einsum over a huge sum
MAX_FLAT_OPS = 10  # keep flattened einsums inside the DP planner's comfort zone

_LETTERS = string.ascii_letters
_HUGE = 1e30


def factor_outputs(builder: Builder, outputs, dims) -> list:
    """Run the factoring passes over `outputs` = [(node, edge_order), ...].

    Returns a new outputs list with the same edge orders; `builder` is the
    hash-consing factory the nodes live in (new nodes are interned there).
    `dims` maps sympy symbols to concrete sizes (as in codegen.specialize).
    """
    if not FACTOR:
        return list(outputs)
    rw = _Rewriter(builder, dims)
    nodes = [n for n, _ in outputs]
    for _ in range(MAX_SWEEPS):
        new_nodes = rw.sweep(nodes)
        if all(a is b for a, b in zip(new_nodes, nodes)):
            break
        nodes = new_nodes
    return [(n, order) for n, (_, order) in zip(nodes, outputs)]


class _Rewriter:
    def __init__(self, builder: Builder, dims: dict):
        self.b = builder
        self._dims = dims
        self._dim_cache: dict = {}
        self._score_cache: dict = {}
        self.counts: dict[int, int] = {}

    # ---- infrastructure ---------------------------------------------------

    def dim_of(self, expr) -> int:
        if (hit := self._dim_cache.get(expr)) is not None:
            return hit
        e = sympy.sympify(expr)
        v = e.subs(self._dims) if e.free_symbols else e
        r = int(v)
        self._dim_cache[expr] = r
        return r

    def sweep(self, roots: list[Node]) -> list[Node]:
        """One bottom-up rebuild of the DAG, applying local rules at each node."""
        order = toposort(list(roots))
        counts: dict[int, int] = {}
        ein_counts: dict[int, int] = {}
        for nd in order:
            for op in nd.operands():
                counts[id(op)] = counts.get(id(op), 0) + 1
                if isinstance(nd, EinsumNode):
                    ein_counts[id(op)] = ein_counts.get(id(op), 0) + 1
        for r in roots:
            counts[id(r)] = counts.get(id(r), 0) + 1
        self.counts = counts
        self.ein_counts = ein_counts

        memo: dict[int, Node] = {}
        for nd in order:
            new_ops = [memo[id(op)] for op in nd.operands()]
            cur = self._rebuild(nd, new_ops)
            for _ in range(16):
                # New nodes inherit the old node's consumer count (heuristic;
                # exact counts are recomputed at the start of every sweep).
                self.counts.setdefault(id(cur), self.counts.get(id(nd), 1))
                nxt = self._local(cur)
                if nxt is cur:
                    break
                cur = nxt
            self.counts.setdefault(id(cur), self.counts.get(id(nd), 1))
            memo[id(nd)] = cur
        return [memo[id(r)] for r in roots]

    def _rebuild(self, nd: Node, ops: list[Node]) -> Node:
        """Reconstruct `nd` with rewritten operands (hash-consing returns the
        identical node when nothing changed)."""
        if not ops:
            return nd
        if isinstance(nd, EinsumNode):
            return self.b.einsum(
                ops, list(nd.in_subs), nd.out_subs, dict(enumerate(nd.wire_dims)), nd.weight, nd.constraints
            )
        if isinstance(nd, LinearNode):
            return self.b.linear(ops, list(nd.perms), list(nd.weights))
        if isinstance(nd, MapNode):
            return self.b.map(nd.op, nd.params, ops, list(nd.perms))
        if isinstance(nd, GatherNode):
            if nd.op == "gather":
                return self.b.gather(ops[0], ops[1], nd.axis)
            return self.b.one_hot(ops[0], nd.dims[0])
        if isinstance(nd, ReduceNode):
            return self.b.reduce(nd.op, nd.axes, ops[0])
        return nd

    def _local(self, nd: Node) -> Node:
        if isinstance(nd, EinsumNode):
            out = self._absorb(nd)
            if out is not nd:
                return out
            out = self._flatten(nd)
            if out is not nd:
                return out
            return self._distribute(nd)
        if isinstance(nd, LinearNode):
            out = self._cleanup_linear(nd)
            if out is not nd:
                return out
            return self._hoist(nd)
        return nd

    def _zero(self, dims) -> Node:
        return self.b.const("zero", (), tuple(dims))

    # ---- cost model ---------------------------------------------------------

    def _numel(self, dims) -> float:
        r = 1.0
        for d in dims:
            r *= self.dim_of(d)
        return r

    def _escore(self, in_subs, out_subs, wire_dims) -> float:
        """Score of contracting operands (wire subscripts) down to out_subs,
        using the same planner codegen uses: flops + MEM_WEIGHT * writes."""
        subs = tuple(tuple(s) for s in in_subs)
        if not subs:
            return 0.0
        op_wires = set().union(*map(set, subs))
        out = tuple(w for w in dict.fromkeys(out_subs) if w in op_wires)
        letters: dict = {}

        def let(w):
            if w not in letters:
                letters[w] = _LETTERS[len(letters)]
            return letters[w]

        try:
            in_eqs = ["".join(let(w) for w in s) for s in subs]
            out_eq = "".join(let(w) for w in out)
        except IndexError:  # >52 distinct wires: refuse to score, block rewrites
            return _HUGE
        shapes = tuple(tuple(self.dim_of(wire_dims[w]) for w in s) for s in subs)
        key = (",".join(in_eqs) + "->" + out_eq, shapes)
        if (hit := self._score_cache.get(key)) is not None:
            return hit

        out_numel = 1.0
        for w in out:
            out_numel *= self.dim_of(wire_dims[w])
        if len(subs) == 1:
            if sorted(in_eqs[0]) == sorted(out_eq):
                score = 0.0  # identity or pure permutation: a view
            else:
                flops = 1.0
                for s in shapes[0]:
                    flops *= s
                score = flops + MEM_WEIGHT * out_numel
        else:
            eq = key[0]
            info = None
            try:
                if len(subs) <= 12:
                    _, info = oe.contract_path(
                        eq, *shapes, shapes=True, optimize=oe.DynamicProgramming(minimize="combo")
                    )
                else:
                    _, info = oe.contract_path(eq, *shapes, shapes=True, optimize="auto")
            except Exception:
                info = None
            if info is None:
                flops = float(len(subs))
                for w in letters:
                    flops *= self.dim_of(wire_dims[w])
                score = flops + MEM_WEIGHT * out_numel
            else:
                score = float(info.opt_cost) + MEM_WEIGHT * float(sum(info.size_list))
        self._score_cache[key] = score
        return score

    def _linear_score(self, nd: LinearNode) -> float:
        n = self._numel(nd.dims)
        return len(nd.terms) * n + MEM_WEIGHT * n

    def _node_score(self, nd: Node) -> float:
        if isinstance(nd, EinsumNode):
            return self._escore(nd.in_subs, nd.out_subs, nd.wire_dims)
        if isinstance(nd, LinearNode):
            return self._linear_score(nd)
        return 0.0

    # ---- pass 3: trivial absorption ------------------------------------------

    def _absorb(self, n: EinsumNode) -> Node:
        """Eliminate delta-constant operands (wire aliasing), fold `ones`
        operands into the weight (with the dimension factor for summed-out
        wires), and annihilate on zero operands."""
        if n.weight == 0 or any(isinstance(op, ConstNode) and op.kind == "zero" for op in n.ops):
            return self._zero(n.dims)

        parent: dict[int, int] = {}

        def find(w: int) -> int:
            parent.setdefault(w, w)
            while parent[w] != w:
                parent[w] = parent[parent[w]]
                w = parent[w]
            return w

        weight = n.weight
        changed = False
        items = list(zip(n.ops, n.in_subs))
        progress = True
        while progress:
            progress = False
            kept = []
            for op, subs in items:
                if isinstance(op, EinsumNode) and not op.ops and not op.constraints:
                    # ones (a pure broadcast einsum): fold into the weight.
                    weight = weight * op.weight
                    changed = progress = True
                    continue
                if isinstance(op, ConstNode) and op.kind == "delta":
                    classes = {find(w) for w in subs}
                    out_occ = sum(1 for w in n.out_subs if find(w) in classes)
                    if out_occ <= 1:
                        # Alias all the delta's wires; the delta vanishes.
                        cs = sorted(classes)
                        for c in cs[1:]:
                            parent[find(c)] = find(cs[0])
                        changed = progress = True
                        continue
                    # >= 2 output positions: a genuine diagonal embedding —
                    # the (small, hoisted) eye constant is the right emission.
                kept.append((op, subs))
            items = kept
        if not changed:
            return n

        new_subs = [tuple(find(w) for w in s) for _, s in items]
        new_out = tuple(find(w) for w in n.out_subs)
        wire_dims: dict[int, object] = {}
        for w, d in enumerate(n.wire_dims):
            wire_dims.setdefault(find(w), d)

        rows = []
        for coeffs, const in n.constraints:
            merged: dict[int, object] = {}
            for w, c in coeffs:
                r = find(w)
                merged[r] = sympy.sympify(merged.get(r, 0)) + sympy.sympify(c)
            merged = {w: c for w, c in merged.items() if c != 0}
            if not merged:
                if sympy.sympify(const) != 0:
                    return self._zero(n.dims)  # unsatisfiable indicator
                continue
            rows.append((tuple(merged.items()), const))

        # Orphaned wire classes (no operand, output or constraint touches
        # them) are free summations of 1: a factor of their size each.
        live = {w for s in new_subs for w in s} | set(new_out)
        for coeffs, _ in rows:
            live |= {w for w, _ in coeffs}
        for w in {find(w) for w in range(len(n.wire_dims))} - live:
            weight = weight * sympy.sympify(wire_dims[w])

        return self.b.einsum([op for op, _ in items], new_subs, new_out, wire_dims, weight, rows)

    # ---- einsum-into-einsum flattening ----------------------------------------

    @staticmethod
    def _splice(ops, in_subs, wire_dims, rows, child: EinsumNode, child_subs):
        """Append `child`'s operands/constraints to einsum parts under
        construction: child's out wires alias the parent wires `child_subs`,
        its internal wires become fresh parent wires. Returns child.weight."""
        base = (max(wire_dims) + 1) if wire_dims else 0
        wmap = {tw: child_subs[a] for a, tw in enumerate(child.out_subs)}
        inner = {w2 for s in child.in_subs for w2 in s}
        inner |= {w2 for cs, _ in child.constraints for w2, _ in cs}
        for tw in sorted(inner):
            if tw not in wmap:
                wmap[tw] = base
                wire_dims[base] = child.wire_dims[tw]
                base += 1
        for op, s in zip(child.ops, child.in_subs):
            ops.append(op)
            in_subs.append(tuple(wmap[w2] for w2 in s))
        for cs, const in child.constraints:
            rows.append((tuple((wmap[w2], c) for w2, c in cs), const))
        return child.weight

    def _flatten(self, n: EinsumNode) -> Node:
        """Fuse a single-consumer EinsumNode operand into its parent when the
        planner's score does not get worse. This undoes contraction boundaries
        that symbolic simplification froze into the tree (e.g. a 4-index
        Jacobian einsum feeding a reduction einsum), so the path planner —
        and the distribution pass — see the whole contraction."""
        for p, op in enumerate(n.ops):
            if not (isinstance(op, EinsumNode) and op.ops):
                continue
            if len(set(op.out_subs)) != len(op.out_subs):
                continue
            if len(n.ops) - 1 + len(op.ops) > MAX_FLAT_OPS:
                continue
            single = self.counts.get(id(op), 2) == 1
            ops = [o for q, o in enumerate(n.ops) if q != p]
            in_subs = [tuple(s) for q, s in enumerate(n.in_subs) if q != p]
            wire_dims = dict(enumerate(n.wire_dims))
            rows = list(n.constraints)
            w = self._splice(ops, in_subs, wire_dims, rows, op, n.in_subs[p])
            cand = self.b.einsum(ops, in_subs, n.out_subs, wire_dims, n.weight * w, rows)
            while isinstance(cand, EinsumNode):
                nxt = self._absorb(cand)
                if nxt is cand:
                    break
                cand = nxt
            parent = self._escore(n.in_subs, n.out_subs, n.wire_dims)
            opscore = self._node_score(op)
            cnt = self.counts.get(id(op), 2)
            if single:
                # The child dies with us: its cost is on the keep side, and a
                # tie is a win (one node fewer, planner sees more).
                ok = self._node_score(cand) <= parent + opscore
            elif self.ein_counts.get(id(op), 0) == cnt:
                # Every consumer is an einsum judging with these same scores:
                # if they all flatten, the child dies — amortize its cost.
                # (This is what dissolves a huge shared Jacobian intermediate
                # that every consumer is better off recomputing around.)
                ok = self._node_score(cand) <= parent + opscore / cnt
            else:
                # Some consumer keeps the child alive no matter what: the
                # flattened form must clearly beat reading it dense.
                ok = self._node_score(cand) <= SHARED_FLAT_MARGIN * parent
            if ok:
                return cand
        return n

    # ---- pass 2: distribution decision ----------------------------------------

    def _distribute(self, n: EinsumNode) -> Node:
        """einsum(..., Linear(T_1..T_k)) -> Linear(einsum(..., T_t)) when the
        cost model strictly approves. Terms are flattened into the new einsums
        when that duplicates no work, so the path planner (not the Sum
        boundary) decides the intermediates."""
        for p, L in enumerate(n.ops):
            if not isinstance(L, LinearNode) or len(L.terms) > MAX_DIST_TERMS:
                continue
            cnt_L = max(1, self.counts.get(id(L), 1))
            # The Linear's materialization — and that of terms consumed only
            # by it — is amortized over its consumers: if they all distribute
            # (they judge with the same scores), it dies, and each consumer
            # claims its share of the savings.
            keep = self._escore(n.in_subs, n.out_subs, n.wire_dims)
            keep += self._linear_score(L) / cnt_L
            private = {id(t) for t in L.terms if self.counts.get(id(t), 2) == 1}
            dist = self._numel(n.dims) * (len(L.terms) + MEM_WEIGHT)
            built = []
            ps = n.in_subs[p]
            for t, pm, w in zip(L.terms, L.perms, L.weights):
                subs_t = [0] * len(pm)
                for j, a in enumerate(pm):
                    subs_t[a] = ps[j]
                node_t, absorbed = self._make_term(n, p, t, subs_t, w, cnt_L)
                built.append(node_t)
                dist += self._node_score(node_t)
                if id(t) in private and not absorbed:
                    dist += self._node_score(t) / cnt_L
            for t in L.terms:
                if id(t) in private:
                    keep += self._node_score(t) / cnt_L
            if dist < DIST_MARGIN * keep:
                nodes = [nd for nd in built if not (isinstance(nd, ConstNode) and nd.kind == "zero")]
                if not nodes:
                    return self._zero(n.dims)
                m = n.order
                return self.b.linear(nodes, [tuple(range(m))] * len(nodes), [1] * len(nodes))
        return n

    def _make_term(self, n: EinsumNode, p: int, t: Node, subs_t: list, w, cnt_L: int):
        """Build einsum(n with operand p replaced by Sum-term t). Returns
        (node, absorbed): `absorbed` when t was flattened into the einsum.
        Both the leaf and the flattened form are built; the score decides."""

        def build(splice: bool):
            ops = [op for q, op in enumerate(n.ops) if q != p]
            in_subs = [tuple(s) for q, s in enumerate(n.in_subs) if q != p]
            wire_dims = dict(enumerate(n.wire_dims))
            weight = n.weight * sympy.sympify(w)
            rows = list(n.constraints)
            if splice:
                weight = weight * self._splice(ops, in_subs, wire_dims, rows, t, subs_t)
            else:
                ops.append(t)
                in_subs.append(tuple(subs_t))
            node = self.b.einsum(ops, in_subs, n.out_subs, wire_dims, weight, rows)
            while isinstance(node, EinsumNode):
                nxt = self._absorb(node)
                if nxt is node:
                    break
                node = nxt
            return node

        can_splice = (
            isinstance(t, EinsumNode)
            and t.ops
            and len(set(t.out_subs)) == len(t.out_subs)
            and len(n.ops) - 1 + len(t.ops) <= MAX_FLAT_OPS
        )
        leaf = build(False)
        if not can_splice:
            return leaf, False
        flat = build(True)
        s_flat, s_leaf = self._node_score(flat), self._node_score(leaf)
        if self.counts.get(id(t), 2) == 1:
            # t is consumed only through L: if every consumer splices, t dies —
            # its amortized cost is the tie-breaking bonus.
            ok = s_flat <= s_leaf + self._node_score(t) / cnt_L
        else:
            # t survives elsewhere: recomputing it here must clearly win.
            ok = s_flat <= SHARED_FLAT_MARGIN * s_leaf
        return (flat, True) if ok else (leaf, False)

    # ---- pass 1: un-distribution (hoisting) -------------------------------------

    def _cleanup_linear(self, L: LinearNode) -> Node:
        """Drop zero terms, flatten single-consumer nested Linears."""
        m = len(L.dims)
        terms, perms, ws = [], [], []
        changed = False
        for t, pm, w in zip(L.terms, L.perms, L.weights):
            if sympy.sympify(w) == 0 or (isinstance(t, ConstNode) and t.kind == "zero"):
                changed = True
                continue
            if isinstance(t, LinearNode) and self.counts.get(id(t), 2) == 1:
                for t2, pm2, w2 in zip(t.terms, t.perms, t.weights):
                    terms.append(t2)
                    perms.append(tuple(pm2[pm[j]] for j in range(m)))
                    ws.append(sympy.sympify(w) * sympy.sympify(w2))
                changed = True
                continue
            terms.append(t)
            perms.append(pm)
            ws.append(w)
        if not changed:
            return L
        if not terms:
            return self._zero(L.dims)
        return self.b.linear(terms, perms, ws)

    def _hoist(self, L: LinearNode) -> Node:
        """sum_t w_t einsum(A, B_t) -> einsum(A, sum_t w_t B_t) for the best
        common operand A whose axis roles agree across a group of terms."""
        m = len(L.dims)
        if len(L.terms) < 2:
            return L

        # A term view is (ops, in_subs, out_subs, wire_dims, ein_weight).
        # Terms that are not plain einsums (or are shared with other parts of
        # the DAG, or carry constraints) participate atomically: the whole
        # term is its own single operand.
        views = []
        for t, pm, w in zip(L.terms, L.perms, L.weights):
            plain = (
                isinstance(t, EinsumNode)
                and not t.constraints
                and len(set(t.out_subs)) == len(t.out_subs)
                and self.counts.get(id(t), 2) == 1
            )
            if plain:
                views.append(
                    (list(t.ops), [tuple(s) for s in t.in_subs], tuple(t.out_subs),
                     dict(enumerate(t.wire_dims)), t.weight, pm, w)
                )
            else:
                idsubs = tuple(range(m))
                views.append(
                    ([t], [idsubs], idsubs, {a: t.dims[a] for a in range(m)},
                     sympy.Integer(1), pm, w)
                )

        # Group candidate occurrences by (operand identity, role signature).
        # Roles: the L output axes followed by the candidate's axes; the
        # signature is the partition of roles into shared wires.
        groups: dict[tuple, list] = {}
        for i, (ops, in_subs, out_subs, wdims, _ewt, pm, _w) in enumerate(views):
            out_wire = [out_subs[pm[j]] for j in range(m)]
            seen_here: set = set()
            for o, opn in enumerate(ops):
                if isinstance(opn, ConstNode) or (isinstance(opn, EinsumNode) and not opn.ops):
                    continue  # hoisting a constant/ones saves nothing
                roles = out_wire + list(in_subs[o])
                lab: dict = {}
                sig = tuple(lab.setdefault(w2, len(lab)) for w2 in roles)
                key = (id(opn), sig)
                if key in seen_here:
                    continue  # one occurrence per term per signature
                seen_here.add(key)
                groups.setdefault(key, []).append((i, o, opn))

        def member_classes(sig, i, o):
            """(roles_w, class_wire) for occurrence (i, o) under signature sig."""
            _ops, in_subs, out_subs, _wdims, _ewt, pm, _w = views[i]
            out_wire = [out_subs[pm[j]] for j in range(m)]
            roles_w = out_wire + list(in_subs[o])
            class_wire = {}
            for r, c in enumerate(sig):
                class_wire.setdefault(c, roles_w[r])
            return roles_w, class_wire

        best = None
        for (nid, sig), members in groups.items():
            if len(members) < 2:
                continue
            n = members[0][2]
            C = max(sig) + 1
            class_dim = {}
            for r, c in enumerate(sig):
                if c not in class_dim:
                    class_dim[c] = L.dims[r] if r < m else n.dims[r - m]
            # The inner Linear only carries classes some Rest actually
            # touches; classes seen only by A (or by nobody: broadcast) are
            # wired directly through the outer einsum.
            live: set = set()
            for i, o, _ in members:
                _, in_subs, _, _, _, _, _ = views[i]
                rest_wires = {w2 for q, s in enumerate(in_subs) if q != o for w2 in s}
                roles_w, _ = member_classes(sig, i, o)
                live |= {c for r, c in enumerate(sig) if roles_w[r] in rest_wires}
            live_order = sorted(live)
            interface = self._numel([class_dim[c] for c in live_order])
            numel_L = self._numel(L.dims)

            keep = 0.0
            hoist = 0.0
            for i, o, _ in members:
                ops, in_subs, out_subs, wdims, _ewt, pm, _w = views[i]
                keep += self._escore(in_subs, out_subs, wdims) + numel_L
                _, class_wire = member_classes(sig, i, o)
                rest_subs = [s for q, s in enumerate(in_subs) if q != o]
                rest_out = tuple(class_wire[c] for c in live_order)
                hoist += self._escore(rest_subs, rest_out, wdims) + interface
            outer_in = [tuple(sig[m:]), tuple(live_order)]
            outer_out = tuple(sig[:m])
            outer_wd = {c: class_dim[c] for c in range(C)}
            hoist += self._escore(outer_in, outer_out, outer_wd)
            gain = keep - hoist
            if gain >= 0 and (best is None or gain > best[0]):
                best = (gain, sig, members, n, class_dim, live_order)
        if best is None:
            return L

        _gain, sig, members, n, class_dim, live_order = best
        member_ids = {i for i, _, _ in members}
        rest_nodes, rest_perms, rest_ws = [], [], []
        for i, o, _ in members:
            ops, in_subs, out_subs, wdims, ewt, pm, w = views[i]
            _, class_wire = member_classes(sig, i, o)
            rest_ops = [op for q, op in enumerate(ops) if q != o]
            rest_subs = [s for q, s in enumerate(in_subs) if q != o]
            rest_out = tuple(class_wire[c] for c in live_order)
            if (
                len(rest_ops) == 1
                and ewt == 1
                and len(set(rest_subs[0])) == len(rest_subs[0])
                and sorted(rest_subs[0]) == sorted(rest_out)
            ):
                # Pure permutation: feed the operand straight into the Linear
                # and let its perm do the transpose (a view, not a copy).
                rest_nodes.append(rest_ops[0])
                rest_perms.append(tuple(rest_subs[0].index(w2) for w2 in rest_out))
                rest_ws.append(sympy.sympify(w))
            else:
                rest_nodes.append(self.b.einsum(rest_ops, rest_subs, rest_out, wdims, ewt))
                rest_perms.append(tuple(range(len(live_order))))
                rest_ws.append(sympy.sympify(w))
        inner = self.b.linear(rest_nodes, rest_perms, rest_ws)
        outer = self.b.einsum(
            [n, inner],
            [tuple(sig[m:]), tuple(live_order)],
            tuple(sig[:m]),
            {c: class_dim[c] for c in range(max(sig) + 1)},
        )

        left = [
            (t, pm, w)
            for i, (t, pm, w) in enumerate(zip(L.terms, L.perms, L.weights))
            if i not in member_ids
        ]
        terms = [outer] + [t for t, _, _ in left]
        perms = [tuple(range(m))] + [pm for _, pm, _ in left]
        ws = [1] + [w for _, _, w in left]
        return self.b.linear(terms, perms, ws)
