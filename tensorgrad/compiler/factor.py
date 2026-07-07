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
from typing import Any, Sequence, cast

import opt_einsum as oe
import sympy

from tensorgrad.compiler import adjoint as _adj
from tensorgrad.utils import DisjointSets
from tensorgrad.compiler.adjoint import splice_child
from tensorgrad.compiler.ir import (
    Dim,
    Builder,
    ConstNode,
    EinsumNode,
    GatherNode,
    InputNode,
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
# Per-kernel dispatch cost in element-op units (~5us of eager launch overhead
# at ~1e10 elem-ops/s). Used in the whole-program score that picks the best
# sweep, and as the absolute hurdle a shared-child flatten must clear (see
# _flatten). It must NOT be added to the local rewrite scores as a per-node
# price: that *rewards* inlining shared children (fewer nodes), the exact
# duplication the flatten hurdle exists to prevent.
OP_OVERHEAD = 50_000.0
# Distribution must strictly beat keeping the Sum operand dense; hoisting
# accepts ties. This asymmetry (hysteresis) makes the rewrite pair acyclic.
DIST_MARGIN = 0.98
# (No margin lets a shared operand be recomputed inside a consumer anymore:
# shared children flatten only when inflated — see _flatten / _make_term.)
MAX_SWEEPS = 20  # sweeps early-exit at the fixpoint; deep module graphs need ~10
STALE_SWEEPS = 3  # stop after this many sweeps with no new best program score
MAX_DIST_TERMS = 8  # never distribute an einsum over a huge sum
MAX_FLAT_OPS = 10  # keep flattened einsums inside the DP planner's comfort zone

# ---- adjoint (reverse-mode) chain collapse -------------------------------
# Forward-mode-shaped Jacobian chains (gradients of deep parameters) are
# collapsed by the one-shot pre-pass in compiler/adjoint.py — that module's
# docstring is the design note, and it owns the inflation knob
# (INFLATE_MARGIN). The sweeps below deliberately do NOT attempt that
# collapse themselves: the local cost model cannot approve it (peeling ONE
# chain boundary improves the local score by <2%, measured 0.2% at the
# GPT-2 chain head, below DIST_MARGIN), and unconditional sweep overrides
# were measured to churn candidates for 100+ seconds per sweep on deep
# stacks. The sweeps' only adjoint-awareness is the _hoist guard: never
# re-extract an inflated operand out of a collapsed einsum.

_LETTERS = string.ascii_letters
_HUGE = 1e30


def factor_outputs(builder: Builder, outputs: list, dims: dict, collapse: bool = True) -> list:
    """Run the factoring passes over `outputs` = [(node, edge_order), ...].

    Returns a new outputs list with the same edge orders; `builder` is the
    hash-consing factory the nodes live in (new nodes are interned there).
    `dims` maps sympy symbols to concrete sizes (as in codegen.specialize).
    `collapse=False` skips the adjoint chain collapse: it is a ONE-SHOT
    pre-pass, and re-running it on an already-collapsed program re-splices
    shared cotangent segments into their consumers (measured on 3-layer
    gpt-nano: a second collapse minted ~900 new distinct values).
    """
    if not FACTOR:
        return list(outputs)
    # One-shot reverse-mode chain collapse FIRST (see adjoint.py): the cost
    # model below must never see forward-mode Jacobian nodes — scoring and
    # candidate-building around them is what made deep stacks intractable.
    if collapse:
        outputs = _adj.collapse_chains(builder, outputs, dims)
    rw = _Rewriter(builder, dims)
    nodes = [n for n, _ in outputs]
    # Baseline scale for the inflation test: the largest program input or
    # output. Anything INFLATE_MARGIN above this is a transient Jacobian.
    base = 1.0
    for nd in toposort(list(nodes)):
        if isinstance(nd, InputNode):
            base = max(base, rw._numel(nd.dims))
    for nd in nodes:
        base = max(base, rw._numel(nd.dims))
    rw._base = base
    # The local rules do not share one Lyapunov function: hoist/distribute
    # judge different nodes against a context that the other rule changes, so
    # sweeps can orbit instead of converging (measured on 3-layer GPT: the
    # program bottoms out mid-run, then climbs for the remaining sweeps).
    # Score every sweep's whole program and return the best one seen; stop
    # once the orbit has clearly stopped finding improvements.
    best_nodes, best_score = nodes, _program_score(rw, nodes)
    stale = 0
    for _ in range(MAX_SWEEPS):
        new_nodes = rw.sweep(nodes)
        if all(a is b for a, b in zip(new_nodes, nodes)):
            break
        nodes = new_nodes
        score = _program_score(rw, nodes)
        if score < best_score * (1.0 - 1e-9):
            best_nodes, best_score, stale = nodes, score, 0
        else:
            stale += 1
            if stale >= STALE_SWEEPS:
                break
    return [(n, order) for n, (_, order) in zip(best_nodes, outputs)]


def _program_score(rw: "_Rewriter", roots: list[Node]) -> float:
    """Total cost of the program under the sweeps' own model: einsum/linear
    nodes score as flops + memory writes (_node_score); map/reduce/gather
    nodes pay their output writes so rewrites cannot hide work in them."""
    total = 0.0
    for nd in toposort(list(roots)):
        s = rw._node_score(nd)
        if s == 0.0 and isinstance(nd, (MapNode, ReduceNode, GatherNode)):
            s = MEM_WEIGHT * rw._numel(nd.dims) + OP_OVERHEAD
        total += s
    return total


class _Rewriter:
    def __init__(self, builder: Builder, dims: dict):
        self.b = builder
        self._dims = dims
        self._dim_cache: dict = {}
        self._score_cache: dict = {}
        self.counts: dict[int, int] = {}
        # canonical-order einsum twin -> first node registered under it
        self._cse_seen: dict[int, Node] = {}
        # Largest program input/output numel (set by factor_outputs).
        self._base = 1.0

    def _inflated(self, dims: Sequence) -> bool:
        """A forward-mode-shaped transient: bigger than every program input
        and output by INFLATE_MARGIN (nothing that big enters or leaves)."""
        return self._numel(dims) > _adj.INFLATE_MARGIN * self._base

    # ---- infrastructure ---------------------------------------------------

    def dim_of(self, expr: Any) -> int:
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
            out = self._cse_einsum(nd)
            if out is not nd:
                return out
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
            out = self._sink_linear(nd)
            if out is not nd:
                return out
            return self._hoist(nd)
        if isinstance(nd, MapNode):
            out = self._normalize_map(nd)
            if out is not nd:
                return out
            return self._sink_map(nd)
        return nd

    def _normalize_map(self, nd: MapNode) -> Node:
        """Map(x, perm) -> view(Map(x)): the permutation moves out of the
        (possibly expensive) elementwise kernel into a free view einsum, and
        permuted duplicates of the same Map hash-cons into ONE kernel (a
        transposed tanh(x) next to tanh(x) was measured as a second full
        tanh pass in the gelu gradient)."""
        if len(nd.ops) != 1 or nd.perms[0] == tuple(range(nd.order)):
            return nd
        core = self.b.map(nd.op, nd.params, [nd.ops[0]])
        return self.b.einsum(
            [core],
            [tuple(range(nd.order))],
            tuple(nd.perms[0]),
            dict(enumerate(nd.ops[0].dims)),
        )

    def _cse_einsum(self, n: EinsumNode) -> Node:
        """Unify einsums that are structurally equal up to operand order.
        Hash-consing misses them (the intern key is order-sensitive), so ±
        duplicates of one contraction — ubiquitous in expanded second
        derivatives — never meet in Linear like-term merging.

        Deliberately CONSERVATIVE: an einsum is only ever replaced by a
        previously-seen equal node, never reordered in place. Reordering
        operands changes pairwise product grouping and hence fp rounding,
        which breaks the saturation-exact zeros the stabilization pass
        arranges (measured: 5e-3 error in the fp32 gelu gradient at |x|=200
        under blanket canonical sorting)."""
        if len(n.ops) < 2:
            return n
        order = sorted(
            range(len(n.ops)), key=lambda i: (self.b.node_index(n.ops[i]), n.in_subs[i])
        )
        # The interned canonical-order twin is only a lookup key; if it was
        # never registered as someone's canonical form, n stays untouched.
        canon = self.b.einsum(
            [n.ops[i] for i in order],
            [tuple(n.in_subs[i]) for i in order],
            n.out_subs,
            dict(enumerate(n.wire_dims)),
            n.weight,
            n.constraints,
        )
        seen = self._cse_seen.get(id(canon))
        if seen is None:
            self._cse_seen[id(canon)] = n
            return n
        return seen

    def _zero(self, dims: Sequence) -> Node:
        return self.b.const("zero", (), tuple(dims))

    # ---- cost model ---------------------------------------------------------

    def _numel(self, dims: Sequence) -> float:
        r = 1.0
        for d in dims:
            r *= self.dim_of(d)
        return r

    def _escore(self, in_subs: Sequence, out_subs: Sequence, wire_dims: Any) -> float:
        """Score of contracting operands (wire subscripts) down to out_subs,
        using the same planner codegen uses: flops + MEM_WEIGHT * writes."""
        subs = tuple(tuple(s) for s in in_subs)
        if not subs:
            return 0.0
        op_wires = set().union(*map(set, subs))
        out = tuple(w for w in dict.fromkeys(out_subs) if w in op_wires)
        letters: dict = {}

        def let(w: int) -> str:
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
                # The scoring cap stays at 12 even though collapsed adjoint
                # einsums run larger: _escore is called ~10^5 times per deep
                # program and DP at 13-16 clique-connected operands costs
                # ~ms each (measured 100s+ sweeps at 3 transformer layers).
                # Codegen's one-shot emission planner uses DP up to 16.
                if len(subs) <= 12:
                    _, info = oe.contract_path(
                        eq, *shapes, shapes=True, optimize=oe.DynamicProgramming(minimize="combo")
                    )
                elif len(subs) <= 16:
                    _, info = oe.contract_path(eq, *shapes, shapes=True, optimize="auto")
                else:
                    # 'greedy' above 16 operands: only the adjoint pass's
                    # collapsed chains reach this size, auto's branching
                    # search costs seconds per call at 20+ operands, and
                    # greedy follows a cotangent chain fine (the small end
                    # is always the locally cheapest contraction).
                    _, info = oe.contract_path(eq, *shapes, shapes=True, optimize="greedy")
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

        # Union-find over wire ids (utils.DisjointSets: union(x, y) keeps
        # y's root as representative).
        aliases: DisjointSets[int, Any] = DisjointSets()
        find = aliases.find

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
                if isinstance(op, LinearNode) and len(op.terms) == 1:
                    # single-term Linear: a scaled/permuted view — fold the
                    # scalar into the weight, the perm into the subs (kills
                    # materialized ±w*y copies feeding contractions).
                    (t,) = op.terms
                    (pm,) = op.perms
                    new_subs: list[Any] = [None] * t.order
                    for j in range(len(pm)):
                        new_subs[pm[j]] = subs[j]
                    weight = weight * sympy.sympify(op.weights[0])
                    kept.append((t, tuple(new_subs)))
                    changed = progress = True
                    continue
                if isinstance(op, ConstNode) and op.kind == "delta":
                    classes = {find(w) for w in subs}
                    out_occ = sum(1 for w in n.out_subs if find(w) in classes)
                    if out_occ <= 1:
                        # Alias all the delta's wires; the delta vanishes.
                        cs = sorted(classes)
                        for c in cs[1:]:
                            aliases.union(c, cs[0])
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
        wire_dims: dict[int, Dim] = {}
        for w, d in enumerate(n.wire_dims):
            wire_dims.setdefault(find(w), d)

        rows = []
        for coeffs, const in n.constraints:
            merged: dict[int, Any] = {}
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

    # Einsum-into-einsum splicing is shared with the adjoint pre-pass.
    _splice = staticmethod(splice_child)

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
            if sum(1 for o in n.ops if o is op) > 1:
                continue  # a power of op: flattening one occurrence splits
                # the two spellings apart (see the square guard in _distribute)
            cnt = self.counts.get(id(op), 2)
            if cnt != 1 and not self._inflated(op.dims) and self.ein_counts.get(id(op), 0) != cnt:
                # Some consumer keeps the child alive no matter what:
                # recomputing it here duplicates work the program still pays
                # for elsewhere. Never worth it for a non-inflated child.
                continue
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
            if cnt == 1:
                # The child dies with us: its cost is on the keep side, and a
                # tie is a win (one node fewer, planner sees more).
                ok = self._node_score(cand) <= parent + opscore
            elif self._inflated(op.dims):
                # Inflated shared child (a transient Jacobian we must not
                # materialize): every consumer judges with these same scores,
                # so if all flatten it dies — amortize its cost.
                ok = self._node_score(cand) <= parent + opscore / cnt
            else:
                # Shared child that dies if every einsum consumer flattens.
                # Amortized-cost ties are NOT enough here: dissolving a
                # shared value also destroys cross-chain sharing that the SZ
                # consolidation pass would otherwise harvest, a cost this
                # local model cannot see. Demand a clear absolute win (one
                # dispatch worth of element-ops) so only genuinely expensive
                # intermediates (the LayerNorm-style dense Jacobians) are
                # recomputed around. (Measured both ways on gpt-nano vs the
                # researcher gates: tie-flattening shredded the shared
                # softmax forwards into 54 gradient chains — 26x step time —
                # while forbidding shared flattening outright lost the
                # hand-fused-parity gate on the LayerNorm gradient.)
                ok = self._node_score(cand) <= parent + opscore / cnt - OP_OVERHEAD
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
            if sum(1 for op in n.ops if op is L) > 1:
                # L appears twice: the einsum is a POWER of L (elementwise
                # squares in optimizer algebra: v += (1-B2)*g*g). Distributing
                # one occurrence turns x^2 into cross terms — quadratic term
                # blowup AND it breaks the IEEE x*x >= 0 guarantee (measured:
                # AdamW second moments came out at -1e-12 for analytically
                # zero gradients, and sqrt(-eps) = NaN on the next step).
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

    def _make_term(self, n: EinsumNode, p: int, t: Node, subs_t: list, w: Any, cnt_L: int) -> tuple:
        """Build einsum(n with operand p replaced by Sum-term t). Returns
        (node, absorbed): `absorbed` when t was flattened into the einsum.
        Both the leaf and the flattened form are built; the score decides."""

        def build(splice: bool) -> Node:
            ops = [op for q, op in enumerate(n.ops) if q != p]
            in_subs = [tuple(s) for q, s in enumerate(n.in_subs) if q != p]
            wire_dims = dict(enumerate(n.wire_dims))
            weight = n.weight * sympy.sympify(w)
            rows = list(n.constraints)
            if splice:
                # (cast: build(True) is only reached when can_splice verified
                # that t is an EinsumNode)
                weight = weight * self._splice(ops, in_subs, wire_dims, rows, cast(EinsumNode, t), subs_t)
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
            # t survives elsewhere: recomputing it here duplicates work that
            # every other consumer still pays for and destroys value-sharing
            # (see _flatten). Only an inflated transient — something we must
            # not materialize anyway — may be spliced regardless.
            ok = self._inflated(t.dims) and s_flat <= s_leaf + self._node_score(t)
        return (flat, True) if ok else (leaf, False)

    # ---- pass 1: un-distribution (hoisting) -------------------------------------

    def _cleanup_linear(self, L: LinearNode) -> Node:
        """Normalize and shrink a LinearNode:
          - drop zero-weight / Zero terms;
          - absorb pure-view einsum terms (fold their permutation and scalar
            weight into the Linear's perm/weight — views are free, and seeing
            through them is what exposes like terms);
          - canonicalize pure-ones terms to a single weight-1 ones node;
          - flatten nested Linears: single-consumer ones unconditionally,
            shared ones when like-term merging keeps the term count from
            growing (this is the cancellation that collapses expanded ±tanh /
            ±softmax second-derivative algebra);
          - merge like terms (same node, same perm: weights add)."""
        m = len(L.dims)

        def absorb(t: Node, pm: tuple, w: Any) -> tuple:
            while (
                isinstance(t, EinsumNode)
                and len(t.ops) == 1
                and not t.constraints
                and len(t.out_subs) == len(t.in_subs[0]) == len(set(t.in_subs[0]))
                and set(t.out_subs) == set(t.in_subs[0])
            ):
                amap = [t.in_subs[0].index(w2) for w2 in t.out_subs]
                pm = tuple(amap[pm[j]] for j in range(m))
                w = w * sympy.sympify(t.weight)
                t = t.ops[0]
            if isinstance(t, EinsumNode) and not t.ops and not t.constraints and m > 0:
                # ones: constant under any perm — canonical output-order node
                w = w * sympy.sympify(t.weight)
                t = self.b.einsum([], [], tuple(range(m)), {j: L.dims[j] for j in range(m)})
                pm = tuple(range(m))
            return t, pm, w

        def unit_twin(t: Node) -> tuple:
            """The weight-1 twin of a weighted einsum (for weight-agnostic
            like-term keys). Codegen's step cache shares the contraction
            steps between the twin and the weighted original."""
            if not (isinstance(t, EinsumNode) and t.ops and sympy.sympify(t.weight) != 1):
                return t, sympy.Integer(1)
            return (
                self.b.einsum(
                    list(t.ops),
                    [tuple(s) for s in t.in_subs],
                    t.out_subs,
                    dict(enumerate(t.wire_dims)),
                    1,
                    t.constraints,
                ),
                sympy.sympify(t.weight),
            )

        def merged(ents: list) -> list:
            acc: dict = {}
            keys: list = []
            for t, pm, w in ents:
                key = (id(t), pm)
                if key in acc:
                    acc[key] = (t, pm, acc[key][2] + w)
                else:
                    acc[key] = (t, pm, w)
                    keys.append(key)
            return [
                acc[k]
                for k in keys
                if sympy.sympify(acc[k][2]) != 0
                and not (isinstance(acc[k][0], ConstNode) and acc[k][0].kind == "zero")
            ]

        entries = []
        changed = False
        for t, pm, w in zip(L.terms, L.perms, L.weights):
            w = sympy.sympify(w)
            t2, pm2, w2 = absorb(t, tuple(pm), w)
            changed |= t2 is not t or pm2 != tuple(pm) or w2 != w
            t, pm, w = t2, pm2, w2
            if w == 0 or (isinstance(t, ConstNode) and t.kind == "zero"):
                changed = True
                continue
            if isinstance(t, LinearNode) and self.counts.get(id(t), 2) == 1:
                for t3, pm3, w3 in zip(t.terms, t.perms, t.weights):
                    entries.append(
                        absorb(t3, tuple(pm3[pm[j]] for j in range(m)), w * sympy.sympify(w3))
                    )
                changed = True
                continue
            entries.append((t, pm, w))

        ents2 = merged(entries)
        changed |= len(ents2) != len(entries)
        entries = ents2

        # Shared nested Linears: flatten when the merged result is no larger
        # (equal is a wash per consumer; smaller means terms cancelled).
        progress = True
        while progress:
            progress = False
            for i, (t, pm, w) in enumerate(entries):
                if not isinstance(t, LinearNode):
                    continue
                flat = [
                    absorb(t3, tuple(pm3[pm[j]] for j in range(m)), w * sympy.sympify(w3))
                    for t3, pm3, w3 in zip(t.terms, t.perms, t.weights)
                ]
                cand = merged(entries[:i] + flat + entries[i + 1 :])
                if len(cand) <= len(entries):
                    entries = cand
                    changed = progress = True
                    break

        if not changed:
            return L
        if not entries:
            return self._zero(L.dims)
        return self.b.linear(*map(list, zip(*entries)))

    # ---- broadcast sinking -----------------------------------------------------

    def _broadcast_axes(self, t: Node) -> set:
        """Output axes of an EinsumNode along which the value is constant:
        their wires appear in no operand and no constraint row (codegen emits
        them as expand views)."""
        if not isinstance(t, EinsumNode) or len(set(t.out_subs)) != len(t.out_subs):
            return set()
        used = {w for s in t.in_subs for w in s}
        used |= {w for cs, _ in t.constraints for w, _ in cs}
        return {a for a, w in enumerate(t.out_subs) if w not in used}

    def _sink_map(self, nd: MapNode) -> Node:
        """Map(broadcast(x)) -> broadcast(Map(x)): elementwise ops commute
        with replication, so the op runs on the pre-expand core. This is what
        keeps LayerNorm's rsqrt(var + eps) at (B,) instead of (B,V)."""
        if len(nd.ops) != 1:
            return nd
        E = nd.ops[0]
        bax = self._broadcast_axes(E)
        if not bax:
            return nd
        assert isinstance(E, EinsumNode)  # nonempty _broadcast_axes implies EinsumNode
        keep = [a for a in range(E.order) if a not in bax]
        core_out = tuple(E.out_subs[a] for a in keep)
        core = self.b.einsum(
            list(E.ops),
            [tuple(s) for s in E.in_subs],
            core_out,
            dict(enumerate(E.wire_dims)),
            E.weight,
            E.constraints,
        )
        mp = self.b.map(nd.op, nd.params, [core])
        pm = nd.perms[0]
        out_subs = tuple(E.out_subs[pm[k]] for k in range(nd.order))
        return self.b.einsum([mp], [core_out], out_subs, dict(enumerate(E.wire_dims)))

    def _sink_linear(self, L: LinearNode) -> Node:
        """A Linear whose EVERY term is broadcast along a common set of output
        axes is itself a broadcast: compute the smaller core Linear and expand
        (the eps*ones + var/n pattern feeding pow in normalization layers)."""
        m = len(L.dims)
        if m == 0:
            return L
        baxes: set = set(range(m))
        for t, pm in zip(L.terms, L.perms):
            bt = self._broadcast_axes(t)
            baxes &= {j for j in range(m) if pm[j] in bt}
            if not baxes:
                return L
        keep_j = [j for j in range(m) if j not in baxes]
        new_terms, new_perms = [], []
        for t, pm in zip(L.terms, L.perms):
            # (cast: every term passed _broadcast_axes above, so it is an EinsumNode)
            t = cast(EinsumNode, t)
            drop_a = {pm[j] for j in baxes}
            keep_a = [a for a in range(t.order) if a not in drop_a]
            core = self.b.einsum(
                list(t.ops),
                [tuple(s) for s in t.in_subs],
                tuple(t.out_subs[a] for a in keep_a),
                dict(enumerate(t.wire_dims)),
                t.weight,
                t.constraints,
            )
            cidx = {a: i for i, a in enumerate(keep_a)}
            new_terms.append(core)
            new_perms.append(tuple(cidx[pm[j]] for j in keep_j))
        inner = self.b.linear(new_terms, new_perms, list(L.weights))
        return self.b.einsum(
            [inner], [tuple(keep_j)], tuple(range(m)), {j: L.dims[j] for j in range(m)}
        )

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
            if (
                isinstance(t, EinsumNode)
                and not t.constraints
                and len(set(t.out_subs)) == len(t.out_subs)
                and self.counts.get(id(t), 2) == 1
            ):
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

        def member_classes(sig: tuple, i: int, o: int) -> tuple:
            """(roles_w, class_wire) for occurrence (i, o) under signature sig."""
            _ops, in_subs, out_subs, _wdims, _ewt, pm, _w = views[i]
            out_wire = [out_subs[pm[j]] for j in range(m)]
            roles_w = out_wire + list(in_subs[o])
            class_wire: dict[int, int] = {}
            for r, c in enumerate(sig):
                class_wire.setdefault(c, roles_w[r])
            return roles_w, class_wire

        best = None
        for (nid, sig), members in groups.items():
            if len(members) < 2 or not sig:
                # Empty signature: an order-0 operand in an order-0 Linear —
                # the shared node is already computed once (hash-consing);
                # there are no wires to hoist through.
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
            # Never hoist a transient Jacobian back out of collapsed adjoint
            # einsums, and never create an inflated inner-Linear interface:
            # this is the hysteresis that keeps the adjoint overrides in
            # _flatten/_distribute (which ignore the shared cost model) from
            # oscillating with the tie-accepting hoist.
            if self._inflated(n.dims) or interface > _adj.INFLATE_MARGIN * self._base:
                continue

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
