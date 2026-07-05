"""Adjoint chain collapse: recover reverse-mode contraction order (#29).

Symbolic gradients of parameters deep under a stack of nonlinearities lower
into FORWARD-MODE-shaped Jacobian chains: einsums separated by LinearNodes
(residual adds, LayerNorm centering, softmax-Jacobian sums) whose node
outputs drag the wrt-parameter's axes through every layer above. At GPT-2
block dims (d=768, seq=256, batch=8) the raw lowered gradient program
totals ~400 TB with a 59 PB rank-5 node; a 3-layer gpt-nano lowers 1110
nodes above 200 MB. The compact reverse-mode order — contract the small
cotangent end first and push it through the chain — is a legal einsum
reassociation through delta wires (the chain alternates linear maps with
diagonal, Hadamard-joined factors).

Why a dedicated pass: factor.py's cost-model rewriter cannot find this
order. Collapsing ONE Linear boundary leaves the next huge node read dense,
so each local step improves the score by <2% (measured 0.2% at the GPT-2
chain head) — a valley whose payoff only appears when the whole chain is
gone. Unconditional sweep overrides cross the valley but churn (every
candidate interned and scored per sweep: 100+ second sweeps and gigabytes
of candidates at 3 layers), and naive full expansion of a head into
monomials is EXPONENTIAL in the boundary count (each residual add, LN
centering and softmax-Jacobian sum forks every partial). This pass instead
mirrors reverse-mode autodiff itself, iterating two structural,
scoring-free rules to a fixpoint over the DAG:

  R (cotangent step) at a small-output einsum reading an inflated operand
    X: contract everything EXCEPT X down to the interface it shares with X
    — an explicit hash-consed node u, the cotangent — then dissolve one
    boundary of X: splice (X einsum) or distribute (X Linear) einsum(u, X).
    Compression keeps every op list tiny, and because u is a NODE, all
    branches and all heads that reach the same context share it.

  M (cotangent merge) at a LinearNode whose terms are einsums reading the
    SAME inflated operand X in the same role: Σ_i einsum(u_i, X) becomes
    einsum(Σ_i u_i, X). This is u = Σ contributions in reverse-mode; it is
    what stops a softmax Jacobian's diag(s) - s sᵀ fork (and every residual
    fork above it) from doubling the downstream work per level.

Each fixpoint iteration applies one boundary step per chain, so the
iteration count is the chain depth (bounded), and the DAG never holds
scored candidates — the pass is pure structure. The general factoring
sweeps then run on an activation-sized DAG.

Inflation test: a node is a transient Jacobian when its numel exceeds every
program input and output by INFLATE_MARGIN — nothing that big enters or
leaves the program, so materializing it is memory-dominated no matter what
the flop model says. Heads keep NON-inflated operands as shared leaves:
those are exactly the forward activations and the shared cotangent pieces,
so the collapse terminates at reverse mode's natural memory frontier.

Every rewrite is einsum reassociation / distribution over a sum —
semantics-preserving, verified by Schwartz-Zippel fingerprints and against
torch.autograd in tests/compiler/test_reverse_mode.py.
"""

import sympy

from tensorgrad.compiler.ir import (
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

# A node is INFLATED when its numel exceeds every program input/output by
# this margin. 32 clears the largest legitimate forward intermediates
# (attention scores, MLP hidden activations) by >2x on the models measured.
INFLATE_MARGIN = 32.0
# Caps on any einsum this pass builds; hitting one freezes the operand (it
# stays dense) — a fallback, never an error. Codegen's einsum emitter is
# limited to 52 distinct wires.
MAX_OPS = 24
MAX_WIRES = 40
# Fixpoint iterations: one chain boundary dissolves per iteration, so this
# bounds the collapsible chain depth (a transformer block is ~8 boundaries).
MAX_ROUNDS = 64


def splice_child(ops, in_subs, wire_dims, rows, child: EinsumNode, child_subs):
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


class _Collapser:
    def __init__(self, builder: Builder, dims: dict):
        self.b = builder
        self._dims = dims
        self._numel_cache: dict = {}
        self.base = 1.0
        self.changed = False

    def _numel(self, dims_tuple) -> float:
        r = 1.0
        for d in dims_tuple:
            if (hit := self._numel_cache.get(d)) is None:
                e = sympy.sympify(d)
                hit = int(e.subs(self._dims) if e.free_symbols else e)
                self._numel_cache[d] = hit
            r *= hit
        return r

    def inflated(self, node: Node) -> bool:
        return self._numel(node.dims) > INFLATE_MARGIN * self.base

    def _winflated(self, wires, wire_dims) -> bool:
        return self._numel([wire_dims[w] for w in wires]) > INFLATE_MARGIN * self.base

    # ---- rule R: one cotangent step at a head einsum -----------------------

    def _pick_boundary(self, n: EinsumNode):
        """The operand to dissolve: an inflated Linear/Einsum (einsums must
        have distinct out wires; ones-einsums are absorption's job)."""
        for i, op in enumerate(n.ops):
            if not self.inflated(op):
                continue
            if isinstance(op, LinearNode):
                return i
            if isinstance(op, EinsumNode) and op.ops and len(set(op.out_subs)) == len(op.out_subs):
                return i
        return None

    def _compress(self, n: EinsumNode, i: int):
        """Contract every operand except `i` down to the wires they share
        with operand i or with the output — the cotangent u. Returns
        (u_node, u_subs) or None (interface inflated / nothing to gain).
        Leaves operands that share no wire with u's interface... they are
        included too (they only scale/broadcast; u absorbs them)."""
        rest = [(op, subs) for q, (op, subs) in enumerate(zip(n.ops, n.in_subs)) if q != i]
        if not rest:
            return ()  # nothing to compress: dissolve X without a cotangent
        if len(rest) == 1 and not n.constraints:
            return rest[0]
        x_wires = set(n.in_subs[i])
        rest_wires = {w for _, s in rest for w in s}
        row_wires = {w for cs, _ in n.constraints for w, _ in cs}
        keep = sorted(rest_wires & (x_wires | set(n.out_subs) | row_wires))
        if self._winflated(keep, n.wire_dims):
            return None
        u = self.b.einsum(
            [op for op, _ in rest],
            [tuple(s) for _, s in rest],
            tuple(keep),
            dict(enumerate(n.wire_dims)),
        )
        return u, tuple(keep)

    def _step_head(self, n: EinsumNode) -> Node:
        i = self._pick_boundary(n)
        if i is None:
            return n
        X = n.ops[i]
        comp = self._compress(n, i)
        if comp is None:
            return n
        ctx = [comp] if comp else []  # comp == () means: no cotangent context
        x_subs = tuple(n.in_subs[i])
        rows = list(n.constraints)
        wire_dims = dict(enumerate(n.wire_dims))

        if isinstance(X, EinsumNode):
            if len(ctx) + len(X.ops) > MAX_OPS:
                return n
            ops = [u for u, _ in ctx]
            in_subs = [tuple(us) for _, us in ctx]
            new_rows = list(rows)
            new_wd = dict(wire_dims)
            w = splice_child(ops, in_subs, new_wd, new_rows, X, x_subs)
            if len(new_wd) > MAX_WIRES:
                return n
            self.changed = True
            return self.b.einsum(ops, in_subs, n.out_subs, new_wd, n.weight * w, new_rows)

        # X Linear: distribute einsum(u, X) over its terms (one boundary; the
        # next iteration's R handles what the terms contain).
        built = []
        for t, pm, w in zip(X.terms, X.perms, X.weights):
            subs_t = [0] * t.order
            for j in range(len(pm)):
                subs_t[pm[j]] = x_subs[j]
            ops = [u for u, _ in ctx]
            in_subs = [tuple(us) for _, us in ctx]
            new_rows = list(rows)
            new_wd = dict(wire_dims)
            weight = n.weight * sympy.sympify(w)
            if (
                isinstance(t, EinsumNode)
                and t.ops
                and len(set(t.out_subs)) == len(t.out_subs)
                and len(ctx) + len(t.ops) <= MAX_OPS
            ):
                weight = n.weight * sympy.sympify(w) * splice_child(ops, in_subs, new_wd, new_rows, t, subs_t)
                if len(new_wd) > MAX_WIRES:
                    ops = [u for u, _ in ctx] + [t]
                    in_subs = [tuple(us) for _, us in ctx] + [tuple(subs_t)]
                    new_rows, new_wd = list(rows), dict(wire_dims)
                    weight = n.weight * sympy.sympify(w)
            else:
                ops.append(t)
                in_subs.append(tuple(subs_t))
            built.append(self.b.einsum(ops, in_subs, n.out_subs, new_wd, weight, new_rows))
        built = [nd for nd in built if not (isinstance(nd, ConstNode) and nd.kind == "zero")]
        self.changed = True
        if not built:
            return self.b.const("zero", (), tuple(n.dims))
        if len(built) == 1:
            return built[0]
        m = n.order
        return self.b.linear(built, [tuple(range(m))] * len(built), [1] * len(built))

    # ---- rule M: merge cotangent branches sharing a chain tail --------------

    def _merge_linear(self, L: LinearNode) -> Node:
        """Group terms of L that are einsums reading the SAME inflated
        operand X with the same role wiring; each group Σ_i w_i einsum(C_i, X)
        becomes einsum(Linear(u_i), X) — the reverse-mode cotangent sum. Only
        fires for groups of >= 2 (otherwise R alone handles the term)."""
        m = len(L.dims)
        if m == 0 or len(L.terms) < 2:
            return L

        def touches_inflated(t: Node) -> bool:
            if self.inflated(t):
                return True
            if isinstance(t, EinsumNode):
                return any(self.inflated(op) for op in t.ops)
            if isinstance(t, LinearNode):
                return any(touches_inflated(t3) for t3 in t.terms)
            return False

        # Flatten nested Linear terms first (R wraps every dissolved boundary
        # in a fresh Linear, so sibling cotangent branches hide one level
        # down; the merge below can only see einsums side by side). Composed
        # perms/weights, bounded fan-in. ONLY in service of the collapse —
        # i.e. when inflated nodes are involved: regrouping an unrelated
        # small sum perturbs fp product grouping, which breaks the
        # saturation-exact zeros the stabilization pass arranges (measured
        # 5e-3 in the fp32 gelu gradient at |x|=200).
        if any(isinstance(t, LinearNode) for t in L.terms) and (
            sum(len(t.terms) if isinstance(t, LinearNode) else 1 for t in L.terms) <= 64
        ) and any(touches_inflated(t) for t in L.terms):
            terms, perms, weights = [], [], []
            for t, pm, w in zip(L.terms, L.perms, L.weights):
                if isinstance(t, LinearNode):
                    for t3, pm3, w3 in zip(t.terms, t.perms, t.weights):
                        terms.append(t3)
                        perms.append(tuple(pm3[pm[j]] for j in range(m)))
                        weights.append(sympy.sympify(w) * sympy.sympify(w3))
                else:
                    terms.append(t)
                    perms.append(tuple(pm))
                    weights.append(w)
            self.changed = True
            return self.b.linear(terms, perms, weights)
        groups: dict = {}
        for idx, (t, pm, w) in enumerate(zip(L.terms, L.perms, L.weights)):
            if not (isinstance(t, EinsumNode) and t.ops and not t.constraints):
                continue
            if len(set(t.out_subs)) != len(t.out_subs):
                continue
            for o, opn in enumerate(t.ops):
                if not self.inflated(opn):
                    continue
                if not isinstance(opn, (EinsumNode, LinearNode, MapNode)):
                    continue
                if len(set(t.in_subs[o])) != len(t.in_subs[o]):
                    continue  # diagonal read of X: skip
                # Role signature: L output axes then X axes, partitioned by
                # shared wires (as in factor._hoist).
                out_wire = [t.out_subs[pm[j]] for j in range(m)]
                roles = out_wire + list(t.in_subs[o])
                lab: dict = {}
                sig = tuple(lab.setdefault(w2, len(lab)) for w2 in roles)
                groups.setdefault((id(opn), sig), []).append((idx, o, opn))
                break  # one inflated operand per term is enough
        for (nid, sig), members in groups.items():
            if len(members) < 2:
                continue
            X = members[0][2]
            # Build u_i per member: contract the term's other operands down
            # to the (out ∪ shared-with-X) wires, in ROLE order so all u_i
            # align; skip the group if any interface inflates.
            role_order = None
            u_parts = []
            ok = True
            for idx, o, _ in members:
                t = L.terms[idx]
                pm = L.perms[idx]
                out_wire = [t.out_subs[pm[j]] for j in range(m)]
                roles = out_wire + list(t.in_subs[o])
                rest = [(op2, s2) for q, (op2, s2) in enumerate(zip(t.ops, t.in_subs)) if q != o]
                rest_wires = {w2 for _, s2 in rest for w2 in s2}
                # u carries ONLY wires the rest operands actually touch —
                # output axes fed by X (the wrt-param axes of a gradient)
                # flow through the outer einsum from X, never through u.
                keep_roles = [r for r in range(len(roles)) if roles[r] in rest_wires]
                # dedupe roles by first occurrence (a wire may appear twice)
                seen: set = set()
                kr = []
                for r in keep_roles:
                    if sig[r] not in seen:
                        seen.add(sig[r])
                        kr.append(r)
                if role_order is None:
                    role_order = [sig[r] for r in kr]
                elif [sig[r] for r in kr] != role_order:
                    ok = False
                    break
                keep_wires = [roles[r] for r in kr]
                if self._winflated(keep_wires, t.wire_dims):
                    ok = False
                    break
                u = self.b.einsum(
                    [op2 for op2, _ in rest],
                    [tuple(s2) for _, s2 in rest],
                    tuple(keep_wires),
                    dict(enumerate(t.wire_dims)),
                    t.weight,
                )
                u_parts.append((u, L.weights[idx]))
            if not ok or role_order is None:
                continue
            k = len(role_order)
            inner = self.b.linear(
                [u for u, _ in u_parts], [tuple(range(k))] * len(u_parts), [w for _, w in u_parts]
            )
            # Outer einsum: wires = role labels. u carries role_order; X
            # carries sig[m:]; output carries sig[:m].
            t0 = L.terms[members[0][0]]
            pm0 = L.perms[members[0][0]]
            o0 = members[0][1]
            out_wire0 = [t0.out_subs[pm0[j]] for j in range(m)]
            roles0 = out_wire0 + list(t0.in_subs[o0])
            role_dim = {}
            for r, c in enumerate(sig):
                if c not in role_dim:
                    role_dim[c] = t0.wire_dims[roles0[r]]
            outer = self.b.einsum(
                [inner, X],
                [tuple(role_order), tuple(sig[m:])],
                tuple(sig[:m]),
                role_dim,
            )
            member_ids = {idx for idx, _, _ in members}
            left_terms = [outer] + [t for q, t in enumerate(L.terms) if q not in member_ids]
            left_perms = [tuple(range(m))] + [
                tuple(p) for q, p in enumerate(L.perms) if q not in member_ids
            ]
            left_ws = [1] + [w for q, w in enumerate(L.weights) if q not in member_ids]
            self.changed = True
            return self.b.linear(left_terms, left_perms, left_ws)
        return L

    # ---- fixpoint over the DAG ----------------------------------------------

    def run(self, outputs):
        nodes = [n for n, _ in outputs]
        for nd in toposort(list(nodes)):
            if isinstance(nd, InputNode):
                self.base = max(self.base, self._numel(nd.dims))
        for nd in nodes:
            self.base = max(self.base, self._numel(nd.dims))

        order0 = toposort(list(nodes))
        if not any(self.inflated(nd) for nd in order0):
            return list(outputs)  # nothing forward-mode-shaped: touch nothing

        initial = len(order0)
        for _ in range(MAX_ROUNDS):
            self.changed = False
            memo: dict[int, Node] = {}
            order = toposort(list(nodes))
            if len(order) > 8 * initial + 512:
                # (A single-block program with one gradient per parameter
                # legitimately grows ~5x mid-collapse before M re-merges the
                # branches; 8x is comfortably above that and far below the
                # geometric multi-block blowup.)
                # KNOWN FRONTIER: gradients through MULTIPLE stacked blocks.
                # R expands each head top-down, which enumerates monomial
                # branches; M re-merges them only within one LinearNode, so
                # cotangent contributions that arise in different Linears —
                # in particular across the many gradients that share the same
                # upper-layer chains — never accumulate, and the branch count
                # grows geometrically with stack depth (measured x1.5/round
                # at 3 transformer layers, while a single block converges in
                # ~10 rounds with FEWER nodes than it started with). The fix
                # is true per-node adjoint accumulation — ONE cotangent node
                # per inflated node, summed over all its consumers, pushed
                # through Linear boundaries as Jacobian-factor matrices
                # (delta/ones contexts for bare and broadcast terms). Until
                # then: abort cleanly, hand the ORIGINAL DAG to the sweeps.
                return list(outputs)
            for nd in order:
                ops = [memo[id(op)] for op in nd.operands()]
                cur = _rebuild(self.b, nd, ops)
                if isinstance(cur, EinsumNode) and not self.inflated(cur):
                    cur = self._step_head(cur)
                elif isinstance(cur, LinearNode) and not self.inflated(cur):
                    cur = self._merge_linear(cur)
                memo[id(nd)] = cur
            nodes = [memo[id(n)] for n in nodes]
            if not self.changed:
                break
        return [(n, edge_order) for n, (_, edge_order) in zip(nodes, outputs)]


def _rebuild(b: Builder, nd: Node, ops: list[Node]) -> Node:
    """Reconstruct `nd` with rewritten operands (hash-consing returns the
    identical node when nothing changed). Mirrors factor._Rewriter._rebuild."""
    if not ops:
        return nd
    if isinstance(nd, EinsumNode):
        return b.einsum(
            ops, list(nd.in_subs), nd.out_subs, dict(enumerate(nd.wire_dims)), nd.weight, nd.constraints
        )
    if isinstance(nd, LinearNode):
        return b.linear(ops, list(nd.perms), list(nd.weights))
    if isinstance(nd, MapNode):
        return b.map(nd.op, nd.params, ops, list(nd.perms))
    if isinstance(nd, GatherNode):
        if nd.op == "gather":
            return b.gather(ops[0], ops[1], nd.axis)
        return b.one_hot(ops[0], nd.dims[0])
    if isinstance(nd, ReduceNode):
        return b.reduce(nd.op, nd.axes, ops[0])
    return nd


def collapse_chains(builder: Builder, outputs, dims) -> list:
    """Entry point: reverse-mode chain collapse over `outputs` =
    [(node, edge_order), ...]. Fixpoint of the R/M rules above."""
    return _Collapser(builder, dims).run(outputs)
