"""Adjoint chain collapse: recover reverse-mode contraction order (#29, #31).

Symbolic gradients of parameters deep under a stack of nonlinearities lower
into FORWARD-MODE-shaped Jacobian chains: einsums separated by LinearNodes
(residual adds, LayerNorm centering, softmax-Jacobian sums) whose node
outputs drag the wrt-parameter's axes through every layer above. At GPT-2
block dims (d=768, seq=256, batch=8) the raw lowered gradient program
totals ~400 TB with a 59 PB rank-5 node; a 3-layer gpt-nano lowers 841 GB
with 3024 inflated nodes. The compact reverse-mode order — contract the
small cotangent end first and push it through the chain — is a legal
einsum reassociation through delta wires (the chain alternates linear maps
with diagonal, Hadamard-joined factors).

Why a dedicated pass: factor.py's cost-model rewriter cannot find this
order. Collapsing ONE Linear boundary leaves the next huge node read dense,
so each local step improves the score by <2% (measured 0.2% at the GPT-2
chain head) — a valley whose payoff only appears when the whole chain is
gone. Unconditional sweep overrides cross the valley but churn (every
candidate interned and scored per sweep: 100+ second sweeps and gigabytes
of candidates at 3 layers), and naive full expansion of a head into
monomials is EXPONENTIAL in the boundary count (each residual add, LN
centering and softmax-Jacobian sum forks every partial into a diamond).

The pass is textbook reverse-mode AD expressed as an IR transform, in two
stages:

  A (per-node adjoint accumulation, _Accumulate — the workhorse). The
    inflated nodes form a linear region: measured on transformer stacks,
    every einsum reads at most ONE inflated operand (product rule — a first
    derivative has exactly one chain factor per monomial) and every
    LinearNode combines them additively, so each small head (a gradient
    output's final contraction) is a LINEAR function of every region node
    it reaches. For each head, walk its region in REVERSE topological
    order keeping an accumulator {region node X -> cotangent contributions
    u_i}: contributions are grouped by their canonical interface signature
    (which X axes feed which head axes / contract which cotangent axes),
    merged into ONE u_X = Σ u_i per signature, and pushed a single
    boundary further: splice (X einsum) or distribute (X Linear), then
    re-contract everything but the next chain node into the next cotangent.
    Diamonds (residual adds) therefore cost one merge instead of doubling
    the branch count — this is exactly u = Σ_consumers J_cᵀ u_c, and the
    walk emits one Linear of frontier-sized pieces per head. Cotangents are
    hash-consed NODES, so the many gradient outputs that share upper-layer
    chains share every pushed cotangent program-globally. Two rules keep
    the cotangents thin and mergeable: (1) walks START only at true seeds —
    a small einsum whose region read passes through other small einsums (a
    TOWER strand, e.g. the loss-side softmax-Jacobian contractions above a
    ∂logits/∂x segment) is spliced into its top consumer's walk instead of
    being walked itself, so its open axes (seq, vocab) contract at the
    consumer instead of riding the whole region (measured 33x cotangent
    fat at 3 gpt-nano layers); (2) only path-INDEPENDENT nodes (original
    program nodes) may defer as passengers — a pass-created cotangent is
    part of the merge signature and deferring one killed accumulation along
    every chain it rode. Anything the walk cannot push through (an einsum
    reading two region operands, a spliced interface that inflates, op/wire
    caps) degrades to a dense read of that operand — never an error, never
    worse than the input program; a second sweep without tower deferral
    rescues strands whose top consumer bailed on a cap.

  R/M fixpoint (_Collapser — retained as a backstop for shapes A skips):
    R (cotangent step) at a small-output einsum reading an inflated operand
    X: contract everything EXCEPT X down to the shared interface (the
    cotangent u), then dissolve one boundary of X. M (cotangent merge) at a
    LinearNode whose terms read the SAME inflated X in the same role:
    Σ_i einsum(u_i, X) becomes einsum(Σ_i u_i, X). R/M re-merge only within
    one LinearNode, so multi-block stacks fork geometrically — that is the
    failure stage A removes; the growth guard in run() stays as a safety
    net and should not fire on transformer stacks anymore.

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
                # Safety net. R expands each head top-down, which enumerates
                # monomial branches; M re-merges them only within one
                # LinearNode, so on multi-block stacks the branch count used
                # to grow geometrically (measured x1.5/round at 3 transformer
                # layers). Stage A (_Accumulate) now collapses those stacks
                # with true per-node adjoint accumulation BEFORE this
                # fixpoint runs, so the guard should never fire on
                # transformer stacks; if it does, abort cleanly and hand the
                # incoming DAG to the sweeps.
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


class _Accumulate(_Collapser):
    """Stage A: per-node adjoint accumulation (#31; module docstring).

    One shot, no fixpoint. For each head (small einsum reading a region
    node), walk the region reachable from it in reverse topological order,
    accumulate ONE cotangent per region node per interface signature, and
    push it one boundary down per node. Exactness invariant: at every step
    head == Σ pieces + Σ pending descriptors, where a descriptor
    (U, u_subs, x_subs, out_subs, dims, w, rows) denotes the einsum
    w * einsum([U, X], [u_subs, x_subs], out_subs) under constraint rows.
    Merging a signature group is linearity of einsum in U; dissolving an
    einsum boundary is associativity (splice + re-contract); dissolving a
    Linear boundary is distribution. All are exact.
    """

    # Caps: a head whose walk exceeds these is left untouched (fall back to
    # the R/M fixpoint / dense form). Contributions per region node cannot
    # exceed its consumer count unless signatures fail to align, which is
    # the geometric-growth mode accumulation exists to prevent.
    MAX_DESCS = 64
    MAX_PIECES = 256
    # Operands riding beside the cotangent uncontracted (see _split below);
    # deferring must beat folding by PAX_BIAS in interface size
    # (an interface win below the inflation margin is not worth the merge
    # fragmentation different passengers cause).
    MAX_PASSENGERS = 6
    PAX_BIAS = 32.0
    # Waist-splice recursion bound (see _build_cotangent).
    MAX_DEPTH = 8

    # ---- region ------------------------------------------------------------

    def _region(self, order) -> dict:
        """id(node) -> node for every dissolvable transient Jacobian:
        inflated einsums with operands and distinct out wires (splice needs
        a well-defined aliasing), and inflated LinearNodes. Inflated Map/
        Reduce/Gather/Const nodes stay dense leaves (nonlinear or opaque)."""
        region = {}
        for nd in order:
            if not self.inflated(nd):
                continue
            if isinstance(nd, EinsumNode) and nd.ops and len(set(nd.out_subs)) == len(nd.out_subs):
                region[id(nd)] = nd
            elif isinstance(nd, LinearNode):
                region[id(nd)] = nd
        return region

    # ---- descriptors -------------------------------------------------------

    def _split(self, rest, x_subs, out_subs, rows, wire_dims):
        """Contract the context `rest` = [(op, subs), ...] of a region read
        down to a cotangent U over the interface it shares with the chain
        operand / output / constraints / passengers. Output axes that do not
        ride the chain operand are covered two ways, and the smaller
        interface wins:

          * fold their carriers into U (right when the carrier CONTRACTS
            other context away — the lm_head read that turns a logits
            cotangent into a (b,s,v) one), or
          * defer the carriers as PASSENGERS riding beside the cotangent
            uncontracted (right when the carrier is a saved forward
            activation or a gather one-hot, contracted LAST — the outer
            product / scatter that ends every backward pass).

        Returns (U, u_subs, pax) or None when no interface fits self.allow
        (the reverse-mode frontier)."""
        xw = set(x_subs)
        row_wires = {wr for cs, _ in rows for wr, _ in cs}
        rest = [(op, tuple(s)) for op, s in rest]

        def interface(fold, pax):
            pax_wires = {wr for _, s in pax for wr in s}
            keep = sorted(
                {wr for _, s in fold for wr in s}
                & (xw | set(out_subs) | row_wires | pax_wires)
            )
            return keep, self._numel([wire_dims[wr] for wr in keep])

        # Fold by default: folding keeps contributions MERGEABLE (different
        # paths defer different carriers, and descriptors only accumulate
        # when their passenger sets match — measured 39 distinct signatures
        # at one node when deferring was preferred). Defer only on an
        # PAX_BIAS interface win or when folding blows
        # the frontier: those carriers (gather one-hots, the saved forward
        # activation of an outer-product gradient) enter the chain at ONE
        # node, identical along every path, so merging survives them.
        # Only ORIGINAL program nodes may ride as passengers: a node this
        # pass created (a pushed cotangent) is path-DEPENDENT, and a
        # passenger's identity is part of the merge signature — deferring
        # one killed accumulation along every chain it rode (measured 948
        # unmerged vocab-carrying cotangents on 3-layer gpt-nano, 4.3 GB).
        deferred = set(out_subs) - xw
        fold_keep, fold_size = interface(rest, [])
        best = (fold_size, fold_keep, rest, []) if fold_size <= self.allow else None
        pax_ids = {
            q for q, (op, s) in enumerate(rest)
            if (deferred & set(s)) and id(op) in self._frozen
        }
        pax = [rest[q] for q in sorted(pax_ids)]
        if pax and len(pax) <= self.MAX_PASSENGERS:
            fold = [rest[q] for q in range(len(rest)) if q not in pax_ids]
            keep, size = interface(fold, pax)
            if size <= self.allow and (best is None or size * self.PAX_BIAS < best[0]):
                best = (size, keep, fold, pax)
        if best is None:
            if len(self.stats) < 256:  # debug telemetry, bounded
                key = "split:" + ",".join(sorted(str(wire_dims[wr]) for wr in fold_keep))
                self.stats[key] = self.stats.get(key, 0) + 1
            return None
        _, keep, f, p = best
        if not f:
            return self.b.scalar(1), (), tuple(p)
        U = self._build_cotangent(f, tuple(keep), wire_dims)
        return U, tuple(keep), tuple(p)

    def _build_cotangent(self, fold, keep, wire_dims):
        """Contract `fold` = [(op, subs), ...] down to the `keep` interface.

        WAIST handling: a fold operand may itself be a small einsum reading
        a region node — a bottleneck the region re-inflates from (e.g. the
        loss-side (b,s,v)-shaped read of a shared ∂x_k/∂x_j segment) — or a
        TOWER member reading one transitively (self.S). Computing such a
        node standalone would push its own output axes (vocab!) down the
        whole segment. Reverse mode instead contracts the incoming cotangent
        with the waist's context FIRST: splice the waist into this
        contraction — the region node becomes a direct operand — and
        recursively collapse the resulting einsum as a head. The waist's
        standalone definition then dies with the region."""
        ops = [op for op, _ in fold]
        in_subs = [tuple(s) for _, s in fold]
        if self._depth < self.MAX_DEPTH:
            wd = dict(wire_dims)
            sp_ops: list = []
            sp_subs: list = []
            sp_rows: list = []
            weight = sympy.Integer(1)
            spliced = False
            for op, s in fold:
                if (
                    isinstance(op, EinsumNode)
                    and op.ops
                    and not self.inflated(op)
                    and len(set(op.out_subs)) == len(op.out_subs)
                    and any(id(o) in self.region or id(o) in self.S for o in op.ops)
                    and len(sp_ops) + len(op.ops) + 4 <= MAX_OPS
                ):
                    weight = weight * splice_child(sp_ops, sp_subs, wd, sp_rows, op, tuple(s))
                    spliced = True
                else:
                    sp_ops.append(op)
                    sp_subs.append(tuple(s))
            if spliced and len(wd) <= MAX_WIRES:
                U = self.b.einsum(sp_ops, sp_subs, keep, wd, weight, sp_rows)
                return self._recollapse(U)
        # No waist to splice — but the fold may read a region node DIRECTLY
        # (the second operand of a composed ∂x_k/∂x_j · ∂x_j/∂x_i product,
        # see _dissolve): the built cotangent is then itself a head, and
        # collapsing it here is the u·(A·B) = (u·A)·B reassociation.
        return self._recollapse(self.b.einsum(ops, in_subs, keep, wire_dims))

    def _recollapse(self, U):
        """Recursively collapse a freshly built cotangent that still reads a
        region or tower node (bounded by MAX_DEPTH)."""
        if (
            self._depth < self.MAX_DEPTH
            and isinstance(U, EinsumNode)
            and any(id(o) in self.region or id(o) in self.S for o in U.ops)
        ):
            self._depth += 1
            try:
                new = self._rewrite_head(U, self.region)
            finally:
                self._depth -= 1
            if new is not None:
                return new
        return U

    def _canon(self, U, u_subs, x_subs, out_subs, wire_dims, w, rows, pax):
        """Canonicalize a contribution to label space: wires relabeled by
        first occurrence over out_subs + x_subs (the shared interface), then
        u_subs, then passenger wires, then any leftover constraint wires.
        Returns (sig, U, w, (u_l, x_l, out_l, dims_list, rows_l, pax_l));
        descriptors merge iff their sig matches — same roles, same wiring,
        same passengers, same constraints."""
        lab: dict = {}
        for wr in tuple(out_subs) + tuple(x_subs) + tuple(u_subs):
            lab.setdefault(wr, len(lab))
        pax = sorted(
            pax, key=lambda ps: (self.b.node_index(ps[0]), tuple(lab.get(wr, -1) for wr in ps[1]))
        )
        for _, s in pax:
            for wr in s:
                lab.setdefault(wr, len(lab))
        nonce = 0
        for cs, _ in rows:
            for wr, _c in cs:
                if wr not in lab:
                    # Leftover row wires have no canonical order; make the
                    # signature unique so such descriptors never merge.
                    lab[wr] = len(lab)
                    nonce = id(rows)
        out_l = tuple(lab[wr] for wr in out_subs)
        x_l = tuple(lab[wr] for wr in x_subs)
        u_l = tuple(lab[wr] for wr in u_subs)
        pax_l = tuple((p, tuple(lab[wr] for wr in s)) for p, s in pax)
        dims_list = [None] * len(lab)
        for wr, a in lab.items():
            dims_list[a] = wire_dims[wr]
        rows_l = tuple(
            (tuple((lab[wr], c) for wr, c in cs), const) for cs, const in rows
        )
        rows_sig = tuple(
            sorted((tuple(sorted((wr, str(c)) for wr, c in cs)), str(const)) for cs, const in rows_l)
        )
        pax_sig = tuple((self.b.node_index(p), s) for p, s in pax_l)
        # The u component of the signature is order-free (u wires come out
        # of _split sorted by ARBITRARY local wire ids); _merge aligns the
        # members' axes to the first member's order with Linear perms.
        sig = (out_l, x_l, tuple(sorted(u_l)), pax_sig,
               tuple(str(d) for d in dims_list), rows_sig, nonce)
        return (sig, U, sympy.sympify(w), (u_l, x_l, out_l, tuple(dims_list), rows_l, pax_l))

    def _merge(self, group):
        """Σ_i w_i einsum([U_i, X], shared wiring) == einsum([Σ_i w_i U_i, X],
        same wiring): sum the cotangents as one LinearNode (weights ride the
        Linear — never wrap a big cotangent in a scaling copy), axes aligned
        by u-wire label to the FIRST member's order (whose layout the caller
        reads the merged node with). Weights with free dim symbols cannot
        ride a LinearNode; those fold into the member via a weighted
        identity einsum (rare: mean factors and friends)."""
        u0 = group[0][3][0]
        parts, perms, weights = [], [], []
        for _sig, U, w, layout in group:
            if w.free_symbols:
                ids = tuple(range(U.order))
                U = self.b.einsum([U], [ids], ids, {a: U.dims[a] for a in ids}, weight=w)
                w = sympy.Integer(1)
            parts.append(U)
            weights.append(w)
            u_l = layout[0]
            perms.append(tuple(u_l.index(lab) for lab in u0))
        if len(parts) == 1 and weights[0] == 1:
            return parts[0]
        return self.b.linear(parts, perms, weights)

    # ---- towers --------------------------------------------------------------

    def _inline_s(self, ops, in_subs, wd, rows):
        """Splice every operand in self.S (absorbable region-reading tower
        einsum; see _sweep) into einsum parts under construction, mutating
        the lists in place, transitively (a spliced tower node's operands
        may be tower nodes one level down). Returns the accumulated weight
        factor, or None on an op/wire cap breach."""
        w = sympy.Integer(1)
        q = 0
        while q < len(ops):
            op = ops[q]
            if id(op) not in self.S:
                q += 1
                continue
            if len(ops) - 1 + len(op.ops) > MAX_OPS:
                return None
            s = tuple(in_subs[q])
            del ops[q], in_subs[q]
            w = w * splice_child(ops, in_subs, wd, rows, op, s)
            if len(wd) > MAX_WIRES:
                return None
            # spliced operands land at the tail; rescan from q (the former
            # q+1 shifted down) — the tail is reached by the same scan.
        return w

    def _inline_towers(self, n: EinsumNode):
        """Inline every tower operand of head `n` so its region reads become
        direct. Returns `n` unchanged when it has none, the rewritten node
        (possibly const-folded), or None on cap breach."""
        if not any(id(op) in self.S for op in n.ops):
            return n
        ops = list(n.ops)
        in_subs = [tuple(s) for s in n.in_subs]
        rows = list(n.constraints)
        wd = dict(enumerate(n.wire_dims))
        w = self._inline_s(ops, in_subs, wd, rows)
        if w is None:
            return None
        return self.b.einsum(ops, in_subs, n.out_subs, wd, n.weight * w, rows)

    # ---- one boundary ------------------------------------------------------

    def _dissolve(self, Y, Um, layout, region, contributions, pieces):
        """Push the merged cotangent read einsum([Um, pax..., Y], ...) one
        boundary into Y. Emits new descriptors for region operands and
        finished pieces for frontier ones; falls back to a dense read of Y
        when the boundary cannot be dissolved (caps, multi-region reads,
        inflated interface)."""
        u_l, x_l, out_l, dims_list, rows_l, pax_l = layout
        dims_map = dict(enumerate(dims_list))
        pax_ops = [p for p, _ in pax_l]
        pax_subs = [s for _, s in pax_l]

        def dense_read(why):
            self.stats[why] = self.stats.get(why, 0) + 1
            pieces.append(
                self.b.einsum([Um, *pax_ops, Y], [u_l, *pax_subs, x_l], out_l, dims_map, 1, rows_l)
            )

        if isinstance(Y, LinearNode):
            for t, pm, wt in zip(Y.terms, Y.perms, Y.weights):
                if isinstance(t, ConstNode) and t.kind == "zero":
                    continue
                subs_t = [0] * t.order
                for j in range(len(pm)):
                    subs_t[pm[j]] = x_l[j]
                if id(t) in region:
                    contributions.setdefault(id(t), []).append(
                        self._canon(Um, u_l, tuple(subs_t), out_l, dims_map, wt, rows_l, pax_l)
                    )
                else:
                    pieces.append(
                        self.b.einsum([Um, *pax_ops, t], [u_l, *pax_subs, tuple(subs_t)],
                                      out_l, dims_map, sympy.sympify(wt), rows_l)
                    )
            return

        # Y EinsumNode: splice its operands onto the read, then re-contract
        # everything except the (single) next region operand into the next
        # cotangent + passengers.
        if 1 + len(pax_ops) + len(Y.ops) > MAX_OPS:
            dense_read("dense:max_ops")
            return
        ops = [Um, *pax_ops]
        in_subs = [u_l, *pax_subs]
        wd = dict(dims_map)
        rows = list(rows_l)
        wY = splice_child(ops, in_subs, wd, rows, Y, x_l)
        if len(wd) > MAX_WIRES:
            dense_read("dense:max_wires")
            return
        # Y's operands may include tower waists (small einsums whose region
        # read is one or more levels down); inline them so the next region
        # node surfaces as a direct operand.
        wS = self._inline_s(ops, in_subs, wd, rows)
        if wS is None:
            dense_read("dense:tower_caps")
            return
        wY = wY * wS
        rslots = [q for q in range(1, len(ops)) if id(ops[q]) in region]
        if not rslots:
            # Chain bottom: the piece is frontier-sized.
            pieces.append(self.b.einsum(ops, in_subs, out_l, wd, wY, rows))
            return
        # A composed read (the lowering's ∂x_k/∂x_j · ∂x_j/∂x_i products at
        # deep stacks) has SEVERAL region operands: push the cotangent into
        # one and fold the others into the next cotangent, whose own region
        # read collapses recursively (_build_cotangent/_recollapse) — the
        # u·(A·B) = (u·A)·B reassociation. Pick the feasible slot with the
        # smallest interface.
        best = None
        for q in rslots:
            s2 = tuple(in_subs[q])
            rest = [(ops[a], tuple(in_subs[a])) for a in range(len(ops)) if a != q]
            sp = self._split(rest, s2, out_l, rows, wd)
            if sp is None:
                continue
            size = self._numel([wd[w] for w in sp[1]])
            if best is None or size < best[0]:
                best = (size, q, sp)
        if best is None:
            dense_read("dense:interface" if len(rslots) == 1 else "dense:multi_region")
            return
        _, q, (U2, u2_subs, pax2) = best
        X2, s2 = ops[q], tuple(in_subs[q])
        contributions.setdefault(id(X2), []).append(
            self._canon(U2, u2_subs, s2, out_l, wd, wY, tuple(rows), pax2)
        )

    # ---- one head ----------------------------------------------------------

    def _rewrite_head(self, n: EinsumNode, region: dict):
        """Reverse-mode program for one head, or None to leave it alone."""
        n = self._inline_towers(n)
        if n is None:
            return None
        if not isinstance(n, EinsumNode):
            return n  # towers const-folded away: that IS the head's value
        slots = [i for i, op in enumerate(n.ops) if id(op) in region]
        if not slots:
            self.stats["head:multislot"] = self.stats.get("head:multislot", 0) + 1
            return None
        # Several region operands (a composed-product read): seed on the
        # feasible slot with the smallest interface; the other region
        # operands fold into the seed cotangent and collapse recursively
        # (_build_cotangent/_recollapse).
        wd = dict(enumerate(n.wire_dims))
        best = None
        for i in slots:
            rest = [(op, tuple(s)) for q, (op, s) in enumerate(zip(n.ops, n.in_subs)) if q != i]
            sp = self._split(rest, tuple(n.in_subs[i]), tuple(n.out_subs), n.constraints, wd)
            if sp is None:
                continue
            size = self._numel([wd[w] for w in sp[1]])
            if best is None or size < best[0]:
                best = (size, i, sp)
        if best is None:
            self.stats["head:seed"] = self.stats.get("head:seed", 0) + 1
            return None
        _, i, (U, u_subs, pax) = best
        seed = self._canon(
            U, u_subs, tuple(n.in_subs[i]), tuple(n.out_subs),
            wd, n.weight, n.constraints, pax,
        )
        contributions = {id(n.ops[i]): [seed]}
        pieces: list = []
        reach = [nd for nd in toposort([n.ops[i]]) if id(nd) in region]
        for Y in reversed(reach):
            descs = contributions.pop(id(Y), None)
            if not descs:
                continue
            if len(descs) > self.MAX_DESCS:
                self.stats["head:max_descs"] = self.stats.get("head:max_descs", 0) + 1
                return None
            groups: dict = {}
            for d in descs:
                groups.setdefault(d[0], []).append(d)
            if len(groups) > 1:
                key = f"multigroup:{len(groups)}"
                self.stats[key] = self.stats.get(key, 0) + 1
            for group in groups.values():
                Um = self._merge(group)
                self._dissolve(Y, Um, group[0][3], region, contributions, pieces)
            if len(pieces) > self.MAX_PIECES:
                self.stats["head:max_pieces"] = self.stats.get("head:max_pieces", 0) + 1
                return None
        if not pieces:
            return self.b.const("zero", (), tuple(n.dims))
        m = n.order
        return self.b.linear(pieces, [tuple(range(m))] * len(pieces), [1] * len(pieces))

    # ---- the pass ----------------------------------------------------------

    def run(self, outputs):
        self.stats: dict = {}  # fallback/skip telemetry, for debugging only
        self.allow = 0.0  # computed by the first sweep, then held fixed
        # Sweep 1 defers absorbable tower nodes to their top consumers'
        # walks (defer_towers); sweep 2 rescues any strand whose top bailed
        # on a cap by walking the survivors standalone, exactly the
        # pre-deferral behavior — never worse than the input program.
        outputs = self._sweep(outputs, defer_towers=True)
        return self._sweep(outputs, defer_towers=False)

    def _sweep(self, outputs, defer_towers):
        nodes = [n for n, _ in outputs]
        order0 = toposort(list(nodes))
        for nd in order0:
            if isinstance(nd, InputNode):
                self.base = max(self.base, self._numel(nd.dims))
        for nd in nodes:
            self.base = max(self.base, self._numel(nd.dims))
        if not any(self.inflated(nd) for nd in order0):
            return list(outputs)
        region = self.region = self._region(order0)
        if not region:
            return list(outputs)
        self._depth = 0
        # Nodes whose identity is path-independent (original program nodes
        # and their rebuilt images below) — the only legal passengers.
        self._frozen = {id(nd) for nd in order0}
        # Waists — small einsums whose value feeds a region node — are NOT
        # heads: computing them standalone pushes their own output axes
        # down the region. They are contracted into the cotangents of the
        # heads above instead (_build_cotangent) and die with the region.
        waists: set = set()
        for nd in order0:
            if id(nd) in region:
                for op in nd.operands():
                    waists.add(id(op))
        # Towers (S): small dissolvable einsums whose region read is direct
        # or passes through other tower nodes — the strands between the
        # region and the true seeds (gradient outputs, the terms of loss-
        # side Linear diamonds). Walking one as a head pushes ITS output
        # axes down the whole region (measured on 3-layer gpt-nano: (seq,
        # vocab) riding every deep-parameter cotangent at 33x reverse
        # mode's size), so a tower node whose every consumer is itself a
        # tower node defers to its consumers' walks (_inline_towers /
        # _inline_s) and dies with the region as dead code.
        S: dict = {}
        for nd in order0:
            if (
                isinstance(nd, EinsumNode)
                and nd.ops
                and not self.inflated(nd)
                and len(set(nd.out_subs)) == len(nd.out_subs)
                and any(id(op) in region or id(op) in S for op in nd.ops)
            ):
                S[id(nd)] = nd
        self.S = S
        consumers: dict = {}
        for nd in order0:
            for op in nd.operands():
                consumers.setdefault(id(op), []).append(nd)
        out_ids = {id(n) for n in nodes}
        # The reverse-mode frontier: cotangents are activation-shaped, so
        # gate pushed interfaces with the same INFLATE_MARGIN but measured
        # against the largest forward intermediate too, not only inputs and
        # outputs — on batched small models the activations (and J-factor
        # products like (b,s,key,h,hs), or the (b,s',v)-batched cotangents
        # behind a loss-side bottleneck) dwarf every input while remaining
        # exactly what reverse mode materializes. Computed on the FIRST
        # sweep only: the second sweep's program contains this pass's own
        # cotangents, which must not widen the frontier.
        if not self.allow:
            act_base = 1.0
            for nd in order0:
                if not isinstance(nd, (InputNode, ConstNode)) and not self.inflated(nd):
                    act_base = max(act_base, self._numel(nd.dims))
            self.allow = INFLATE_MARGIN * max(self.base, act_base)
        memo: dict[int, Node] = {}
        for nd in order0:
            ops = [memo[id(op)] for op in nd.operands()]
            cur = _rebuild(self.b, nd, ops)
            deferred_tower = (
                defer_towers
                and id(nd) in S
                and id(nd) not in out_ids
                and (cons := consumers.get(id(nd)))
                and all(id(c) in S for c in cons)
            )
            was_head = False
            if (
                id(nd) not in waists
                and not deferred_tower
                and isinstance(cur, EinsumNode)
                and not self.inflated(cur)
                and any(id(op) in region or id(op) in S for op in cur.ops)
            ):
                new = self._rewrite_head(cur, region)
                if new is not None:
                    cur = new
                    was_head = True
            if id(nd) in region and cur is not nd:
                # A rewritten head below (a chain WAIST — a small bottleneck
                # inside the region, e.g. a loss-side (b,s,vocab) contraction
                # the wte chains re-inflate from) changed this chain node's
                # identity; it is still the same transient Jacobian, so keep
                # it dissolvable for the heads above.
                region[id(cur)] = cur
            if (
                id(nd) in S
                and cur is not nd
                and not was_head
                and isinstance(cur, EinsumNode)
                and cur.ops
                and len(set(cur.out_subs)) == len(cur.out_subs)
            ):
                # A rebuilt tower node is still the same tower strand.
                S[id(cur)] = cur
            self._frozen.add(id(cur))
            memo[id(nd)] = cur
        return [(memo[id(n)], edge_order) for n, (_, edge_order) in zip(nodes, outputs)]


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
    [(node, edge_order), ...]. Stage A (per-node adjoint accumulation)
    first, then the R/M fixpoint as a backstop for anything A skipped."""
    outputs = _Accumulate(builder, dims).run(outputs)
    return _Collapser(builder, dims).run(outputs)
