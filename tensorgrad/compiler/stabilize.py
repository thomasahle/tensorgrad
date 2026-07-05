"""IR stabilization pass: re-fuse numerically unstable expanded forms.

Researchers (and simplify({"expand_softmax": True})) spell softmax, log-softmax
and tanh out of primitives:

    softmax(x) = exp(x) * pow(sum(exp(x)), -1)
    tanh(y)    = (exp(y) - exp(-y)) * pow(exp(y) + exp(-y), -1)
    lse(x)     = log(sum(exp(x)))

These forms are exact algebra but overflow float32 for |x| >~ 88 (exp -> inf,
inf/inf -> nan), while the fused kernels (torch.softmax, torch.tanh,
max-shifted logsumexp) are exact at any magnitude. This pass runs on the IR
DAG after CSE and the factoring pass and recognizes the expanded patterns —
including the shapes they take inside gradient graphs — rewriting them into
stable nodes:

1. RATIO FUSION (on EinsumNode). An operand R = pow(Z, -k), k >= 1, is a
   reciprocal "denominator". First, an aligned co-operand that IS Z cancels
   exactly (Z * Z^-k = Z^-(k-1) pointwise — the shape 1/softmax * softmax
   takes inside log-derivative graphs). Then two denominator families are
   recognized:

   a. Z = w_z * sum_axes(exp(X))  (softmax family). Every co-operand that is
      *the same* exp(X) node, aligned with R on Z's kept axes, consumes one
      power: exp(X)[a] * Z^-1[kept a] == softmax(X)[a] / w_z. Each match
      replaces the exp occurrence with a ReduceNode(softmax) and decrements
      the power (k occurrences of exp with pow(Z,-k) is the classic gradient
      term s_i s_j = e_i e_j / Z^2).

   b. Z = a*(exp(Y) + exp(-Y))  (tanh/cosh family, the exp-spelled tanh and
      sigmoid). A co-operand aligned with R is *divided* by Z through a
      recursive rewriter: alpha*exp(Y) + beta*exp(-Y) over the same Y becomes
      (alpha+beta)/(2a) + (alpha-beta)/(2a) * tanh(Y) — covering sinh/cosh
      = tanh, exp/cosh = 1 +- tanh (sigmoid), and cosh/cosh = 1 — and the
      division distributes through LinearNodes and free (non-contracted)
      operands of nested EinsumNodes. Residual pow(Z, -k) powers are left
      alone: 1/cosh underflows *to the correctly-rounded 0*, it is the
      inf * 0 pairing that produced nan.

   Both rewrites are pointwise identities on the operand list, so they are
   valid inside any contraction, including gradient terms where the ratio's
   axes are contracted away.

2. PUSH-THROUGH (on EinsumNode). Expanded-softmax *gradients* factor the
   numerator and its denominator onto opposite sides of a node boundary:
   - einsum(exp(X), Linear(t_1..t_m)) where each t_i holds a
     pow(sum exp(X), -k): the exp is multiplied into every term (splicing
     into term einsums) and the ratio fusion is run there; the rewrite
     commits only if EVERY term absorbed its exp into a softmax (otherwise
     it would just duplicate an unstable exp);
   - einsum(A, C) where C is an einsum holding pow(A, -k) (or, for A an
     exp, pow of its sum): A is spliced into C and fused there — this is
     the 1/softmax * softmax cancellation of d log(softmax) split across
     two contractions.

3. LOG FUSION (on MapNode(log)):
   - log(softmax) (also through a pure-view einsum of a softmax) becomes
     ReduceNode(log_softmax);
   - log(w_z * sum(exp(X))) becomes the max-shifted stable logsumexp
     m + log(sum(exp(X - m))) + log(w_z), with m = max over the summed axes
     (requires w_z provably positive).

Every rewrite preserves semantics exactly up to float rounding (verified by
randomized equivalence tests in tests/compiler/test_stabilize.py); the point
is that the stable form's rounding does not overflow.
"""

from typing import cast

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

# Master toggle so tests can compare stabilized vs raw programs.
STABILIZE = True

MAX_SWEEPS = 6
MAX_DIV_DEPTH = 8


def _identity(k: int) -> tuple:
    return tuple(range(k))


def stabilize_outputs(builder: Builder, outputs) -> list:
    """Run the stabilization pass over `outputs` = [(node, edge_order), ...].

    Returns a new outputs list with the same edge orders; new nodes are
    interned in `builder`. Purely structural (no concrete dims needed)."""
    if not STABILIZE:
        return list(outputs)
    st = _Stabilizer(builder)
    nodes = [n for n, _ in outputs]
    for _ in range(MAX_SWEEPS):
        new_nodes = st.sweep(nodes)
        if all(a is b for a, b in zip(new_nodes, nodes)):
            break
        nodes = new_nodes
    return [(n, order) for n, (_, order) in zip(nodes, outputs)]


class _Stabilizer:
    def __init__(self, builder: Builder):
        self.b = builder
        self._div_memo: dict = {}
        self._neg_memo: dict = {}
        # log nodes CREATED by the logsumexp rewrite (already max-shifted);
        # without this the rule would re-fire on its own output every sweep.
        self._log_done: set = set()

    # ---- infrastructure -------------------------------------------------

    def sweep(self, roots: list[Node]) -> list[Node]:
        """One bottom-up rebuild of the DAG, applying local rules at each node."""
        order = toposort(list(roots))
        memo: dict[int, Node] = {}
        for nd in order:
            new_ops = [memo[id(op)] for op in nd.operands()]
            cur = self._rebuild(nd, new_ops)
            for _ in range(8):
                nxt = self._local(cur)
                if nxt is cur:
                    break
                cur = nxt
            memo[id(nd)] = cur
        return [memo[id(r)] for r in roots]

    def _rebuild(self, nd: Node, ops: list[Node]) -> Node:
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
            out = self._fuse_ratios(nd)
            if out is not nd:
                return out
            out = self._push_exp(nd)
            if out is not nd:
                return out
            return self._push_base(nd)
        if isinstance(nd, MapNode):
            return self._rewrite_log(nd)
        return nd

    # ---- structural helpers ---------------------------------------------

    def _negate(self, n: Node) -> Node:
        """The hash-consed canonical negation of `n`: `_negate(a) is b` proves
        b == -a elementwise (builder interning makes it an identity check)."""
        if (hit := self._neg_memo.get(id(n))) is not None:
            return hit
        if isinstance(n, LinearNode):
            r = self.b.linear(list(n.terms), [tuple(p) for p in n.perms], [-w for w in n.weights])
        elif isinstance(n, EinsumNode):
            r = self.b.einsum(
                list(n.ops),
                [tuple(s) for s in n.in_subs],
                n.out_subs,
                dict(enumerate(n.wire_dims)),
                -n.weight,
                n.constraints,
            )
        elif isinstance(n, ConstNode) and n.kind == "scalar":
            r = self.b.scalar(-n.params[0])
        else:
            r = self.b.linear([n], [_identity(n.order)], [-1])
        self._neg_memo[id(n)] = r
        return r

    def _exp_pair(self, node: Node):
        """node == alpha*exp(Y)[p] + beta*exp(-Y)[p] -> (Y, p, alpha, beta)."""
        if not (isinstance(node, LinearNode) and len(node.terms) == 2):
            return None
        t1, t2 = node.terms
        p1, p2 = node.perms
        if p1 != p2:
            return None
        for t in (t1, t2):
            if not (isinstance(t, MapNode) and t.op == "exp" and t.perms[0] == _identity(t.order)):
                return None
        # (casts: the loop above verified both terms are exp MapNodes)
        y1, y2 = cast(MapNode, t1).ops[0], cast(MapNode, t2).ops[0]
        if self._negate(y1) is y2 or self._negate(y2) is y1:
            return (y1, p1, node.weights[0], node.weights[1])
        return None

    def _sum_exp(self, Z: Node):
        """Z == w_z * sum over some axes of exp(X)
        -> (E, X, reduced_axes, axis_of_out, w_z)."""
        if not (isinstance(Z, EinsumNode) and len(Z.ops) == 1 and not Z.constraints):
            return None
        E = Z.ops[0]
        if not (isinstance(E, MapNode) and E.op == "exp" and E.perms[0] == _identity(E.order)):
            return None
        subs = Z.in_subs[0]
        if len(set(subs)) != len(subs) or len(set(Z.out_subs)) != len(Z.out_subs):
            return None
        if not set(Z.out_subs) <= set(subs):
            return None
        red = tuple(a for a in range(E.order) if subs[a] not in Z.out_subs)
        if not red or Z.weight == 0:
            return None
        a_of_out = tuple(subs.index(w) for w in Z.out_subs)
        return E, E.ops[0], red, a_of_out, Z.weight

    # ---- rule 1: ratio fusion ---------------------------------------------

    def _fuse_ratios(self, n: EinsumNode) -> Node:
        ops = list(n.ops)
        in_subs = [tuple(s) for s in n.in_subs]
        weight = n.weight
        changed = False
        progress = True
        while progress:
            progress = False
            for p in range(len(ops)):
                factor = self._consume_power(ops, in_subs, p)
                if factor is not None:
                    weight = weight * factor
                    changed = progress = True
                    break
        if not changed:
            return n
        return self.b.einsum(
            ops, in_subs, n.out_subs, dict(enumerate(n.wire_dims)), weight, n.constraints
        )

    def _consume_power(self, ops: list, in_subs: list, p: int):
        """If ops[p] is pow(Z, -k) with Z a recognized denominator, try to
        consume ONE power against another operand (in place). Returns the
        scalar weight factor the rewrite contributes, or None."""
        R = ops[p]
        if not (isinstance(R, MapNode) and R.op == "pow" and len(R.ops) == 1):
            return None
        k = R.params[0]
        try:
            ki = int(k)
        except (TypeError, ValueError):
            return None
        if ki != k or ki >= 0:
            return None
        Z = R.ops[0]
        prm = R.perms[0]
        if len(set(prm)) != len(prm):
            return None
        sr = in_subs[p]
        # zwire[j] = the parent wire carrying Z's axis j
        zwire = [None] * Z.order
        for t, j in enumerate(prm):
            zwire[j] = sr[t]

        def shrink():
            if ki + 1 == 0:
                del ops[p], in_subs[p]
            else:
                ops[p] = self.b.map("pow", (ki + 1,), [Z], [prm])

        # exact cancellation: Z itself as a fully aligned co-operand
        for q in range(len(ops)):
            if q == p or ops[q] is not Z or list(in_subs[q]) != zwire:
                continue
            if ki + 1 == 0:
                for idx in sorted((p, q), reverse=True):
                    del ops[idx], in_subs[idx]
            else:
                ops[p] = self.b.map("pow", (ki + 1,), [Z], [prm])
                del ops[q], in_subs[q]
            return sympy.Integer(1)

        # (a) softmax family: exp(X)[a] * (w_z * sum exp(X))^-1[kept a]
        se = self._sum_exp(Z)
        if se is not None:
            E, X, red, a_of_out, w_z = se
            for q in range(len(ops)):
                if q == p or ops[q] is not E:
                    continue
                s2 = in_subs[q]
                if len(set(s2)) != len(s2):
                    continue
                if all(s2[a_of_out[j]] == zwire[j] for j in range(Z.order)):
                    ops[q] = self.b.reduce("softmax", red, X)
                    shrink()
                    return sympy.Integer(1) / sympy.sympify(w_z)
            return None

        # (c) softmax denominator: exp(X) * softmax(X)^-1 == sum-exp(X),
        #     constant over the softmax axes (they drop from the operand's
        #     wires). Gradient-of-log graphs are full of exp/softmax pairs
        #     split by factoring; each match consumes one power. No new
        #     instability: the sum-exp overflows exactly when the already
        #     present exp operand does.
        if isinstance(Z, ReduceNode) and Z.op == "softmax":
            X = Z.ops[0]
            for q in range(len(ops)):
                if q == p:
                    continue
                E = ops[q]
                if not (
                    isinstance(E, MapNode)
                    and E.op == "exp"
                    and E.ops[0] is X
                    and E.perms[0] == tuple(range(E.order))
                ):
                    continue
                if list(in_subs[q]) != zwire:
                    continue
                kept = tuple(a for a in range(Z.order) if a not in Z.axes)
                zsum = self.b.einsum(
                    [E], [tuple(range(E.order))], kept, dict(enumerate(E.dims))
                )
                ops[q] = zsum
                in_subs[q] = tuple(zwire[a] for a in kept)
                shrink()
                return sympy.Integer(1)
            return None

        # (b) tanh family: divide an aligned co-operand by a*(exp(Y)+exp(-Y))
        pe = self._exp_pair(Z)
        if pe is not None and pe[2] == pe[3] and pe[2] != 0:
            for q in range(len(ops)):
                if q == p:
                    continue
                s2 = in_subs[q]
                pos, ok = [], True
                for j in range(Z.order):
                    if s2.count(zwire[j]) != 1:
                        ok = False
                        break
                    pos.append(s2.index(zwire[j]))
                if not ok:
                    continue
                d = self._div(ops[q], tuple(pos), Z, pe, 0)
                if d is not None:
                    ops[q] = d
                    shrink()
                    return sympy.Integer(1)
        return None

    # ---- recursive division by a cosh denominator ---------------------------

    def _div(self, node: Node, pos: tuple, P: Node, pe, depth: int):
        """Return node' == node / P, with P axis j aligned to node axis pos[j],
        or None. P == a*(exp(Y) + exp(-Y))[pperm], pe = (Y, pperm, a, a)."""
        if depth > MAX_DIV_DEPTH:
            return None
        key = (id(node), pos, id(P))
        if key in self._div_memo:
            return self._div_memo[key]
        res = self._div_impl(node, pos, P, pe, depth)
        self._div_memo[key] = res
        return res

    def _div_impl(self, node: Node, pos: tuple, P: Node, pe, depth: int):
        Y, pperm, _a, _ = pe
        negY = self._negate(Y)
        aligned = len(pos) == node.order  # pos is injective by construction

        # alpha*exp(Y) + beta*exp(-Y) over the SAME Y (covers node is P: -> 1)
        ep = self._exp_pair(node) if aligned else None
        if ep is not None:
            Y2, p2, alpha, beta = ep
            if all(p2[pos[j]] == pperm[j] for j in range(len(pos))):
                if Y2 is Y:
                    return self._half_tanh(node, pos, pe, alpha, beta)
                if Y2 is negY or self._negate(Y2) is Y:
                    return self._half_tanh(node, pos, pe, beta, alpha)

        # bare exp(+-Y): exp(Y)/(2a cosh(Y)) = (1 +- tanh(Y)) / (2a)
        if aligned and isinstance(node, MapNode) and node.op == "exp":
            (Y2,) = node.ops
            pm = node.perms[0]
            if all(pm[pos[j]] == pperm[j] for j in range(len(pos))):
                if Y2 is Y:
                    return self._half_tanh(node, pos, pe, 1, 0)
                if Y2 is negY or self._negate(Y2) is Y:
                    return self._half_tanh(node, pos, pe, 0, 1)

        # Linear: division distributes over the terms (one P total)
        if aligned and isinstance(node, LinearNode):
            new_terms = []
            for t, pm in zip(node.terms, node.perms):
                ti = self._div(t, tuple(pm[pos[j]] for j in range(len(pos))), P, pe, depth + 1)
                if ti is None:
                    return None
                new_terms.append(ti)
            return self.b.linear(new_terms, [tuple(pm) for pm in node.perms], list(node.weights))

        # Einsum: divide one operand that carries all of P's (free) wires
        if isinstance(node, EinsumNode) and node.ops:
            pw = [node.out_subs[pos[j]] for j in range(len(pos))]
            if len(set(pw)) != len(pw):
                return None
            for q, (opq, sq) in enumerate(zip(node.ops, node.in_subs)):
                u, ok = [], True
                for w in pw:
                    if sq.count(w) != 1:
                        ok = False
                        break
                    u.append(sq.index(w))
                if not ok:
                    continue
                d = self._div(opq, tuple(u), P, pe, depth + 1)
                if d is not None:
                    ops2 = list(node.ops)
                    ops2[q] = d
                    return self.b.einsum(
                        ops2,
                        [tuple(s) for s in node.in_subs],
                        node.out_subs,
                        dict(enumerate(node.wire_dims)),
                        node.weight,
                        node.constraints,
                    )
        return None

    def _half_tanh(self, node: Node, pos: tuple, pe, alpha, beta) -> Node:
        """(alpha*exp(Y) + beta*exp(-Y)) / (a*(exp(Y)+exp(-Y)))
        == (alpha+beta)/(2a) + (alpha-beta)/(2a) * tanh(Y), in node's axes."""
        Y, pperm, a, _ = pe
        a = sympy.sympify(a)
        c1 = (sympy.sympify(alpha) + sympy.sympify(beta)) / (2 * a)
        c2 = (sympy.sympify(alpha) - sympy.sympify(beta)) / (2 * a)
        n = node.order
        perm_t = [0] * n
        for j in range(n):
            perm_t[pos[j]] = pperm[j]
        terms, perms, ws = [], [], []
        if c2 != 0:
            terms.append(self.b.map("tanh", (), [Y], [tuple(perm_t)]))
            perms.append(_identity(n))
            ws.append(c2)
        if c1 != 0 or not terms:
            ones = self.b.einsum([], [], _identity(n), dict(enumerate(node.dims)))
            terms.append(ones)
            perms.append(_identity(n))
            ws.append(c1)
        return self.b.linear(terms, perms, ws)

    # ---- rule 2: push exp through a Linear operand ---------------------------

    def _push_exp(self, n: EinsumNode) -> Node:
        """einsum(exp(X), Linear(t_1..t_m), ...) -> einsum(Linear(t_i * exp(X)), ...)
        when EVERY term absorbs the exp into a softmax via ratio fusion."""
        for p, (E, se) in enumerate(zip(n.ops, n.in_subs)):
            if not (isinstance(E, MapNode) and E.op == "exp" and E.perms[0] == _identity(E.order)):
                continue
            if len(set(se)) != len(se):
                continue
            for q, (L, sl) in enumerate(zip(n.ops, n.in_subs)):
                if q == p or not isinstance(L, LinearNode):
                    continue
                if len(set(sl)) != len(sl) or not set(se) <= set(sl):
                    continue
                new_L = self._push_terms(E, se, L, sl)
                if new_L is None:
                    continue
                ops = [op for r, op in enumerate(n.ops) if r not in (p, q)] + [new_L]
                subs = [tuple(s) for r, s in enumerate(n.in_subs) if r not in (p, q)] + [tuple(sl)]
                return self.b.einsum(
                    ops, subs, n.out_subs, dict(enumerate(n.wire_dims)), n.weight, n.constraints
                )
        return n

    def _push_terms(self, E: Node, se: tuple, L: LinearNode, sl: tuple):
        terms, perms = [], []
        for t, pm in zip(L.terms, L.perms):
            r = self._mul_fuse(t, pm, E, se, sl)
            if r is None:
                return None
            t2, pm2 = r
            terms.append(t2)
            perms.append(pm2)
        return self.b.linear(terms, perms, list(L.weights))

    def _mul_fuse(self, t: Node, pm: tuple, E: Node, se: tuple, sl: tuple):
        """Build t * E (aligned through the Linear's perm) and fuse; succeeds
        only if no bare E operand remains. Returns (node, perm) or None."""
        if isinstance(t, EinsumNode) and t.ops and len(set(t.out_subs)) == len(t.out_subs):
            # splice E into the term's own wire space
            e_subs = tuple(t.out_subs[pm[sl.index(w)]] for w in se)
            cand = self.b.einsum(
                list(t.ops) + [E],
                [tuple(s) for s in t.in_subs] + [e_subs],
                t.out_subs,
                dict(enumerate(t.wire_dims)),
                t.weight,
                t.constraints,
            )
            fused = self._fuse_ratios(cand) if isinstance(cand, EinsumNode) else cand
            if self._holds_exp(fused, E):
                return None
            return fused, pm
        if isinstance(t, MapNode) and t.op == "pow":
            # wrap (t, E) in a fresh einsum in the Linear's axis space
            nl = len(sl)
            subs_t = tuple(pm.index(a) for a in range(t.order))
            e_subs = tuple(sl.index(w) for w in se)
            wire_dims = {j: t.dims[pm[j]] for j in range(nl)}
            cand = self.b.einsum([t, E], [subs_t, e_subs], _identity(nl), wire_dims)
            fused = self._fuse_ratios(cand) if isinstance(cand, EinsumNode) else cand
            if self._holds_exp(fused, E):
                return None
            return fused, _identity(nl)
        return None

    @staticmethod
    def _holds_exp(node: Node, E: Node) -> bool:
        if node is E:
            return True
        if isinstance(node, (EinsumNode, LinearNode)):
            return any(op is E for op in node.operands())
        return False

    @staticmethod
    def _is_neg_pow(op: Node) -> bool:
        if not (isinstance(op, MapNode) and op.op == "pow" and len(op.ops) == 1):
            return False
        k = op.params[0]
        try:
            ki = int(k)
        except (TypeError, ValueError):
            return False
        return ki == k and ki < 0

    def _push_base(self, n: EinsumNode) -> Node:
        """einsum(A, C, ...) with C an einsum holding pow(A, -k) (or, for
        A = exp(X), pow(w_z * sum exp(X), -k)): splice A into C and fuse.
        This co-locates numerator/denominator pairs that the factoring pass
        left on opposite sides of a contraction boundary."""
        for p, (A, sa) in enumerate(zip(n.ops, n.in_subs)):
            if isinstance(A, ConstNode) or (isinstance(A, EinsumNode) and not A.ops):
                continue
            if len(set(sa)) != len(sa):
                continue
            for q, (C, sc) in enumerate(zip(n.ops, n.in_subs)):
                if q == p or not (isinstance(C, EinsumNode) and C.ops):
                    continue
                if len(set(sc)) != len(sc) or not set(sa) <= set(sc):
                    continue
                if len(set(C.out_subs)) != len(C.out_subs):
                    continue
                # quick precheck: C holds a reciprocal whose base pairs with A
                def pairs(R: Node) -> bool:
                    if not self._is_neg_pow(R):
                        return False
                    Z = cast(MapNode, R).ops[0]  # _is_neg_pow verified R is a pow MapNode
                    if Z is A:
                        return True
                    se = self._sum_exp(Z)
                    return se is not None and se[0] is A
                if not any(pairs(op) for op in C.ops):
                    continue
                a_subs = tuple(C.out_subs[sc.index(w)] for w in sa)
                cand = self.b.einsum(
                    list(C.ops) + [A],
                    [tuple(s) for s in C.in_subs] + [a_subs],
                    C.out_subs,
                    dict(enumerate(C.wire_dims)),
                    C.weight,
                    C.constraints,
                )
                fused = self._fuse_ratios(cand) if isinstance(cand, EinsumNode) else cand
                if self._holds_exp(fused, A):
                    continue  # A survived: pushing would only duplicate it
                ops = [op for r, op in enumerate(n.ops) if r not in (p, q)] + [fused]
                subs = [tuple(s) for r, s in enumerate(n.in_subs) if r not in (p, q)] + [tuple(sc)]
                return self.b.einsum(
                    ops, subs, n.out_subs, dict(enumerate(n.wire_dims)), n.weight, n.constraints
                )
        return n

    # ---- rule 3: log fusion --------------------------------------------------

    def _rewrite_log(self, n: MapNode) -> Node:
        if n.op != "log" or len(n.ops) != 1:
            return n
        (S,) = n.ops
        pl = n.perms[0]

        # log(softmax) -> log_softmax (also through a pure-view einsum)
        if isinstance(S, ReduceNode) and S.op == "softmax":
            ls = self.b.reduce("log_softmax", S.axes, S.ops[0])
            return self._permuted(ls, pl)
        if (
            isinstance(S, EinsumNode)
            and len(S.ops) == 1
            and isinstance(S.ops[0], ReduceNode)
            and S.ops[0].op == "softmax"
            and S.weight == 1
            and not S.constraints
            and set(S.in_subs[0]) <= set(S.out_subs)  # pure view: no contraction
        ):
            sm = S.ops[0]
            ls = self.b.reduce("log_softmax", sm.axes, sm.ops[0])
            view = self.b.einsum(
                [ls], [tuple(S.in_subs[0])], S.out_subs, dict(enumerate(S.wire_dims))
            )
            return self._permuted(view, pl)

        # log(w_z * sum exp(X)) -> max-shifted stable logsumexp
        se = self._sum_exp(S)
        if se is not None:
            if id(n) in self._log_done or self._already_shifted(se[1]):
                return n
            E, X, red, a_of_out, w_z = se
            w_z = sympy.sympify(w_z)
            if w_z.is_positive is not True:
                return n
            kept = [a for a in range(E.order) if a not in red]
            M = self.b.reduce("max", red, X)  # axes: kept, in X order
            x_dims = dict(enumerate(X.dims))
            mb = self.b.einsum([M], [tuple(kept)], _identity(E.order), x_dims)
            D = self.b.linear([X, mb], [_identity(E.order)] * 2, [1, -1])
            S2 = self.b.einsum([self.b.map("exp", (), [D])], [_identity(E.order)], a_of_out, x_dims)
            L2 = self.b.map("log", (), [S2])
            self._log_done.add(id(L2))
            m_perm = tuple(kept.index(a) for a in a_of_out)
            terms, perms = [M, L2], [m_perm, _identity(S.order)]
            ws: list = [1, 1]
            if w_z != 1:
                ones = self.b.einsum([], [], _identity(S.order), dict(enumerate(S.dims)))
                terms.append(ones)
                perms.append(_identity(S.order))
                ws.append(sympy.log(w_z))
            return self._permuted(self.b.linear(terms, perms, ws), pl)
        return n

    @staticmethod
    def _already_shifted(X: Node) -> bool:
        """True when X has the max-shifted form Y - broadcast(max(Y)): the
        logsumexp rewrite must not re-fire on its own output. Structural (not
        per-instance memo): the pass runs more than once per specialization."""
        if not (isinstance(X, LinearNode) and len(X.terms) == 2):
            return False
        for i in (0, 1):
            if sympy.sympify(X.weights[i]) != -1:
                continue
            mb, other = X.terms[i], X.terms[1 - i]
            if isinstance(mb, EinsumNode) and len(mb.ops) == 1:
                mb = mb.ops[0]
            if isinstance(mb, ReduceNode) and mb.op == "max" and mb.ops[0] is other:
                return True
        return False

    def _permuted(self, node: Node, perm: tuple) -> Node:
        """Apply a MapNode-style perm (out axis k = node axis perm[k]) as a view."""
        if perm == _identity(node.order):
            return node
        return self.b.einsum(
            [node], [_identity(node.order)], tuple(perm), dict(enumerate(node.dims))
        )
