"""IR consolidation: Schwartz-Zippel value numbering modulo axis permutation.

Why hash-consing and canonical forms are not enough: the 3-layer gpt-nano
loss+53grads program factors to 2925 compute nodes of which 83% are PRIVATE
to a single gradient output — every parameter's adjoint walk re-derives the
same downward cotangent chain (dL/dx at each residual boundary), but each
gradient was symbolically derived, simplified and factored separately, so
the shared chain appears as 53 strands that compute EQUAL TENSORS through
DIFFERENT contraction groupings and axis orders. No structural canonical
form unifies them (measured: a full recursive quotient modulo axis
permutation removes only 568 of 2925 nodes); the equality is algebraic, not
syntactic.

So this pass merges by VALUE. Every node is evaluated exactly mod P at
K_TRIALS seeded random points (random small dims, random integer variable
tensors — the szfp machinery, whose per-step exactness makes fingerprints
of algebraically equal nodes provably collide), and nodes whose values agree
at every trial under one consistent axis permutation are merged: the
topologically first member stays, every other member's consumers are rewired
to read it through the permutation (a free view). Non-polynomial atoms
(exp/tanh/softmax/...) are seeded random tensors keyed by their inputs'
PERM-CANONICAL fingerprints and generated in canonical orientation, so the
atom semantics is permutation-EQUIVARIANT exactly like the real ops
(tanh(x)ᵀ and tanh(xᵀ) fingerprint equal); gather/one_hot atoms stay
orientation-exact (conservative: missed merges only).

Soundness: a false merge requires two genuinely different polynomials to
agree at K_TRIALS independent random points under some size-preserving
permutation — probability <= (n_perms * deg/P)^K per pair — AND to survive
an independent fresh-seed gate: after the rewrite, the pass re-evaluates
the original and consolidated outputs side by side at GATE_TRIALS new
points and REFUSES the whole rewrite on any mismatch (falling back to the
input program). The gate also turns any bookkeeping bug in the rewiring
into a no-op instead of a miscompile. Merging never changes what a kept
node computes — consumers just stop reading redundant twins — so float
rounding of surviving nodes is unchanged and exact-cancellation
arrangements from the stabilization pass are preserved (twins collapsing
onto one node can only make cancellations MORE exact).

Measured on the 3-layer gpt-nano natural-API program (loss + 53 grads):
2925 -> ~1000 compute nodes, the per-head cotangent chains collapsing into
one shared reverse-mode spine.
"""

from itertools import permutations

import numpy as np

from tensorgrad.compiler import szfp as _sz
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

# Master toggle (tests compare consolidated vs raw programs).
CONSOLIDATE = True

# Random evaluation points used for grouping, and for the independent
# accept/reject gate. More trials = exponentially smaller false-merge
# probability; each trial costs one exact evaluation of the DAG at tiny
# random dims (2..5), independent of the real compile dims.
K_TRIALS = 3
GATE_TRIALS = 2
SEED = 0x5EED

# Only nodes of order <= this participate in modulo-permutation matching
# (n_perms grows factorially); higher-order nodes still merge when their
# orientations match exactly.
MAX_PERM_ORDER = 6


# ---------------------------------------------------------------------------
# Permutation-equivariant trial evaluation
# ---------------------------------------------------------------------------


def _perms_mapping(src_shape: tuple, dst_shape: tuple) -> list:
    """All permutations p with transpose(src, p).shape == dst_shape, i.e.
    src_shape[p[j]] == dst_shape[j]. Only axes of equal size may move onto
    each other, so the count is the product of factorials of the
    size-multiplicities, not order!."""
    n = len(src_shape)
    if sorted(src_shape) != sorted(dst_shape):
        return []
    if n <= 1:
        return [tuple(range(n))]
    return [
        p
        for p in permutations(range(n))
        if tuple(src_shape[a] for a in p) == tuple(dst_shape)
    ]


def _sort_perms(shape: tuple) -> list:
    """Permutations that bring `shape` into sorted (ascending) order — the
    canonical orientation. Twins in ANY axis order canonicalize into the
    same sorted-shape orientation."""
    return _perms_mapping(tuple(shape), tuple(sorted(shape)))


def _canon_arr(arr: np.ndarray):
    """(canonical bytes, perm) — the lexicographically smallest transpose
    over the sorted-shape orientations. Two arrays equal up to an axis
    permutation (in any axis order) have EQUAL canonical bytes."""
    if arr.ndim <= 1:
        return arr.tobytes(), tuple(range(arr.ndim))
    best = None
    best_p = None
    for p in _sort_perms(arr.shape):
        b = np.ascontiguousarray(np.transpose(arr, p)).tobytes()
        if best is None or b < best:
            best, best_p = b, p
    assert best is not None and best_p is not None  # _sort_perms is never empty
    return best, best_p


def _inv_perm(p: tuple) -> tuple:
    q = [0] * len(p)
    for i, a in enumerate(p):
        q[a] = i
    return tuple(q)


def _eval_map_equivariant(node: MapNode, vals, assign, ctx) -> np.ndarray:
    """Like szfp's map atoms, but keyed by the inputs' perm-canonical
    fingerprints and generated in canonical orientation: the atom function
    commutes with axis permutation, exactly like a real elementwise op."""
    aligned = []
    for opnd, perm in zip(node.ops, node.perms):
        arr = vals[id(opnd)]
        if perm != tuple(range(len(perm))):
            arr = np.transpose(arr, perm)
        aligned.append(arr)
    if node.op == "pow":
        k = _sz._as_int_exponent(node.params[0])
        if k is not None:
            return _sz._eval_pow_int(aligned[0], k)
    # Joint canonicalization: ONE permutation applied to all aligned inputs
    # (they share the output orientation), chosen by the first input's
    # canonical form with the rest as tie-breakers.
    perms = _sort_perms(aligned[0].shape) if aligned[0].ndim <= MAX_PERM_ORDER else [
        tuple(range(aligned[0].ndim))
    ]
    best = None
    best_p = None
    for p in perms:
        key = tuple(np.ascontiguousarray(np.transpose(a, p)).tobytes() for a in aligned)
        if best is None or key < best:
            best, best_p = key, p
    assert best is not None and best_p is not None  # perms is never empty
    key = ("atom-map-eq", node.op, repr(node.params), tuple(_sz._h(b) for b in best))
    core = _sz._rand_tensor(tuple(aligned[0].shape[a] for a in best_p), *key)
    return np.transpose(core, _inv_perm(best_p))


def _eval_reduce_equivariant(node: ReduceNode, vals, assign, ctx) -> np.ndarray:
    """Shape-preserving reductions (softmax family): atom keyed by the
    operand's canonical fingerprint plus the reduced axes in canonical
    coordinates. max/argmax (shape-changing) stay orientation-exact."""
    arr = vals[id(node.ops[0])]
    if node.op in ("softmax", "log_softmax") and arr.ndim <= MAX_PERM_ORDER:
        b, p = _canon_arr(arr)
        ip = _inv_perm(p)
        axes_c = tuple(sorted(ip[a] for a in node.axes))
        key = ("atom-reduce-eq", node.op, axes_c, _sz._h(b))
        core = _sz._rand_tensor(tuple(arr.shape[a] for a in p), *key)
        return np.transpose(core, ip)
    dims = arr.shape if node.op in ("softmax", "log_softmax", "argsort") else tuple(
        d for i, d in enumerate(arr.shape) if i not in node.axes
    )
    key = ("atom-reduce", node.op, node.axes, _sz._vhash(arr), dims)
    return _sz._rand_tensor(dims, *key)


def _eval_node(node: Node, vals, assign, ctx) -> np.ndarray:
    if isinstance(node, MapNode):
        return _eval_map_equivariant(node, vals, assign, ctx)
    if isinstance(node, ReduceNode):
        return _eval_reduce_equivariant(node, vals, assign, ctx)
    return _sz._eval_node(node, vals, assign, ctx)


def _eval_all(nodes: list, ctx) -> dict:
    """One trial: exact mod-P values for every node (id -> array)."""
    syms = set()
    for n in nodes:
        syms |= _sz._node_syms(n)
    floors = _sz._row_const_floors(nodes)
    assign = {
        s: (lo := max(_sz._DIM_LO, floors.get(s.name, 0)))
        + _sz._h(ctx, "dim", s.name, floors.get(s.name, 0)) % (max(_sz._DIM_HI, lo + 3) - lo + 1)
        for s in syms
    }
    vals: dict[int, np.ndarray] = {}
    for n in nodes:
        vals[id(n)] = _eval_node(n, vals, assign, ctx)
    return vals


def _trials(nodes: list, k: int, seed) -> list[dict]:
    out = []
    for trial in range(k):
        for salt in range(_sz._MAX_RETRIES):
            try:
                out.append(_eval_all(nodes, (seed, trial, salt)))
                break
            except _sz._Retry:
                continue
        else:
            raise RuntimeError(f"consolidate: trial {trial} exceeded retries")
    return out


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def consolidate_outputs(builder: Builder, outputs) -> list:
    """Merge nodes whose values agree (mod P, K_TRIALS points) under one
    consistent axis permutation. `outputs` is [(node, edge_order), ...];
    returns the same structure. Falls back to `outputs` unchanged if the
    fresh-seed gate detects any output difference."""
    if not CONSOLIDATE:
        return list(outputs)
    roots = [n for n, _ in outputs]
    nodes = toposort(list(roots))
    try:
        trials = _trials(nodes, K_TRIALS, SEED)
    except RuntimeError:
        return list(outputs)

    # Group compute nodes by their perm-canonical value across all trials.
    groups: dict[tuple, list] = {}
    for n in nodes:
        if isinstance(n, (InputNode, ConstNode)):
            continue
        parts = []
        for vals in trials:
            arr = vals[id(n)]
            if arr.ndim <= MAX_PERM_ORDER:
                b, _ = _canon_arr(arr)
                shape_key = tuple(sorted(arr.shape))
            else:
                b = np.ascontiguousarray(arr).tobytes()
                shape_key = arr.shape
            parts.append((shape_key, _sz._h(b)))
        groups.setdefault(tuple(parts), []).append(n)

    # For every non-representative member, ONE permutation rho with
    # member value == transpose(rep value, rho) at EVERY trial
    # (member axis a <-> rep axis rho[a]), and matching SYMBOLIC dims
    # (random small dims can collide numerically across different symbols).
    # No consistent rho -> no merge.
    merge: dict[int, tuple] = {}  # id(member) -> (rep, rho)
    for members in groups.values():
        if len(members) < 2:
            continue
        rep = members[0]
        rep_sdims = tuple(str(d) for d in rep.dims)
        for m in members[1:]:
            if m.order != rep.order:
                continue
            m_sdims = tuple(str(d) for d in m.dims)
            cand = None
            if rep.order > MAX_PERM_ORDER:
                # exact-orientation grouping above: identity is the only
                # permutation ever tried for high-order nodes
                perms = [tuple(range(rep.order))]
            else:
                perms = _perms_mapping(trials[0][id(rep)].shape, trials[0][id(m)].shape)
            for p in perms:
                if tuple(rep_sdims[p[a]] for a in range(m.order)) != m_sdims:
                    continue
                if all(
                    np.array_equal(np.transpose(vals[id(rep)], p), vals[id(m)])
                    for vals in trials
                ):
                    cand = p
                    break
            if cand is not None:
                merge[id(m)] = (rep, cand)
    if not merge:
        return list(outputs)

    # ---- rebuild sweep -----------------------------------------------------
    # R: id(original) -> (final node, rho) with original axis a <-> final
    # axis rho[a]. Merged members resolve through their representative.
    R: dict[int, tuple] = {}

    def res(op: Node) -> tuple:
        return R.get(id(op), (op, tuple(range(op.order))))

    def rebuild(nd: Node):
        """Rebuild nd with replaced operands; returns (node, rho)."""
        if isinstance(nd, (InputNode, ConstNode)):
            return nd, tuple(range(nd.order))
        ident = tuple(range(nd.order))
        if isinstance(nd, EinsumNode):
            ops, in_subs = [], []
            for op, subs in zip(nd.ops, nd.in_subs):
                fop, rho = res(op)
                new_subs = [0] * len(subs)
                for a, w in enumerate(subs):
                    new_subs[rho[a]] = w
                ops.append(fop)
                in_subs.append(tuple(new_subs))
            return (
                builder.einsum(
                    ops, in_subs, nd.out_subs, dict(enumerate(nd.wire_dims)),
                    nd.weight, nd.constraints,
                ),
                ident,
            )
        if isinstance(nd, LinearNode):
            terms, perms = [], []
            for t, pm in zip(nd.terms, nd.perms):
                ft, rho = res(t)
                terms.append(ft)
                perms.append(tuple(rho[pm[j]] for j in range(nd.order)))
            return builder.linear(terms, perms, list(nd.weights)), ident
        if isinstance(nd, MapNode):
            ops, perms = [], []
            for op, pm in zip(nd.ops, nd.perms):
                fop, rho = res(op)
                ops.append(fop)
                perms.append(tuple(rho[pm[j]] for j in range(nd.order)))
            return builder.map(nd.op, nd.params, ops, perms), ident
        if isinstance(nd, ReduceNode):
            fop, rho = res(nd.ops[0])
            new_axes = tuple(sorted(rho[a] for a in nd.axes))
            new_node = builder.reduce(nd.op, new_axes, fop)
            if nd.op in ("softmax", "log_softmax", "argsort"):
                return new_node, rho
            kept_old = [a for a in range(nd.ops[0].order) if a not in nd.axes]
            kept_new = [b for b in range(fop.order) if b not in new_axes]
            return new_node, tuple(kept_new.index(rho[a]) for a in kept_old)
        if isinstance(nd, GatherNode):
            if nd.op == "one_hot":
                fop, rho = res(nd.ops[0])
                new_node = builder.one_hot(fop, nd.dims[0])
                return new_node, (0,) + tuple(1 + rho[a] for a in range(fop.order))
            table, idx = nd.ops
            ft, rho_t = res(table)
            fi, rho_i = res(idx)
            new_axis = rho_t[nd.axis]
            new_node = builder.gather(ft, fi, new_axis)
            k_old, k_new = idx.order, fi.order

            def new_pos_table(b):  # new-table axis -> position in new output
                return b if b < new_axis else b - 1 + k_new

            rho_out = []
            for a in range(table.order):  # old output = table axes with axis -> idx block
                if a == nd.axis:
                    rho_out.extend(new_axis + rho_i[j] for j in range(k_old))
                else:
                    rho_out.append(new_pos_table(rho_t[a]))
            return new_node, tuple(rho_out)
        return nd, ident

    for nd in nodes:
        if isinstance(nd, (InputNode, ConstNode)):
            continue
        hit = merge.get(id(nd))
        if hit is not None:
            rep, rho_vm = hit
            rep_final, rho_rep = R[id(rep)]
            R[id(nd)] = (rep_final, tuple(rho_rep[rho_vm[a]] for a in range(nd.order)))
        else:
            R[id(nd)] = rebuild(nd)

    # Outputs keep their edge order EXACTLY (the runtime pairs positions
    # with the pre-pass edge names): a root that merged into a permuted
    # representative is read through a pure permutation-view einsum, which
    # codegen emits as a free view.
    new_outputs = []
    for node, order in outputs:
        fnode, rho = res(node)
        if rho == tuple(range(len(rho))):
            new_outputs.append((fnode, order))
            continue
        subs = tuple(range(fnode.order))
        out_subs = tuple(rho[a] for a in range(len(order)))
        view = builder.einsum(
            [fnode], [subs], out_subs, dict(enumerate(fnode.dims))
        )
        new_outputs.append((view, order))

    # ---- independent fresh-seed gate ----------------------------------------
    both = [n for n, _ in outputs] + [n for n, _ in new_outputs]
    try:
        for trial in range(GATE_TRIALS):
            for salt in range(_sz._MAX_RETRIES):
                try:
                    vals = _eval_all(toposort(list(both)), (SEED ^ 0xA5A5A5, trial, salt))
                    break
                except _sz._Retry:
                    continue
            else:
                return list(outputs)
            for (o_n, o_e), (n_n, n_e) in zip(outputs, new_outputs):
                a = vals[id(o_n)]
                b = vals[id(n_n)]
                perm = tuple(n_e.index(name) for name in o_e)
                if not np.array_equal(a, np.transpose(b, perm)):
                    return list(outputs)
    except RuntimeError:
        return list(outputs)
    return new_outputs
