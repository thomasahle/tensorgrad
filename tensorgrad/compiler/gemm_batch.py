"""Shared-operand GEMM batching: k matmuls that read the same operand
become ONE.

The motivating pattern is attention's projections -- x @ Wq, x @ Wk,
x @ Wv (and their gradient transposes) are separate einsum nodes reading
the same activation; each is a separate GEMM launch that Inductor cannot
fuse (library calls), while XLA batches them. This pass groups 2-operand
einsums that are identical except for the non-shared operand, stacks
those operands along a new leading wire (a `stack` cell), reruns the ONE
batched einsum through the ordinary contraction machinery (opt_einsum +
the matmul cells -- no new GEMM code), and hands each original consumer a
zero-cost `select` view. FLOP-neutral by construction; misses degrade
performance, never correctness.

v1 scope: no constraints, unit weight, identical in/out wiring and
identical non-shared operand shapes across the group, no repeated wires,
and a size ceiling (batching huge GEMMs trades cache locality for a
launch, a bad deal).
"""

from collections import defaultdict

import sympy

from tensorgrad.compiler.ir import Builder, EinsumNode, Node, toposort
from tensorgrad.compiler.peepholes import _rebuild

# The trade is fewer launches for k-times-larger transients, which is only
# a good deal while the transients stay cache-sized. Ceiling on BOTH the
# stacked operand and the batched output (elements; 2^18 = 1 MB fp32).
# Memory-critical programs (the factor tests pin peak-intermediate bounds)
# must stay untouched.
_SIZE_CEILING = 1 << 18

# DEFAULT OFF -- measured a 69% REGRESSION on the motivating gpt target
# (eager, 2-thread CPU): torch.stack is a per-step copy (weights are step
# inputs, so the stack cannot hoist), and the batched contraction renders
# through torch.matmul/einsum paths (CPU bmm is a slice loop, not one
# GEMM), i.e. strictly worse than the k separate mm cells it replaced.
# Kept as an opt-in experiment; the viable follow-up is the CAT
# formulation -- concatenate along the FREE axis so the result is ONE true
# torch.mm -- which needs flattened-wire reshapes at the IR level.
GEMM_BATCHING = False


def _numel(dims, dim_map) -> int:
    total = 1
    for d in dims:
        if isinstance(d, int):
            total *= d
            continue
        e = sympy.sympify(d).subs(dim_map)
        if not e.is_Integer:
            return 1 << 62  # unresolved symbolic size: treat as too big
        total *= int(e)
    return total


def _signature(nd: EinsumNode, side: int):
    """Group key: identical wiring and shared operand; members differ only
    in the non-shared operand (which must be shape-identical)."""
    other = nd.ops[1 - side]
    return (
        id(nd.ops[side]),
        side,
        nd.in_subs,
        nd.out_subs,
        nd.wire_dims,
        tuple(other.dims),
    )


def _eligible(nd: Node) -> bool:
    if not isinstance(nd, EinsumNode) or len(nd.ops) != 2:
        return False
    if nd.constraints or nd.weight != 1:
        return False
    for subs in nd.in_subs:
        if len(set(subs)) != len(subs):
            return False  # diagonals: einsum territory
    return True


def batch_shared_gemms(builder: Builder, outputs: list, dims: dict) -> list:
    if not GEMM_BATCHING:
        return outputs
    dim_map = {sympy.Symbol(k) if isinstance(k, str) else k: v for k, v in dims.items()}
    roots = [n for n, _ in outputs]
    order = toposort(roots)

    groups: dict[tuple, list[EinsumNode]] = defaultdict(list)
    for nd in order:
        if _eligible(nd):
            for side in (0, 1):
                groups[_signature(nd, side)].append(nd)

    # A node may appear in two groups (once per side); claim greedily by
    # group size so each node is batched at most once.
    claimed: set[int] = set()
    plans: list[tuple[tuple, list[EinsumNode]]] = []
    for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        members = [m for m in members if id(m) not in claimed]
        if len(members) < 2:
            continue
        k = len(members)
        if k * _numel(members[0].ops[1 - key[1]].dims, dim_map) > _SIZE_CEILING:
            continue  # stacked operand too large
        if k * _numel(members[0].dims, dim_map) > _SIZE_CEILING:
            continue  # batched output too large
        for m_ in members:
            claimed.add(id(m_))
        plans.append((key, members))
    if not plans:
        return outputs

    # Build the batched replacement for each plan.
    rep: dict[int, Node] = {}
    for key, members in plans:
        side = key[1]
        shared = members[0].ops[side]
        others = [m.ops[1 - side] for m in members]
        stacked = builder.fused_fwd(
            "stack", {}, others, (len(others),) + tuple(others[0].dims)
        )
        proto = members[0]
        all_wires = [w for subs in proto.in_subs for w in subs] + list(proto.out_subs)
        k_wire = max(all_wires) + 1
        in_subs = list(proto.in_subs)
        in_subs[1 - side] = (k_wire,) + tuple(proto.in_subs[1 - side])
        out_subs = (k_wire,) + tuple(proto.out_subs)
        wire_dims = dict(enumerate(proto.wire_dims)) | {k_wire: len(others)}
        ops = [None, None]
        ops[side] = shared
        ops[1 - side] = stacked
        big = builder.einsum(ops, in_subs, out_subs, wire_dims, proto.weight, proto.constraints)
        for i, m_ in enumerate(members):
            rep[id(m_)] = builder.fused_fwd(
                "select", {"index": i}, [big], tuple(m_.dims)
            )

    # Memoized topological rebuild through Builder methods (stabilize's
    # discipline), remapping consumers onto the select views.
    memo: dict[int, Node] = {}
    for nd in order:
        ops = [memo.get(id(op), op) for op in nd.operands()]
        cur = rep.get(id(nd)) or _rebuild(builder, nd, ops)
        memo[id(nd)] = cur
    return [(memo[id(n)], o) for n, o in outputs]
