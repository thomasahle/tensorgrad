"""Linear-algebra technology-mapping peepholes on the IR.

Two rewrites that map algebra the language keeps explicit onto the better
library kernel -- the same philosophy as the stabilize pass (exp/sum-exp
-> softmax), applied to linalg:

  * inverse-then-contract:  einsum(inv(A), b) contracting one matrix edge
    of the inverse with a VECTOR b becomes solve(A, b) -- one
    torch.linalg.solve instead of a materialized inverse plus a matmul
    (faster, and numerically the right thing). There is deliberately no
    F.solve in the language: users write the algebra inverse(A) @ b and
    the compiler picks the kernel.

  * log-of-det:  log(det(A)) becomes slogdet(A) -- log|K| of a 128-point
    GP kernel genuinely underflows float32 as a det, while slogdet
    computes it exactly.

Both are purely syntactic, node-local rewrites (no cost model, no
search); misses degrade performance, never correctness. The pass follows
stabilize's discipline: a memoized topological rebuild through Builder
methods, so every surviving node stays interned and indexed; orphaned
inverse/det nodes are dropped by codegen's dead-line pruning when nothing
else reads them.
"""

from typing import Any, Optional

from tensorgrad.compiler.ir import (
    Builder,
    EinsumNode,
    FusedBwdNode,
    FusedFwdNode,
    GatherNode,
    LinearNode,
    MapNode,
    Node,
    ReduceNode,
    toposort,
)


def _try_solve(b: Builder, node: Node) -> Optional[Node]:
    """einsum contracting ONE matrix edge of inverse(A) with a vector rhs
    -> solve(A, rhs). v1 scope: plain matrix inverse (order 2, no batch),
    vector right-hand side, no constraints/diagonals/weights."""
    if not isinstance(node, EinsumNode) or len(node.ops) != 2:
        return None
    if node.constraints or node.weight != 1:
        return None
    for ii in (0, 1):
        inv = node.ops[ii]
        if isinstance(inv, FusedFwdNode) and inv.cell_name == "inverse" and inv.order == 2:
            break
    else:
        return None
    rhs = node.ops[1 - ii]
    inv_subs, rhs_subs = node.in_subs[ii], node.in_subs[1 - ii]
    if rhs.order != 1 or len(set(inv_subs)) != 2:
        return None
    (rhs_wire,) = rhs_subs
    if rhs_wire not in inv_subs or rhs_wire in node.out_subs:
        return None  # the rhs axis must contract with the inverse
    keep_wire = inv_subs[0] if inv_subs[1] == rhs_wire else inv_subs[1]
    if node.out_subs != (keep_wire,):
        return None
    # Which inverse axis contracted? The inverse cell's convention
    # (cells.py) puts axes (e2, e1) with inv[e2, e1] = inv(A[e1, e2]), so
    # contracting axis 1 computes A^-1 rhs (plain solve) and contracting
    # axis 0 computes A^-T rhs (solve against the transpose).
    transposed = inv_subs.index(rhs_wire) == 0
    A = inv.ops[0]  # already aligned (e1, e2) by the inverse cell's lowering
    return b.fused_fwd("solve", {"transposed": transposed}, [A, rhs], node.dims)


def _try_slogdet(b: Builder, node: Node) -> Optional[Node]:
    """log(det(A)) -> slogdet(A)[1] (log|det|; SPD kernels in mind)."""
    if not (isinstance(node, MapNode) and node.op == "log" and len(node.ops) == 1):
        return None
    det = node.ops[0]
    if isinstance(det, FusedFwdNode) and det.cell_name == "det":
        return b.fused_fwd("slogdet", {}, list(det.ops), node.dims)
    return None


def _rebuild(b: Builder, nd: Node, ops: list) -> Node:
    """Reconstruct `nd` over remapped operands through Builder methods
    (mirrors stabilize._rebuild), so interning and node indexing hold."""
    if not ops or all(o is p for o, p in zip(ops, nd.operands())):
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
    if isinstance(nd, FusedFwdNode):
        return b.fused_fwd(nd.cell_name, nd.params_dict(), ops, nd.dims, nd.layout, nd.which)
    if isinstance(nd, FusedBwdNode):
        return b.fused_bwd(nd.cell_name, nd.which, nd.params_dict(), ops, nd.dims, nd.layout)
    return nd


def linalg_peepholes(builder: Builder, outputs: list) -> list:
    roots = [n for n, _ in outputs]
    order = toposort(roots)
    memo: dict[int, Node] = {}
    changed = False
    for nd in order:
        ops = [memo.get(id(op), op) for op in nd.operands()]
        cur = _rebuild(builder, nd, ops)
        new = _try_solve(builder, cur) or _try_slogdet(builder, cur)
        if new is not None:
            cur = new
        if cur is not nd:
            changed = True
        memo[id(nd)] = cur
    if not changed:
        return outputs
    return [(memo[id(n)], o) for n, o in outputs]
