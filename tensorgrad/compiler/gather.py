"""Gather formation as a cost-driven IR pass (task #43).

The index_select rewrite — sum_v [idx == v] * T[..., v, ...] ==
T[..., idx, ...] — is a COST decision (dense one-hot contraction vs an
integer table lookup), so under the law "lowering makes no cost
decisions" it lives here, as the first pass of specialize(), not inside
lower.py's einsum construction. The pipeline point is identical
(post-lowering, pre-factoring), so the emitted programs are unchanged;
only the layer responsible moved.

The mirrored gradient pattern (class wire free, idx wires contracted) is
deliberately left as an einsum: codegen's scatter peephole (_try_scatter)
turns it into zeros().index_add_. A carrier that is purely STRUCTURAL —
an einsum of delta/ones constants, e.g. the gradient's
Delta(v,v') x Delta(d,d') jacobian — is never fused into: gathering would
materialize the dense structure (vocab^2 * d^2 for an embedding-table
gradient) that the factoring pass exists to dissolve.

Not szfp-gated: forming a GatherNode replaces the one_hot atom + exact
einsum with an atom-gather key — the documented atom-vocabulary
false-negative class (same as stabilize/cell formation), so an exact
before/after gate would always refuse.
"""

from tensorgrad.compiler.ir import (
    Builder,
    ConstNode,
    EinsumNode,
    FusedBwdNode,
    FusedFwdNode,
    GatherNode,
    LinearNode,
    MapNode,
    Node,
    ReduceNode,
)


def _structural(op: Node) -> bool:
    if isinstance(op, ConstNode):
        return True
    if isinstance(op, EinsumNode):
        return all(_structural(o) for o in op.ops)
    return False


def _fuse_in_einsum(b: Builder, node: EinsumNode) -> Node:
    """Apply the index_select rewrite inside one einsum; returns the node
    itself when nothing fused."""
    ops = list(node.ops)
    in_subs = [tuple(s) for s in node.in_subs]
    constrained = {w for row, _ in node.constraints for w, _ in row}
    changed_any = False
    changed = True
    while changed:
        changed = False
        for i, (op, subs) in enumerate(zip(ops, in_subs)):
            if not (isinstance(op, GatherNode) and op.op == "one_hot"):
                continue
            if len(set(subs)) != len(subs):
                continue  # diagonal one-hot: keep the dense fallback
            wc = subs[0]  # the class wire (one_hot's axis 0 is the class axis)
            if wc in node.out_subs or wc in constrained:
                continue  # class wire free: scatter/jacobian territory
            carriers = [j for j in range(len(ops)) if j != i and wc in in_subs[j]]
            if len(carriers) != 1 or in_subs[carriers[0]].count(wc) != 1:
                continue  # not a plain table contraction over the class wire
            if _structural(ops[carriers[0]]):
                continue  # delta-built jacobian: factoring dissolves it
            j = carriers[0]
            axis = in_subs[j].index(wc)
            ops[j] = b.gather(ops[j], op.ops[0], axis)
            in_subs[j] = in_subs[j][:axis] + tuple(subs[1:]) + in_subs[j][axis + 1 :]
            del ops[i], in_subs[i]
            changed = changed_any = True
            break
    if not changed_any:
        return node
    return b.einsum(
        ops,
        in_subs,
        tuple(node.out_subs),
        dict(enumerate(node.wire_dims)),
        node.weight,
        list(node.constraints),
    )



def form_gathers(b: Builder, outputs: list) -> list:
    """Rewrite every einsum in the program, bottom-up with identity-preserving
    memoization (unchanged subgraphs keep their nodes, so hash-cons sharing
    survives)."""
    from tensorgrad.compiler.ir import toposort

    memo: dict[int, Node] = {}
    order = toposort([n for n, _ in outputs])
    for node in order:
        kids = node.operands()
        new_kids = [memo.get(id(k), k) for k in kids]
        cur = node if all(a is c for a, c in zip(kids, new_kids)) else b.with_ops(node, new_kids)
        if isinstance(cur, EinsumNode):
            cur = _fuse_in_einsum(b, cur)
        memo[id(node)] = cur
    return [(memo.get(id(n), n), o) for n, o in outputs]
