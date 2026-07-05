"""Global physical-layout assignment for the compiler IR.

tensorgrad semantics never fix an axis order: IR nodes have *logical* axes
(positions in node.dims), but the torch tensor that materializes a node may
store those axes in any physical order. This pass chooses, for every node,
the physical order that lets the emitting kernel produce it FOR FREE and
lets its consumers read it WITHOUT permute-copies.

A layout is a tuple `phys` with phys[p] = the logical axis stored at
physical position p (a permutation of range(node.order)).

Preferences (votes) flow from consumers to producers in one reverse-topo
sweep — greedy propagation outward from the "hard" cells:

  * matmul-shaped einsums (see `matmul_groups`) can only cheaply produce
    (batch..., M..., N...) block layouts, and want their operands as
    (batch..., M..., K...) / (batch..., K..., N...) — with the batch block
    leading and identically ordered, because BLAS absorbs a transpose of
    the two trailing groups per operand but batch dims must stay leading;
  * softmax/log_softmax and single-operand reductions prefer the reduced
    axes physically last (contiguous reduction);
  * elementwise nodes (Linear/Map) have no preference of their own: they
    adopt their consumers' order and pass it through to every operand, so
    whole residual chains agree on one order and add without views;
  * program outputs vote for the logical order (the runtime returns axes
    in edge order);
  * inputs, constants and gather-family nodes are pinned to the logical
    order (inputs arrive canonically; constants are built row-major).

Votes are advisory: codegen honors `phys` exactly, inserting a (rare)
permute view where a producer could not comply, so disagreement costs a
view — never wrong results.
"""

from typing import Sequence, cast

from tensorgrad.compiler.ir import (
    ConstNode,
    EinsumNode,
    GatherNode,
    InputNode,
    LinearNode,
    MapNode,
    Node,
    ReduceNode,
)


def matmul_groups(node: Node):
    """If `node` is a 2-operand einsum expressible as a (batched) matrix
    multiply, return (batch, m, k, n) wire lists; else None.

      batch: wires in both operands and the output
      k:     wires in both operands, contracted (not in output)
      m:     wires only in operand 0, kept
      n:     wires only in operand 1, kept

    Rejected: constraint rows (they rewrite operands into strided views),
    repeated wires (diagonals), wires contracted from a single operand
    (pre-reductions), broadcast-only output wires, no contracted wire
    (pure Hadamard/outer — TensorIterator territory), and pure batched
    dots (m and n both empty — `(A*B).sum()` beats a degenerate bmm).
    One-hot operands are left to the scatter peephole.
    """
    if not isinstance(node, EinsumNode) or node.constraints or len(node.ops) != 2:
        return None
    s1, s2 = node.in_subs
    out = node.out_subs
    if len(set(s1)) != len(s1) or len(set(s2)) != len(s2) or len(set(out)) != len(out):
        return None
    if any(isinstance(op, GatherNode) and op.op == "one_hot" for op in node.ops):
        return None
    set1, set2, oset = set(s1), set(s2), set(out)
    if not oset <= (set1 | set2):
        return None  # broadcast-only output wires
    if (set1 - set2 - oset) or (set2 - set1 - oset):
        return None  # wire summed out of a single operand: pre-reduction
    k = [w for w in s1 if w in set2 and w not in oset]
    if not k:
        return None
    batch = [w for w in out if w in set1 and w in set2]
    m = [w for w in out if w in set1 and w not in set2]
    n = [w for w in out if w in set2 and w not in set1]
    if not m and not n:
        return None
    return batch, m, k, n


def _majority(votes: list) -> tuple:
    """Most common vote; ties broken by first cast."""
    counts: dict[tuple, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    for v in votes:
        if counts[v] == best:
            return v
    raise AssertionError


def _pinned(op: Node) -> bool:
    return isinstance(op, (InputNode, ConstNode, GatherNode))


def _split_blocks(wires, batch, m, n):
    """Split a block-form wire order into (b_ord, m_ord, n_ord); None if the
    order is not (batch, m-block, n-block) / (batch, n-block, m-block)."""
    nb = len(batch)
    if set(wires[:nb]) != set(batch):
        return None
    rest = wires[nb:]
    if set(rest[: len(m)]) == set(m):
        return list(wires[:nb]), list(rest[: len(m)]), list(rest[len(m):])
    if set(rest[: len(n)]) == set(n):
        return list(wires[:nb]), list(rest[len(n):]), list(rest[: len(n)])
    return None


def _cell_feasible(node: EinsumNode, groups, wires) -> bool:
    """Can this block-form output order be produced by mm/bmm WITHOUT
    permuting a PINNED operand? (Pinned operands lie in logical order; a
    block's intra order must be a subsequence match. Flexible operands are
    assumed to comply — they receive our votes.)"""
    batch, m, k, n = groups
    split = _split_blocks(wires, batch, m, n)
    if split is None:
        return False
    b_ord, m_ord, n_ord = split
    for i, (op, subs) in enumerate(zip(node.ops, node.in_subs)):
        if not _pinned(op):
            continue
        entry = list(subs)  # pinned = logical order
        if [w for w in entry if w in batch] != b_ord:
            return False
        grp, grp_ord = (m, m_ord) if i == 0 else (n, n_ord)
        if [w for w in entry if w in grp] != grp_ord:
            return False
    if all(_pinned(op) for op in node.ops):
        ka = [w for w in node.in_subs[0] if w in k]
        kb = [w for w in node.in_subs[1] if w in k]
        if ka != kb:
            return False
    return True


def _decide(node: Node, votes: list) -> tuple:
    """Pick the physical layout for `node` given its consumers' votes."""
    ident = tuple(range(node.order))
    if isinstance(node, (InputNode, ConstNode, GatherNode)):
        return ident  # pinned: canonical arrival / row-major build

    if isinstance(node, EinsumNode):
        groups = matmul_groups(node)
        if groups is not None:
            batch, m, k, n = groups
            out = node.out_subs
            # A cell can only produce block layouts for free, and only with
            # intra-block orders its operands deliver. Priority:
            #   1. block-form votes not permuting a pinned operand;
            #   2. with pinned operands: the layout they naturally produce
            #      (the disagreeing consumer takes a permute view instead of
            #      the mm silently degrading to einsum);
            #   3. any block-form vote (flexible operands comply via votes);
            #   4. non-block votes as-is (attention-scores case: emission
            #      falls back to einsum, which writes any order for free);
            #   5. canonical blocks by output appearance.
            block_votes = [
                v for v in votes if _split_blocks([out[j] for j in v], batch, m, n) is not None
            ]
            feas = [v for v in block_votes if _cell_feasible(node, groups, [out[j] for j in v])]
            if feas:
                return _majority(feas)
            ea = list(node.in_subs[0]) if _pinned(node.ops[0]) else None
            eb = list(node.in_subs[1]) if _pinned(node.ops[1]) else None
            if ea or eb:
                ref = [out[j] for j in _majority(block_votes)] if block_votes else list(out)
                # (cast: `if ea or eb` above guarantees the chosen list is not None)
                b_ord = [w for w in cast(list, ea if ea is not None else eb) if w in batch]
                if ea and eb and [w for w in eb if w in batch] != b_ord:
                    b_ord = None  # incompatible pinned batch orders: no bmm
                if b_ord is not None:
                    m_ord = [w for w in (ea if ea is not None else ref) if w in m]
                    n_ord = [w for w in (eb if eb is not None else ref) if w in n]
                    return tuple(out.index(w) for w in b_ord + m_ord + n_ord)
            if block_votes:
                return _majority(block_votes)
            if votes:
                return _majority(votes)
            return tuple(out.index(w) for w in batch + m + n)
        return _majority(votes) if votes else ident

    if isinstance(node, ReduceNode) and node.op in ("softmax", "log_softmax"):
        if votes:
            return _majority(votes)
        # unconstrained: reduced axes last (contiguous softmax)
        return tuple(a for a in ident if a not in node.axes) + tuple(node.axes)

    # Linear / Map / max / argmax: produced elementwise or by reduction in
    # whatever order we ask for.
    return _majority(votes) if votes else ident


def _vote_operands(node: Node, phys: tuple, votes: dict) -> None:
    """Given node's decided layout, record desired layouts for its operands."""

    def vote(op, want):
        # Pinned kinds ignore votes in _decide; recording them is harmless.
        votes.setdefault(id(op), []).append(tuple(want))

    if isinstance(node, LinearNode):
        for term, perm in zip(node.terms, node.perms):
            vote(term, tuple(perm[j] for j in phys))
    elif isinstance(node, MapNode):
        for op, perm in zip(node.ops, node.perms):
            vote(op, tuple(perm[j] for j in phys))
    elif isinstance(node, ReduceNode):
        (op,) = node.ops
        if node.op in ("softmax", "log_softmax"):
            vote(op, phys)
        else:  # max/argmax: operand = kept axes (in output order) + reduced last
            kept = [a for a in range(op.order) if a not in node.axes]
            vote(op, tuple(kept[j] for j in phys) + tuple(node.axes))
    elif isinstance(node, EinsumNode):
        groups = matmul_groups(node)
        if groups is not None:
            batch, m, k, n = groups
            wires = [node.out_subs[j] for j in phys]
            split = _split_blocks(wires, batch, m, n)
            b_ord, m_ord, n_ord = split if split is not None else (list(batch), list(m), list(n))
            s1, s2 = node.in_subs
            want1 = list(b_ord) + list(m_ord) + list(k)
            want2 = list(b_ord) + list(k) + list(n_ord)
            vote(node.ops[0], tuple(s1.index(w) for w in want1))
            vote(node.ops[1], tuple(s2.index(w) for w in want2))
        elif not node.constraints and len(set(node.out_subs)) == len(node.out_subs):
            # Non-cell einsums vote ONLY when the emitted kernel actually
            # touches operands in output order — i.e. when it lowers to a
            # TensorIterator expression, not to contraction machinery (which
            # absorbs any operand order in its equation):
            #  * pure Hadamard/broadcast products (`a * b.view * ...`): every
            #    operand wants to lie exactly as the output does;
            #  * Hadamard-reduce `(A*B).sum(dims)`: kept wires in output
            #    order, reduced wires last (contiguous reduction), the SAME
            #    reduced order on both operands;
            #  * single-operand reductions `.sum(dim=...)`: likewise.
            if not node.ops or any(len(set(s)) != len(s) for s in node.in_subs):
                return  # pure ones / diagonal extraction: no clean preference
            pout = [node.out_subs[j] for j in phys]
            oset = set(node.out_subs)
            sets = [set(s) for s in node.in_subs]
            hadamard = all(ss <= oset for ss in sets)
            had_reduce = len(sets) == 2 and sets[0] == sets[1] and oset < sets[0]
            if not (hadamard or had_reduce or len(node.ops) == 1):
                return
            red_ref = [w for w in node.in_subs[0] if w not in oset]
            for op, subs in zip(node.ops, node.in_subs):
                kept = [w for w in pout if w in set(subs)]
                red = red_ref if had_reduce else [w for w in subs if w not in oset]
                vote(op, tuple(subs.index(w) for w in kept + red))


def _solid(term: Node) -> bool:
    """Terms whose layout is 'real': broadcast-expand einsums (pure ones, or
    ops covering only part of the output wires) are free views that follow
    any layout, so they never drive an elementwise node's refinement."""
    if isinstance(term, EinsumNode):
        covered = set()
        for subs in term.in_subs:
            covered |= set(subs)
        return set(term.out_subs) <= covered and bool(term.ops)
    return True


def _free_expand(node: Node) -> bool:
    """Nodes emitted as pure unsqueeze/expand views of at most one small
    operand: torch.full ones and single-operand broadcasts. Realigning them
    to their consumers costs nothing (the view is rebuilt, never copied)."""
    if not isinstance(node, EinsumNode) or node.constraints:
        return False
    if not node.ops:
        return True
    if len(node.ops) != 1:
        return False
    (subs,) = node.in_subs
    return len(set(subs)) == len(subs) and set(subs) < set(node.out_subs)


def _refine(node: Node, phys: dict):
    """Forward refinement for weakly-decided nodes: derive the layout the
    node's operands make FREE to produce. Returns a phys tuple or None."""

    def phys_of(op):
        p = phys.get(id(op))
        return p if p is not None else tuple(range(op.order))

    if isinstance(node, EinsumNode):
        if node.constraints or len(set(node.out_subs)) != len(node.out_subs):
            return None
        groups = matmul_groups(node)
        if groups is not None:
            batch, m, k, n = groups
            a_ord = [node.in_subs[0][q] for q in phys_of(node.ops[0])]
            b_ord = [node.in_subs[1][q] for q in phys_of(node.ops[1])]
            a_batch = [w for w in a_ord if w in batch]
            b_batch = [w for w in b_ord if w in batch]
            if a_batch != b_batch:
                return None
            m_ord = [w for w in a_ord if w in m]
            n_ord = [w for w in b_ord if w in n]
            return tuple(node.out_subs.index(w) for w in a_batch + m_ord + n_ord)
        # Hadamard chains: follow a full-shape operand's physical order.
        cands = []
        oset = set(node.out_subs)
        for op, subs in zip(node.ops, node.in_subs):
            if len(set(subs)) == len(subs) and set(subs) == oset:
                entry = [subs[q] for q in phys_of(op)]
                cands.append(tuple(node.out_subs.index(w) for w in entry))
        return _majority(cands) if cands else None

    if isinstance(node, (LinearNode, MapNode)):
        terms = node.terms if isinstance(node, LinearNode) else node.ops
        cands = []
        for term, perm in zip(terms, node.perms):
            if not _solid(term):
                continue
            inv = {a: j for j, a in enumerate(perm)}  # term axis -> node axis
            cands.append(tuple(inv[a] for a in phys_of(term)))
        return _majority(cands) if cands else None

    if isinstance(node, ReduceNode):
        (op,) = node.ops
        physO = phys_of(op)
        if node.op in ("softmax", "log_softmax"):
            return physO
        kept = [a for a in range(op.order) if a not in node.axes]
        renum = {a: j for j, a in enumerate(kept)}
        return tuple(renum[a] for a in physO if a not in node.axes)

    return None


def assign_layouts(
    order: list[Node],
    outputs: list[tuple[Node, tuple]],
    var_orders: "Sequence[tuple]" = (),
) -> dict[int, tuple]:
    """Three sweeps; returns {id(node): phys permutation}.

    1. Reverse-topo voting: consumers vote their operands' layouts (cells
       vote block layouts; reductions vote reduced-last; elementwise nodes
       pass their own layout through). Nodes decided WITHOUT any consumer
       vote are 'weak' — their fallback layout is arbitrary, so weak
       elementwise nodes cast no votes (an arbitrary order must never pin a
       producer; cells/reductions still vote: their preference is real).
    2. Forward refinement: weak nodes re-derive their layout from what their
       (now decided) operands produce for free — elementwise chains follow
       their inputs, matmul cells emit whatever block layout their operands
       already lie in.
    3. Free expands (torch.full / bias broadcasts) are realigned to their
       final consumers: rebuilding an unsqueeze/expand view is free.

    `outputs` are (node, edge_order) pairs; `var_orders` the input variables'
    canonical edge orders (in the runtime's lookup order). An output whose
    edge set matches a variable's (typically its gradient) votes for the
    VARIABLE's edge order — the runtime aligns it to that order anyway, so
    landing there physically keeps the backward chain in the same layout as
    the forward nodes it shares work with. Other outputs express no
    preference: the return statement un-permutes with a free view."""
    votes: dict[int, list] = {}
    for node, edge_order in outputs:
        for var_edges in var_orders:
            if set(edge_order) == set(var_edges) and len(edge_order) == len(var_edges):
                want = tuple(edge_order.index(e) for e in var_edges)
                votes.setdefault(id(node), []).append(want)
                break

    phys: dict[int, tuple] = {}
    weak: set[int] = set()
    for node in reversed(order):
        vs = votes.get(id(node), [])
        p = _decide(node, vs)
        phys[id(node)] = p
        if not vs:
            weak.add(id(node))
            # An unvoted node's own layout is arbitrary — it must not pin its
            # producers. Only real structural preferences still vote: matmul
            # cells (block layouts) and softmax (reduced-last).
            real_pref = (isinstance(node, EinsumNode) and matmul_groups(node) is not None) or (
                isinstance(node, ReduceNode) and node.op in ("softmax", "log_softmax")
            )
            if not real_pref:
                continue
        _vote_operands(node, p, votes)

    for node in order:  # forward: operands refined before consumers
        if id(node) in weak and not isinstance(node, (InputNode, ConstNode, GatherNode)):
            p2 = _refine(node, phys)
            if p2 is not None:
                phys[id(node)] = p2

    votes2: dict[int, list] = {}
    for node in order:
        _vote_operands(node, phys[id(node)], votes2)
    for node in order:
        if _free_expand(node) and (vs := votes2.get(id(node))):
            phys[id(node)] = _majority(vs)
    return phys
