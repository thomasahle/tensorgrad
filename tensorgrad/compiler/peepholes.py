"""Technology-mapping peepholes: one generic driver over the cell registry.

EDA technology mapping treats cells as DATA: a library cell carries its
own matching pattern, and the mapper is a single engine that walks the
netlist trying every cell's pattern (LLVM's instruction selection makes
the same move). Here each FusedCell may define `match(builder, node)`
(compiler/cells.py); this pass is the engine -- a memoized topological
rebuild through Builder methods (stabilize's discipline, so interning and
node indexing stay sound). Adding a mapping touches only the cell.

Current patterns: inverse-then-contract -> torch.linalg.solve, and
log(det(A)) -> torch.linalg.slogdet. Misses degrade performance, never
correctness; orphaned producer nodes are dropped by codegen's dead-line
pruning.
"""

from tensorgrad.compiler.cells import CELLS, FusedCell
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



def linalg_peepholes(builder: Builder, outputs: list) -> list:
    matchers = [c for c in CELLS.values() if type(c).match is not FusedCell.match]
    if not matchers:
        return outputs
    roots = [n for n, _ in outputs]
    order = toposort(roots)
    memo: dict[int, Node] = {}
    changed = False
    for nd in order:
        ops = [memo.get(id(op), op) for op in nd.operands()]
        cur = builder.with_ops(nd, ops)
        for cell in matchers:
            new = cell.match(builder, cur)
            if new is not None:
                cur = new
                break
        if cur is not nd:
            changed = True
        memo[id(nd)] = cur
    if not changed:
        return outputs
    return [(memo[id(n)], o) for n, o in outputs]
