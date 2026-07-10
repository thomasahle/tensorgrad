"""Lowering from tensorgrad Tensor trees to the compiler IR.

All requested output tensors are lowered into ONE hash-consed DAG, so the
forward pass shared between a loss and its gradients is computed once.

The central trick: a Product is lowered by running union-find over its edge
names, where every Delta factor unions the edges it touches. Deltas thus
never become tensors — they become shared einsum indices (hyper-edges).
Only a Delta with two or more *free* output edges (a true diagonal
embedding, e.g. the output of d(diag(x))/dx) falls back to an explicit
constant, which is hoisted and built once per shape-specialization.
"""

from typing import Any, Optional

import sympy

from tensorgrad import functions as F
from tensorgrad.utils import DisjointSets
from tensorgrad.tensor import (
    Delta,
    Derivative,
    Function,
    Product,
    Rename,
    Sum,
    Tensor,
    Variable,
    Zero,
)
from tensorgrad.compiler.ir import (
    Builder,
    ConstNode,
    Dim,
    EinsumNode,
    GatherNode,
    Node,
)
from tensorgrad.compiler.affine import Affine
from tensorgrad.compiler.cells import CELLS, _FusedFunction, _FusedVJP

# Elementwise function signatures with "scalar" shape (edges pass through).
_SIMPLE_OPS = {
    "exp": "exp",
    "log": "log",
    "relu": "relu",
    "sign": "sign",
    "abs": "abs",
    "gt0": "gt0",
    "tanh": "tanh",
    "erf": "erf",
}


class Lowerer:
    def __init__(self, builder: Optional[Builder] = None):
        self.b = builder or Builder()
        # Cache: id(tensor) -> (node, edge_order). Tensor __eq__ is graph
        # isomorphism (expensive); identity is a cheap first-level cache.
        # Structural CSE still happens in the Builder via hash-consing.
        self._memo: dict[int, tuple[Node, tuple[str, ...]]] = {}

    def lower(self, t: Tensor) -> tuple[Node, tuple[str, ...]]:
        """Lower a tensor; returns (node, edge_order) where edge_order names
        each axis of the node with the tensor's free edge names."""
        if (hit := self._memo.get(id(t))) is not None:
            return hit
        res = self._lower(t)
        node, order = res
        assert set(order) == set(t.edges), f"{order=} vs {tuple(t.edges)=}"
        assert len(order) == node.order
        self._memo[id(t)] = res
        return res

    def _lower(self, t: Tensor) -> tuple[Node, tuple[str, ...]]:
        if isinstance(t, Variable):
            return self.b.input(t), tuple(t.edges)
        if isinstance(t, Rename):
            node, order = self.lower(t.tensor)
            return node, tuple(t.mapping.get(e, e) for e in order)
        if isinstance(t, Zero):
            return self.b.const("zero", (), tuple(t.shape.values())), tuple(t.edges)
        if isinstance(t, Delta):
            return self._lower_delta(t)
        if isinstance(t, Affine) and t.range_rows:
            # Range rows (inequality indicators, #44 stage 1) are not in the
            # einsum-constraint vocabulary: the whole Affine lowers as one
            # hoisted dense indicator constant (built once per specialization
            # by codegen, evaluated exactly mod P by szfp).
            return self._lower_affine_const(t)
        if isinstance(t, (F.Convolution, Affine)):
            # A standalone structured tensor is just a one-factor product; its
            # constraint rows become the einsum node's constraints and codegen
            # materializes the indicator (there is nothing to contract with).
            return self._lower_product(Product([t]))
        if isinstance(t, F.Reshape):
            edges = tuple(t.edges)
            dims = tuple(t.shape[e] for e in edges)
            return self.b.const("reshape", dims, dims), edges
        if isinstance(t, Sum):
            return self._lower_sum(t)
        if isinstance(t, Product):
            return self._lower_product(t)
        if isinstance(t, Function):
            return self._lower_function(t)
        if isinstance(t, Derivative):
            raise NotImplementedError(
                "Cannot compile a Derivative node. Call .simplify() or .full_simplify() first."
            )
        raise NotImplementedError(f"No lowering for {type(t).__name__}: {t}")

    # ---- Affine with range rows (dense indicator constant) -------------

    def _lower_affine_const(self, t: Affine) -> tuple[Node, tuple[str, ...]]:
        edges = tuple(t.edges)
        rows_param = tuple(
            ("eq", tuple(sorted((edges.index(e), c) for e, c in coeffs.items())), const)
            for coeffs, const in t.rows
        ) + tuple(
            ("range", tuple(sorted((edges.index(e), c) for e, c in coeffs.items())), k, X)
            for coeffs, k, X in t.range_rows
        )
        dims = tuple(t.shape[e] for e in edges)
        return self.b.const("affine", (rows_param,), dims), edges

    # ---- Delta --------------------------------------------------------

    def _lower_delta(self, t: Delta) -> tuple[Node, tuple[str, ...]]:
        """A *standalone* Delta (not inside a Product)."""
        edges = tuple(t.edges)
        if t.order == 0:
            return self.b.scalar(t.size), ()
        if t.order == 1:
            # Ones vector: einsum with a single pure-output wire (expanded view).
            return self.b.einsum([], [], (0,), {0: t.size}), edges
        # order >= 2: an actual diagonal tensor as *output*. Rare; materialize.
        return self.b.const("delta", (t.size, t.order), tuple(t.shape.values())), edges

    # ---- Sum ----------------------------------------------------------

    def _lower_sum(self, t: Sum) -> tuple[Node, tuple[str, ...]]:
        out_order = tuple(t.edges)
        terms, perms, weights = [], [], []
        for w, term in zip(t.weights, t.terms):
            node, order = self.lower(term)
            # perm[j] = position in `order` of out_order[j]
            perms.append(tuple(order.index(e) for e in out_order))
            terms.append(node)
            weights.append(w)
        return self.b.linear(terms, perms, weights), out_order

    # ---- Product (Delta elimination lives here) ------------------------

    def _lower_product(self, t: Product) -> tuple[Node, tuple[str, ...]]:
        out_order = tuple(t.edges)
        weight = sympy.Integer(1)

        # Union-find over edge names in this product's scope. This is the
        # derived special case of affine-row elimination: a row of the form
        # e1 - e2 = 0 is solved immediately by aliasing the two wires.
        # (utils.DisjointSets: union(x, y) keeps y's root as representative.)
        aliases: DisjointSets[str, Any] = DisjointSets()
        find = aliases.find

        # General affine rows (edge-name form): (coeffs {edge: sympy}, const).
        raw_rows: list[tuple[dict[str, Any], Any]] = []
        edge_dims: dict[str, Dim] = {}

        operands_t: list[Tensor] = []
        for ft in t.factors:
            if isinstance(ft, Delta):
                if ft.order == 0:
                    weight *= ft.size
                else:
                    edges = list(ft.edges)
                    for e in edges[1:]:
                        aliases.union(e, edges[0])
            elif isinstance(ft, F.Convolution):
                # C[i, k, o] = 1 iff i = dilation*k + stride*o (stride/dilation
                # currently 1 in tensorgrad, but the row form supports them).
                stride = getattr(ft, "stride", 1)
                raw_rows.append(
                    ({ft.input_name: 1, ft.kernel_name: -1, ft.output_name: -stride}, 0)
                )
                for e in ft.edges:
                    edge_dims.setdefault(e, ft.shape[e])
            elif isinstance(ft, Affine) and not ft.range_rows:
                for coeffs, const in ft.rows:
                    raw_rows.append((dict(coeffs), const))
                for e in ft.edges:
                    edge_dims.setdefault(e, ft.shape[e])
            else:
                # (an Affine WITH range rows lands here too: it lowers as a
                # dense indicator operand — see _lower_affine_const)
                operands_t.append(ft)

        # Wire = union-find class. Assign wire ids.
        wire_of: dict[str, int] = {}
        wire_dims: dict[int, Dim] = {}

        def wire(e: str, dim: Any) -> int:
            r = find(e)
            if r not in wire_of:
                wire_of[r] = len(wire_of)
            w = wire_of[r]
            wire_dims.setdefault(w, dim)
            return w

        ops, in_subs = [], []
        for ft in operands_t:
            node, order = self.lower(ft)
            in_subs.append(tuple(wire(e, ft.shape[e]) for e in order))
            ops.append(node)

        shape = t.shape
        out_subs = [wire(e, shape[e]) for e in out_order]

        # If a wire carries more than one free output edge, that's a diagonal
        # embedding (e.g. d diag(x)/dx). einsum can't repeat an output index,
        # so keep one representative and link the extras through explicit
        # (hoisted) identity constants. This is the only case where a Delta
        # materializes.
        counts: dict[int, int] = {}
        for w in out_subs:
            counts[w] = counts.get(w, 0) + 1
        for w, c in counts.items():
            if c > 1:
                first = out_subs.index(w)
                for pos in range(len(out_subs)):
                    if pos != first and out_subs[pos] == w:
                        nw = len(wire_dims)
                        wire_dims[nw] = wire_dims[w]
                        size = wire_dims[w]
                        eye = self.b.const("delta", (size, 2), (size, size))
                        ops.append(eye)
                        in_subs.append((w, nw))
                        out_subs[pos] = nw

        # Map affine rows from edge names to wires. Delta-aliasing may have
        # merged two of a row's edges into one wire: coefficients add up, and
        # a row that collapses to 0 = 0 is dropped, while 0 = c (c != 0) means
        # the indicator is unsatisfiable and the whole product is zero.
        constraints = []
        for coeffs, const in raw_rows:
            wire_coeffs: dict[int, Any] = {}
            for e, c in coeffs.items():
                w = wire(e, edge_dims[e])
                wire_coeffs[w] = sympy.sympify(wire_coeffs.get(w, 0)) + sympy.sympify(c)
            wire_coeffs = {w: c for w, c in wire_coeffs.items() if sympy.sympify(c) != 0}
            if not wire_coeffs:
                if sympy.sympify(const) != 0:
                    zero = self.b.const("zero", (), tuple(t.shape[e] for e in out_order))
                    return zero, out_order
                continue
            constraints.append((tuple(wire_coeffs.items()), const))

        if not ops and not out_subs and not constraints:
            return self.b.scalar(weight), ()

        # Gather formation (index_select) is a COST decision and lives in
        # compiler/gather.py as the first pass of specialize() — the law:
        # lowering makes no cost decisions (task #43).

        node = self.b.einsum(ops, in_subs, tuple(out_subs), wire_dims, weight, constraints)
        return node, out_order

    # ---- Function -----------------------------------------------------

    def _lower_function(self, t: Function) -> tuple[Node, tuple[str, ...]]:
        sig = t.signature

        if isinstance(sig, F._PowerFunction):
            node, order = self.lower(t.inputs[0])
            k = sig.k
            return self.b.map("pow", (k,), [node]), order

        if sig.name in _SIMPLE_OPS:
            node, order = self.lower(t.inputs[0])
            return self.b.map(_SIMPLE_OPS[sig.name], (), [node]), order

        if isinstance(sig, F._EqualFunction) or sig.name == "equal":
            n1, o1 = self.lower(t.inputs[0])
            n2, o2 = self.lower(t.inputs[1])
            out_order = tuple(t.edges)
            perms = [tuple(o1.index(e) for e in out_order), tuple(o2.index(e) for e in out_order)]
            return self.b.map("equal", (), [n1, n2], perms), out_order

        if isinstance(sig, _FusedFunction):
            return CELLS[sig.cell_name].lower_fwd(self, t)

        if isinstance(sig, F._MatrixInverseFunction):
            return CELLS["inverse"].lower_fwd(self, t)

        if isinstance(sig, F._DeterminantFunction):
            return CELLS["det"].lower_fwd(self, t)

        if isinstance(sig, _FusedVJP):
            return CELLS[sig.cell_name].lower_bwd(self, t)

        if isinstance(sig, F._ArgMaxFunction):
            node, order = self.lower(t.inputs[0])
            axis = order.index(sig.dim)
            out_order = tuple(e for e in order if e != sig.dim)
            return self.b.reduce("argmax", (axis,), node), out_order

        if isinstance(sig, F._ArgSortFunction):
            node, order = self.lower(t.inputs[0])
            axis = order.index(sig.dim)
            return self.b.reduce("argsort", (axis,), node), order

        if isinstance(sig, F._OneHotFunction):
            # inputs = (idx, size_carrier); the carrier only fixes the number
            # of classes, so it is not lowered at all.
            inode, iorder = self.lower(t.inputs[0])
            node = self.b.one_hot(inode, t.shape[sig.eq_edge])
            return node, (sig.eq_edge,) + iorder

        if isinstance(sig, F._MaxFunction):
            node, order = self.lower(t.inputs[0])
            (dims,) = sig.inputs
            axes = tuple(order.index(e) for e in dims)
            out_order = tuple(e for i, e in enumerate(order) if i not in axes)
            return self.b.reduce("max", axes, node), out_order

        raise NotImplementedError(
            f"No lowering for function {sig.name!r} ({type(sig).__name__}). "
            "If this signature is normally removed by simplification, call .full_simplify() first."
        )


def lower_program(tensors: list[Tensor]) -> tuple[Builder, list[tuple[Node, tuple[str, ...]]]]:
    """Lower several output tensors into one shared DAG."""
    lw = Lowerer()
    outputs = [lw.lower(t) for t in tensors]
    return lw.b, outputs
