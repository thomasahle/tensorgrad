"""Intermediate representation for the tensorgrad compiler.

The IR is a DAG of nodes, hash-consed so that structurally identical
subcomputations are represented by a single node (global CSE across all
outputs). Axes are positional; symbolic sizes (sympy Symbols/expressions)
are attached per axis. Edge *names* from tensorgrad only exist at the
boundary (inputs/outputs) — inside the IR everything is positional wires.

Delta tensors never become nodes. During lowering of a Product, a Delta
merely unions the wires of the edges it touches (union-find). This is what
guarantees no identity/copy tensor is ever materialized, and no einsum is
ever asked to build a `batch x batch` diagonal blowup that a Delta merely
"aliases".
"""

from dataclasses import dataclass
from fractions import Fraction
from numbers import Number
from typing import Any, Callable, Iterable, Optional, Sequence, SupportsFloat, TypeVar, Union, cast

import sympy
from sympy import Symbol

Dim = Union[Symbol, sympy.Expr, int]

_N = TypeVar("_N", bound="Node")


@dataclass(frozen=True, eq=False)
class Node:
    """Base IR node. Nodes are unique-by-construction (hash-consed in Builder),
    so identity comparison is structural comparison."""

    dims: tuple[Dim, ...]  # symbolic size of each output axis

    @property
    def order(self) -> int:
        return len(self.dims)

    def operands(self) -> tuple["Node", ...]:
        return ()


@dataclass(frozen=True, eq=False)
class InputNode(Node):
    """A user-supplied Variable. `axes` names are the Variable's canonical
    edge order; the runtime aligns the provided torch tensor to this order.
    `sym` lists groups of axis positions the Variable declares interchangeable
    (Variable symmetries); szfp draws values symmetric under them, making
    symmetry-using identities (x == x.T) fingerprint-visible."""

    var_name: str = ""
    sym: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, eq=False)
class FusedFwdNode(Node):
    """Forward of a fused technology-mapping cell (tensorgrad/compiler/cells.py).
    `cell_name` selects the cell; `params` is a sorted hashable tuple of its
    scalar + edge-role arguments; `layout` is cell-defined lowering metadata
    (operand permutations into canonical order, batch-axis count); `which`
    selects one result of a multi-output kernel (0 for single-output cells;
    AdamW emits w'/m'/v' as which 0/1/2 sharing one fused call). Opaque to
    differentiation, pinned in layout, an atom in szfp -- the generic node
    behind sdpa/layer_norm/gelu/adamw/... ."""
    cell_name: str = ""
    params: tuple = ()
    layout: tuple = ()
    which: int = 0
    ops: tuple = ()
    def operands(self) -> tuple["Node", ...]:
        return self.ops
    def params_dict(self) -> dict:
        return dict(self.params)


@dataclass(frozen=True, eq=False)
class FusedBwdNode(Node):
    """Reverse VJP of a fused cell w.r.t. input `which`."""
    cell_name: str = ""
    which: int = 0
    params: tuple = ()
    layout: tuple = ()
    ops: tuple = ()
    def operands(self) -> tuple["Node", ...]:
        return self.ops
    def params_dict(self) -> dict:
        return dict(self.params)


@dataclass(frozen=True, eq=False)
class ConstNode(Node):
    """A constant tensor that only depends on dimension sizes.
    Built once per shape-specialization and closed over by the generated code.

    kind:
      - 'zero':    zeros of shape dims
      - 'delta':   order-k diagonal tensor (params=(size, k)); only used as a
                   *fallback* when a Delta has >=2 free output edges.
      - 'conv':    convolution structure tensor (params=(w_in, k_size, w_out))
      - 'reshape': reshape/copy structure tensor
      - 'scalar':  0-dim scalar with value params[0] (a sympy expression in dims)
    """

    kind: str = ""
    params: tuple = ()


@dataclass(frozen=True, eq=False)
class EinsumNode(Node):
    """A single (multi-ary) contraction over `wires`.

    in_subs[i] gives, for each axis of operand i, the wire it is attached to.
    out_subs gives the wire of each output axis. Wires appearing in operands
    but not in out_subs are summed over. Wires appearing ONLY in out_subs are
    broadcast (codegen expands, which is a view — no memory).
    Repeated wires within one operand denote diagonal extraction.
    `weight` is a scalar (sympy expression in the dims) multiplied onto the
    result; order-0 Deltas and Sum-of-one-term weights fold into it.

    `constraints` are affine indicator factors: each row is
    (((wire, coeff), ...), const) meaning the implicit 0/1 factor
    [sum_i coeff_i * wire_i == const] multiplies the einsum body. Structured
    tensors (Convolution, Affine, non-aliasing Deltas) lower to rows here;
    codegen eliminates rows into strided views or materializes indicators.
    Coeffs/consts are sympy expressions resolved at shape specialization.
    """

    ops: tuple[Node, ...] = ()
    in_subs: tuple[tuple[int, ...], ...] = ()
    out_subs: tuple[int, ...] = ()
    wire_dims: tuple[Dim, ...] = ()  # wire id -> dim (wires are 0..n-1)
    weight: sympy.Expr = sympy.Integer(1)
    constraints: tuple = ()

    def operands(self) -> tuple["Node", ...]:
        return self.ops


@dataclass(frozen=True, eq=False)
class LinearNode(Node):
    """Weighted sum of terms, each permuted into the output axis order.
    perms[i][j] = which axis of term i lands on output axis j (after the
    term has been broadcast to the full output shape; terms always have the
    full edge set post-simplify, tensorgrad broadcasts via Ones-products)."""

    terms: tuple[Node, ...] = ()
    perms: tuple[tuple[int, ...], ...] = ()
    weights: tuple = ()  # numbers (Fraction/int/float)

    def operands(self) -> tuple["Node", ...]:
        return self.terms


@dataclass(frozen=True, eq=False)
class MapNode(Node):
    """Elementwise op. For multi-input ops, perms[i] aligns operand i's axes
    to the output axis order (identity for single-input ops)."""

    op: str = ""  # exp, log, relu, gt0, sign, abs, pow, equal
    params: tuple = ()  # e.g. (k,) for pow
    ops: tuple[Node, ...] = ()
    perms: tuple[tuple[int, ...], ...] = ()

    def operands(self) -> tuple["Node", ...]:
        return self.ops


@dataclass(frozen=True, eq=False)
class GatherNode(Node):
    """Integer-indexed lookups.

    op == "gather": ops = (table, idx); `axis` is the indexed axis of the
    table. Output axes = table axes with `axis` replaced by all idx axes
    (idx holds integral values stored as floats).

    op == "one_hot": ops = (idx,); output axes = (num_classes,) + idx axes,
    value [idx == class]. Codegen defers materialization: an EinsumNode that
    contracts a one_hot over all its idx axes is emitted as an index_add_
    scatter instead, so the dense one-hot is only built if some other
    consumer actually needs it.
    """

    op: str = ""  # gather, one_hot
    axis: int = 0
    ops: tuple[Node, ...] = ()

    def operands(self) -> tuple["Node", ...]:
        return self.ops


@dataclass(frozen=True, eq=False)
class ReduceNode(Node):
    """Non-linear reductions that einsum can't express: argmax, max, softmax.
    `axes` are positions in the operand to reduce over (or apply softmax over).
    Output axes = operand axes minus reduced axes (softmax keeps all axes)."""

    op: str = ""  # argmax, max, softmax
    axes: tuple[int, ...] = ()
    ops: tuple[Node, ...] = ()

    def operands(self) -> tuple["Node", ...]:
        return self.ops


class Builder:
    """Hash-consing factory for IR nodes. Two structurally identical nodes
    (same class, params, operand identities, canonical wire structure) are
    the same object. This gives global CSE across all outputs for free."""

    def __init__(self) -> None:
        self._cache: dict = {}
        # id(node) -> insertion index, to give deterministic topo/canonical order
        self._index: dict[int, int] = {}
        # var_name -> the tensorgrad Variable object (for runtime binding).
        # Typed Any: the IR deliberately does not import tensorgrad.tensor.
        self.input_vars: dict[str, Any] = {}

    def _intern(self, key: tuple, make: Callable[[], _N]) -> _N:
        if (node := self._cache.get(key)) is None:
            node = make()
            self._cache[key] = node
            self._index[id(node)] = len(self._index)
        return node

    def node_index(self, node: Node) -> int:
        return self._index[id(node)]

    # ---- constructors -------------------------------------------------

    def input(self, var: Any) -> InputNode:
        name = var.name
        dims = tuple(var.shape.values())
        if name in self.input_vars and self.input_vars[name] is not var:
            if not self.input_vars[name] == var:
                raise ValueError(f"Two different Variables share the name {name!r}")
        self.input_vars[name] = var
        edges = list(var.shape.keys())
        sym = tuple(
            sorted(
                tuple(sorted(edges.index(e) for e in group))
                for group in (getattr(var, "symmetries", None) or set())
                if len(group) > 1
            )
        )
        return self._intern(("in", name, dims, sym), lambda: InputNode(dims, name, sym))

    def const(self, kind: str, params: tuple, dims: tuple) -> ConstNode:
        params = tuple(params)
        return self._intern(("const", kind, params, tuple(dims)), lambda: ConstNode(tuple(dims), kind, params))

    def scalar(self, expr: Any) -> ConstNode:
        expr = sympy.sympify(expr)
        return self.const("scalar", (expr,), ())

    def einsum(
        self,
        ops: Sequence[Node],
        in_subs: Sequence[tuple[int, ...]],
        out_subs: tuple[int, ...],
        wire_dims: dict[int, Dim],
        weight: Union[int, float, Fraction, sympy.Expr] = 1,
        constraints: Iterable[tuple[Any, Any]] = (),
    ) -> Node:
        # (cast: sympify of a numeric scalar yields an Expr; the stub says Basic)
        w_expr = cast(sympy.Expr, sympy.sympify(weight))
        # Constant folding: 0-dim scalar operands multiply into the weight
        # instead of becoming einsum operands.
        kept_ops, kept_subs = [], []
        for op, subs in zip(ops, in_subs):
            if isinstance(op, ConstNode) and op.kind == "scalar":
                w_expr = w_expr * op.params[0]
            elif isinstance(op, EinsumNode) and not op.ops and not op.out_subs and not op.constraints:
                w_expr = w_expr * op.weight
            else:
                kept_ops.append(op)
                kept_subs.append(subs)
        ops, in_subs = kept_ops, kept_subs
        if not ops and not out_subs and not constraints:
            return self.scalar(w_expr)
        # Canonicalize wire ids by first occurrence, so isomorphic einsum
        # structures share a node regardless of the wire numbers used.
        remap: dict[int, int] = {}
        for subs in list(in_subs) + [out_subs]:
            for w in subs:
                if w not in remap:
                    remap[w] = len(remap)
        for coeffs, _const in constraints:
            for w in sorted(dict(coeffs)):
                if w not in remap:
                    remap[w] = len(remap)
        in_subs = tuple(tuple(remap[w] for w in subs) for subs in in_subs)
        out_subs = tuple(remap[w] for w in out_subs)
        wdims = tuple(dim for w, dim in sorted(((nw, wire_dims[ow]) for ow, nw in remap.items())))

        # Normalize constraint rows: remapped wires, sign-normalized (first
        # coeff positive), deterministically sorted.
        norm_rows = []
        for coeffs, const in constraints:
            items = sorted((remap[w], sympy.sympify(c)) for w, c in dict(coeffs).items())
            const = sympy.sympify(const)
            if items and items[0][1].could_extract_minus_sign():
                items = [(w, -c) for w, c in items]
                const = -const
            norm_rows.append((tuple(items), const))
        norm_rows = tuple(sorted(norm_rows, key=lambda r: (tuple((w, str(c)) for w, c in r[0]), str(r[1]))))

        # Trivial case: identity einsum over a single operand -> pass through.
        if (
            len(ops) == 1
            and w_expr == 1
            and not norm_rows
            and in_subs[0] == out_subs
            and len(set(out_subs)) == len(out_subs)
        ):
            return ops[0]

        dims = tuple(wdims[w] for w in out_subs)
        rows_key = tuple((tuple((w, str(c)) for w, c in row), str(const)) for row, const in norm_rows)
        key = ("ein", tuple(id(op) for op in ops), in_subs, out_subs, wdims, w_expr, rows_key)
        return self._intern(
            # (cast: pyright widens the captured w_expr in the closure; it is always an Expr here)
            key, lambda: EinsumNode(dims, tuple(ops), in_subs, out_subs, wdims, cast(sympy.Expr, w_expr), norm_rows)
        )

    def linear(self, terms: Sequence[Node], perms: Sequence[tuple[int, ...]], weights: Sequence) -> Node:
        if len(terms) == 1 and weights[0] == 1 and perms[0] == tuple(range(len(perms[0]))):
            return terms[0]
        # Canonical order: sort by (node index, perm) for deterministic CSE.
        order = sorted(range(len(terms)), key=lambda i: (self.node_index(terms[i]), perms[i]))
        terms = tuple(terms[i] for i in order)
        perms = tuple(tuple(perms[i]) for i in order)
        weights = tuple(weights[i] for i in order)
        t0, p0 = terms[0], perms[0]
        dims = tuple(t0.dims[p0[j]] for j in range(len(p0)))
        key = ("lin", tuple(id(t) for t in terms), perms, tuple(map(str, weights)))
        return self._intern(key, lambda: LinearNode(dims, terms, perms, weights))

    def map(self, op: str, params: tuple, ops: Sequence[Node], perms: Optional[Sequence] = None) -> Node:
        # Constant folding: elementwise ops on symbolic scalars stay symbolic.
        if len(ops) == 1 and isinstance(ops[0], ConstNode) and ops[0].kind == "scalar":
            (expr,) = ops[0].params
            folded = {
                "pow": lambda: expr ** params[0],
                "exp": lambda: sympy.exp(expr),
                "log": lambda: sympy.log(expr),
                "abs": lambda: sympy.Abs(expr),
                "sign": lambda: sympy.sign(expr),
            }.get(op)
            if folded is not None:
                return self.scalar(folded())
        if perms is None:
            perms = [tuple(range(op_.order)) for op_ in ops]
        perms = tuple(tuple(p) for p in perms)
        t0, p0 = ops[0], perms[0]
        dims = tuple(t0.dims[p0[j]] for j in range(len(p0)))
        key = ("map", op, tuple(params), tuple(id(o) for o in ops), perms)
        return self._intern(key, lambda: MapNode(dims, op, tuple(params), tuple(ops), perms))

    def gather(self, table: Node, idx: Node, axis: int) -> Node:
        dims = table.dims[:axis] + idx.dims + table.dims[axis + 1 :]
        key = ("gather", id(table), id(idx), axis)
        return self._intern(key, lambda: GatherNode(dims, "gather", axis, (table, idx)))

    def one_hot(self, idx: Node, num_classes: Dim) -> Node:
        dims = (num_classes,) + idx.dims
        key = ("one_hot", id(idx), num_classes)
        return self._intern(key, lambda: GatherNode(dims, "one_hot", 0, (idx,)))

    def fused_fwd(
        self, cell_name: str, params: dict, ops: Sequence[Node], dims: Sequence[Dim],
        layout: tuple = (), which: int = 0,
    ) -> Node:
        pt = tuple(sorted(params.items()))
        key = ("fused_fwd", cell_name, pt, layout, which, tuple(id(o) for o in ops))
        return self._intern(key, lambda: FusedFwdNode(tuple(dims), cell_name, pt, layout, which, tuple(ops)))

    def fused_bwd(
        self, cell_name: str, which: int, params: dict, ops: Sequence[Node], dims: Sequence[Dim],
        layout: tuple = (),
    ) -> Node:
        pt = tuple(sorted(params.items()))
        key = ("fused_bwd", cell_name, which, pt, layout, tuple(id(o) for o in ops))
        return self._intern(key, lambda: FusedBwdNode(tuple(dims), cell_name, which, pt, layout, tuple(ops)))

    def with_ops(self, nd: Node, ops: list[Node]) -> Node:
        """Reconstruct `nd` with replaced operands through the interning
        constructors (hash-consing returns the identical node when nothing
        changed). THE one rebuild vocabulary: every IR-to-IR pass uses this
        instead of a private copy, so a new Node type is added HERE once —
        the previous five per-pass copies each covered a different subset,
        and three silently kept STALE operands for fused-cell nodes (a
        latent miscompile class the szfp gates were quietly absorbing).
        Unknown compute nodes RAISE rather than silently self-return."""
        if not ops or all(o is p for o, p in zip(ops, nd.operands())):
            return nd
        if isinstance(nd, EinsumNode):
            return self.einsum(
                ops, [tuple(s) for s in nd.in_subs], tuple(nd.out_subs),
                dict(enumerate(nd.wire_dims)), nd.weight, list(nd.constraints),
            )
        if isinstance(nd, LinearNode):
            return self.linear(ops, [tuple(p) for p in nd.perms], list(nd.weights))
        if isinstance(nd, MapNode):
            return self.map(nd.op, nd.params, ops, [tuple(p) for p in nd.perms])
        if isinstance(nd, GatherNode):
            if nd.op == "gather":
                return self.gather(ops[0], ops[1], nd.axis)
            return self.one_hot(ops[0], nd.dims[0])
        if isinstance(nd, ReduceNode):
            return self.reduce(nd.op, nd.axes, ops[0])
        if isinstance(nd, FusedFwdNode):
            return self.fused_fwd(nd.cell_name, nd.params_dict(), ops, nd.dims, nd.layout, nd.which)
        if isinstance(nd, FusedBwdNode):
            return self.fused_bwd(nd.cell_name, nd.which, nd.params_dict(), ops, nd.dims, nd.layout)
        raise NotImplementedError(f"with_ops: cannot rebuild {type(nd).__name__}")

    def reduce(self, op: str, axes: tuple[int, ...], operand: Node) -> Node:
        axes = tuple(sorted(axes))
        if op in ("softmax", "log_softmax", "argsort"):
            dims = operand.dims
        else:  # argmax, max: reduced axes are removed
            dims = tuple(d for i, d in enumerate(operand.dims) if i not in axes)
        key = ("red", op, axes, id(operand))
        return self._intern(key, lambda: ReduceNode(dims, op, axes, (operand,)))


def toposort(outputs: list[Node]) -> list[Node]:
    """Return all nodes reachable from `outputs` in dependency order."""
    seen: dict[int, Node] = {}
    order: list[Node] = []

    def visit(n: Node) -> None:
        if id(n) in seen:
            return
        seen[id(n)] = n
        for op in n.operands():
            visit(op)
        order.append(n)

    for out in outputs:
        visit(out)
    return order


def to_float(w: Any) -> float:
    """Convert a weight (Fraction, int, sympy) to a python float."""
    if isinstance(w, Fraction):
        return w.numerator / w.denominator
    if isinstance(w, Number):
        # (cast: concrete runtime Numbers all support float(); the ABC does not, statically)
        return float(cast(SupportsFloat, w))
    return float(sympy.sympify(w))
