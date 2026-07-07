"""Fused technology-mapping cells, as data.

The paper's §7 casts the backend as EDA technology mapping: a
technology-independent netlist (the einsum DAG) is mapped onto a
*characterized cell library*. A fused kernel -- attention, layer norm, GELU
-- is one cell. Historically each such cell was ~8 files of near-identical
wiring (a Function signature, an IR node pair, a lowering case, a reverse
VJP hook, a codegen emitter, a layout pin, an szfp opaque atom, an
evaluate() oracle); the duplication is what let the GELU layout-pin bug
slip in (a copy-paste that missed one of three pin sites).

Here a cell is instead a single object implementing `FusedCell`. The
compiler carries *generic* `_FusedFunction` / `_FusedVJP` signatures and
`FusedFwdNode` / `FusedBwdNode` IR nodes that name their cell; every
pipeline stage dispatches once, through the cell, to the target-specific
behavior. Adding a cell -- or swapping the whole library for a different
target (a Triton GPU library is a second registry) -- touches this file
only.

Contracts a cell must honor:
  * forward is opaque to differentiation (`derivative` raises); gradients
    come from the reverse VJP, so a cell is only usable inside a compiled
    gradient FAMILY, never a lone forward-mode d/dinput.
  * `eval_*` is the value definition -- the oracle the szfp verifier and
    the interpreter agree on. It must match the emitted kernel bit-for-bit
    on the target, because a cell that miscompiles a gradient is the
    cardinal failure.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from tensorgrad.tensor import Function, FunctionSignature, Tensor


# ---------------------------------------------------------------------------
# Generic signatures. A FunctionSignature is distinguished by its `name`
# string, so the cell name AND every distinguishing parameter (scale, eps,
# approximate, the VJP's input index) must appear in the name -- otherwise
# two structurally-identical-but-semantically-different calls merge.
# ---------------------------------------------------------------------------


def _pkey(params: dict) -> str:
    return ",".join(f"{k}={params[k]}" for k in sorted(params))


class _FusedFunction(FunctionSignature):
    """Forward of a fused cell. `cell_name` selects the cell; `params`
    carries its scalar arguments (scale, eps, approximate, ...)."""

    def __init__(self, cell_name: str, edges, out_edges, params: dict):
        self.cell_name = cell_name
        self.params = params
        super().__init__(f"{cell_name}[{_pkey(params)}]", frozenset(edges), out_edges)

    def derivative(self, i: int, new_edges=None) -> FunctionSignature:
        raise NotImplementedError(
            f"F.{self.cell_name} (fused) gradients use reverse mode: compile a gradient "
            f"FAMILY, as any training loop does. A lone forward-mode d/dinput is not fused."
        )


class _FusedVJP(FunctionSignature):
    """Reverse VJP of a fused cell w.r.t. input `which`; inputs are the cell's
    original inputs plus the cotangent u; output has that input's edges."""

    def __init__(self, cell_name: str, which: int, edges, in_edges, params: dict):
        self.cell_name = cell_name
        self.which = which
        self.params = params
        super().__init__(
            f"{cell_name}_vjp[i={which},{_pkey(params)}]", frozenset(edges), in_edges
        )

    def derivative(self, i: int, new_edges=None) -> FunctionSignature:
        raise NotImplementedError(
            f"Second-order fused {self.cell_name} is not fused; use the composite form."
        )


# ---------------------------------------------------------------------------
# The cell interface + registry.
# ---------------------------------------------------------------------------


class FusedCell:
    name: str = ""

    # -- language side: build the forward Function and its reverse VJP -------
    def build(self, inputs: tuple[Tensor, ...], params: dict) -> Tensor:
        raise NotImplementedError

    def vjp(self, inputs: tuple[Tensor, ...], which: int, u: Tensor, params: dict) -> Tensor:
        raise NotImplementedError

    # -- compiler side: lowering to IR (cell owns its edge->wire layout) -----
    def lower_fwd(self, lower, t: Function):
        raise NotImplementedError

    def lower_bwd(self, lower, t: Function):
        raise NotImplementedError

    # -- backend side: emit target kernels + the value oracle ----------------
    def emit_fwd(self, cg, node, name: str, names) -> str:
        raise NotImplementedError

    def emit_bwd(self, cg, node, name: str, names) -> str:
        raise NotImplementedError

    def eval_fwd(self, params: dict, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def eval_bwd(self, params: dict, which: int, inputs: tuple[torch.Tensor, ...],
                 u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


CELLS: dict[str, FusedCell] = {}


def register(cell_cls):
    """Class decorator: register ONE instance of the cell under its name."""
    inst = cell_cls()
    CELLS[inst.name] = inst
    return cell_cls


def cell_of(node_or_sig: Any) -> FusedCell:
    return CELLS[node_or_sig.cell_name]


# ---------------------------------------------------------------------------
# GELU: the elementwise archetype -- one input, one gradient, no saved state,
# no reshape (edges pass straight through). The simplest possible cell.
# ---------------------------------------------------------------------------


@register
class _GeluCell(FusedCell):
    name = "gelu"

    def build(self, inputs, params):
        (t,) = inputs
        sig = _FusedFunction(self.name, t.edges, (frozenset(t.edges),), params)
        return Function(sig, (t,), dict(t.shape))

    def vjp(self, inputs, which, u, params):
        (x,) = inputs
        sig = _FusedVJP(self.name, 0, x.edges, (frozenset(x.edges), frozenset(x.edges)), params)
        return Function(sig, (x, u), dict(x.shape))

    def lower_fwd(self, lower, t):
        params = cast(_FusedFunction, t.signature).params
        n, o = lower.lower(t.inputs[0])
        dims = tuple(t.shape[e] for e in o)
        return lower.b.fused_fwd(self.name, params, [n], dims), o

    def lower_bwd(self, lower, t):
        params = cast(_FusedVJP, t.signature).params
        xn, xo = lower.lower(t.inputs[0])
        un, uo = lower.lower(t.inputs[1])
        u_al = lower.b.linear([un], [tuple(uo.index(e) for e in xo)], [1]) if uo != xo else un
        dims = tuple(t.shape[e] for e in xo)
        return lower.b.fused_bwd(self.name, 0, params, [xn, u_al], dims), xo

    def emit_fwd(self, cg, node, name, names):
        a = node.params_dict()["approximate"]
        return f"{name} = torch.nn.functional.gelu({cg._logical(node.ops[0], names)}, approximate='{a}')"

    def emit_bwd(self, cg, node, name, names):
        a = node.params_dict()["approximate"]
        return (f"{name} = torch.ops.aten.gelu_backward({cg._logical(node.ops[1], names)}, "
                f"{cg._logical(node.ops[0], names)}, approximate='{a}')")

    def eval_fwd(self, params, inputs):
        (x,) = inputs
        names = x.names
        return torch.nn.functional.gelu(x.rename(None), approximate=params["approximate"]).rename(*names)

    def eval_bwd(self, params, which, inputs, u):
        (x,) = inputs
        names = x.names
        u_al = u.align_to(*names).rename(None)
        g = torch.ops.aten.gelu_backward(u_al, x.rename(None), approximate=params["approximate"])  # pyright: ignore[reportCallIssue]
        return g.rename(*names)
