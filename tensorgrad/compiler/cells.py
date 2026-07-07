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

from typing import Any, Sequence, cast

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
    """Forward of a fused cell. `cell_name` selects the cell; `params` carries
    its scalar arguments (scale, eps, approximate, ...); `out_idx` selects one
    result of a multi-output kernel (0 for single-output cells -- the name is
    then unchanged, so existing cells are byte-identical)."""

    def __init__(self, cell_name: str, edges: Any, out_edges: Any, params: dict, out_idx: int = 0):
        self.cell_name = cell_name
        self.params = params
        self.out_idx = out_idx
        suffix = f",out={out_idx}" if out_idx else ""
        super().__init__(f"{cell_name}[{_pkey(params)}{suffix}]", frozenset(edges), out_edges)

    def derivative(self, i: int, new_edges: Any = None) -> FunctionSignature:
        raise NotImplementedError(
            f"F.{self.cell_name} (fused) gradients use reverse mode: compile a gradient "
            f"FAMILY, as any training loop does. A lone forward-mode d/dinput is not fused."
        )


class _FusedVJP(FunctionSignature):
    """Reverse VJP of a fused cell w.r.t. input `which`; inputs are the cell's
    original inputs plus the cotangent u; output has that input's edges."""

    def __init__(self, cell_name: str, which: int, edges: Any, in_edges: Any, params: dict):
        self.cell_name = cell_name
        self.which = which
        self.params = params
        super().__init__(
            f"{cell_name}_vjp[i={which},{_pkey(params)}]", frozenset(edges), in_edges
        )

    def derivative(self, i: int, new_edges: Any = None) -> FunctionSignature:
        raise NotImplementedError(
            f"Second-order fused {self.cell_name} is not fused; use the composite form."
        )


# ---------------------------------------------------------------------------
# The cell interface + registry.
# ---------------------------------------------------------------------------


class FusedCell:
    name: str = ""
    n_diff: int = 1  # number of leading inputs the reverse VJP differentiates
                     # (trailing inputs -- e.g. an attention mask -- are not)

    # -- language side: build the forward Function and its reverse VJP -------
    def build(self, inputs: Sequence[Tensor], params: dict) -> Any:
        raise NotImplementedError  # single-output cells return a Tensor; adamw a tuple

    def vjp(self, inputs: Sequence[Tensor], which: int, u: Tensor, params: dict) -> Tensor:
        raise NotImplementedError

    # -- compiler side: lowering to IR (cell owns its edge->wire layout) -----
    def lower_fwd(self, lower: Any, t: Function) -> Any:
        raise NotImplementedError

    def lower_bwd(self, lower: Any, t: Function) -> Any:
        raise NotImplementedError

    # -- backend side: emit target kernels + the value oracle ----------------
    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any) -> str:
        raise NotImplementedError

    def emit_bwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any) -> str:
        raise NotImplementedError

    def eval_fwd(self, params: dict, inputs: tuple[torch.Tensor, ...], out_idx: int = 0) -> torch.Tensor:
        raise NotImplementedError

    def eval_bwd(self, params: dict, which: int, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError


    # -- mapper side: the cell's IR pattern. EDA cells carry their own
    # matching pattern; the peephole pass (compiler/peepholes.py) is one
    # generic engine over the registry, so adding a mapping touches only
    # the cell. Return the replacement node built through the Builder, or
    # None when `node` is not an instance of the pattern. Misses degrade
    # performance, never correctness.
    def match(self, b: Any, node: Any) -> Any:
        return None


CELLS: dict[str, FusedCell] = {}


def register(cell_cls: type) -> type:
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

    def build(self, inputs: Sequence[Tensor], params: dict) -> Any:
        (t,) = inputs
        sig = _FusedFunction(self.name, t.edges, (frozenset(t.edges),), params)
        return Function(sig, (t,), dict(t.shape))

    def vjp(self, inputs: Sequence[Tensor], which: int, u: Tensor, params: dict) -> Tensor:
        (x,) = inputs
        sig = _FusedVJP(self.name, 0, x.edges, (frozenset(x.edges), frozenset(x.edges)), params)
        return Function(sig, (x, u), dict(x.shape))

    def lower_fwd(self, lower: Any, t: Function) -> Any:
        params = cast(_FusedFunction, t.signature).params
        n, o = lower.lower(t.inputs[0])
        dims = tuple(t.shape[e] for e in o)
        return lower.b.fused_fwd(self.name, params, [n], dims), o

    def lower_bwd(self, lower: Any, t: Function) -> Any:
        params = cast(_FusedVJP, t.signature).params
        xn, xo = lower.lower(t.inputs[0])
        un, uo = lower.lower(t.inputs[1])
        u_al = lower.b.linear([un], [tuple(uo.index(e) for e in xo)], [1]) if uo != xo else un
        dims = tuple(t.shape[e] for e in xo)
        return lower.b.fused_bwd(self.name, 0, params, [xn, u_al], dims), xo

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        a = node.params_dict()["approximate"]
        return f"{name} = torch.nn.functional.gelu({cg._logical(node.ops[0], names)}, approximate='{a}')"

    def emit_bwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        a = node.params_dict()["approximate"]
        return (f"{name} = torch.ops.aten.gelu_backward({cg._logical(node.ops[1], names)}, "
                f"{cg._logical(node.ops[0], names)}, approximate='{a}')")

    def eval_fwd(self, params: dict, inputs: Sequence[torch.Tensor], out_idx: int = 0) -> torch.Tensor:
        (x,) = inputs
        names = x.names
        return torch.nn.functional.gelu(x.rename(None), approximate=params["approximate"]).rename(*names)

    def eval_bwd(self, params: dict, which: int, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        x, u = inputs[0], inputs[1]
        names = x.names
        u_al = u.align_to(*names).rename(None)
        g = torch.ops.aten.gelu_backward(u_al, x.rename(None), approximate=params["approximate"])  # pyright: ignore[reportCallIssue]
        return g.rename(*names)


# ---------------------------------------------------------------------------
# Cells with reshape + saved-state caching. SDPA and layer norm share a shape:
# reshape operands into the aten kernel's canonical layout, call the fused
# forward (returning out plus saved state), reshape out back; the backward
# reuses the forward's saved state (or recomputes it in a backward-only
# program -- NEVER via a lambda, which would break torch.compile fullgraph)
# and returns all input grads in one call, deduped per site. The forward's
# saved-state temp and the backward's tuple live in generic per-codegen
# caches keyed by (cell, site).
# ---------------------------------------------------------------------------


def _fmt(x: Any) -> str:  # lazy handle to codegen's weight formatter (avoids import cycle)
    from tensorgrad.compiler.codegen_torch import _fmt_weight
    return _fmt_weight(x)


def _tup(xs: Any) -> str:
    from tensorgrad.compiler.codegen_torch import _tup as t
    return t(xs)


def _prod(xs: Any) -> Any:
    from tensorgrad.compiler.codegen_torch import _prod as p
    return p(xs)


def _lu_prelude(cg: Any, node: Any, names: Any) -> tuple:
    """Shared LU factorization for an operand consumed by several linalg
    cells (codegen pre-scans consumers into cg._lu_shared): the first cell
    at a site emits torch.linalg.lu_factor once; the rest reuse it. This is
    the factorization reuse jax's solve/slogdet adjoints get implicitly."""
    key = ("lu", names[id(node.ops[0])])
    tmp = cg._fused_fwd_cache.get(key)
    prelude = ""
    if tmp is None:
        tmp = f"_lu{len(cg._fused_fwd_cache)}"
        cg._fused_fwd_cache[key] = tmp
        prelude = f"{tmp} = torch.linalg.lu_factor({cg._logical(node.ops[0], names)}); "
    return prelude, tmp


@register
class _SDPACell(FusedCell):
    name = "sdpa"
    n_diff = 3  # q, k, v differentiate; a trailing mask does not
    FWD = "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu"
    BWD = "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward"

    def build(self, inputs: Sequence[Tensor], params: dict) -> Any:
        seq, key, hs, batch = params["seq"], params["key"], params["hs"], params["batch"]
        out = batch | {seq, hs}
        ins = [batch | {seq, hs}, batch | {key, hs}, batch | {key, hs}]
        if params["has_mask"]:
            ins.append(frozenset({seq, key}))
        sig = _FusedFunction(self.name, out, tuple(frozenset(i) for i in ins), params)
        q = inputs[0]
        shape_out = {**{e: q.shape[e] for e in batch}, seq: q.shape[seq], hs: q.shape[hs]}
        return Function(sig, tuple(inputs), shape_out)

    def vjp(self, inputs: Sequence[Tensor], which: int, u: Tensor, params: dict) -> Tensor:
        seq, key, hs, batch = params["seq"], params["key"], params["hs"], params["batch"]
        q, k = inputs[0], inputs[1]
        ins = [q, k, inputs[2], u] + ([inputs[3]] if params["has_mask"] else [])
        in_edges = [batch | {seq, hs}, batch | {key, hs}, batch | {key, hs}, frozenset(u.edges)]
        if params["has_mask"]:
            in_edges.append(frozenset({seq, key}))
        out_i = (batch | {seq, hs}) if which == 0 else (batch | {key, hs})
        sig = _FusedVJP(self.name, which, out_i, tuple(frozenset(e) for e in in_edges), params)
        role = seq if which == 0 else key
        shape_out = {e: (q.shape[e] if which == 0 else k.shape[e])
                     for e in [*sorted(batch), role, hs]}
        return Function(sig, tuple(ins), shape_out)

    def lower_fwd(self, lower: Any, t: Function) -> Any:
        p = cast(_FusedFunction, t.signature).params
        batch = sorted(p["batch"])
        q_order = batch + [p["seq"], p["hs"]]
        kv_order = batch + [p["key"], p["hs"]]
        ops, perms = [], []
        for oi, order in [(0, q_order), (1, kv_order), (2, kv_order)]:
            n, o = lower.lower(t.inputs[oi])
            ops.append(n); perms.append(tuple(o.index(e) for e in order))
        if p["has_mask"]:
            mn, mo = lower.lower(t.inputs[3])
            ops.append(mn); perms.append(tuple(mo.index(e) for e in [p["seq"], p["key"]]))
        out_order = tuple(batch + [p["seq"], p["hs"]])
        dims = tuple(t.shape[e] for e in out_order)
        node = lower.b.fused_fwd(self.name, p, ops, dims, layout=(len(batch), tuple(perms)))
        return node, out_order

    def lower_bwd(self, lower: Any, t: Function) -> Any:
        sig = cast(_FusedVJP, t.signature)
        p, which = sig.params, sig.which
        batch = sorted(p["batch"])
        specs = [(0, batch + [p["seq"], p["hs"]]), (1, batch + [p["key"], p["hs"]]),
                 (2, batch + [p["key"], p["hs"]]), (3, batch + [p["seq"], p["hs"]])]  # 3 = u
        ops, perms = [], []
        for oi, oedges in specs:
            n, o = lower.lower(t.inputs[oi])
            ops.append(n); perms.append(tuple(o.index(e) for e in oedges))
        if p["has_mask"]:
            mn, mo = lower.lower(t.inputs[4])
            ops.append(mn); perms.append(tuple(mo.index(e) for e in [p["seq"], p["key"]]))
        role = p["seq"] if which == 0 else p["key"]
        out_order = tuple(t.edges)
        res_perm = tuple((batch + [role, p["hs"]]).index(e) for e in out_order)
        dims = tuple(t.shape[e] for e in out_order)
        node = lower.b.fused_bwd(self.name, which, p, ops, dims,
                                 layout=(len(batch), tuple(perms), res_perm))
        return node, out_order

    def _qkv(self, cg: Any, node: Any, names: Any, dim_of: Any) -> tuple:
        nb, perms = node.layout[0], node.layout[1]
        p0 = perms[0]
        bsz = _prod([dim_of(node.ops[0].dims[a]) for a in p0[:nb]]) if nb else 1
        S = dim_of(node.ops[0].dims[p0[nb]])
        E = dim_of(node.ops[0].dims[p0[nb + 1]])
        K = dim_of(node.ops[1].dims[perms[1][nb]])
        def r(i: int, seqlen: Any) -> str:
            return f"{cg._sdpa_canon(node.ops[i], perms[i], names)}.reshape({bsz}, 1, {seqlen}, {E})"
        return r(0, S), r(1, K), r(2, K), (bsz, S, K, E)

    def _site(self, node: Any, names: Any, mask_idx: int) -> tuple:
        p = node.params_dict()
        return ("sdpa", names[id(node.ops[0])], names[id(node.ops[1])], names[id(node.ops[2])],
                p["scale"], p["has_mask"], names[id(node.ops[mask_idx])] if p["has_mask"] else None)

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        p = node.params_dict()
        q4, k4, v4, _ = self._qkv(cg, node, names, dim_of)
        mask = cg._sdpa_canon(node.ops[3], node.layout[1][3], names) if p["has_mask"] else "None"
        bshape = _tup(str(dim_of(d)) for d in node.dims)
        tmp = f"_sdpaf{len(cg._fused_fwd_cache)}"
        cg._fused_fwd_cache[self._site(node, names, 3)] = tmp
        return (f"{tmp} = {self.FWD}({q4}, {k4}, {v4}, 0.0, False, "
                f"attn_mask={mask}, scale={_fmt(p['scale'])}); {name} = {tmp}[0].reshape({bshape})")

    def emit_bwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        p = node.params_dict()
        nb, perms, res_perm = node.layout
        q4, k4, v4, (bsz, S, K, E) = self._qkv(cg, node, names, dim_of)
        u4 = f"{cg._sdpa_canon(node.ops[3], perms[3], names)}.reshape({bsz}, 1, {S}, {E})"
        mask = cg._sdpa_canon(node.ops[4], perms[4], names) if p["has_mask"] else "None"
        gkey = ("sdpa", names[id(node.ops[0])], names[id(node.ops[1])], names[id(node.ops[2])],
                names[id(node.ops[3])], p["scale"], p["has_mask"],
                names[id(node.ops[4])] if p["has_mask"] else None)
        prelude = ""
        tmp = cg._fused_bwd_cache.get(gkey)
        if tmp is None:
            tmp = f"_sdpab{len(cg._fused_bwd_cache)}"
            cg._fused_bwd_cache[gkey] = tmp
            kw = f"0.0, False, attn_mask={mask}, scale={_fmt(p['scale'])}"
            fwd_tmp = cg._fused_fwd_cache.get(self._site(node, names, 4))
            if fwd_tmp is None:
                fwd_tmp = f"_sdpar{len(cg._fused_fwd_cache)}"
                cg._fused_fwd_cache[self._site(node, names, 4)] = fwd_tmp
                prelude = f"{fwd_tmp} = {self.FWD}({q4}, {k4}, {v4}, {kw}); "
            prelude += (f"{tmp} = {self.BWD}({u4}, {q4}, {k4}, {v4}, "
                        f"{fwd_tmp}[0], {fwd_tmp}[1], {kw}); ")
        role = S if node.which == 0 else K
        bdims = [str(dim_of(node.ops[0].dims[a])) for a in perms[0][:nb]]
        gshape = _tup(bdims + [str(role), str(E)])
        return (f"{prelude}{name} = {tmp}[{node.which}]"
                f".reshape({gshape}).permute({_tup(map(str, res_perm))})")

    def eval_fwd(self, params: dict, inputs: Sequence[torch.Tensor], out_idx: int = 0) -> torch.Tensor:
        from tensorgrad.extras.evaluate import _eval_sdpa_fwd
        return _eval_sdpa_fwd(params, inputs)

    def eval_bwd(self, params: dict, which: int, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        from tensorgrad.extras.evaluate import _eval_sdpa_bwd
        return _eval_sdpa_bwd(params, which, inputs)


@register
class _LayerNormCell(FusedCell):
    name = "layer_norm"
    n_diff = 3  # x, weight, bias
    FWD = "torch.ops.aten.native_layer_norm"
    BWD = "torch.ops.aten.native_layer_norm_backward"

    def build(self, inputs: Sequence[Tensor], params: dict) -> Any:
        dim, batch = params["dim"], params["batch"]
        out = batch | {dim}
        ins = [batch | {dim}, frozenset({dim}), frozenset({dim})]
        sig = _FusedFunction(self.name, out, tuple(frozenset(i) for i in ins), params)
        x = inputs[0]
        shape_out = {**{e: x.shape[e] for e in batch}, dim: x.shape[dim]}
        return Function(sig, tuple(inputs), shape_out)

    def vjp(self, inputs: Sequence[Tensor], which: int, u: Tensor, params: dict) -> Tensor:
        dim, batch = params["dim"], params["batch"]
        x = inputs[0]
        ins = (x, inputs[1], inputs[2], u)
        in_edges = [batch | {dim}, frozenset({dim}), frozenset({dim}), frozenset(u.edges)]
        out_i = (batch | {dim}) if which == 0 else frozenset({dim})
        sig = _FusedVJP(self.name, which, out_i, tuple(in_edges), params)
        if which == 0:
            shape_out = {**{e: x.shape[e] for e in batch}, dim: x.shape[dim]}
        else:
            shape_out = {dim: x.shape[dim]}
        return Function(sig, ins, shape_out)

    def lower_fwd(self, lower: Any, t: Function) -> Any:
        p = cast(_FusedFunction, t.signature).params
        dim, batch = p["dim"], sorted(p["batch"])
        xn, xo = lower.lower(t.inputs[0])
        wn, wo = lower.lower(t.inputs[1])
        bn, bo = lower.lower(t.inputs[2])
        perms = [tuple(xo.index(e) for e in batch + [dim]), (wo.index(dim),), (bo.index(dim),)]
        out_order = tuple(batch + [dim])
        dims = tuple(t.shape[e] for e in out_order)
        node = lower.b.fused_fwd(self.name, p, [xn, wn, bn], dims, layout=(len(batch), tuple(perms)))
        return node, out_order

    def lower_bwd(self, lower: Any, t: Function) -> Any:
        sig = cast(_FusedVJP, t.signature)
        p, which = sig.params, sig.which
        dim, batch = p["dim"], sorted(p["batch"])
        specs = [(0, batch + [dim]), (1, [dim]), (2, [dim]), (3, batch + [dim])]  # 3 = u
        ops, perms = [], []
        for oi, oedges in specs:
            n, o = lower.lower(t.inputs[oi])
            ops.append(n); perms.append(tuple(o.index(e) for e in oedges))
        out_order = tuple(t.edges)
        grad_canon = (batch + [dim]) if which == 0 else [dim]
        res_perm = tuple(grad_canon.index(e) for e in out_order)
        dims = tuple(t.shape[e] for e in out_order)
        node = lower.b.fused_bwd(self.name, which, p, ops, dims,
                                 layout=(len(batch), tuple(perms), res_perm))
        return node, out_order

    def _xwb(self, cg: Any, node: Any, names: Any, dim_of: Any) -> tuple:
        nb, perms = node.layout[0], node.layout[1]
        p0 = perms[0]
        rows = _prod([dim_of(node.ops[0].dims[a]) for a in p0[:nb]]) if nb else 1
        D = dim_of(node.ops[0].dims[p0[nb]])
        x2d = f"{cg._sdpa_canon(node.ops[0], perms[0], names)}.reshape({rows}, {D})"
        w1d = f"{cg._sdpa_canon(node.ops[1], perms[1], names)}.reshape({D})"
        b1d = f"{cg._sdpa_canon(node.ops[2], perms[2], names)}.reshape({D})"
        return x2d, w1d, b1d, (rows, D)

    def _site(self, node: Any, names: Any) -> tuple:
        return ("layer_norm", names[id(node.ops[0])], names[id(node.ops[1])],
                names[id(node.ops[2])], node.params_dict()["eps"])

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        eps = _fmt(node.params_dict()["eps"])
        x2d, w1d, b1d, (rows, D) = self._xwb(cg, node, names, dim_of)
        bshape = _tup(str(dim_of(d)) for d in node.dims)
        tmp = f"_lnf{len(cg._fused_fwd_cache)}"
        cg._fused_fwd_cache[self._site(node, names)] = tmp
        return (f"{tmp} = {self.FWD}({x2d}, [{D}], {w1d}, {b1d}, {eps}); "
                f"{name} = {tmp}[0].reshape({bshape})")

    def emit_bwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        nb, perms, res_perm = node.layout
        eps = _fmt(node.params_dict()["eps"])
        x2d, w1d, b1d, (rows, D) = self._xwb(cg, node, names, dim_of)
        u2d = f"{cg._sdpa_canon(node.ops[3], perms[3], names)}.reshape({rows}, {D})"
        gkey = ("layer_norm", names[id(node.ops[0])], names[id(node.ops[1])],
                names[id(node.ops[2])], names[id(node.ops[3])], node.params_dict()["eps"])
        prelude = ""
        tmp = cg._fused_bwd_cache.get(gkey)
        if tmp is None:
            tmp = f"_lnb{len(cg._fused_bwd_cache)}"
            cg._fused_bwd_cache[gkey] = tmp
            fwd_tmp = cg._fused_fwd_cache.get(self._site(node, names))
            if fwd_tmp is None:
                fwd_tmp = f"_lnr{len(cg._fused_fwd_cache)}"
                cg._fused_fwd_cache[self._site(node, names)] = fwd_tmp
                prelude = f"{fwd_tmp} = {self.FWD}({x2d}, [{D}], {w1d}, {b1d}, {eps}); "
            prelude += (f"{tmp} = {self.BWD}({u2d}, {x2d}, [{D}], {fwd_tmp}[1], {fwd_tmp}[2], "
                        f"{w1d}, {b1d}, [True, True, True]); ")
        if node.which == 0:  # grad_x: (rows, D) -> (batch..., dim) canonical
            bdims = [str(dim_of(node.ops[0].dims[a])) for a in perms[0][:nb]]
            gshape = _tup(bdims + [str(D)])
        else:  # grad_weight / grad_bias: (D,) -> (dim,)
            gshape = _tup([str(D)])
        return (f"{prelude}{name} = {tmp}[{node.which}]"
                f".reshape({gshape}).permute({_tup(map(str, res_perm))})")

    def eval_fwd(self, params: dict, inputs: Sequence[torch.Tensor], out_idx: int = 0) -> torch.Tensor:
        from tensorgrad.extras.evaluate import _eval_layer_norm_fwd
        return _eval_layer_norm_fwd(params, inputs)

    def eval_bwd(self, params: dict, which: int, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        from tensorgrad.extras.evaluate import _eval_layer_norm_bwd
        return _eval_layer_norm_bwd(params, which, inputs)


# ---------------------------------------------------------------------------
# AdamW: the first NON-differentiable, MULTI-OUTPUT cell. It proves the
# abstraction is "any kernel", not just forward+reverse primitives. The
# optimizer update is a leaf in the training step (nothing differentiates it:
# n_diff=0, no VJP), and it returns three results (w', m', v') from ONE fused
# call, selected by the node's `which`.
#
# Crucially it also removes the AdamW gradient-doubling: written as algebra,
# g = grad(loss, w) is consumed at three spots (m' uses g, v' uses g*g), and
# simplify duplicates the g-subtree, so the whole backward is computed twice.
# As this opaque cell, g is a SINGLE operand -- computed once. The doubling
# cannot occur.
#
# Value definition (matches examples/mingpt.py's `adamw`):
#   m' = b1*m + (1-b1)*g
#   v' = b2*v + (1-b2)*g*g
#   w' = w*decay - lr*(c1*m') / (sqrt(c2*v') + eps)
# where c1, c2 are runtime scalar inputs (1/(1-b^t)); decay is a per-param
# compile-time constant (1 - lr*wd on matrices, 1 on biases/gains).
# ---------------------------------------------------------------------------


@register
class _AdamWCell(FusedCell):
    name = "adamw"
    n_diff = 0  # the optimizer update is a leaf; nothing differentiates it

    def build(self, inputs: Sequence[Tensor], params: dict) -> Any:
        # inputs: (w, g, m, v, c1, c2); outputs: (w', m', v')
        w = inputs[0]
        in_edges = tuple(frozenset(x.edges) for x in inputs)
        outs = []
        for i in range(3):
            sig = _FusedFunction(self.name, w.edges, in_edges, params, out_idx=i)
            outs.append(Function(sig, tuple(inputs), dict(w.shape)))
        return tuple(outs)

    def lower_fwd(self, lower: Any, t: Function) -> Any:
        sig = cast(_FusedFunction, t.signature)
        out = tuple(t.edges)
        ops = []
        for inp in t.inputs:
            n, o = lower.lower(inp)
            if o and tuple(o) != out:  # bring each operand to the output edge order
                n = lower.b.linear([n], [tuple(o.index(e) for e in out)], [1])
            ops.append(n)
        dims = tuple(t.shape[e] for e in out)
        return lower.b.fused_fwd(self.name, sig.params, ops, dims, which=sig.out_idx), out

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        p = node.params_dict()
        w, g, m, v, c1, c2 = [cg._logical(o, names) for o in node.ops]
        site = ("adamw",) + tuple(names[id(o)] for o in node.ops)
        tmp = cg._fused_fwd_cache.get(site)
        prelude = ""
        if tmp is None:
            # Emit the three results ONCE per site; each which selects its own.
            tmp = f"_adamw{len(cg._fused_fwd_cache)}"
            cg._fused_fwd_cache[site] = tmp
            b1, b2, lr, eps, decay = p["b1"], p["b2"], p["lr"], p["eps"], p["decay"]
            prelude = (
                f"{tmp}_m = {b1!r}*{m} + {(1.0 - b1)!r}*{g}; "
                f"{tmp}_v = {b2!r}*{v} + {(1.0 - b2)!r}*{g}*{g}; "
                f"{tmp}_w = {w}*{decay!r} - {lr!r}*({c1}*{tmp}_m)/(torch.sqrt({c2}*{tmp}_v) + {eps!r}); "
            )
        sel = ("w", "m", "v")[node.which]
        return f"{prelude}{name} = {tmp}_{sel}"

    def eval_fwd(self, params: dict, inputs: Sequence[torch.Tensor], out_idx: int = 0) -> torch.Tensor:
        w, g, m, v, c1, c2 = inputs
        names = w.names
        wv, gv, mv, vv = (x.align_to(*names).rename(None) for x in (w, g, m, v))
        c1v, c2v = c1.rename(None), c2.rename(None)  # 0-d scalars, broadcast
        b1, b2, lr, eps, decay = params["b1"], params["b2"], params["lr"], params["eps"], params["decay"]
        m_new = b1 * mv + (1.0 - b1) * gv
        v_new = b2 * vv + (1.0 - b2) * gv * gv
        w_new = wv * decay - lr * (c1v * m_new) / (torch.sqrt(c2v * v_new) + eps)
        return (w_new, m_new, v_new)[out_idx].rename(*names)


# ---------------------------------------------------------------------------
# Matrix inverse: a cell that is NOT opaque to differentiation. F.inverse
# keeps its classic signature whose SYMBOLIC derivative (the cookbook
# identity d(K^-1) = -K^-1 dK K^-1) expands during simplification, so
# lowering only ever sees forward applications -- the cell contributes just
# the backend mapping. It shows the registry's span: differentiable leaf
# (this), opaque forward+reverse pairs (sdpa/layer_norm/gelu), and
# multi-output optimizer updates (adamw) are all "cells".
#
# Edge convention (matches the interpreter oracle in extras/evaluate.py):
# the output carries the input's two edges SWAPPED, so contracting
# same-name edges with the original cancels: out[.., j:e2, i:e1] =
# inv(in[.., i:e1, j:e2]) -- orientation-free since inv(M^T) = inv(M)^T.
# ---------------------------------------------------------------------------


@register
class _InverseCell(FusedCell):
    name = "inverse"
    n_diff = 0  # differentiation happened symbolically, upstream of lowering

    def lower_fwd(self, lower: Any, t: Function) -> Any:
        e1, e2 = sorted(t.signature.edges)
        n, o = lower.lower(t.inputs[0])
        rest = [e for e in o if e not in (e1, e2)]
        want = tuple(rest + [e1, e2])
        if want != tuple(o):  # align operand to (batch..., e1, e2)
            n = lower.b.linear([n], [tuple(o.index(e) for e in want)], [1])
        out_order = tuple(rest + [e2, e1])  # swapped naming = cancel convention
        dims = tuple(t.shape[e] for e in out_order)
        return lower.b.fused_fwd(self.name, {}, [n], dims), out_order

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        A = cg._logical(node.ops[0], names)
        if id(node.ops[0]) in getattr(cg, "_lu_shared", ()):
            pre, lu = _lu_prelude(cg, node, names)
            n_ = dim_of(node.dims[-1])
            return f"{pre}{name} = torch.linalg.lu_solve({lu}[0], {lu}[1], torch.eye({n_}, dtype={A}.dtype))"
        return f"{name} = torch.linalg.inv({A})"


@register
class _DetCell(FusedCell):
    """Determinant over two named edges -> torch.linalg.det, broadcasting
    over the rest. Like the inverse cell, differentiation is symbolic and
    upstream (the cookbook rule d det(A) = det(A) A^-T lives on the
    signature), so the cell is a forward mapping only. Orientation-free:
    det(A^T) = det(A)."""
    name = "det"
    n_diff = 0

    def lower_fwd(self, lower: Any, t: Function) -> Any:
        (edges,) = t.signature.inputs
        e1, e2 = sorted(edges)
        n, o = lower.lower(t.inputs[0])
        rest = [e for e in o if e not in (e1, e2)]
        want = tuple(rest + [e1, e2])
        if want != tuple(o):
            n = lower.b.linear([n], [tuple(o.index(e) for e in want)], [1])
        out_order = tuple(rest)
        dims = tuple(t.shape[e] for e in out_order)
        return lower.b.fused_fwd(self.name, {}, [n], dims), out_order

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        return f"{name} = torch.linalg.det({cg._logical(node.ops[0], names)})"


@register
class _SolveCell(FusedCell):
    """torch.linalg.solve, created ONLY by the linalg peephole
    (compiler/peepholes.py) from inverse-then-contract patterns -- there is
    deliberately no F.solve: users write the algebra inverse(A) @ b and the
    compiler picks the kernel, exactly like exp/sum-exp becomes softmax.
    ops = (A, b) with A pre-aligned (e1, e2) by the inverse cell's lowering;
    params carry whether the contraction hit the transposed side."""
    name = "solve"
    n_diff = 0

    def match(self, b: Any, node: Any) -> Any:
        """einsum contracting ONE matrix edge of inverse(A) with a vector
        rhs -> solve(A, rhs). v1 scope: plain matrix inverse (order 2, no
        batch), vector right-hand side, no constraints/diagonals/weights."""
        from tensorgrad.compiler.ir import EinsumNode, FusedFwdNode

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
        # Which inverse axis contracted? The inverse cell's convention puts
        # axes (e2, e1) with inv[e2, e1] = inv(A[e1, e2]): axis 1 -> A^-1 rhs
        # (plain solve), axis 0 -> A^-T rhs (solve against the transpose).
        transposed = inv_subs.index(rhs_wire) == 0
        A = inv.ops[0]  # aligned (e1, e2) by the inverse cell's lowering
        return b.fused_fwd("solve", {"transposed": transposed}, [A, rhs], node.dims)

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        A = cg._logical(node.ops[0], names)
        b = cg._logical(node.ops[1], names)
        adj = node.params_dict()["transposed"]
        if id(node.ops[0]) in getattr(cg, "_lu_shared", ()):
            pre, lu = _lu_prelude(cg, node, names)
            return (f"{pre}{name} = torch.linalg.lu_solve({lu}[0], {lu}[1], "
                    f"{b}.unsqueeze(-1), adjoint={adj}).squeeze(-1)")
        if adj:
            A = f"{A}.transpose(-2, -1)"
        return f"{name} = torch.linalg.solve({A}, {b})"


@register
class _SlogdetCell(FusedCell):
    """log|det(A)| via torch.linalg.slogdet, created ONLY by the linalg
    peephole from log(det(A)). det(K) of a moderate SPD kernel underflows
    float32 while its log is a perfectly ordinary number; slogdet computes
    the log directly."""
    name = "slogdet"
    n_diff = 0

    def match(self, b: Any, node: Any) -> Any:
        """log(det(A)) -> slogdet(A)[1] (log|det|; SPD kernels in mind)."""
        from tensorgrad.compiler.ir import FusedFwdNode, MapNode

        if not (isinstance(node, MapNode) and node.op == "log" and len(node.ops) == 1):
            return None
        det = node.ops[0]
        if isinstance(det, FusedFwdNode) and det.cell_name == "det":
            return b.fused_fwd("slogdet", {}, list(det.ops), node.dims)
        return None

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        if id(node.ops[0]) in getattr(cg, "_lu_shared", ()):
            pre, lu = _lu_prelude(cg, node, names)
            return f"{pre}{name} = {lu}[0].diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)"
        return f"{name} = torch.linalg.slogdet({cg._logical(node.ops[0], names)})[1]"


@register
class _StackCell(FusedCell):
    """torch.stack of same-shaped operands along a new leading axis; created
    only by the gemm-batching pass (compiler/gemm_batch.py). One kernel
    replaces k operand reads; the batched einsum downstream replaces k GEMM
    launches with one."""
    name = "stack"
    n_diff = 0

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        parts = ", ".join(cg._logical(op, names) for op in node.ops)
        return f"{name} = torch.stack(({parts}))"


@register
class _SelectCell(FusedCell):
    """A zero-cost view selecting index i along the leading axis of a
    batched result; created only by the gemm-batching pass."""
    name = "select"
    n_diff = 0

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        i = node.params_dict()["index"]
        return f"{name} = {cg._logical(node.ops[0], names)}[{i}]"


@register
class _SoftplusCell(FusedCell):
    """log(1 + exp(x)) -> torch.nn.functional.softplus (task #53). The raw
    composition overflows float32 at x ~ 89; softplus computes it exactly.
    Same contract as the other stabilization patterns: the DERIVATIVE was
    already taken symbolically upstream, so only the surviving forward
    composition is rewritten -- misses degrade numerics headroom, never
    correctness of what does compile."""
    name = "softplus"
    n_diff = 0

    def match(self, b: Any, node: Any) -> Any:
        from tensorgrad.compiler.ir import EinsumNode, LinearNode, MapNode

        if not (isinstance(node, MapNode) and node.op == "log" and len(node.ops) == 1):
            return None
        lin = node.ops[0]
        if not (isinstance(lin, LinearNode) and len(lin.terms) == 2):
            return None
        if tuple(lin.weights) != (1, 1):
            return None

        def is_ones(t: Any) -> bool:
            return isinstance(t, EinsumNode) and not t.ops and not t.constraints and t.weight == 1

        for i in (0, 1):
            exp_ = lin.terms[i]
            if (isinstance(exp_, MapNode) and exp_.op == "exp" and len(exp_.ops) == 1
                    and is_ones(lin.terms[1 - i])):
                x = exp_.ops[0]
                perm = lin.perms[i]
                if tuple(perm) != tuple(range(len(perm))):
                    x = b.linear([x], [tuple(perm)], [1])
                return b.fused_fwd("softplus", {}, [x], node.dims)
        return None

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        return f"{name} = torch.nn.functional.softplus({cg._logical(node.ops[0], names)})"
