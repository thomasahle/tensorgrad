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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Optional, Sequence, cast

import torch

from tensorgrad.tensor import Delta, Function, FunctionSignature, Product, Rename, Sum, Tensor

if TYPE_CHECKING:  # runtime imports of ir stay function-local (import cycle)
    from tensorgrad.compiler.ir import MapNode


# ---------------------------------------------------------------------------
# Definitions: the derived composition each cell is a fusion OF, as data.
#
# tensorgrad unfolds composites at construction (F.softmax IS exp/Σexp); a
# Definition is the inverse direction packaged as a rule: how to RECOGNIZE
# an instance of the derived composition in a user tree and rebuild it as
# the fused cell (Burstall-Darlington fold). Consumed by the definition-
# folding pass (compiler/fold.py) today and, unchanged, by a future
# saturation engine where `derived == fused` is just an equality rule.
#
# `candidates` proposes (inputs, params) assignments for an anchor node; it
# never decides correctness -- the engine value-gates every proposal and
# falls back to the derived form, so a wrong proposal costs a missed fold,
# never a wrong program. The `site` argument is the engine's capability
# object (fold.FoldSite): per-node feature bits, cached fp64 values, and
# DAG-deduped subtree iteration -- cells declare, the engine provides.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Definition:
    cell: str  # CELLS registry key
    root_types: tuple[type, ...]  # tensor classes a match root may have
    features: frozenset[str]  # feature bits required somewhere under the root
    feature_preds: Mapping[str, Callable[[str], bool]]  # bit -> predicate on signature.name
    derived: Callable[[Sequence[Tensor], dict], Tensor]  # the composition, from primitives
    fused: Callable[[Sequence[Tensor], dict], Tensor]  # thin wrapper over the cell's build
    candidates: Callable[[Tensor, Any], Iterator[tuple[tuple[Tensor, ...], dict]]]
    priority: int = 0  # overlap tiebreak (larger regions should also win by size)


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

    # -- language side: the derived composition this cell fuses, as a rule.
    # Consumed by the definition-folding pass (compiler/fold.py); cells
    # without one simply never fold. See the Definition docstring above.
    def definition(self) -> Optional[Definition]:
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

    def definition(self) -> Definition:
        # gelu(x, "tanh") constructs as Sum(w=[1/2], [Product[x', 1+tanh(...)',
        # Delta]]): the Hadamard multiply is a Delta join over renamed copies
        # of the operands. The argument is the non-tanh-carrying operand.
        def derived(ins: Sequence[Tensor], params: dict) -> Tensor:
            import tensorgrad.functions as F
            return F.gelu(ins[0], approximate=params["approximate"], fused=False)

        def fused(ins: Sequence[Tensor], params: dict) -> Tensor:
            return self.build(tuple(ins), dict(params))

        def candidates(anchor: Tensor, site: Any) -> Any:
            if not (isinstance(anchor, Sum) and len(anchor.terms) == 1):
                return
            if [float(w) for w in anchor.weights] != [0.5]:
                return
            (prod,) = anchor.terms
            if not isinstance(prod, Product):
                return
            ops = [f for f in prod.factors if not isinstance(f, Delta)]
            if len(ops) != 2:
                return
            # The argument is the non-tanh-chain operand -- but at depth BOTH
            # operands can carry the tanh feature (the argument of a later
            # gelu contains earlier gelus), so order by preference and let
            # the value gate arbitrate.
            ordered = sorted(ops, key=lambda o: site.has_feature(o, "tanh"))
            for cand in ordered:
                arg = cand
                while isinstance(arg, Rename):  # the Delta join renames the operand copy
                    arg = arg.tensor
                yield (arg,), {"approximate": "tanh"}

        return Definition(
            cell=self.name,
            root_types=(Sum,),
            features=frozenset({"tanh"}),
            feature_preds={"tanh": lambda n: n == "tanh"},
            derived=derived,
            fused=fused,
            candidates=candidates,
            priority=0,
        )


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

    def definition(self) -> Definition:
        # Attention constructs as dot(softmax(scale*dot(q,k,hs) + mask, key), v)
        # whose softmax has already unfolded to exp/Σexp: the anchor Product
        # contains exactly ONE exp whose argument splits into a scores term
        # (it contracts the feature edge) and a mask term. Extraction is EDGE
        # ALGEBRA, never name matching -- F.dot renames its contraction edge
        # internally, so hs is recovered as (q.edges ∩ k.edges) - scores.edges
        # from candidate projections. scale is inferred numerically from the
        # labeling values. Every proposal is value-gated against F.sdpa.
        def derived(ins: Sequence[Tensor], params: dict) -> Tensor:
            import tensorgrad.functions as F
            q, k, v, *rest = ins
            seq, key, hs = params["seq"], params["key"], params["hs"]
            scores = F.dot(q, k, dim=hs) * params["scale"]
            if rest:
                scores = scores + rest[0]
            att = F.softmax(scores, dim=key)
            return F.dot(att, v, dim=key)

        def fused(ins: Sequence[Tensor], params: dict) -> Tensor:
            import tensorgrad.functions as F
            q, k, v, *rest = ins
            return F.sdpa(
                q, k, v,
                seq=params["seq"], key=params["key"], hs=params["hs"],
                mask=rest[0] if rest else None, scale=params["scale"],
            )

        def candidates(anchor: Tensor, site: Any) -> Any:
            import tensorgrad.functions as F
            from tensorgrad.compiler.fold import _kids
            if not isinstance(anchor, Product):
                return
            nodes = list(site.subtrees(anchor))
            exps = [
                t for t in nodes
                if isinstance(t, Function) and getattr(t.signature, "name", "") == "exp"
            ]
            if not exps:
                return
            # At depth the anchor's subtree contains EARLIER layers' exps
            # (they ride in through the activations). The anchor's own
            # softmax exp is the OUTERMOST one -- subtrees() is postorder,
            # so try from the back; wrong exps die at the keyset/ratio/gate
            # checks. Cap at 3 to bound the work.
            splits: list[tuple[Tensor, Optional[Tensor], set[int]]] = []
            for e in reversed(exps[-3:]):
                # Nodes containing THIS exp are downstream of its softmax and
                # can never be its q/k/v (deep layers' true projections DO
                # contain EARLIER exps, so a blanket no-exp filter is wrong).
                contains = {id(e)}
                for t in nodes:  # postorder: children decided first
                    if any(id(c) in contains for c in _kids(t)):
                        contains.add(id(t))
                arg = e.inputs[0]
                if isinstance(arg, Sum) and len(arg.terms) == 2:
                    splits += [(arg.terms[0], arg.terms[1], contains),
                               (arg.terms[1], arg.terms[0], contains)]
                else:
                    splits.append((arg, None, contains))
            aedges = set(anchor.edges)
            # Subtree sizes order the pool LARGEST-FIRST: the true q/k/v are
            # the dot's full operands; bias/projection sub-pieces are smaller
            # lookalikes that would otherwise burn the try cap (measured: a
            # biased block puts ~10 same-signature handles in the pool).
            size = {id(t): 0 for t in nodes}
            for t in nodes:  # postorder: children before parents
                size[id(t)] = 1 + sum(size.get(id(c), 0) for c in _kids(t))

            for scores, mask, contains in splits:
                sc = set(scores.edges)
                # key = the scores edge the anchor contracts away (softmax@v)
                keyset = sc - aedges
                if len(keyset) != 1:
                    continue
                (key,) = keyset
                sv = site.pvalue(scores)
                if sv is None:
                    continue
                # Operand handles that survive in the tree: DIRECT nodes
                # already carrying `key` (bare k/v Variables stay visible
                # inside dot's Rename wrappers), and CLEAN pre-rename
                # composites (dot freshens the in-tree operands' batch/head
                # edges, so k/v are re-derived as t.rename(seq->key),
                # inverting the construction). Every proposal value-gates.
                pool: list[Tensor] = []
                direct: list[Tensor] = []
                for t in nodes:
                    if isinstance(t, Delta) or not 3 <= len(t.edges) <= 4:
                        continue
                    if id(t) in contains:
                        continue  # downstream of this softmax: not q/k/v
                    if isinstance(t, Function) and getattr(
                        t.signature, "name", ""
                    ).startswith("pow"):
                        continue
                    if len(set(t.edges) - sc) != 1:
                        continue
                    (direct if key in t.edges else pool).append(t)
                pool.sort(key=lambda t: -size[id(t)])
                direct.sort(key=lambda t: -size[id(t)])

                # For masked attention the mask core {seq,key} PINS seq --
                # no guessing (each wrong guess costs a full k sweep).
                pinned_seqs = None
                if mask is not None:
                    pinned_seqs = sorted(
                        {e for t in site.subtrees(mask) if len(t.edges) == 2
                         and key in t.edges for e in t.edges if e != key}
                    )

                tried = 0
                for q in pool:
                    (hs,) = set(q.edges) - sc
                    if hs not in aedges:
                        continue  # sdpa's output carries hs: junk extras fail
                    seqs = pinned_seqs if pinned_seqs else sorted(set(q.edges) & sc)
                    for seq in seqs:
                        if seq not in q.edges:
                            continue
                        ks = [t for t in direct if hs in t.edges]
                        for k0 in pool:
                            if seq in k0.edges and hs in k0.edges:
                                try:
                                    ks.append(k0.rename(**{seq: key}))
                                except Exception:
                                    pass
                        for k in ks:
                            if k is q:
                                continue
                            tried += 1
                            if tried > 96:
                                return
                            # scale: value(scores)/value(dot(q,k)) constant
                            dv = site.pvalue(F.dot(q, k, dim=hs))
                            if dv is None:
                                continue
                            try:
                                ratio = sv.rename(None) / dv.align_to(*sv.names).rename(None)
                            except Exception:
                                continue
                            scale = float(ratio.flatten()[0])
                            if not torch.allclose(
                                ratio, torch.full_like(ratio, scale), rtol=1e-6, atol=1e-9
                            ):
                                continue
                            params = {
                                "seq": seq, "key": key, "hs": hs,
                                "batch": frozenset(q.edges) - {seq, hs},
                                "scale": scale, "has_mask": mask is not None,
                            }
                            vs = [t for t in direct if hs in t.edges]
                            for v0 in pool:
                                if seq in v0.edges and hs in v0.edges:
                                    try:
                                        vs.append(v0.rename(**{seq: key}))
                                    except Exception:
                                        pass
                            for v in vs:
                                if key not in v.edges or hs not in v.edges:
                                    continue
                                if mask is None:
                                    yield (q, k, v), dict(params)
                                    continue
                                # mask is broadcast over batch/head; its
                                # {seq,key} core is what F.sdpa consumes;
                                # outermost candidates first (weights kept)
                                cores = [
                                    t for t in site.subtrees(mask)
                                    if set(t.edges) == {seq, key}
                                ]
                                for mc in reversed(cores[-4:]):
                                    yield (q, k, v, mc), dict(params)

        return Definition(
            cell=self.name,
            root_types=(Product,),
            features=frozenset({"exp"}),
            feature_preds={"exp": lambda n: n == "exp"},
            derived=derived,
            fused=fused,
            candidates=candidates,
            priority=2,
        )


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

    def definition(self) -> Definition:
        # Hand-written layer norm constructs as
        #   Sum(w=[1,1], [Product[Rename(norm), Rename(g), Delta], bias-term])
        # where norm = (x - mean)/sqrt(var + eps) carries the pow(1/2) anchor
        # and the bias term is the bare Variable or its broadcast Product
        # b ⊗ ones (order-1 Delta caps). g and b are extracted LOCALLY from
        # the anchor's top two levels; only x (the centering Sum's +1 term),
        # and eps (tiny Sum weights under the sqrt, plus fallbacks) are
        # searched -- every proposal is value-gated by the engine.
        def derived(ins: Sequence[Tensor], params: dict) -> Tensor:
            import tensorgrad.functions as F
            x, g, b = ins
            dim, eps = params["dim"], params["eps"]
            xc = x - F.mean(x, dim=dim, keepdims=True)
            var = F.mean(xc * xc, dim=dim, keepdims=True)
            return xc / F.sqrt(var + eps) * g + b

        def fused(ins: Sequence[Tensor], params: dict) -> Tensor:
            import tensorgrad.functions as F
            x, g, b = ins
            return F.layer_norm(x, dim=params["dim"], weight=g, bias=b, eps=params["eps"])

        def _ones(t: Tensor) -> bool:
            """All-ones broadcast caps: order-1 Deltas, possibly nested in
            Products/Renames (broadcasting builds Product[Delta(seq)] etc.)."""
            if isinstance(t, Delta):
                return t.order == 1
            if isinstance(t, Rename):
                return _ones(t.tensor)
            if isinstance(t, Product):
                return bool(t.factors) and all(_ones(f) for f in t.factors)
            return False

        def _core(t: Tensor) -> Tensor:
            """Unwrap Renames and broadcast Products (b ⊗ ones-caps)."""
            while True:
                if isinstance(t, Rename):
                    t = t.tensor
                    continue
                if isinstance(t, Product):
                    real = [f for f in t.factors if not _ones(f)]
                    if len(real) == 1:
                        t = real[0]
                        continue
                return t

        def candidates(anchor: Tensor, site: Any) -> Any:
            if not (isinstance(anchor, Sum) and len(anchor.terms) == 2):
                return
            if [float(w) for w in anchor.weights] != [1.0, 1.0]:
                return
            with_sqrt = [t for t in anchor.terms if site.has_feature(t, "sqrt")]
            if len(with_sqrt) != 1:
                return
            prod = with_sqrt[0]
            (bias_term,) = [t for t in anchor.terms if t is not prod]
            b = _core(bias_term)
            if not isinstance(prod, Product):
                return
            ops = [f for f in prod.factors if not isinstance(f, Delta)]
            if len(ops) != 2:
                return
            norm_ops = [o for o in ops if site.has_feature(o, "sqrt")]
            if len(norm_ops) != 1:
                return
            norm = norm_ops[0]
            g = _core([o for o in ops if o is not norm_ops[0]][0])
            if len(g.edges) != 1 or set(g.edges) != set(b.edges):
                return
            (dim,) = g.edges
            batch = frozenset(anchor.edges) - {dim}
            # x: the +1 term of a centering Sum in the norm subtree, with the
            # anchor's full edge set. subtrees() is postorder, so a DEEP site
            # (whose x contains earlier layers) lists earlier layers'
            # centerings first -- the site's own is the LAST/outermost; try
            # from the back. eps: tiny positive Sum weights under the sqrt
            # (the `var + eps` shift), then standard fallbacks.
            xs, epss = [], []
            for s in site.subtrees(norm):
                if isinstance(s, Sum):
                    ws = [float(w) for w in s.weights]
                    if len(s.terms) == 2 and ws == [1.0, -1.0]:
                        if set(s.terms[0].edges) == set(anchor.edges):
                            xs.append(s.terms[0])
                    epss += [w for w in ws if 0 < w < 1e-2]
            for eps in dict.fromkeys(epss + [1e-5, 1e-6]):  # ordered dedup
                for x in reversed(xs[-3:]):
                    yield (x, g, b), {"dim": dim, "batch": batch, "eps": eps}

        return Definition(
            cell=self.name,
            root_types=(Sum,),
            features=frozenset({"sqrt"}),
            feature_preds={"sqrt": lambda n: n.startswith("pow(k=Fraction(1, 2")},
            derived=derived,
            fused=fused,
            candidates=candidates,
            priority=1,
        )


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

    def definition(self) -> Definition:
        # AdamW-as-algebra (examples/mingpt.py):
        #   m' = b1*m + (1-b1)*g
        #   v' = b2*v + (1-b2)*g*g
        #   w' = w*decay - lr*(c1*m')/(sqrt(c2*v') + eps)
        # The anchor is w' -- Sum(2 terms, weights [decay, -lr]) with a sqrt
        # below -- and the m'/v' Sums are id-shared subtrees of it, matched
        # as ALIASES: one cell call replaces all three roots (the gradient g
        # is consumed once, so the backward is never duplicated). g is an
        # unresolved Derivative at fold time; the engine's labeling treats
        # it as an opaque random atom, keeping the value gate sound.
        def derived(ins: Sequence[Tensor], params: dict) -> Tensor:
            import tensorgrad.functions as F
            w, g, m, v, c1, c2 = ins
            b1, b2 = params["b1"], params["b2"]
            m2 = b1 * m + (1 - b1) * g
            v2 = b2 * v + (1 - b2) * g * g
            return w * params["decay"] - params["lr"] * (c1 * m2) / (
                F.sqrt(c2 * v2) + params["eps"]
            )

        def fused(ins: Sequence[Tensor], params: dict) -> Any:
            return self.build(tuple(ins), dict(params))  # (w', m', v')

        def _core(t: Tensor) -> Tensor:
            while isinstance(t, Rename):
                t = t.tensor
            return t

        def _shallow(t: Tensor, max_depth: int = 8) -> Any:
            """Bounded-depth walk: the optimizer algebra is shallow; the
            gradient subtree below it is arbitrarily deep and must not be
            searched (or even walked -- it can be the whole backward)."""
            from tensorgrad.compiler.fold import _kids
            stack = [(t, 0)]
            seen: set[int] = set()
            while stack:
                n, dep = stack.pop()
                if id(n) in seen or dep > max_depth:
                    continue
                seen.add(id(n))
                yield n
                for k in _kids(n):
                    stack.append((k, dep + 1))

        def _unweight(t: Tensor) -> tuple[float, Tensor]:
            """Peel singleton weighted Sums: scalar multiplication constructs
            `c * t` as Sum([t],[c]), so compile-time constants (decay, lr,
            beta) live one level inside the terms they scale."""
            scale = 1.0
            while isinstance(t, Sum) and len(t.terms) == 1:
                scale *= float(t.weights[0])
                t = t.terms[0]
            return scale, t

        def _moment(s: Tensor) -> Any:
            """Recognize b*prev + (...)*g-ish 2-term Sums. Compile-time
            scalars fold at INCONSISTENT depths (singleton-Sum wrappers,
            weights inside Hadamard products), so only the prev-side weight
            b is read syntactically; the other side's scalar is wherever
            construction put it -- the value gate arbitrates. Returns
            (b, prev_core, other_core) or None."""
            if not (isinstance(s, Sum) and len(s.terms) == 2):
                return None
            w0, t0 = _unweight(s.terms[0])
            b0 = float(s.weights[0]) * w0
            if not 0 < b0 < 1:
                return None
            _, t1 = _unweight(s.terms[1])
            return b0, _core(t0), _core(t1)

        def candidates(anchor: Tensor, site: Any) -> Any:
            import tensorgrad.functions as F
            from tensorgrad.tensor import Variable
            if not (isinstance(anchor, Sum) and len(anchor.terms) == 2):
                return
            ws = [float(w) for w in anchor.weights]
            for wi, ui in ((0, 1), (1, 0)):
                w_scale, w = _unweight(anchor.terms[wi])
                u_scale, uterm = _unweight(anchor.terms[ui])
                decay = ws[wi] * w_scale
                if not 0.5 < decay <= 1.0 + 1e-12:
                    continue
                w = _core(w)
                # moment Sums sit shallow in the update term
                moments = []
                for s in _shallow(uterm):
                    mo = _moment(s)
                    if mo is not None:
                        moments.append((s, mo))
                # c1/c2: order-0 Variables in the (shallow) update term
                cs = [n for n in _shallow(uterm) if isinstance(n, Variable) and not n.edges]
                cs = list(dict.fromkeys(cs))
                epss = []
                for s in _shallow(uterm):
                    if isinstance(s, Sum):
                        epss += [x for x in ([float(y) for y in s.weights]) if 0 < x < 1e-2]
                epss = list(dict.fromkeys(epss + [1e-8]))[:3]
                uv = site.pvalue(uterm)
                if uv is None:
                    continue
                tried = 0
                for m_node, (b1, m, g) in moments:
                    for v_node, (b2, v, _gv) in moments:
                        if v_node is m_node:
                            continue
                        for c1 in cs:
                            for c2 in cs:
                                if c2 is c1:
                                    continue
                                for eps in epss:
                                    tried += 1
                                    if tried > 48:
                                        return
                                    # lr hides wherever construction folded
                                    # it; read it off the constant ratio of
                                    # the update term to a probe built from
                                    # the FOUND nodes (sdpa's scale trick).
                                    try:
                                        probe = (c1 * m_node) / (F.sqrt(c2 * v_node) + eps)
                                        pv = site.pvalue(probe)
                                        if pv is None:
                                            continue
                                        ratio = uv.rename(None) / pv.align_to(*uv.names).rename(None)
                                        # sqrt(c2*v') is NaN where the random
                                        # draws make v' negative -- on BOTH
                                        # sides identically; read the hidden
                                        # scalar off the finite entries.
                                        finite = ratio[torch.isfinite(ratio)]
                                        if finite.numel() < max(4, ratio.numel() // 4):
                                            continue
                                        hidden = float(finite[0])
                                        if not torch.allclose(
                                            finite, torch.full_like(finite, hidden),
                                            rtol=1e-6, atol=1e-9,
                                        ):
                                            continue
                                    except Exception:
                                        continue
                                    lr = -(ws[ui] * u_scale * hidden)
                                    if not 0 < lr < 1:
                                        continue
                                    params = {
                                        "b1": b1, "b2": b2, "lr": lr,
                                        "eps": eps, "decay": decay,
                                    }
                                    yield (
                                        (w, g, m, v, c1, c2),
                                        params,
                                        {1: m_node, 2: v_node},
                                    )

        return Definition(
            cell=self.name,
            root_types=(Sum,),
            features=frozenset({"sqrt"}),
            feature_preds={"sqrt": lambda n: n.startswith("pow(k=Fraction(1, 2")},
            derived=derived,
            fused=fused,
            candidates=candidates,
            priority=3,
        )


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


@register
class _SigmoidCell(FusedCell):
    """exp(x) / (1 + exp(x)) -> torch.sigmoid (task #59; completes #53).

    This is the shape of softplus's DERIVATIVE, where it arrives smeared:
    the derivative wraps exp(x) in a diagonal einsum and the reciprocal
    (pow(-1) of 1+exp(x)) joins one node later, so the raw composition
    overflows float32 at x ~ 89 even after the forward is stabilized.
    The pattern matches an einsum that multiplies recip(1+exp(x)) with the
    SAME exp(x) -- directly, or as a factor of an inner einsum (the
    derivative's diag) -- and substitutes sigmoid(x), dropping the
    reciprocal operand (its wires must be a subset of the exp-carrier's,
    i.e. a pure hadamard join)."""
    name = "sigmoid"
    n_diff = 0

    @staticmethod
    def _recip_of_one_plus_exp(op: Any) -> MapNode | None:
        """Return the exp MapNode if `op` is pow(-1)(1 + exp(x)), else None."""
        from tensorgrad.compiler.ir import EinsumNode, LinearNode, MapNode

        if not (isinstance(op, MapNode) and op.op == "pow" and op.params == (-1,)):
            return None
        lin = op.ops[0]
        if not (isinstance(lin, LinearNode) and len(lin.terms) == 2 and tuple(lin.weights) == (1, 1)):
            return None
        for i in (0, 1):
            t = lin.terms[i]
            o = lin.terms[1 - i]
            if (isinstance(t, MapNode) and t.op == "exp" and len(t.ops) == 1
                    and isinstance(o, EinsumNode) and not o.ops and not o.constraints and o.weight == 1):
                return t
        return None

    def match(self, b: Any, node: Any) -> Any:
        from tensorgrad.compiler.ir import EinsumNode

        if not isinstance(node, EinsumNode) or node.constraints or len(node.ops) < 2:
            return None
        for ri, rop in enumerate(node.ops):
            exp_node = self._recip_of_one_plus_exp(rop)
            if exp_node is None:
                continue
            r_wires = set(node.in_subs[ri])
            for oi, oop in enumerate(node.ops):
                if oi == ri or not r_wires <= set(node.in_subs[oi]):
                    continue
                carrier = None
                if oop is exp_node:
                    carrier = b.fused_fwd("sigmoid", {}, list(exp_node.ops), exp_node.dims)
                elif isinstance(oop, EinsumNode) and not oop.constraints and any(
                    o is exp_node for o in oop.ops
                ):
                    sig = b.fused_fwd("sigmoid", {}, list(exp_node.ops), exp_node.dims)
                    inner_ops = [sig if o is exp_node else o for o in oop.ops]
                    carrier = b.einsum(
                        inner_ops, list(oop.in_subs), oop.out_subs,
                        dict(enumerate(oop.wire_dims)), oop.weight, oop.constraints,
                    )
                if carrier is None:
                    continue
                new_ops = [carrier if j == oi else op for j, op in enumerate(node.ops) if j != ri]
                new_subs = [node.in_subs[j] for j in range(len(node.ops)) if j != ri]
                return b.einsum(
                    new_ops, new_subs, node.out_subs,
                    dict(enumerate(node.wire_dims)), node.weight, node.constraints,
                )
        return None

    def emit_fwd(self, cg: Any, node: Any, name: str, names: Any, dim_of: Any = None) -> str:
        return f"{name} = torch.sigmoid({cg._logical(node.ops[0], names)})"
