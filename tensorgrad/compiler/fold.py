"""Definition folding: recognize derived compositions, rewrite to fused cells.

tensorgrad unfolds composites at construction -- F.softmax IS exp/Σexp, the
demo's layer norm is three lines of mean/sqrt -- so by the time a program
reaches the compiler, the fast fused kernels' patterns exist only as soups of
primitives. This pass runs the other direction (a Burstall-Darlington FOLD):
find subtrees that compute an instance of a cell's declared Definition
(compiler/cells.py) and rebuild them as the fused cell, BEFORE gradients
resolve, so reverse mode differentiates through the cell's verified VJP and
the whole training step gets the fused forward AND backward kernels.

Where it runs: runtime.CompiledProgram, on the pristine user trees (gradient
outputs are still lazy Derivative nodes; we fold inside them). It must run
pre-normalization: simplification flattens e.g. the softmax@v boundary, after
which no matcher can recover attention (measured; see the sdpa IR peephole's
contortions in codegen_torch._match_sdpa for what post-hoc matching costs).

Safety model -- a fold may only ever cost speed, never correctness:
  * every candidate is VALUE-GATED: the cell is built and numerically
    compared (float64, distinct random dims per symbol, GATE_SEEDS seeds)
    against the anchor subtree; any mismatch, exception or shape drift
    silently skips the candidate;
  * the rewrite is two-phase: all matching happens against the PRISTINE
    tree, then non-overlapping matches apply in one rebuild (greedy
    in-place rewriting measurably destroys outer patterns' anchors);
  * a final fresh-seed gate re-compares each rebuilt root against its
    original (consolidate.py's refusal shape) and drops ALL folds on any
    mismatch;
  * programs whose gradients would not resolve through the reverse-mode
    family sweep (singleton or nested Derivative families -- see
    reverse.py's family collection) are skipped whole: a folded cell inside
    forward-mode step_derivative would raise (cells are reverse-only).

Gradient equivalence itself is deliberately NOT runtime-gated: the cells'
VJPs are machine-verified against the derived gradients by the permanent
tests (tests/compiler/test_fold.py) and every benchmark's import-time
training gate. NOTE: szfp.verify_rewrite can NOT gate folds -- szfp assigns
fused and derived forms different atoms BY DESIGN (szfp.py), so it would
refuse every correct fold; the float gate here is the right tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

import torch

from tensorgrad.compiler.cells import CELLS, Definition
from tensorgrad.extras.evaluate import Context
from tensorgrad.tensor import Delta, Derivative, Function, Product, Rename, Sum, Tensor, Variable

FOLD = True  # module kill-switch, like STABILIZE / CONSOLIDATE
GATE_SEEDS = 2  # independent labelings a candidate must match at
FINAL_SEED = 0xF01D  # fresh-seed whole-root refusal gate
MAX_GATE_EVALS_PER_ANCHOR = 64
GATE_RTOL, GATE_ATOL = 1e-8, 1e-10  # float64 evaluation


# ---------------------------------------------------------------------------
# Tree access: the five composite types the rebuild knows how to reconstruct.
# Everything else (Variable, Delta, Zero, Affine, Expectation, ...) is a leaf
# -- if we cannot rebuild inside it, we must not match inside it.
# ---------------------------------------------------------------------------


def _kids(t: Tensor) -> Sequence[Tensor]:
    if isinstance(t, Sum):
        return t.terms
    if isinstance(t, Product):
        return t.factors
    if isinstance(t, Rename):
        return (t.tensor,)
    if isinstance(t, Function):
        return tuple(t.inputs)
    if isinstance(t, Derivative):
        return (t.tensor,)
    return ()


def _postorder(roots: Sequence[Tensor]) -> list[Tensor]:
    """Unique nodes, children before parents, DAG-deduped by id. The list
    keeps every node alive so id() keys stay valid for the whole pass."""
    order: list[Tensor] = []
    seen: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(r, False) for r in roots]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for k in _kids(node):
            if id(k) not in seen:
                stack.append((k, False))
    return order


class _FoldContext(Context):
    """extras.evaluate.Context with an ID-KEYED memo instead of the base
    class's isomorphism-keyed cache. Two reasons: (1) the base cache runs a
    VF2 isomorphisms() remap on every hit, which grinds on automorphism-rich
    graphs (a multi-head block's head/batch symmetries); (2) under __debug__
    every hit RE-EVALUATES the subtree, super-linear on shared DAGs. The
    fold pass holds the postorder node list alive, so ids are stable; memo
    entries keep a reference to freshly built candidate tensors too. NaNs
    are values here, not errors -- the gate compares, it never interprets."""

    def __init__(self, values: dict, dims: dict, seed: int):
        super().__init__(values, dims)
        self._memo: dict[int, tuple[Tensor, torch.Tensor]] = {}
        self._atoms: dict[tuple, torch.Tensor] = {}
        self._gen = torch.Generator().manual_seed(seed ^ 0xDE21)

    def _derivative_atom(self, tensor: Tensor) -> torch.Tensor:
        """Unresolved gradients are OPAQUE RANDOM ATOMS (szfp's move for
        transcendentals). Keyed by structure, NOT by object id: Hadamard
        construction RENAMES operand copies, and Derivative._rename rebuilds
        a fresh object -- isomorphic twins must share one draw or the two
        sides of a gate see different gradients. The twin correspondence
        comes from canon's refined EDGE COLORS (iso-invariant and
        name-insensitive, so corresponding edges of twins agree even when
        size symbols repeat, e.g. a square d,d weight); the base is indexed
        in color order and mapped through each twin's own color order."""
        from tensorgrad.structure import canon_info

        info = canon_info(tensor)
        items = sorted((info.refined_colors[e], e) for e in tensor.shape)
        cols = [c for c, _ in items]
        if len(set(cols)) == len(cols):
            key = (info.coarse_fp, tuple(cols))
            base = self._atoms.get(key)
            if base is None:
                sizes = [int(tensor.shape[e].subs(self.dims)) for _, e in items]
                base = torch.randn(sizes, generator=self._gen, dtype=torch.float64)
                self._atoms[key] = base
            return base.rename(*[e for _, e in items]).align_to(*tensor.edges)
        # Automorphic edges (repeated colors) make the correspondence
        # ambiguous: per-id draw (twins then mismatch and the gate refuses
        # the fold -- safe, never wrong).
        edges = list(tensor.shape.keys())
        sizes = [int(tensor.shape[e].subs(self.dims)) for e in edges]
        return torch.randn(sizes, generator=self._gen, dtype=torch.float64).rename(*edges)

    def evaluate(self, tensor: Tensor) -> torch.Tensor:  # noqa: D102 -- see base
        hit = self._memo.get(id(tensor))
        if hit is not None:
            return hit[1]
        if isinstance(tensor, Derivative):
            res = self._derivative_atom(tensor)
        else:
            res = self._evaluate(tensor)
            for k, v in tensor.shape.items():
                if v not in self.dims:
                    self.dims[v] = res.size(k)  # type: ignore[call-overload]  # named dim
        self._memo[id(tensor)] = (tensor, res)
        return res


@dataclass
class FoldSite:
    """Capability object handed to Definition.candidates: per-node feature
    bits, DAG-deduped subtree iteration and cached values -- cells declare
    patterns, the engine provides the tooling."""

    _feats: dict[int, frozenset]
    _ctx: _FoldContext  # signed seed-0 labeling context
    _pctx: Optional[_FoldContext] = None  # positive-draw context (finite)

    def has_feature(self, node: Tensor, bit: str) -> bool:
        return bit in self._feats.get(id(node), frozenset())

    def subtrees(self, node: Tensor) -> Iterator[Tensor]:
        yield from _postorder([node])

    def value(self, node: Tensor) -> Optional[torch.Tensor]:
        """Cached float64 value of `node` at the seed-0 labeling, or None."""
        try:
            return self._ctx.evaluate(node)
        except Exception:
            return None

    def pvalue(self, node: Tensor) -> Optional[torch.Tensor]:
        """Value at the POSITIVE-draw labeling: finite through sqrt/log
        regions, the right oracle for scalar-ratio inference."""
        try:
            return (self._pctx or self._ctx).evaluate(node)
        except Exception:
            return None


@dataclass
class _Match:
    root: Tensor
    definition: Definition
    inputs: tuple[Tensor, ...]
    params: dict
    consumed: frozenset  # node ids between root and the extracted inputs
    index: int  # discovery order (deterministic tiebreak)
    # Multi-output cells (adamw): sibling tree nodes rewritten to OTHER
    # outputs of the same cell call. node id -> (node, out_idx); the node
    # reference keeps the id alive.
    aliases: dict[int, tuple[Tensor, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine phases
# ---------------------------------------------------------------------------


def _definitions() -> list[Definition]:
    defs = [d for c in CELLS.values() if (d := c.definition()) is not None]
    defs.sort(key=lambda d: -d.priority)
    return defs


def _derivative_families_ok(tensors: Sequence[Tensor], order: list[Tensor]) -> bool:
    """True iff every Derivative in the program will resolve through the
    reverse-mode family sweep (scalar base, Variable wrt, family size >= 2).
    Mirrors reverse.resolve_shared_gradients' recognition: Derivative nodes
    may sit ANYWHERE in the outputs (optimizers embed gradients in update
    algebra), collected without descending into a Derivative's own subtree.
    Singleton families and nested derivatives resolve through forward-mode
    step_derivative, where a fused cell raises -- so those programs are
    skipped whole. Errs toward False: a skip preserves behavior."""
    outer: list[Derivative] = []
    seen: set[int] = set()
    stack = list(tensors)
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Derivative):
            outer.append(t)
            continue  # reverse.py does not descend into Derivatives either
        stack.extend(_kids(t))
    families: dict[int, set[int]] = {}
    for d in outer:
        if not isinstance(d.x, Variable) or d.tensor.order != 0:
            return False
        if any(isinstance(n, Derivative) for n in _postorder([d.tensor])):
            return False  # nested derivative (hessian-shaped): forward-mode
        families.setdefault(id(d.tensor), set()).add(id(d.x))
    # Families are counted by DISTINCT wrt-variables: Hadamard construction
    # mints renamed Derivative twins (g*g), which would inflate a singleton
    # gradient into an apparent family and sneak past this gate.
    return all(len(xs) >= 2 for xs in families.values())


def _features(order: list[Tensor], preds: dict) -> dict[int, frozenset]:
    feats: dict[int, frozenset] = {}
    for node in order:
        bits: set = set()
        for k in _kids(node):
            bits |= feats[id(k)]
        if isinstance(node, Function):
            name = getattr(node.signature, "name", "")
            for bit, pred in preds.items():
                if bit not in bits and pred(name):
                    bits.add(bit)
        feats[id(node)] = frozenset(bits)
    return feats


def _dims_for(order: list[Tensor]) -> dict:
    """Distinct small sizes per free symbol (sorted by name): distinctness
    breaks the edge symmetries the matchers rely on (seq vs key vs head).

    Every symbol in the program gets a size (Affine nodes demand their dims
    explicitly -- they produce tensors, so nothing upstream can teach the
    Context their sizes)."""
    syms: set = set()
    for node in order:
        for s in node.shape.values():
            syms |= getattr(s, "free_symbols", set())
    return {s: 3 + i for i, s in enumerate(sorted(syms, key=str))}


def _context(
    order: list[Tensor], dims: dict, seed: int, positive: bool = False
) -> _FoldContext:
    """A labeling context with random float64 values for every Variable.
    Evaluation is LAZY -- nothing is computed until a gate asks (the eager
    whole-program labeling variant taxed every compile in the test suite;
    the Context's isomorphism cache already shares subtree work between the
    anchors and candidates that do get evaluated).

    positive=True draws every value strictly positive. Needed because the
    Hadamard-as-Delta-einsum turns ONE NaN into ALL-NaN (0*NaN terms enter
    every output sum), so a sqrt of a random-signed moment poisons an
    entire optimizer region and signed gates go vacuous there; the positive
    context keeps those regions finite for both scalar-ratio inference and
    a meaningful third gate."""
    gen = torch.Generator().manual_seed(seed)
    values: dict[Variable, torch.Tensor] = {}
    by_name: dict[str, torch.Tensor] = {}
    for node in order:
        if isinstance(node, Variable) and node not in values:
            edges = list(node.shape.keys())
            try:
                sizes = [int(node.shape[e].subs(dims)) for e in edges]
            except Exception:
                continue
            if node.name not in by_name:  # same-named twins share one value
                val = torch.randn(sizes, generator=gen, dtype=torch.float64)
                if positive or not edges:
                    # 0-dim scalars are always positive (step counts / bias
                    # corrections): a negative draw NaNs sqrt(c*v) even in
                    # the signed contexts.
                    val = val.abs() + 0.5
                by_name[node.name] = val.rename(*edges)
            values[node] = by_name[node.name].rename(*edges)
    return _FoldContext(values, dict(dims), seed)


def _gate(anchor: Tensor, cell: Tensor, ctxs: list[_FoldContext]) -> bool:
    """The candidate value gate: WHEREVER the derived anchor's value is
    defined, the cell must be defined and equal. NaN semantics matter: the
    derived Hadamard-as-Delta-einsum spreads one NaN to the WHOLE region
    (0*NaN terms enter every output sum) while a fused cell keeps NaN
    local, so exact NaN-pattern comparison would refuse correct folds. A
    context where the anchor has no finite entries carries no information
    and is skipped; at least one context must be informative (the positive
    contexts keep sqrt regions finite)."""
    if dict(cell.shape) != dict(anchor.shape):
        return False
    informative = 0
    for ctx in ctxs:
        try:
            a = ctx.evaluate(anchor)
            c = ctx.evaluate(cell).align_to(*a.names)
        except Exception:
            return False
        if a.shape != c.shape:
            return False
        a_r, c_r = a.rename(None), c.rename(None)
        mask = torch.isfinite(a_r)
        if not mask.any():
            continue
        informative += 1
        if not torch.allclose(a_r[mask], c_r[mask], rtol=GATE_RTOL, atol=GATE_ATOL):
            return False
    return informative > 0


def _gate_all(
    anchor: Tensor, outs: Any, aliases: dict[int, Tensor], ctxs: list[_FoldContext]
) -> bool:
    """Gate a (possibly multi-output) candidate: output 0 against the anchor
    and every aliased output against its sibling node."""
    seq = outs if isinstance(outs, tuple) else (outs,)
    if not _gate(anchor, seq[0], ctxs):
        return False
    for out_idx, node in aliases.items():
        if out_idx >= len(seq) or not _gate(node, seq[out_idx], ctxs):
            return False
    return True


def _consumed(root: Tensor, inputs: Sequence[Tensor]) -> frozenset:
    """Ids of nodes the fold replaces: descendants of root, stopping at (and
    excluding) the extracted input subtrees -- those remain live as the
    cell's inputs, so nested matches inside them survive. The stop set covers
    the inputs' whole subtrees because an extracted input may be a FRESH
    wrapper (a candidate-built rename inverting dot's edge freshening) whose
    root is not a tree node -- its in-tree inner nodes still must not count
    as consumed."""
    stop = {id(n) for n in _postorder(list(inputs))}
    out: set[int] = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if id(n) in out or id(n) in stop:
            continue
        out.add(id(n))
        stack.extend(_kids(n))
    return frozenset(out)


def _find_matches(
    order: list[Tensor],
    defs: list[Definition],
    feats: dict[int, frozenset],
    ctxs: list[_FoldContext],
) -> list[_Match]:
    site = FoldSite(feats, ctxs[0], ctxs[-1])  # last ctx = the positive one
    matches: list[_Match] = []
    for d in defs:
        # Outermost-first (reverse postorder): once an anchor matches, inner
        # anchors over the same region die at overlap resolution anyway --
        # trying the outer one first just avoids wasted gate evaluations.
        for node in reversed(order):
            if not isinstance(node, tuple(d.root_types)):
                continue
            if not d.features <= feats[id(node)]:
                continue
            # Lazy anchor labeling: evaluate the anchor once per context; an
            # unevaluable anchor (data-pipeline island at these random dims)
            # is skipped without paying per-candidate exception walks.
            try:
                for ctx in ctxs:
                    ctx.evaluate(node)
            except Exception:
                continue
            evals = 0
            for cand in d.candidates(node, site):
                inputs, params = cand[0], cand[1]
                aliases: dict[int, Tensor] = cand[2] if len(cand) > 2 else {}
                evals += 1
                if evals > MAX_GATE_EVALS_PER_ANCHOR:
                    break
                try:
                    outs = d.fused(inputs, params)
                except Exception:
                    continue
                if _gate_all(node, outs, aliases, ctxs):
                    matches.append(
                        _Match(node, d, tuple(inputs), dict(params),
                               _consumed(node, inputs), len(matches),
                               {id(n): (n, i) for i, n in aliases.items()})  # type: ignore[arg-type]
                    )
                    break  # first gate-passing candidate wins this anchor
    return matches


def _resolve_overlaps(matches: list[_Match]) -> dict[int, _Match]:
    """Largest region first (then declared priority, then discovery order);
    a match whose root lies inside an accepted match's consumed region is
    dropped -- it would have no node to attach to after the rewrite. A
    multi-output match's alias nodes count as claimed roots too."""
    matches = sorted(matches, key=lambda m: (-len(m.consumed), -m.definition.priority, m.index))
    accepted: dict[int, _Match] = {}
    claimed: set[int] = set()
    blocked: set[int] = set()
    for m in matches:
        ids = {id(m.root), *m.aliases}
        if ids & claimed or ids & blocked:
            continue
        accepted[id(m.root)] = m
        claimed |= ids
        blocked |= m.consumed - ids
    return accepted


def _rebuild(
    order: list[Tensor],
    roots: Sequence[Tensor],
    accepted: dict[int, _Match],
    fires: dict[str, int],
) -> tuple[list[Tensor], dict[int, Tensor]]:
    """One bottom-up identity-preserving rebuild over the shared postorder:
    matched roots become their cells (inputs already rebuilt -- postorder),
    everything else is rebuilt only if a child changed, so untouched shared
    subtrees keep their identity (and hash-cons sharing) across all outputs."""
    memo: dict[int, Tensor] = {}

    def reb(t: Tensor) -> Tensor:
        """Rebuild `t` through the memo. In-order nodes hit the memo (their
        children are guaranteed processed -- postorder); a FRESH candidate-
        built wrapper (e.g. the rename inverting dot's edge freshening) is
        not in the memo and rebuilds recursively -- its children are."""
        hit = memo.get(id(t))
        if hit is not None:
            return hit
        ks = [reb(k) for k in _kids(t)]
        if all(a is b for a, b in zip(ks, _kids(t))):
            return t
        if isinstance(t, Sum):
            return Sum(ks, t.weights)
        if isinstance(t, Product):
            return Product(ks)
        if isinstance(t, Rename):
            return ks[0].rename(**t.mapping)
        if isinstance(t, Function):
            return Function(t.signature, ks, dict(t.shape_out))
        assert isinstance(t, Derivative)
        return Derivative(ks[0], t.x, t.new_names)

    alias_of: dict[int, tuple[_Match, int]] = {}
    for m in accepted.values():
        for nid, (_, out_idx) in m.aliases.items():
            alias_of[nid] = (m, out_idx)

    def build(m: _Match, out_idx: int, want: Tensor) -> Optional[Tensor]:
        try:
            ins = tuple(reb(i) for i in m.inputs)
            outs = m.definition.fused(ins, m.params)
            res = outs[out_idx] if isinstance(outs, tuple) else outs
            return res if dict(res.shape) == dict(want.shape) else None
        except Exception:
            return None

    for t in order:
        res = None
        m = accepted.get(id(t))
        if m is not None:
            res = build(m, 0, t)
            if res is not None:
                fires[m.definition.cell] = fires.get(m.definition.cell, 0) + 1
        elif id(t) in alias_of:
            am, out_idx = alias_of[id(t)]
            res = build(am, out_idx, t)
        if res is None:
            memo.pop(id(t), None)
            res = reb(t)
        memo[id(t)] = res
    return [memo[id(r)] for r in roots], memo


def fold_program(
    tensors: tuple[Tensor, ...], verbose: bool = False
) -> tuple[tuple[Tensor, ...], dict[str, int]]:
    """Fold cell definitions across a compile family. Never raises; on any
    doubt returns the input tensors unchanged (`fires` says what happened)."""
    fires: dict[str, int] = {}
    if not FOLD:
        return tensors, fires
    defs = _definitions()
    if not defs:
        return tensors, fires
    try:
        order = _postorder(tensors)
        if not _derivative_families_ok(tensors, order):
            return tensors, {"skipped:unresolved-gradient-family": 1}
        preds: dict = {}
        for d in defs:
            preds.update(d.feature_preds)
        feats = _features(order, preds)
        # Cheap structural pre-check BEFORE any evaluation machinery: most
        # compiles have no plausible anchor at all and must pay ~nothing.
        if not any(
            isinstance(n, tuple(d.root_types)) and d.features <= feats[id(n)]
            for d in defs
            for n in order
        ):
            return tensors, fires

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            dims = _dims_for(order)
            # Signed contexts + positive-draw contexts: the positive ones
            # keep sqrt-carrying regions finite, where the signed gates are
            # uninformative (the derived side is all-NaN -- see _gate).
            ctxs = [_context(order, dims, seed=0xC0FFEE + s) for s in range(GATE_SEEDS)]
            ctxs += [
                _context(order, dims, seed=0xBEEF + s, positive=True)
                for s in range(GATE_SEEDS)
            ]

            matches = _find_matches(order, defs, feats, ctxs)
            if not matches:
                return tensors, fires
            accepted = _resolve_overlaps(matches)
            folded, rmemo = _rebuild(order, tensors, accepted, fires)

            # Final fresh-seed refusal gate (consolidate.py's shape): every
            # root whose ORIGINAL evaluates must have its folded twin match.
            # Checked at a signed AND a positive context (sqrt regions are
            # all-NaN under signed draws, where the check is vacuous).
            fctxs = [
                _context(order, dims, seed=FINAL_SEED),
                _context(order, dims, seed=FINAL_SEED ^ 0x5A5A, positive=True),
            ]
            # The fold rebuilds Derivative nodes (their base folds), so the
            # rebuilt twins are NEW objects -- seed them with the SAME opaque
            # atom values as their originals, or the gate compares different
            # random draws and refuses everything.
            for t in order:
                if isinstance(t, Derivative):
                    new = rmemo.get(id(t))
                    if new is not None and new is not t:
                        for fctx in fctxs:
                            try:
                                fctx._memo[id(new)] = (new, fctx.evaluate(t))
                            except Exception:
                                pass
            for orig0, new0 in zip(tensors, folded):
                # Derivative roots are not evaluable; gate their inner tree
                # (the folding happened there; the wrapper is rebuilt 1:1).
                orig, new = orig0, new0
                while isinstance(orig, Derivative) and isinstance(new, Derivative):
                    orig, new = orig.tensor, new.tensor
                if orig is new:
                    continue
                for fctx in fctxs:
                    try:
                        a = fctx.evaluate(orig)
                    except Exception:
                        fires["skipped:root-gate-unevaluable"] = (
                            fires.get("skipped:root-gate-unevaluable", 0) + 1
                        )
                        continue
                    try:
                        c = fctx.evaluate(new).align_to(*a.names)
                    except Exception as exc:
                        c = None
                        reason = f"folded root unevaluable: {type(exc).__name__}: {exc}"
                    if c is not None:
                        if a.shape != c.shape:
                            reason, c = f"shape {tuple(a.shape)} != {tuple(c.shape)}", None
                        else:
                            # masked semantics, like _gate: where the derived
                            # value is defined, the folded one must equal it
                            a_r, c_r = a.rename(None), c.rename(None)
                            fin = torch.isfinite(a_r)
                            if fin.any() and not torch.allclose(
                                a_r[fin], c_r[fin], rtol=GATE_RTOL, atol=GATE_ATOL
                            ):
                                d = (a_r[fin] - c_r[fin]).abs()
                                reason = f"maxdiff {float(d.max()):.3e}"
                                c = None
                    if c is None:
                        if verbose:
                            print(
                                f"fold: final gate REFUSED at root with edges "
                                f"{sorted(orig.edges)}: {reason}; keeping derived forms"
                            )
                        return tensors, {}
        finally:
            torch.set_default_dtype(prev_dtype)

        if verbose and fires:
            print("fold: " + ", ".join(f"{k} x{v}" for k, v in sorted(fires.items())))
        return tuple(folded), fires
    except Exception:
        if verbose:
            import traceback

            traceback.print_exc()
        return tensors, {"skipped:error": 1}
