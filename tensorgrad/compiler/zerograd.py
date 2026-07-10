"""Zero-gradient pruning: gradients that are provably zero compile to Zero.

Some gradients are exactly zero by SYMMETRY, not by data: the canonical case
is the second bias inside softmax attention — softmax is invariant to a
per-row shift, so d loss / d bk == 0 identically, for every input and every
weight value. A tape-based framework computes that zero numerically every
step (and pays the whole cotangent chain that produces it); a symbolic
library can prove it once at compile time and delete the chain.

Mechanism: BEFORE definition folding (fused cells are opaque atoms to szfp,
so the proof must run on the plain algebra — measured: the real minGPT bk
gradient proves zero unfolded and is unprovable folded), collect the lazy
Derivative families, resolve each family's gradients with the same reverse
sweep the compiler uses (reverse.py), and Schwartz-Zippel zero-test each
gradient: exact arithmetic mod P at k random points, so a nonzero gradient
is misclassified with probability <= (deg/P)^k (~1e-15 at k=3) — the same
trust level as the consolidation pass's value merging. Proven-zero
Derivative nodes are replaced by Zero; simplify then constant-folds the
optimizer algebra above them (m' = b1*m, ...) and lowering emits a free
torch.zeros for anything that survives to an output.

Semantics note: this makes training MORE exact, not different — torch
computes the same gradient as fp32 roundoff (~1e-9, not 0.0), so tiny
parameter drift between frameworks on these entries is expected and covered
by the benchmark training gates' tolerances.

ZERO_PRUNE gates the pass; any failure inside falls back to the unpruned
program (a missed optimization, never a miscompile).
"""

from typing import Optional

from tensorgrad.tensor import Derivative, Sum, Tensor, Zero
from tensorgrad.compiler.reverse import (
    _collect_derivatives,
    _contains_derivative,
    _rebuild,
    _supported,
    _sweep,
)

ZERO_PRUNE = True
MAX_TESTS = 128  # szfp evaluations per program (compile-time budget)


def _family_gradient(base: Tensor, d: Derivative) -> Optional[Tensor]:
    """The resolved gradient of `base` w.r.t. d.x via the reverse sweep, or
    None when the sweep does not support the base."""
    occ = _sweep(base, [d.x])
    if occ is None:
        return None
    post = _supported(base)
    assert post is not None
    from tensorgrad.tensor import Variable

    pieces = [
        occ[id(node)]
        for node in post
        if isinstance(node, Variable) and node == d.x and id(node) in occ
    ]
    if not pieces:
        return Zero(_symmetries=None, **d.x.shape)
    return pieces[0] if len(pieces) == 1 else Sum(pieces)


def prune_zero_gradients(tensors: tuple) -> tuple:
    """Replace provably-zero Derivative nodes with Zero. Runs pre-fold on
    the pristine trees; returns the input unchanged on any failure."""
    if not ZERO_PRUNE:
        return tensors
    try:
        return _prune(tensors)
    except Exception:
        return tensors


def _prune(tensors: tuple) -> tuple:
    from tensorgrad.compiler import szfp
    from tensorgrad.compiler.runtime import normalize_args

    # One shared simplify-args dict: the cross-gradient memo keeps 50+
    # sibling cotangent chains from re-simplifying shared subtrees.
    nargs = normalize_args()

    found: dict[int, Derivative] = {}
    seen: set[int] = set()
    for t in tensors:
        _collect_derivatives(t, found, seen)
    if not found:
        return tensors

    repl: dict[int, Tensor] = {}
    # Gather one normalized gradient per family, then zero-test them ALL in
    # ONE szfp evaluation: sibling cotangent chains share most of their
    # subgraphs, so batching turns 50+ separate lowerings (measured 21s on
    # minGPT) into one shared-DAG evaluation.
    fams: dict[tuple[int, int], list[Derivative]] = {}
    for d in found.values():
        if _contains_derivative(d.tensor):
            continue  # nested chains resolve innermost-first elsewhere
        fams.setdefault((id(d.tensor), id(d.x)), []).append(d)

    cands: list[tuple[list[Derivative], Tensor]] = []
    for key, ds in list(fams.items())[:MAX_TESTS]:
        g = _family_gradient(ds[0].tensor, ds[0])
        if g is None:
            continue
        # normalize resolves derivative signatures so szfp can lower; szfp
        # then decides zero-ness by exact evaluation (it needs a lowerable
        # graph, not a symbolic cancellation).
        cands.append((ds, g.simplify(nargs)))
    if not cands:
        return tensors

    trials = szfp._evaluate([g for _, g in cands], k=3, seed=0)
    for i, (ds, _) in enumerate(cands):
        if all(not tvals[i][2].any() for tvals in trials):
            for d in ds:
                repl[id(d)] = Zero(_symmetries=None, **d.x.shape).rename(**d.new_names)

    if not repl:
        return tensors
    out = []
    memo: dict[int, Tensor] = {}
    for t in tensors:
        try:
            out.append(_rebuild(t, repl, memo))
        except TypeError:
            out.append(t)
    return tuple(out)
