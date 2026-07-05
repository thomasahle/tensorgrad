# The tensorgrad compiler

tensorgrad expressions are symbolic: named edges, exact algebra, derivatives
as rewrites. The compiler turns them into fast straight-line PyTorch programs
— ahead of time, with the calculus done once instead of every step.

```python
from tensorgrad.compiler import compile_to_callable

step = compile_to_callable(loss, *[loss.grad(p) for p in params])
loss_val, *grad_vals = step({x: x_t, y: y_t, **weights}, {batch: 64, ...})
```

That is the whole API. Gradients go in raw (`Derivative` nodes are resolved
internally), inputs and outputs are named torch tensors, and the returned
callable specializes lazily per shape signature: the first call with a given
set of sizes generates and `exec`s a straight-line torch function; later
calls are a dict lookup. `torch_compile=True` hands the generated code to
Inductor (it traces as one graph — see *fullgraph* below).

`examples/mingpt.py` is the showcase: a 3-block GPT trained on the sorting
task with `torch.set_grad_enabled(False)` for the entire run — PyTorch is
used purely as a tensor library.

## Pipeline

```
symbolic Tensor(s)
  │  normalize          resolve Derivative nodes & derivative signatures
  │                     (simplify_for_compile preset, one shared memo)
  ▼
  lower.py              one hash-consed einsum-DAG for ALL outputs;
  │                     Deltas become shared wires via union-find,
  │                     structured tensors become affine constraint rows
  ▼
  adjoint.py            reverse-mode ordering: collapse forward-mode
  │                     Jacobian chains into cotangent accumulation
  ▼
  factor.py             un-distribute / distribute / flatten by
  │                     contraction cost (memory-bounded)
  ▼
  stabilize.py          recognize exp-ratio / log patterns → fused
  │                     softmax, log_softmax, logsumexp, tanh
  ▼
  layout.py             physical axis order chosen globally (named
  │                     edges never fixed one) → zero permute-copies
  ▼
  codegen_torch.py      cells: mm/bmm/addmm/index_select/index_add_,
  │                     einsum only where no cell fits; spec-time
  │                     strides; liveness `del`s; per-shape source
  ▼
  runtime.py            specialization cache, dtype promotion,
                        optional torch.compile(fullgraph=True)
```

Supporting machinery: `canon.py` (compositional isomorphism-invariant
hashing — used for hash-consing and fast equality), `szfp.py` (exact mod-p
Schwartz–Zippel fingerprints; every optimization rewrite class is verified
semantically), `affine.py` (the structured-sparsity language, below).

## Design principles

**No hand-written derivatives for composites.** Only true leaves (`exp`,
`log`, `erf`, `inverse`, …) carry derivative rules. Softmax, layernorm,
gelu written from primitives get fast, stable backward passes *derived* —
the researcher benchmark (`tests/compiler/test_researcher_benchmark.py`)
enforces this, up to the cross-entropy∘softmax Hessian, which simplifies to
the two-term `diag(s) − ssᵀ` (y-free, given `y.with_constraint("simplex")`).

**Structure is born named, never inferred.** `Delta`, `Convolution`,
`Reshape`, gathers are indicators of integer-affine index relations
(`affine.py`): equations, not index data. Equations compose symbolically
(conv∘conv is Gaussian elimination on rows) and compile to `as_strided`
views — zero copies, zero FLOPs. The dense indicator is always a correct
fallback: structure is a fast path, never a correctness requirement.

**The compiler owns physical layout.** Named edges mean no tensor ever had
an axis order; `layout.py` assigns one globally, voting from the cells that
care (matmuls, reductions), so remaining permutes are free views. BLAS
absorbs one transpose per operand, so only genuine copies count as cost.

**Deterministic canonicalization; no search.** Equality is isomorphism,
decided by content-derived fingerprints (VF2 only for rare hash-equal
pairs). Rewrites are cost-greedy and order-independent. Where local scoring
provably cannot work — reverse-mode ordering improves nothing until a whole
chain collapses — the pass is scoring-free and structural (`adjoint.py`).

**Reverse mode is an IR transform.** Symbolic differentiation naturally
yields forward-mode-shaped chains that drag a parameter's axes through
every layer above it. `adjoint.py` restores reverse-mode order: one
cotangent per transient node, accumulated over all consumers, pushed down
one boundary at a time. Measured on a 12-block GPT-2-dims graph: 11.26 EB
of planned intermediates → 19.8 GB (largest node 23 MB).

## Results (CPU, torch as the baseline's backend too)

| workload | vs torch autograd |
| --- | --- |
| layernorm grad (primitives-defined) | **2.8× faster** |
| gelu grad | **1.4× faster** |
| softmax grad | parity |
| conv1d forward+backward | **2.4–4.8× faster** |
| CE∘softmax Hessian-vector product | parity with double-backward |
| MLP loss+grads (Wine) | ~1.6× slower eager, parity compiled |
| 3-block GPT step (loss + 53 grads) | dispatch-bound; consolidation in progress |

Gradients are verified against `torch.autograd` (typically ≤1e-6 relative)
and against the `evaluate()` interpreter; `szfp` pins every rewrite exactly.
One amusing hazard for checkers: attention key-bias gradients are *exactly
zero* by softmax shift-invariance — the symbolic system knows a theorem the
numeric comparison needs an `atol` floor to survive.

## Escape hatches and flags

- `compile_to_callable(..., simplify=False)` — compile a raw structure
  verbatim (developer/testing use).
- Module flags in `codegen_torch.py` / `adjoint.py` / `layout.py`
  (`LAYOUT_ASSIGN`, `MATMUL_CELLS`, `STATIC_STRIDES`, `EMIT_DEL`, …) turn
  individual stages off; equivalence tests use them to check fast paths
  against reference paths.
- `fn._source` on any specialization holds the generated Python for
  inspection.

## Frontiers

- **Consolidation** (in progress): the deep-model monolith is memory-compact
  but dispatch-heavy (~3k small ops); horizontal/vertical fusion toward the
  proven per-module-VJP op count.
- **GPU**: Triton emission for fused affine regions; `F.conv2d`/SDPA
  peepholes become load-bearing on GPU.
- **Tech-mapping phase 2**: cut-based DP covering over a cost-characterized
  cell library (the current cell selection is greedy).
