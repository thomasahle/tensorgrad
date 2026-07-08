# Changelog

## 0.3.0 (unreleased)

The ahead-of-time compiler release. Symbolic gradients (and Hessians) now
lower to fused, straight-line PyTorch programs — no autograd tape.

### Added
- **Ahead-of-time compiler** (`tensorgrad.compiler`): `compile_to_callable`
  turns a loss and its raw symbolic gradients into one fused straight-line
  torch program. `tg.compile` / `tg.grad` with structured (pytree) outputs.
- **Joint reverse-mode resolution**: a whole family of gradients of one loss
  is resolved with a single shared cotangent sweep; the pass restores
  reverse-mode ordering to gradients that arrive forward-mode-shaped.
- **Fused kernels as a `FusedCell` library** (`tensorgrad/compiler/cells.py`):
  `F.sdpa` (flash-attention), `F.layer_norm` (native_layer_norm),
  `F.gelu`, plus softplus/sigmoid stabilization and an AdamW cell — each with
  forward *and* derived reverse backward.
- **Exact verification**: modular Schwartz–Zippel fingerprinting checks every
  optimization rewrite, and a value-numbering pass merges algebraically-equal
  subgraphs behind a fresh-seed refusal gate.
- `Expectation.gaussian` classmethod; convolutions expressed as `Affine` /
  `window`.
- New worked examples: minGPT (trained with `torch.set_grad_enabled(False)`),
  VAE (sample-free, derived block-Newton), DDPM diffusion, Gaussian process,
  Newton optimization.

### Changed
- Per-type operations (simplify, grad, depends_on, substitute) are organized
  as `functools.singledispatch` modules; the node classes hold only intrinsic
  behavior.
- `tg.compile` drops the `inputs=` declaration — edge names carry the
  structure.
- Retired the pre-compiler backends (`to_numpy`, `to_numpy_optimized`,
  `to_pytorch`) in favor of the compiler.
- The whole package is `mypy`-clean and type-checked in CI.

### Fixed
- Packaging: the wheel now ships only `tensorgrad` (0.2.1 also installed
  `tests`, `examples`, `server`, `paper`, `benchmarks`, `scripts` as
  importable top-level packages).
