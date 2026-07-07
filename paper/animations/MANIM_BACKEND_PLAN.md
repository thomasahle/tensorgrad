# Plan: Manim backend + object tracking for automated derivation videos

Goal: `tensor → derivation video draft` with the same visual quality as the
hand-built videos in `paper/animations/`, by reusing the book_layout engine's
geometry and adding (1) a renderer-neutral drawing layer, (2) step-to-step
object correspondence, (3) layout stickiness, and (4) an animation compiler
that encodes the SKILL.md grammar. Target: "expression in, lint-clean draft
video out; 2–3 review rounds instead of 15."

## Assets already in place

- **`tensorgrad/extras/book_layout.py`** — the layout engine. `BookLayout`
  is renderer-neutral data (LNode/LWire/derivs/boxes), calibrated over the
  whole `examples/main.py` corpus (38 regression tests). All quality work
  (side consistency, bracket sizing, loop separation, wire-breaking,
  ∂X tags) lives at the layout level, so every backend inherits it.
- **`paper/animations/SKILL.md`** — the animation grammar distilled from the
  hand-made videos: which transforms are "safe", halo emphasis, spawn/despawn
  idioms, object-persistence discipline. This is the spec the compiler
  targets; the videos are its reference output.
- **`tensorgrad/structure.py`** — per-node Merkle-style fingerprints
  (`canon_info`), `structural_fingerprint`, `is_isomorphic`,
  `isomorphisms(t, other)`. Reusable for cross-step atom matching.
- **Immutability + structural sharing** — simplify rebuilds only the rewrite
  spine; untouched subtrees are the *same Python objects* across steps.
- **Step chains** — `examples_main.py::derivation_steps` (and
  `imgtools.save_steps`) already produce the `expr = step₁ = … ` sequences;
  the derivations doc is the test corpus.
- **Self-diagnostics** — bracket-cut warnings; same idea extends to
  animation lint (below).

## Architecture (5 components, in dependency order)

### 1. Drawing-op IR (`plan/draw` split inside book_layout)

Today `_emit_layout/_emit_boxes/_emit_derivs` interleave *drawing decisions*
(which endpoints, which curve, which label, where) with *TikZ syntax*.
Split them:

```
plan_drawing(layout) -> list[DrawOp]      # all decisions, no syntax
draw_tikz(ops)       -> str               # current output, byte-compatible
draw_manim(ops)      -> manim.VGroup      # new backend (component 2)
```

DrawOp schema (dataclasses; all coordinates in layout units):

- `GlyphOp(node_id, x, y, tex, kind, axis, rotated)` — text glyphs, signs,
  parens (paren height in ex), copydots (as `DotOp`).
- `WireOp(a: EndRef, b: EndRef, curve, style, label, label_pos)` where
  `EndRef = (node_id, role)` with role ∈ `center | paren_l | paren_r |
  north | angle(θ)` — endpoints stay SYMBOLIC. TikZ resolves them to node
  anchors (exactly as today, preserving TeX text metrics); manim resolves
  them against MathTex bounding boxes at build time. `curve` is one of
  `Straight | OutIn(out°, in°, looseness) | Bezier(c1, c2) | Bend(dir, deg)`.
  TikZ maps these 1:1 to its syntax; manim converts `OutIn/Bend` to cubic
  Béziers via the TikZ formula (controls at `0.3915·looseness·dist` along
  the exit/entry angles).
- `StubOp(node_id, direction, dx, dy, label, second_label)` — free-edge
  stubs incl. the `inner::outer` rename double labels.
- `LoopOp(node_id, min_dist, open_down)`, `EllipseOp(fit_points, sep,
  wrt_tag, tag_at_dot)`, `BracketPairOp(fit_points, sep, label)` (E-boxes),
  `ArrowheadOp`, `BareWireOp`, `RingOp`.

Acceptance: the regenerated `examples_main.tex` is **identical (or
whitespace-equivalent) before/after**; 38 tests green. This is a pure
refactor — no visual change permitted.

### 2. Static manim emitter (`tensorgrad/extras/book_manim.py`)

`to_book_manim(tensor, **layout_kwargs) -> VGroup` = layout → plan_drawing →
draw_manim. Key differences from TikZ handled here:

- **Text metrics**: manim measures real MathTex bounding boxes, so EndRef
  resolution uses actual glyph borders (better than our CHAR_W estimates).
- **Fit shapes**: EllipseOp/BracketPairOp compute their own bbox from
  fit_points + resolved mobject boxes (TikZ's `fit=` semantics,
  reimplemented once).
- Style table maps SKILL.md conventions: wire stroke widths, dotted
  application arrows, copydot radius, colors (monochrome default; the
  compiler adds emphasis colors).
- The old `extras/diagram_manim.py` (spring-layout era) is superseded;
  delete or leave with a deprecation note.

Acceptance: side-by-side PNG parity harness — for each corpus example,
render TikZ and manim stills; eyeball + pixel-diff structural landmarks.
New tests: mobject counts match op counts; no NaN coordinates.

### 3. Correspondence matcher (`tensorgrad/extras/correspond.py`)

`match(t_before, t_after) -> Matching` where Matching maps *atom
occurrences* (the same units as `extract_graph` atoms) with tags:

1. **Identity pass (exact, free)**: walk both trees; subtrees that are the
   same Python object (structural sharing) match exactly, including all
   their atoms. Handles most of every step.
2. **Copy detection**: one shared source subtree appearing N times in
   t_after (distribution) → one-to-many matches tagged `copy` →
   `TransformFromCopy`. N-to-one (factoring/merging) → tagged `merge`.
3. **Fingerprint pass (rewrite site)**: remaining atoms matched greedily by
   `canon_info` per-node fingerprint + neighborhood (WL-style, reusing
   structure.py). Ambiguity only among symmetric copies — visually
   interchangeable, so any consistent choice is fine (but make it
   deterministic: sort by size_key).
4. **Births/deaths**: unmatched-new → `birth` (grow in), unmatched-old →
   `death` (fade toward the rule site = the smallest changed region).

Upgrade path (only if needed): `simplify(trace=True)` attaching exact
correspondences per rule in the `simplify.py` rule catalog. Do NOT start
here — fingerprint matching is expected to cover the corpus.

Acceptance: unit tests on hand-checked pairs (notebook0 distribution, main
product rule, factor-out-of-expectation in main21); every atom in both
expressions is classified (matched/copy/merge/birth/death) — nothing
silently dropped (mirror the wire-conservation audit).

### 4. Layout stickiness (temporal coherence)

`layout_any(tensor, prev=(BookLayout, Matching))`. At each choice point,
break near-ties toward minimizing total movement of matched atoms:

- `_choose_spine`: score bonus proportional to agreement with the matched
  atoms' previous left-to-right order (never override a large score gap —
  correctness of the book grammar beats coherence).
- Component order, `solo_side`, stack row order: same tie-break rule.
- Optionally pass through free-edge side assignments (left/right kwargs)
  from the previous step's resolved sides.

Metric: `sum(|pos_k+1(a) − pos_k(match(a))|)` over matched atoms, exposed
by a debug helper so tests can assert coherence improved on the corpus.
Acceptance: static doc output unchanged when `prev=None` (default); with
prev, the notebook0 chain's shared factors keep their order across steps.

### 5. Animation compiler (`paper/animations/auto/compile.py`)

`compile_steps(steps: list[Tensor]) -> manim.Scene`:

- For consecutive steps: layout with stickiness, build both mobject sets
  via component 2 keyed by atom occurrence, then per Matching tag:
  `matched → Transform` (or `.animate.move_to` when the glyph is
  unchanged), `copy → TransformFromCopy`, `merge → Transform many→one`,
  `death → FadeOut` (shrink toward rule site), `birth → FadeIn/GrowFromCenter`.
  Wires: transform path-to-path between the WireOps whose endpoints matched;
  else fade. Signs/parens/brackets: match by position in the term structure.
- Pacing/emphasis per SKILL.md: rule-site halo before the transform, hold
  frames after, `=`-row breathing room.
- **Animation lint** (the "warnings" idea, applied to video): report
  transforms that move objects across more than X units, glyph swaps where
  tex changed under a `Transform` (should be `ReplacementTransform` +
  fade), any unclassified atom. Lint output = the review checklist.

Acceptance: pilot videos (below) pass lint and read correctly frame-by-
frame at 1×.

## Milestones

| # | Deliverable | Verification |
|---|-------------|--------------|
| M1 | DrawOp IR + `draw_tikz`; emission refactored | examples doc byte/visually identical; 38 tests |
| M2 | `book_manim.py` static stills | TikZ/manim parity sheet over the corpus |
| M3 | `correspond.py` matcher | unit tests on 3 hand-checked step pairs; total-classification audit |
| M4 | Layout stickiness | coherence metric drops on notebook0/main chains; static output unchanged without `prev` |
| M5 | Animation compiler + lint | pilot: notebook0 distribution step (copies + sign flip) |
| M6 | Full-chain pilots | `main` (product rule), main12 (∂X quotient rule); review rounds ≤3 |

Order matters: M1 is the only risky refactor (mitigated by byte-diff); M3
is independent of M1/M2 and can be built in parallel if convenient.

## Risks / mitigations

- **M1 regression risk**: the TikZ renderer is heavily calibrated. Mitigate
  with mechanical translation (each `lines.append` → one op + interpreter
  case), byte-diff of `examples_main.tex`, and the 38-test suite.
- **Metric mismatch (TikZ vs manim text sizes)**: EndRefs resolve per
  backend, so wires attach correctly in both; only inter-glyph *spacing*
  (CHAR_W estimates) is shared — acceptable, it's already tuned.
- **Symmetric-copy ambiguity in matching**: deterministic tie-break; the
  lint flags large moves so a human catches genuinely wrong pairings.
- **Scope creep**: no crossing-minimizing router, no interactive editor,
  no `δ` notation switch — out of scope. The raw-Taylor rows (main18/20
  step 1) stay at the documented router limit in video too (the compiler
  may simply skip un-drawable steps like the doc does).

## Open questions (for Thomas)

1. Video style: keep the monochrome book look for auto-drafts, or adopt the
   hand-videos' color emphasis from the start?
2. Should `save_steps` become the shared entry point (doc + video from the
   same chain), retiring the spring-layout `to_tikz` path?
3. Where do auto-drafts land — `paper/animations/auto/` per-example, or a
   build directory outside git?
