---
name: tensor-diagram-videos
description: How to make publication-quality manim videos of tensor-diagram
  derivations for the Tensor Cookbook — the animation grammar, the book's
  notation rules, the visual style, the review workflow, and the manim
  pitfalls, all learned across three videos (trace_delete, kronecker_trace,
  softmax_jacobian) and ~40 review rounds.
---

# Making tensor-diagram derivation videos

The product: a 15–40s silent 1080p60 video in which a Matrix-Cookbook-style
identity is *derived*, not asserted. The standard was set by three videos in
this folder: `trace_delete.mp4`, `kronecker_trace.mp4`, `softmax_jacobian.mp4`,
all rendered from scenes in `advanced_rules.py`.

The one-sentence philosophy: **the animation is only a proof if every frame
is a well-formed diagram and every mobject has an ancestry.** Nothing may
appear, vanish, flip, or morph without a semantic reason.

## The book's notation (non-negotiable)

Check `paper/chapters/functions.tex` before inventing anything. Known rules:

- **Function application**: the function is a plain text node (no circle);
  the *input* edge carries an arrow pointing INTO the function. Solid arrow
  for vector→vector functions (softmax), **densely dotted** for elementwise
  ones (exp, pow_k) — and for elementwise functions the data wire stays on
  the argument (`exp ←··· z ——`, the free wire belongs to z).
- **Sum** = wire terminated by a copy-dot. **diag(v)** = 3-edged copy-dot
  with v hanging off. **1/x** = `pow₋₁` applied by arrow — not a fraction —
  though converting pow↔fraction *on screen* is a good beat (go full circle:
  open with the fraction, work in pow form, return to fractions at the end).
- No brackets when arrows already disambiguate. Scalar-argument functions
  (pow on a sum) still get their application arrow.
- A matrix is a node with two edges; vectors have one (a and b in aᵀXb are
  closed by the trace — count edges before drawing!).
- **Covariance**: a named edge exits the same side in every term and every
  frame. Two free edges never cross incidentally — after the twist
  convention, a crossing MEANS a swap. Rotating a node 180° IS transposition
  (so a turned glyph legitimately reads as Aᵀ — relabel by spinning it
  upright while the ᵀ fades in).
- Claim only what is derived (the softmax video says "Jacobian of softmax",
  not "Hessian of cross-entropy", because the CE connection is never shown).

## Visual style

- White background, `config.frame_height = 5.6` (set in `__init__` before
  `super().__init__`), same pixels → everything ~40% bigger. Title top
  (`to_edge(UP, buff≈0.4)`), gray caption bottom (`Tex … scale(0.62)`,
  `to_edge(DOWN, buff=0.18)`), one clause per beat — the captions ARE the
  voiceover script.
- Semantic color, constant per object across formula AND diagram:
  variables `#C03B2B` (red), function names `#1F6FB2` (blue), recognized
  results `#188A54` (green), derivative apparatus `#B07000` (amber),
  structural wires/dots/signs black. The amber thread is pedagogy: whisker →
  dangling stub → final edge are visibly the same object.
- **Uniform glyph sizes.** No shrunken denominators or mini-copies; retune
  layout instead.
- Glyph halos (`set_background_stroke(color=WHITE, width=6)`), never
  `BackgroundRectangle` masks — rectangles clip *other* wires passing behind
  (visible the moment the scene moves). A curve's clearance around its own
  riding label must be a real parameter gap in the curve, sized from the
  local radius.
- Every node gets clearance — edges stop short of glyph ink, including
  operator glyphs like flatten triangles (stubs may start *under* a
  white-filled node so the fill covers the join).
- Arrows must clear both endpoints: tips short of subscripts (`pow₋₂`),
  tails outside the source glyph. When in doubt, widen the gap between the
  nodes.
- Finish: `Indicate` when a result lands, `Circumscribe` the final formula,
  settle the finished diagram up under its equation, hold ≥2.5s.

## The animation grammar

1. **Every frame is a valid diagram.** Right edge count per tensor, no
   transient crossings, invariants hold mid-animation, not just at
   endpoints (e.g. delayed growth so dangling edges never cross).
2. **Glyphs keep identity.** A symbol persists (and may glide, with
   `path_arc` to hop over others), appears, or disappears — it never smears
   into a different symbol. The one sanctioned morph is a semantic renaming
   (exp absorbing its normalizer to *become* softmax). Titles: build the
   formula in sync with the diagram (X → AXB → Tr(AXB) → ∂·/∂z); keep it
   centered early and slide left as `= …` grows; hand-assemble fractions so
   the ∂ exists as its own animatable glyph; early title stages sit at the
   equation midline, not the numerator slot.
3. **Count-match every transform.** One wire → one wire. If topology
   changes, split the target at the join (de Casteljau at a bezier apex;
   closed loops split into quadrants) so correspondence is 1:1. A mismatch
   makes manim silently duplicate → ghost arcs.
4. **Direction-match: anchored end first.** Transforms map start→start. A
   target curve parameterized from its free end makes the source pivot/flip
   about the wrong end (this bug shipped four times: trace closure, A–X
   edge, j1f, jw2). Build every replacement wire starting from the end that
   stays attached.
5. **`Transform` keeps the SOURCE mobject alive** (wearing the target's
   shape); `ReplacementTransform` swaps the target in. Corollaries: never
   FadeOut/group/animate the *target* of a plain Transform (ghost loops,
   stray minus signs, unmoved edges — this bug shipped five times); after a
   Transform, later code must reference the source name.
6. **`Transform` preserves z-order; `ReplacementTransform` re-layers to
   top.** Use Transform when a morphing wire must stay under labels.
7. **Copies spawn from their parent.** Product rule, chain rule: `.copy()`
   superimposed on the original, then transform the copies out to the new
   rows/groups. Factor swaps arc over each other (`animate(path_arc=…)`).
8. **The whisker is the promised edge.** One whisker per derivative loop
   (per new edge); it survives distribution (loop→loop, dot→dot,
   whisker→whisker at hand-offs) and finally *becomes* the new edge. New
   edges are born already pointing in their final direction and never
   reverse (mirror sub-layouts if needed to keep all j's on one side).
9. **Amber circle = pending derivative; evaluation spends it.** Circling a
   function is applying ∂; when you evaluate (exp′=exp, pow₋₁′=−pow₋₂) the
   circle fades — and the relabel is the semantic event (the minus is BORN
   at pow₋₁′=−pow₋₂: keep ONE sign object centered between terms and flip
   it + → − at that moment). A chain-rule link through a scalar carries no
   d-edge at all.
10. **Dots terminate their edges.** A sliding sum-dot drags its wire
    (shorten in step). Dot-absorption is articulated: slide → merge →
    collapse, with the index-level receipt (Σₖδⱼₖ…) as the caption.
11. **Static↔tracker hand-offs must be seamless both ways**: animate the
    outgoing static to exactly the incoming `always_redraw`'s initial
    state (labels glide with relaxing wires; freeze redraws to statics
    before a group shift).
12. **Bundles are offset curves of ONE center path** (foot-curl → dome →
    foot-curl), strands at ±offset along numeric normals — guaranteed
    parallel, and the top strand takes the *inside* of every turn. Draw the
    pair at zero gap until the bundling operator is resolved; the gap
    opening IS the resolution. Deforming closed curves = one parametric
    curve with keyframed shape parameters (never piecewise arcs), labels
    riding a fixed parameter point through a real gap.
13. **More steps beat magic.** Rewrite big jumps as chains of small ones,
    each with a caption: quotient → product rule with explicit factors;
    pow↔fraction conversions; collect terms onto one line (swapping factors
    so covariance holds) *before* the final recognition; every `=` in the
    classical derivation gets a diagram move, and the object mapping in
    each move is reasoned (z's persist, arrows persist, machinery is
    absorbed into the name it justifies, split wires turn rather than
    respawn).

## Workflow

1. Read the relevant chapter first; the book may already contain the exact
   decomposition/figure (softmax) or pose the identity as an exercise.
2. Storyboard the beats; the user may supply frame-by-frame sketches —
   follow them panel by panel.
3. Build the scene in `advanced_rules.py` (helpers there: `glyph` halo
   text, `farrow` dotted/solid application arrows, `wire`, `cdot`,
   `dcircle`, `dloop` (whisker options), `node`, caption closure,
   `smgroup`-style builders). Fixed coordinate slots per beat; builders
   return dicts of named parts so transforms can be per-piece.
4. Iterate at `-ql`, extracting frames at beat boundaries:
   `ffmpeg -ss T -i video.mp4 -frames:v 1 out.png` (get duration via
   `ffprobe`), and Read them. Check *mid-transition* frames, not just
   settled ones.
5. **Final review at 1080p (`-qh`) — delivery resolution.** 480p
   anti-aliasing hides exactly the shippable-defect class: 1–2px gaps,
   wires grazing glyphs, mask edges. Use PIL crops for pixel-level checks
   of junctions.
6. Ship: copy .py + .mp4 to `paper/animations/`, commit with explicit
   pathspec, push. One commit per review round.
7. Expect ~5–15 user review rounds per video. Each note usually
   generalizes: fix the instance, then apply the principle everywhere
   (e.g. "circle is spent" came from one pow note; uniform font sizes from
   one screenshot).

## Tooling pitfalls

- Patch scripts: `assert old in s` for EVERY replacement; a failed assert
  before `write` means the file is untouched and the subsequent render
  shows the OLD video — check which. Region-scope all `s.index` anchors to
  the class (`base = s.index("class Foo"); s.index(anchor, base)`) — a
  global `index` can match an earlier scene and splice duplicated chunks
  into the file.
- Multiple scenes share `advanced_rules.py`; anchor text ("# ---- title
  ----", "self.cap = None") repeats across scenes.
- The scratchpad can be wiped between sessions; the repo copy is the
  source of truth — restore, then re-apply.
- manim: `MathTex(a, b, c)` part-splitting enables per-glyph color/animate;
  `set_background_stroke` for halos; `FadeOut(m, target_position=…)` fades
  while translating (absorption); `FadeIn(m, target_position=…)` arrives
  from there; `Rotate(m, about_point=…)` for in-place glyph spins;
  `DashedLine(...).add_tip(...)` for dotted arrows; `always_redraw`
  closures over `ValueTracker`s for deforming geometry (keyframe param
  arrays + `np.interp`); `Date`-like nondeterminism absent but LaTeX
  compile cache makes re-renders fast.
- Render one scene: `python -m manim render -qh --disable_caching
  advanced_rules.py SceneName` (venv: `.venv/bin/python`; latexmk not
  needed — manim compiles its own tex).
