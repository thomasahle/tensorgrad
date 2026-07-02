# Audit notes — items for discussion

## 1. Kronecker/vec convention — RESOLVED (top = most significant)

Adopted: the top edge of a bundle is its most significant part (reading top-to-bottom =
reading the classical expression left-to-right). A⊗B keeps A on top, column-major vec
kept (row = fast = bottom port), matching the FFT chapter's existing top-wire-=-MSB
order. Changes: ▷ definition labels swapped (fast index now the bottom edge) + a
convention paragraph incl. the wrap-under rule; (520) A/B swapped in both diagrams;
(yyy) LHS outer factors swapped + RHS inner pairs swapped; Lyapunov chain (272)-(273)
diagrams flipped. Verified unchanged/correct under the new reading: (524), (xxx),
Kronecker-vector products, Khatri–Rao section (now literally correct — was silently
B∗A before), property rows (506)–(515), pure-diagram identities, stacking/direct-sum.
Deferred: figures/kron.pdf_tex factor order (TODO comment in Matrification section).

From a full correctness audit of the book (all chapters, 8 parallel reviews, cross-checked
against the Matrix Cookbook and verified numerically where possible). The mechanical and
clearly-wrong items have already been fixed directly in the chapters; this file collects
the findings that need an author decision. Ordered by importance.

## 1. Kronecker/vec convention clash (systemic)

The flattening triangle is defined (kronecker.tex ~17–28) with the *top* edge as the fast
index, and vec is column-major (~551–557). With these choices, the diagrammatic Kronecker
product drawn everywhere in the chapter (first factor on the *top* port) equals classical
**B ⊗ A**, not A ⊗ B — contradicting the block-matrix definition at ~156–166 where B's
index is fast.

Consequences:
- Pure-⊗ identities (506)–(515) are unaffected (order-symmetric).
- Every identity mixing vec and ⊗ silently means B⊗A. This is why the false
  `A ⊗ (B ⊕ C) = (A ⊗ B) ⊕ (A ⊗ C)` "diagram-checked", and it interacted with the
  (520) error (both now fixed at the formula level; diagrams marked with
  `% TODO(convention)` comments where they only match under the flipped convention).

Options:
1. **Redraw:** put the *first* Kronecker factor on the *bottom* (slow) port in all
   ⊗/∗/• diagrams. Keeps column-major vec and all classical formulas standard.
   Most invasive but cleanest.
2. **Redefine ▷:** make the bottom edge the fast index. Then the Kronecker diagrams are
   right as drawn, but the vec diagrams need X's edges swapped.
3. **Declare it:** add a remark that diagrammatic A⊗B corresponds to classical B⊗A
   (like Kolda–Bader's reverse-lexicographic convention) and keep everything as is.
   Cheapest, but readers copying formulas out of the book will get burned.

Note this also touches the new Tensor Trains chapter only lightly (it uses the triangle
just for bond-pair bundling, where order doesn't matter).

Related decision: the general "preserve covariance/contravariance" rule (edges exiting
left on the LHS should exit left on the RHS) should be stated explicitly as a notation
principle in the intro, and used as the tie-breaker for whichever option is chosen.

## 2. (Anti)symmetrizer normalization — RESOLVED (Penrose convention)

Done: squiggle/thick bar are now normalized idempotent projectors (1/k!·Σ), per Penrose.
Intro shows ½(A+Aᵀ) as "the symmetric part", idempotence holds exactly, determinant
identities are factor-free (det = closed stack, det(I) = Tr(P) = 1), and the free-standing
bar-tensor is ε/√(n!) (Cvitanović birdtrack normalization, cited) so that closed diagrams
evaluate identically bar-by-bar or through the projector. statistics.tex's mixed
third-moment coefficient updated (½ → 3) for the average convention.

<details>Original discussion:</details>

## OLD 2. (Anti)symmetrizer normalization (intro ↔ determinant chapters)

The intro defines symmetrization as the plain sum over permutations (no 1/n!). The
determinant chapter's thick bar is likewise raw Levi-Civita. Under this convention several
identities were off by n! (now fixed by inserting explicit n! / 1/n! factors):
intro's idempotence claim, determinant's "curious property", and the det product rule.

Decision: keep the sum convention with explicit factorials (current state), or switch the
squiggle/bar to normalized operators (1/n! · Σ)? Normalized makes idempotence and the
det-diagram identities factor-free, but breaks the clean `A + Aᵀ = squiggle(A)` example
and the `closed stack = n!·det(A)` derivation. If you switch, `\detstack` should absorb
the 1/n! so rows (19)/(21) stay uniform.

## 3. tensorgrad.tex — RESOLVED (trimmed to design overview)

Done: rewritten as a 121-line chapter (Representation / Canonicalization /
Differentiation / Simplification and Evaluation) with all identifiers verified against
the current source and one executable example (runs, simplifies to 2Xᵀ(XW−Y)).
Side-note: the README's own example has bugs (missing imports, sp. vs tg.) — fix upstream.

<details>Original discussion:</details>

## OLD 3. tensorgrad.tex documents an API that no longer exists

Every code listing in the chapter references identifiers absent from the current
`tensorgrad/` source: `canonical_edge_names`, `t.canon`, `TensorDict`,
`_compute_canonical`, `edge_equivalences`, `original_edges`, class `Copy` (now `Delta`,
tensor.py:739). The chapter also reads as an internal design diary (first-person musings,
one-sentence sections). Options: rewrite against the current classes (Tensor, Variable,
Rename, Delta, Zero, FunctionSignature, Function, Derivative, Product, Sum), or cut the
chapter down to a short "design of tensorgrad" overview and drop the listings.

## 4. Unfinished/stub content (decide: finish, cut, or mark as draft)

- ~~statistics.tex 1–92 notation block~~ — RESOLVED: author picked tilde. Draft block
  deleted, $\tilde x := x - m$ defined once, all 48 ⊖ occurrences swept (two judgment
  sites: Var[Ax]'s definitional minus kept plain; scalar section uses $(y - m_y)$).
- **advanced_derivatives.tex**: mostly empty sections (matrix norms, structured matrices,
  determinant forms, eigenvalues); section title "From Stack Exchange"; zero diagrams in
  a diagram book. The trace identities would shine as diagrams. Merge into
  simple_derivatives or finish?
- **determinant.tex**: dangling "hyper determinants by $\dots$" sentence (~106); table
  rows (18), (20), (22)–(24) have `\dots` where diagrams should be; "Inverses" section is
  a note-to-self. The audit sketched the answer: adj(A)ᵀ = (n−1 copies of A between two
  ε's)/(n−1)!, which also gives Jacobi's formula diagrammatically — worth writing.
- **ml.tex**: pure stub (5 empty sections). Least-squares/Hessian material exists in other
  chapters and could be moved here. The tensor-sketch sentence misdescribes the method
  (it is CountSketch-based, not a random rank-1 projection) — rewrite when filling in;
  the new Tensor Trains chapter's sketching section could be cross-referenced or moved.
- **kronecker.tex ~1623**: empty `\section{Derivatives}`.
- **appendix.tex**: 3 lines, hardcoded "equation 524 or 571" (Matrix Cookbook numbers)
  with no `\ref`; promised proofs absent.
- **decompositions.tex ~213–216**: meta-commentary ("Would this be a nice thing to show?
  Maybe too messy?") — decide and delete. Placeholder tags `(xxx)`/`(yyy)` in
  kronecker.tex similarly need real numbers.

## 4b. Residuals from the fix pass

- Three diagrams in kronecker.tex are marked `% TODO(convention)` — their classical
  formulas are now correct, but the adjacent diagrams only match under the flipped
  convention of item 1. They should be redrawn once item 1 is decided.
- functions.tex eq. (128) prose now uses X, but the external `figures/cos.pdf` still
  shows lowercase x internally.
- `figures/linearityOfExpectation` still labels its result node `M₃`; the text now
  explains it is the raw moment E[X^⊗3], but regenerating the figure with the label
  `E[X^{⊗3}]` would be cleaner.

## 5. Figure-level issues — RESOLVED (minimal path, per author)

No TikZ redraws. The linearityOfExpectation passage now assumes X has mean zero, which
makes the figure's M₃ label exactly correct (no relabeling, no X^⊗3 notation). The
Stein passage already assumed zero mean, so E[XX] = Cov holds as printed; the hidden
draft annotation was deleted from steins.svg (source-only change, exported look intact).
Also added: the "delete the node" rule as a short observation with one figure at the top
of simple_derivatives (framed as an observation, not a theorem, and explicitly not the
chain rule).

## OLD 5. Figure-level issues (source assets)

- **figures/steins.svg** contains hidden draft layers incl. the self-annotation "Not
  quite true. The E[XX] should be Cov" next to a nonzero-mean panel. The cropped
  steins.pdf_tex renders only the safe panel, so the printed book is fine — but the flag
  suggests an unresolved generalization worth revisiting; at minimum clean the svg.
- **figures/linearityOfExpectation**: node labeled `M$_3$` but the text (now fixed) calls
  it the raw third moment E[x^⊗3]; consider relabeling the figure to `E[x^{⊗3}]`.
- The external figures in functions.tex (~553–568: chain_rule_broadcast, multi_input
  _functions, other_functions, cos) were not auditable from source; other_functions.pdf
  should be checked against the corrected higher-determinant-derivative statement.

## 6. Smaller judgment calls

- **functions.tex ~547**: "Interested readers may refer to his blog post" — no cite/URL
  (whose blog? Needs the reference).
- **functions.tex label typography**: edge labels alternate between text (`{d}`) and math
  (`{$n$}`) mode — renders upright vs italic. Pick one (math mode suggested) and sweep.
- **simple_derivatives.tex eq (91)**: inline "edges" written as literal hyphens
  (`$-(X^r)^T-a$`) typeset as minus signs; convert to `\vecmatvec` or drawn edges.
- **intro.tex order-0 copy tensor** (~404): the ambiguity could be resolved concretely
  (•–• = n, so the 0-legged spider must be n for consistency) instead of "may or may not".
- **intro.tex eigendecomposition** (~850–910): silently assumes diagonalizability;
  a footnote would cover the general case.
- **statistics.tex**: `M` vs `M₂` used interchangeably; state `M ≡ M₂` once.
- **tensor_algos.tex ~38,49**: the `dpconv` bounds (O(2ⁿn³) exact / O*(2ⁿ/ε)
  (1+ε)-approx) could not be verified against the cited paper — double-check exponents.
- **decompositions.tex ~210**: "led to the best algorithms by Josh Alman and others" —
  the ω record comes from the laser method on CW tensors, not small ⟨n,m,p⟩
  factorizations; loosen the causal claim.
- **Exercise 1 in intro** (~1098): n overloaded (count of matrices and their dimension).
- **Hardcoded Matrix-Cookbook tags**: table numbers are literal text; fine while numbering
  is frozen, but fragile. (Also: our (382) row generalizes MC's; now annotated.)

## 7. Structural suggestions from comparing with Rakhshan–Rabusseau (arXiv:2605.16610)

Already done: Tensor Trains/MPS chapter (incl. min-cut rank bound, TT-SVD, QTT, and the
recursive-sketching section from Ahle et al. SODA'20). Still worth considering:
- A one-page notation cheat-sheet after the title page.
- Half-colored nodes for orthogonal matrices (makes UᵀU = I visually immediate; would
  also serve the TT-SVD orthogonal-core diagrams).
- Stating the "derivative of a network w.r.t. a core = delete the node (sum over
  occurrences)" principle as a boxed theorem early in simple_derivatives.
- A "probability distributions as tensors" section (marginals = contract with 𝟙,
  Born machines) — natural fit for statistics.tex or ml.tex.
- Cite arXiv:2605.16610 in the intro's related-work paragraph (entry `rakhshan_cookbook`
  already in references.bib) and position the two books relative to each other.
