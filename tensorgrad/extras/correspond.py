"""Step-to-step object correspondence for animating derivations.

``match(before, after)`` classifies every drawable atom occurrence in two
consecutive derivation steps as moved / copy / merge / birth / death --
the input an animation compiler needs to emit Transform /
TransformFromCopy / FadeOut / FadeIn per the SKILL.md grammar
(see paper/animations/MANIM_BACKEND_PLAN.md, component 3).

Two passes:

1. **Identity.** Tensors are immutable and rewrite rules rebuild only the
   spine of the rewrite, so every untouched subtree in ``after`` is the
   SAME Python object as in ``before``. Shared subtrees match exactly,
   atom by atom. A subtree appearing more times after than before means
   the rule copied it (distribution): the surplus occurrences are
   ``copies`` of the first before-occurrence. More before than after
   means a merge (factoring).
2. **Signature.** Atoms not covered (the rewrite site itself) are matched
   greedily by (kind, label) in deterministic path order. Leftovers are
   births (grow in) and deaths (fade out).

Every atom on both sides ends up in exactly one bucket -- ``audit()``
checks this, mirroring the wire-conservation audit in book_layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tensorgrad.structure import structure_of
from tensorgrad.tensor import Delta, Derivative, Function, Variable, Zero
from tensorgrad.extras.expectation import Expectation

Path = tuple[int, ...]


@dataclass(frozen=True)
class Occ:
    """One drawable atom occurrence, addressed by its path from the root
    (child indices), so two occurrences of the same Variable are distinct."""

    path: Path
    kind: str
    label: str


@dataclass
class Matching:
    # (before, after) pairs matched one-to-one -- animate as Transform/move
    moved: list[tuple[Occ, Occ]] = field(default_factory=list)
    # (source-before, new-after): the rule duplicated this atom's subtree
    # -- animate as TransformFromCopy from the source
    copies: list[tuple[Occ, Occ]] = field(default_factory=list)
    # (dying-before, target-after): several before-occurrences collapsed
    # into one -- animate as many-to-one Transform
    merges: list[tuple[Occ, Occ]] = field(default_factory=list)
    births: list[Occ] = field(default_factory=list)  # FadeIn / grow
    deaths: list[Occ] = field(default_factory=list)  # FadeOut

    def audit(self, n_before: int, n_after: int) -> None:
        """Every atom on both sides is classified exactly once."""
        b = len(self.moved) + len(self.merges) + len(self.deaths)
        a = len(self.moved) + len(self.copies) + len(self.merges) + len(self.births)
        if b != n_before or a != n_after:
            raise AssertionError(
                f"correspondence dropped atoms: classified {b}/{n_before}"
                f" before, {a}/{n_after} after"
            )


def _kids(t: Any) -> tuple:
    # the wrt Variable of a Derivative (and the Expectation's variable) are
    # not drawn as atoms -- they appear as the ellipse's dX tag / the E box
    if isinstance(t, Derivative):
        return (t.tensor,)
    if isinstance(t, Expectation):
        return (t.tensor,)
    try:
        return tuple(structure_of(t).children)
    except NotImplementedError:
        return ()


def _is_atom(t: Any) -> bool:
    if isinstance(t, (Variable, Zero, Function)):
        return True
    # an order-2 Delta is a plain wire, not a glyph; other orders draw as
    # copy-dots / scalar rings
    if isinstance(t, Delta):
        return t.order != 2
    return False


def _label(t: Any) -> str:
    if isinstance(t, Variable):
        return t.name
    if isinstance(t, Function):
        return t.signature.name
    if isinstance(t, Delta):
        return "copydot"
    if isinstance(t, Zero):
        return "0"
    return type(t).__name__  # pragma: no cover


def _kind(t: Any) -> str:
    if isinstance(t, Variable):
        return "var"
    if isinstance(t, Function):
        return "func"
    if isinstance(t, Delta):
        return "copydot"
    if isinstance(t, Zero):
        return "zero"
    return "other"  # pragma: no cover


def _atoms_under(t: Any, path: Path = ()) -> list[Occ]:
    out: list[Occ] = []
    if _is_atom(t):
        out.append(Occ(path, _kind(t), _label(t)))
    for i, c in enumerate(_kids(t)):
        out.extend(_atoms_under(c, path + (i,)))
    return out


def _subtree_index(t: Any) -> dict[int, list[tuple[Path, Any]]]:
    """id(subtree) -> occurrences (pre-order). Structural sharing means one
    object can occur at several paths."""
    idx: dict[int, list[tuple[Path, Any]]] = {}

    def rec(n: Any, path: Path) -> None:
        idx.setdefault(id(n), []).append((path, n))
        for i, c in enumerate(_kids(n)):
            rec(c, path + (i,))

    rec(t, ())
    return idx


def match(before: Any, after: Any) -> Matching:
    """Classify every atom occurrence of ``before`` and ``after``."""
    m = Matching()
    before_idx = _subtree_index(before)

    # ---- pass 1: shared-object regions (top-down, maximal) ----
    # pool[obj_id] = unconsumed before-paths for that object
    pool: dict[int, list[Path]] = {
        k: [p for p, _ in v] for k, v in before_idx.items()
    }
    first_before: dict[int, Path] = {
        k: v[0][0] for k, v in before_idx.items()
    }
    consumed_before: list[tuple[Path, Any]] = []  # regions matched in pass 1
    matched_after_regions: list[tuple[Path, Path, Any, str]] = []
    # (before_path, after_path, obj, tag)

    def walk_after(n: Any, path: Path) -> None:
        entries = pool.get(id(n))
        if entries is not None and _shares_ref(n):
            if entries:
                bp = entries.pop(0)
                consumed_before.append((bp, n))
                matched_after_regions.append((bp, path, n, "moved"))
            else:
                matched_after_regions.append(
                    (first_before[id(n)], path, n, "copy"))
            return  # a maximal shared region covers all its atoms
        for i, c in enumerate(_kids(n)):
            walk_after(c, path + (i,))

    def _shares_ref(n: Any) -> bool:
        # scalars / trivial leaves are interned by Python (small ints etc.)
        # -- only trust identity for actual Tensor nodes
        return _is_atom(n) or bool(_kids(n))

    walk_after(after, ())

    covered: set[Path] = set()  # consumed region ROOT paths in before
    for bp, ap, obj, tag in matched_after_regions:
        rel = _atoms_under(obj)
        for occ in rel:
            b_occ = Occ(bp + occ.path, occ.kind, occ.label)
            a_occ = Occ(ap + occ.path, occ.kind, occ.label)
            if tag == "moved":
                m.moved.append((b_occ, a_occ))
            else:
                m.copies.append((b_occ, a_occ))
        if tag == "moved":
            covered.add(bp)

    # before-regions whose object also appears in after but was consumed
    # fewer times: their atoms MERGE into the first after-occurrence
    after_first: dict[int, Path] = {}

    def walk_after_first(n: Any, path: Path) -> None:
        after_first.setdefault(id(n), path)
        for i, c in enumerate(_kids(n)):
            walk_after_first(c, path + (i,))

    walk_after_first(after, ())

    leftover_before: list[Occ] = []

    def walk_before(n: Any, path: Path) -> None:
        if path in covered:
            return  # region already matched one-to-one
        if id(n) in after_first and _shares_ref(n) and pool.get(id(n)) is not None:
            still = pool[id(n)]
            if path in still:
                # object exists in after but this occurrence was surplus
                still.remove(path)
                ap = after_first[id(n)]
                for occ in _atoms_under(n):
                    m.merges.append((
                        Occ(path + occ.path, occ.kind, occ.label),
                        Occ(ap + occ.path, occ.kind, occ.label),
                    ))
                return
        if _is_atom(n):
            leftover_before.append(Occ(path, _kind(n), _label(n)))
        for i, c in enumerate(_kids(n)):
            walk_before(c, path + (i,))

    walk_before(before, ())

    # ---- pass 2: signature matching at the rewrite site ----
    covered_after = {a.path for _, a in m.moved}
    covered_after |= {a.path for _, a in m.copies}
    all_after = _atoms_under(after)
    leftover_after = [o for o in all_after if o.path not in covered_after
                      and o.path not in {a.path for _, a in m.merges}]

    by_sig_after: dict[tuple[str, str], list[Occ]] = {}
    for o in sorted(leftover_after, key=lambda o: o.path):
        by_sig_after.setdefault((o.kind, o.label), []).append(o)
    for o in sorted(leftover_before, key=lambda o: o.path):
        bucket = by_sig_after.get((o.kind, o.label))
        if bucket:
            m.moved.append((o, bucket.pop(0)))
        else:
            m.deaths.append(o)
    m.births.extend(o for bucket in by_sig_after.values() for o in bucket)

    m.audit(len(_atoms_under(before)), len(all_after))
    return m
