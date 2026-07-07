#!/usr/bin/env python
"""Render every example in examples/main.py with the book_layout engine.

    cd paper/book_layout_samples && python examples_main.py

The example functions don't return their tensors -- they hand them to
save_steps / save_as_image / to_tikz. So we monkeypatch those hooks to
CAPTURE the tensor each example builds, run the no-argument examples, and
draw each captured tensor with to_book_tikz. Produces examples_main.tex/.pdf.

Examples that need arguments, or that only exist to compile/run torch
(main11), are skipped; an example whose own source has drifted from the
current API (e.g. main3's F.sum(edges=...)) is reported inline, not fatal.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import tensorgrad.imgtools as imgtools
import tensorgrad.extras.to_tikz as to_tikz_mod
from tensorgrad.tensor import Tensor
from tensorgrad.extras.book_layout import to_book_tikz

HERE = Path(__file__).parent
STYLES = HERE.parent / "chapters" / "tikz-styles.tex"

captured: list[tuple[str, Tensor]] = []
_current = [""]


def _cap(expr, *a, **k):
    if isinstance(expr, Tensor):
        captured.append((_current[0], expr))


imgtools.save_steps = _cap
imgtools.save_as_image = _cap
imgtools.save_steps_old = _cap
to_tikz_mod.to_tikz = lambda t: (_cap(t), "")[1]

import examples.main as M  # noqa: E402  (patched hooks must exist first)

M.save_steps = _cap
M.save_as_image = _cap
M.to_tikz = to_tikz_mod.to_tikz

NAMES = [
    "notebook0", "notebook1", "notebook2", "notebook3",
    "main", "main2", "main3", "main4", "main6", "main7",
    "main9", "main10", "main12", "main13", "main14",
    "main17", "main18", "main20", "main21",
]


def collect() -> list[tuple[str, Tensor | None, str | None]]:
    out: list[tuple[str, Tensor | None, str | None]] = []
    for name in NAMES:
        fn = getattr(M, name, None)
        if fn is None:
            continue
        _current[0] = name
        before = len(captured)
        err = None
        try:
            fn()
        except Exception as e:  # example source may have drifted from the API
            err = f"{type(e).__name__}: {str(e)[:70]}"
        got = captured[before:]
        if got:
            out.append((name, got[-1][1], None))  # the final captured tensor
        else:
            out.append((name, None, err or "no expression captured"))
    return out


def render(rows) -> None:
    body = [r"\section*{examples/main.py rendered with book\_layout}"]
    for name, tensor, err in rows:
        body.append(rf"\subsection*{{\texttt{{{name}}}}}")
        if tensor is None:
            body.append(rf"\texttt{{\small {err}}}")
            continue
        try:
            body.append(r"\noindent")
            body.append(to_book_tikz(tensor, max_width=15, edge_labels=True))
        except Exception:  # unevaluated contracted derivatives etc.
            try:
                body.append(r"\noindent")
                body.append(to_book_tikz(tensor.simplify(), max_width=15, edge_labels=True))
                body[-3] = body[-3].replace("}}", " (simplified)}}")
            except Exception as e:
                body.append(rf"\texttt{{\small draw: {type(e).__name__}: {e}}}")
    tex = "\n".join([
        r"\documentclass{article}",
        r"\usepackage[margin=0.5in]{geometry}",
        r"\usepackage{tikz}\usepackage{amsmath,amssymb}",
        r"\usetikzlibrary{fit,calc}",
        rf"\input{{{STYLES}}}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        *body,
        r"\end{document}",
    ])
    (HERE / "examples_main.tex").write_text(tex)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "examples_main.tex"],
        cwd=HERE, check=False, capture_output=True,
    )
    for junk in list(HERE.glob("*.aux")) + list(HERE.glob("*.log")):
        junk.unlink()


if __name__ == "__main__":
    rows = collect()
    render(rows)
    drawn = sum(1 for _, t, _ in rows if t is not None)
    print(f"examples_main.pdf: {drawn}/{len(rows)} examples drawn")
