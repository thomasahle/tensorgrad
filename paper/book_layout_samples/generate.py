#!/usr/bin/env python
"""Regenerate the book_layout sample sheets (and serve as usage examples).

    cd paper/book_layout_samples && python generate.py

Produces one <name>_sheet.tex + .pdf per group below. Requires a LaTeX
install with the repo's paper/chapters/tikz-styles.tex on the input path.

This doubles as the engine's example gallery: each `case(...)` shows how to
build an expression and render it with `to_book_tikz`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from sympy import symbols

from tensorgrad import Variable, functions as F
from tensorgrad.tensor import Delta, Product, Sum, Zero
from tensorgrad.extras.expectation import Expectation
from tensorgrad.extras.book_layout import to_book_tikz

n = symbols("n")
HERE = Path(__file__).parent
STYLES = HERE.parent / "chapters" / "tikz-styles.tex"


def V(name, **edges):
    return Variable(name, **edges)


# --------------------------------------------------------------------------
# Each group is a list of (caption, expression, kwargs-to-to_book_tikz)
# --------------------------------------------------------------------------

BASIC = [
    (r"matrix product $AB$", V("A", i=n, j=n) @ V("B", j=n, k=n), {}),
    (r"bilinear $a^\top X b$",
     Product([V("a", i=n), V("X", i=n, j=n), V("b", j=n)]), {}),
    (r"trace $\mathrm{Tr}(ABCD)$",
     Product([V("A", i=n, j=n), V("B", j=n, k=n),
              V("C", k=n, l=n), V("D", l=n, i=n)]), {}),
    (r"$\mathrm{diag}(v)$", Product([V("v", m=n), Delta(n, "m", "i", "j")]), {}),
    (r"outer $x x^\top$", Product([V("x", i=n), V("x", j=n)]), {}),
    (r"sum $AB + BA$",
     Sum([V("A", i=n, j=n) @ V("B", j=n, k=n),
          V("B", i=n, j=n) @ V("A", j=n, k=n)]), {}),
]

FUNCTIONS = [
    (r"$\exp(x)$", F.exp(V("x", i=n)), {}),
    (r"$\mathrm{softmax}(x)$", F.softmax(V("x", i=n), dim="i"), {}),
    (r"$(A+B)\,C$",
     Product([Sum([V("A", i=n, j=n), V("B", i=n, j=n)]), V("C", j=n, k=n)]), {}),
    (r"$\mathbb{E}[x x^\top]$",
     Expectation(Product([V("x", i=n), V("x", j=n)]), V("x", i=n)), {}),
    (r"zero matrix", Zero(i=n, j=n), {}),
]


def _derivatives():
    x = V("x", i=n)
    A = V("A", i=n, j=n)
    q = x @ A @ x.rename(i="j")
    return [
        (r"$\partial(x^\top A x)/\partial x$ (loop)", q.grad(x, {"i": "i_"}), {}),
        (r"$\partial^2(x^\top A x)/\partial x^2$ (nested)",
         q.grad(x, {"i": "p"}).grad(x, {"i": "q"}), {}),
        (r"softmax Jacobian (simplified)",
         F.softmax(x, dim="i").grad(x, {"i": "j"}).simplify(), {}),
        (r"$\partial\mathrm{CE}/\partial z$ (fit to 9cm)",
         F.cross_entropy(V("z", i=n), V("y", i=n), dim="i")
             .grad(V("z", i=n), {"i": "j"}).simplify(),
         {"max_width": 9}),
    ]


GROUPS = {"basic": BASIC, "functions": FUNCTIONS, "derivatives": _derivatives()}


def render(name: str, cases) -> None:
    body = [rf"\subsection*{{{name}}}", r"\begin{tabular}{p{5.5cm}l}"]
    for caption, expr, kw in cases:
        body.append(rf"{caption} & {to_book_tikz(expr, **kw)} \\[22pt]")
    body.append(r"\end{tabular}")
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
    tex_path = HERE / f"{name}_sheet.tex"
    tex_path.write_text(tex)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=HERE, check=False, capture_output=True,
    )
    print(f"  wrote {name}_sheet.pdf ({len(cases)} cases)")


if __name__ == "__main__":
    print("Regenerating book_layout sample sheets:")
    for name, cases in GROUPS.items():
        render(name, cases)
    # tidy LaTeX aux files
    for junk in HERE.glob("*.aux"):
        junk.unlink()
    for junk in HERE.glob("*.log"):
        junk.unlink()
    print("done.")
