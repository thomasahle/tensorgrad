"""Affine structured tensors: 0/1 indicators of integer-affine index relations.

    T[e1, ..., ek] = 1  iff  for every row:  sum_i coeff_i * e_i == const

This is the general "language" behind tensorgrad's structured sparse tensors:

    Delta(n; i, j, k)          rows: i - j = 0, j - k = 0
    Convolution(x; z, y)       rows: x - z - y = 0        (i = kernel + output)
    strided/dilated conv       rows: x - z - s*y = 0  /  x - d*z - y = 0
    shift by c                 rows: i - o = c
    basis vector e_c           rows: i = c
    flatten (row-major)        rows: f - W*i - j = 0

New structured tensors need NO new compiler code: the compiler eliminates
rows into strided views (as_strided / pad+flip) when possible and falls back
to a hoisted dense indicator otherwise.
"""

from typing import Union, Any, Mapping, Sequence

import sympy
from sympy import Symbol
import torch

from tensorgrad.structure import Structure
from tensorgrad.tensor import Constant, Tensor  # noqa: F401  (Tensor for docs/typing)

# Coefficient rows: {edge: coeff} == const. Mapping/Sequence (not dict/list)
# so callers may pass narrower value types (dict is invariant in its values).
Rows = Sequence[tuple[Mapping[str, Union[int, Symbol, sympy.Expr]], Union[int, Symbol, sympy.Expr]]]


class Affine(Constant):
    """A constant 0/1 tensor defined by integer-affine equalities on its indices."""

    def __init__(
        self, rows: Rows, *shape0: Symbol,
        _symmetries: None | str | set[frozenset[str]] = None, **shape1: Symbol,
    ):
        shape = self._check_shape(shape0, shape1)
        super().__init__(_symmetries=_symmetries, **shape)
        norm: list[tuple[dict[str, sympy.Expr], sympy.Expr]] = []
        for coeffs, const in rows:
            coeffs = {e: sympy.sympify(c) for e, c in coeffs.items()}
            coeffs = {e: c for e, c in coeffs.items() if c != 0}
            for e in coeffs:
                if e not in shape:
                    raise ValueError(f"Row references edge {e!r} not in shape {set(shape)}")
            norm.append((coeffs, sympy.sympify(const)))
        self.rows = norm

    def _canonical_rows(self) -> tuple:
        return tuple(
            sorted(
                (tuple(sorted((e, str(c)) for e, c in coeffs.items())), str(const))
                for coeffs, const in self.rows
            )
        )

    def __repr__(self) -> str:
        rows = "; ".join(
            " + ".join(f"{c}*{e}" for e, c in sorted(coeffs.items())) + f" = {const}"
            for coeffs, const in self.rows
        )
        return f"Affine({rows}; shape={dict(self.shape)})"

    def __hash__(self) -> int:
        return hash((type(self).__name__, tuple(self.shape.items()), self._canonical_rows()))

    def _rename(self, **kwargs: str) -> "Affine":
        rows = [
            ({kwargs.get(e, e): c for e, c in coeffs.items()}, const) for coeffs, const in self.rows
        ]
        return Affine(rows, **{kwargs.get(e, e): s for e, s in self.shape.items()})

    def structure(self) -> Structure:
        # Two Affines with the same shape but different rows must NOT be
        # isomorphic. Conservative: bake the canonical row signature (with
        # edge names) into the label. This may miss some true isomorphisms
        # under renaming (a CSE opportunity, not a bug).  Orbits/sizes come
        # from the Constant junctions (they are part of the identity: an
        # Affine with declared symmetries differs from one without — the old
        # canon fingerprint ignored symmetries here while the old graph did
        # not, an unsoundness this resolves).
        import dataclasses

        return dataclasses.replace(super().structure(), label=("Affine", self._canonical_rows()))


# ---- derived structured tensors (the re-derivations) -----------------------


def affine_delta(size: Symbol, *edges: str) -> Affine:
    """Delta re-derived: all edges equal."""
    edge_list = list(edges)
    rows = [({edge_list[i]: 1, edge_list[i + 1]: -1}, 0) for i in range(len(edge_list) - 1)]
    return Affine(rows, **{e: size for e in edge_list})


def affine_convolution(*shape0: Symbol, stride: int = 1, dilation: int = 1, **shape1: Symbol) -> Affine:
    """Convolution re-derived: input = dilation*kernel + stride*output.

    Same edge convention as F.Convolution(input, kernel, output); unlike the
    private implementation, stride and dilation actually work.
    """
    shape = Tensor._check_shape(shape0, shape1)
    assert len(shape) == 3, "Convolution needs exactly 3 edges: input, kernel, output"
    (i, _), (k, _), (o, _) = shape.items()
    return Affine([({i: 1, k: -dilation, o: -stride}, 0)], **shape)


def affine_shift(offset: int, **shape1: Symbol) -> Affine:
    """Shift matrix: S[i, o] = 1 iff i = o + offset, so (x @ S)[o] = x[o + offset]."""
    shape = Tensor._check_shape((), shape1)
    assert len(shape) == 2, "Shift needs exactly 2 edges: input, output"
    (i, _), (o, _) = shape.items()
    return Affine([({i: 1, o: -1}, offset)], **shape)


def affine_flatten(radix: Any, **shape1: Symbol) -> Affine:
    """Row-major flatten: F[f, i, j] = 1 iff f = radix*i + j."""
    shape = Tensor._check_shape((), shape1)
    assert len(shape) == 3, "Flatten needs exactly 3 edges: flat, outer, inner"
    (f, _), (i, _), (j, _) = shape.items()
    return Affine([({f: 1, i: -radix, j: -1}, 0)], **shape)


def affine_basis(index: int, **shape1: Symbol) -> Affine:
    """Basis (one-hot) vector: B[i] = 1 iff i == index."""
    shape = Tensor._check_shape((), shape1)
    assert len(shape) == 1
    (e, _) = list(shape.items())[0]
    return Affine([({e: 1}, index)], **shape)


# ---- dense indicator materialization (fallback + oracle) -------------------


def _norm_row(row: tuple) -> tuple[str, Any, Any, Any]:
    """Accept ('eq', coeffs, const), ('range', coeffs, k, X) or the legacy
    bare (coeffs, const) equality form."""
    if len(row) == 2:
        return ("eq", row[0], row[1], None)
    if row[0] == "eq":
        return ("eq", row[1], row[2], None)
    return ("range", row[1], row[2], row[3])


def indicator_tensor(sizes: list[int], rows: Sequence) -> torch.Tensor:
    """Dense 0/1 tensor over axes of the given sizes; rows reference axis
    positions. Equality rows are [sum c_i*w_i == const]; range rows are
    [0 <= sum c_i*w_i + k <= X-1] (from summing an eq row over a free wire).
    Used as the compiler's correctness fallback and by evaluate()."""
    if not sizes:
        ok = True
        for row in rows:
            kind, coeffs, a, b = _norm_row(row)
            ok &= (a == 0) if kind == "eq" else (0 <= a <= b - 1)
        return torch.tensor(1.0 if ok else 0.0)
    grids = torch.meshgrid(*[torch.arange(n) for n in sizes], indexing="ij")
    mask = torch.ones(tuple(sizes), dtype=torch.bool)
    for row in rows:
        kind, coeffs, a, b = _norm_row(row)
        acc = torch.zeros(tuple(sizes), dtype=torch.long)
        for axis, c in coeffs.items():
            acc = acc + int(c) * grids[axis]
        if kind == "eq":
            mask &= acc == int(a)
        else:
            e = acc + int(a)
            mask &= (e >= 0) & (e <= int(b) - 1)
    return mask.to(torch.get_default_dtype())


# ---- evaluate() oracle support ---------------------------------------------

from tensorgrad.extras.evaluate import Context  # noqa: E402


# (explicit register(Affine): with `self` annotated, the bare decorator would
# infer the dispatch class from the FIRST annotation, i.e. Context)
@Context._evaluate.register(Affine)  # type: ignore[attr-defined]  # singledispatchmethod stub loses .register
def _evaluate_affine(self: Context, affine: Affine) -> torch.Tensor:
    edges = list(affine.edges)
    missing = {s for s in affine.shape.values() if s not in self.dims}
    if missing:
        raise ValueError(f"Dims {missing} not supplied to Affine")
    sizes = [self.dims[affine.shape[e]] for e in edges]
    rows = []
    for coeffs, const in affine.rows:
        rows.append(
            (
                {edges.index(e): int(sympy.sympify(c).subs(self.dims)) for e, c in coeffs.items()},
                int(sympy.sympify(const).subs(self.dims)),
            )
        )
    return indicator_tensor(sizes, rows).refine_names(*edges)
