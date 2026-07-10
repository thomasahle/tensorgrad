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
from tensorgrad.tensor import Constant, Delta, Tensor, Zero  # noqa: F401  (Tensor for docs/typing)

# Coefficient rows. Accepted forms:
#   ({edge: coeff}, const)             equality  [sum c_i*e_i == const]
#   ("eq", {edge: coeff}, const)       equality, tagged
#   ("range", {edge: coeff}, k, X)     inequality [0 <= sum c_i*e_i + k <= X-1]
# Mapping/Sequence (not dict/list) so callers may pass narrower value types.
Rows = Sequence[tuple]
_Num = Union[int, Symbol, sympy.Expr]
_Coeffs = Mapping[str, _Num]


class Affine(Constant):
    """A constant 0/1 tensor defined by integer-affine relations on its indices.

    Rows come in two kinds. Equality rows (`self.rows`, the original
    vocabulary) are `(coeffs, const)` meaning [sum c_i*e_i == const]; the
    compiler threads them into einsum constraints and eliminates them into
    strided views. Range rows (`self.range_rows`, task #44) are
    `(coeffs, k, X)` meaning [0 <= sum c_i*e_i + k <= X-1] — inequality
    indicators (tril/triu/causal masks). Stage 1: an Affine with any range
    row lowers as a hoisted dense indicator constant (correct, one
    materialization per specialization); native IR range constraints are
    the follow-up (#46 era)."""

    def __init__(
        self, rows: Rows, *shape0: Symbol,
        _symmetries: None | str | set[frozenset[str]] = None, **shape1: Symbol,
    ):
        shape = self._check_shape(shape0, shape1)
        super().__init__(_symmetries=_symmetries, **shape)
        norm: list[tuple[dict[str, sympy.Expr], sympy.Expr]] = []
        ranges: list[tuple[dict[str, sympy.Expr], sympy.Expr, sympy.Expr]] = []

        def _coeffs(coeffs: Mapping) -> dict[str, sympy.Expr]:
            out = {e: sympy.sympify(c) for e, c in coeffs.items()}
            out = {e: c for e, c in out.items() if c != 0}
            for e in out:
                if e not in shape:
                    raise ValueError(f"Row references edge {e!r} not in shape {set(shape)}")
            return out

        for row in rows:
            if len(row) == 2:  # legacy bare equality
                coeffs, const = row
                norm.append((_coeffs(coeffs), sympy.sympify(const)))
            elif row[0] == "eq":
                norm.append((_coeffs(row[1]), sympy.sympify(row[2])))
            elif row[0] == "range":
                ranges.append((_coeffs(row[1]), sympy.sympify(row[2]), sympy.sympify(row[3])))
            else:
                raise ValueError(f"Unknown Affine row kind {row[0]!r}")
        self.rows = norm
        self.range_rows = ranges

    def _canonical_rows(self) -> tuple:
        eqs = tuple(
            sorted(
                (tuple(sorted((e, str(c)) for e, c in coeffs.items())), str(const))
                for coeffs, const in self.rows
            )
        )
        rngs = tuple(
            sorted(
                (tuple(sorted((e, str(c)) for e, c in coeffs.items())), str(k), str(X))
                for coeffs, k, X in self.range_rows
            )
        )
        return eqs + (("__ranges__",) + rngs if rngs else ())

    def __repr__(self) -> str:
        parts = [
            " + ".join(f"{c}*{e}" for e, c in sorted(coeffs.items())) + f" = {const}"
            for coeffs, const in self.rows
        ] + [
            "0 <= " + " + ".join(f"{c}*{e}" for e, c in sorted(coeffs.items())) + f" + {k} <= {X}-1"
            for coeffs, k, X in self.range_rows
        ]
        return f"Affine({'; '.join(parts)}; shape={dict(self.shape)})"

    def __hash__(self) -> int:
        return hash((type(self).__name__, tuple(self.shape.items()), self._canonical_rows()))

    def _rename(self, **kwargs: str) -> "Affine":
        rows: list = [
            ("eq", {kwargs.get(e, e): c for e, c in coeffs.items()}, const)
            for coeffs, const in self.rows
        ] + [
            ("range", {kwargs.get(e, e): c for e, c in coeffs.items()}, k, X)
            for coeffs, k, X in self.range_rows
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


def affine_ineq(coeffs: _Coeffs, lo: _Num, hi: _Num, **shape1: Symbol) -> Affine:
    """Inequality indicator: T[e...] = 1 iff lo <= sum c_i*e_i <= hi."""
    lo, hi = sympy.sympify(lo), sympy.sympify(hi)
    return Affine([("range", coeffs, -lo, hi - lo + 1)], **shape1)


def affine_tril(*, strict: bool = False, **shape1: Symbol) -> Affine:
    """Lower-triangle indicator over (row, col) — edge ORDER carries the
    roles: T[r, c] = 1 iff r >= c (strict: r > c). The causal attention
    mask [key <= seq] is affine_tril(seq=n, key=n)."""
    shape = Tensor._check_shape((), shape1)
    assert len(shape) == 2, "tril needs exactly 2 edges: row, col"
    (r, sr), (c, _) = shape.items()
    return affine_ineq({r: 1, c: -1}, 1 if strict else 0, sr - 1, **shape1)


def affine_triu(*, strict: bool = False, **shape1: Symbol) -> Affine:
    """Upper-triangle indicator over (row, col): T[r, c] = 1 iff r <= c
    (strict: r < c)."""
    shape = Tensor._check_shape((), shape1)
    assert len(shape) == 2, "triu needs exactly 2 edges: row, col"
    (r, _), (c, sc) = shape.items()
    return affine_ineq({c: 1, r: -1}, 1 if strict else 0, sc - 1, **shape1)


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
# affine must NOT import extras.evaluate at module load: evaluate imports the
# compiler (compiler.cells), so a top-level import here forms a cycle
#   evaluate -> compiler(pkg) -> runtime -> lower -> affine -> evaluate .
# Instead the Affine oracle is registered by a hook that extras.evaluate calls
# at the END of its own init, once Context and its dispatch table exist.


def _register_evaluate_oracle() -> None:
    from tensorgrad.extras.evaluate import Context

    # (explicit register(Affine): with `self` annotated, the bare decorator
    # would infer the dispatch class from the FIRST annotation, i.e. Context)
    @Context._evaluate.register(Affine)  # type: ignore[attr-defined]  # singledispatchmethod stub loses .register
    def _evaluate_affine(self: "Context", affine: Affine) -> torch.Tensor:
        edges = list(affine.edges)
        missing = {s for s in affine.shape.values() if s not in self.dims}
        if missing:
            raise ValueError(f"Dims {missing} not supplied to Affine")
        sizes = [self.dims[affine.shape[e]] for e in edges]

        def sub(x: Any) -> int:
            return int(sympy.sympify(x).subs(self.dims))

        rows: list[tuple] = []
        for coeffs, const in affine.rows:
            rows.append(({edges.index(e): sub(c) for e, c in coeffs.items()}, sub(const)))
        for coeffs, k, X in affine.range_rows:
            rows.append(("range", {edges.index(e): sub(c) for e, c in coeffs.items()}, sub(k), sub(X)))
        return indicator_tensor(sizes, rows).refine_names(*edges)


# ---- contraction closure (#46): one pairwise rule for structural algebra ---


def _as_row_systems(t: Tensor) -> "tuple[list, list, dict] | None":
    """(eq_rows, range_rows, shape) viewing a structural tensor as an Affine.
    Delta of order >= 1 is the all-edges-equal Affine (order 1 = ones, no
    rows); true Affines contribute their rows verbatim."""
    if isinstance(t, Affine):
        return list(t.rows), list(t.range_rows), dict(t.shape)
    if isinstance(t, Delta) and t.order >= 1:
        es = list(t.edges)
        one = sympy.Integer(1)
        eqs = [({es[i]: one, es[i + 1]: -one}, sympy.Integer(0)) for i in range(len(es) - 1)]
        return eqs, [], dict(t.shape)
    return None


def contract_affines(t1: Tensor, t2: Tensor, e: str) -> "list[Tensor] | None":
    """The structural-tensor contraction closure (#46): contracting two
    Affine-representable factors concatenates their row systems and
    eliminates every shared edge over the integers. Each eliminated edge x
    with a unit-coefficient equality row substitutes x = expr into all
    other rows AND emits the range row [0 <= expr <= size_x - 1] (the sum
    over x has a solution only where expr lands in x's index range) —
    Sigma_o [i = o + a] = [0 <= i - a <= X-1], the shifted-window boundary.
    A shared edge in no row at all contributes a multiplicity of size_x
    (an order-0 Delta). Anything it cannot eliminate exactly (non-unit
    coefficients: divisibility conditions; edges appearing only in range
    rows: solution COUNTS vary) returns None and the product stays as-is —
    a missed simplification, never a wrong one. Subsumes window/shift
    composition, Delta-through-Affine contraction, and mask-shift algebra;
    Delta-Delta pairs and identity renames are handled by the earlier
    (cheaper) rules in simplify.PAIR_RULES."""
    if isinstance(t1, Delta) and isinstance(t2, Delta):
        return None  # merge_copy_tensors' domain
    s1, s2 = _as_row_systems(t1), _as_row_systems(t2)
    if s1 is None or s2 is None:
        return None
    eqs = [(dict(c), sympy.sympify(k)) for c, k in s1[0] + s2[0]]
    rngs = [(dict(c), sympy.sympify(k), sympy.sympify(X)) for c, k, X in s1[1] + s2[1]]
    shape = s1[2] | s2[2]
    shared = set(t1.edges) & set(t2.edges)  # pair-exclusive: an edge name
    # occurs at most twice in a product, and _delta_pair_step only pairs
    # factors when exactly these two carry it.
    rem_shape = {f: s for f, s in shape.items() if f not in shared}

    mults: list = []
    todo = set(shared)
    progress = True
    while todo and progress:
        progress = False
        for x in sorted(todo):
            row = next((r for r in eqs if x in r[0] and r[0][x] in (1, -1)), None)
            if row is None:
                continue
            coeffs, const = row
            cx = coeffs[x]
            # x = K - sum_f d_f * f   (exact: cx is a unit)
            K = cx * const
            d = {f: cx * c for f, c in coeffs.items() if f != x}
            eqs.remove(row)

            def _subst_linear(c2: dict, x: str = x, d: dict = d, K: sympy.Expr = K) -> "tuple[dict, sympy.Expr]":
                """Rewrite sum c2_f*f eliminating x; returns (coeffs, delta_const)
                where the form gains +delta_const on the constant-offset side."""
                cx2 = c2.pop(x)
                for f, df in d.items():
                    c2[f] = sympy.sympify(c2.get(f, 0)) - cx2 * df
                c2 = {f: c for f, c in c2.items() if sympy.sympify(c) != 0}
                return c2, cx2 * K

            for i, (c2, k2) in enumerate(list(eqs)):
                if x in c2:
                    nc, dk = _subst_linear(dict(c2))
                    eqs[i] = (nc, k2 - dk)  # sum c*f = const - cx2*K
            for i, (c2, k2, X2) in enumerate(list(rngs)):
                if x in c2:
                    nc, dk = _subst_linear(dict(c2))
                    rngs[i] = (nc, k2 + dk, X2)  # offset side gains +cx2*K
            # the summed index must land in range: 0 <= K - sum d*f <= size_x - 1
            neg = {f: -df for f, df in d.items() if sympy.sympify(df) != 0}
            trivial = (
                len(neg) == 1
                and next(iter(neg.values())) == 1
                and K == 0
                and shape.get(next(iter(neg))) == shape[x]
            )
            if not trivial:
                rngs.append((neg, K, sympy.sympify(shape[x])))
            todo.discard(x)
            progress = True
    for x in sorted(todo):
        if any(x in c for c, _ in eqs) or any(x in c for c, _, _ in rngs):
            return None  # non-unit or range-only occurrence: not exactly eliminable
        mults.append(Delta(shape[x]))  # free summed index: multiplicity size_x
        todo.discard(x)

    # Degenerate rows: decide concrete ones now; an unsatisfiable row zeroes
    # the whole contraction.
    out_eqs, out_rngs = [], []
    for c, k in eqs:
        if not c:
            if k == 0:
                continue
            if k.is_number:
                return [Zero(**rem_shape)]
        out_eqs.append((c, k))
    for c, k, X in rngs:
        if not c and k.is_number and X.is_number:
            if 0 <= k <= X - 1:
                continue
            return [Zero(**rem_shape)]
        out_rngs.append((c, k, X))

    out: list[Tensor] = list(mults)
    if not out_eqs and not out_rngs:
        out += [Delta(s, f) for f, s in rem_shape.items()]  # pure ones
    else:
        rows = [("eq", c, k) for c, k in out_eqs] + [("range", c, k, X) for c, k, X in out_rngs]
        out.append(Affine(rows, **rem_shape))
    return out


# Register in the pairwise product-rule catalog (the same on-import
# convention as _dispatch_simplify.register for whole-node rules).
from tensorgrad import simplify as _simplify_catalog  # noqa: E402

if contract_affines not in _simplify_catalog.PAIR_RULES:
    _simplify_catalog.PAIR_RULES.append(contract_affines)
