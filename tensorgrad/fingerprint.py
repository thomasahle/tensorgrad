"""Tensor-level Schwartz-Zippel fingerprints (v2): the tensor tier's OWN
exact evaluator, incremental per node.

`szfp(t)` returns a cached bundle `(weight-symbol support, digest, is-zero)`
or None (abstention — the node opts out of semantic rewrites and keeps the
structural merge key). The evaluator walks the tree through the structure()
child accessor and gives each node type its value semantics directly —
NOT by lowering to the compiler IR. The IR has its own fingerprint engine
(compiler/szfp.py) for the pass gates; the two tiers share the algorithm
family but deliberately not digests: they are separate semantics with
separate trusted bases, reconciled — like everything else — by the
compiled-vs-evaluate property harness.

The algorithm, per node, at k trials over F_P (P = 2^19 - 1, exact int64
arithmetic — no rounding, so reassociation cannot leak):

  * Variable: seeded random tensor keyed by (seed, trial, name, dims) —
    program-independent, so separately built expressions over the same
    variables agree. Declared symmetry groups (<= 6 axes) are drawn as
    orbit sums: a uniform point of the SYMMETRIC subspace, the correct
    Schwartz-Zippel domain for a declared-symmetric variable.
  * Delta / Zero: exact structural values. An order-0 Delta(n) is the
    scalar n itself (weight channel).
  * Sum: exact weighted combination; weights reduced exactly (ints,
    Fractions, floats via their exact binary Fraction, sympy expressions
    under the trial's dim assignment, irrational numbers as their binary64
    image).
  * Product: exact einsum over the edge graph, greedy pairwise folding
    (connectivity-ordered — left-to-right explodes on flattened chains),
    mod-P reduction between steps.
  * Function: an integer power evaluates exactly (square-and-multiply mod
    P — no Fermat exponent reduction, so x^P and x stay distinct);
    everything else is an OPAQUE ATOM: a seeded random tensor keyed by
    (signature, input value-hashes, dims). Atom-blindness to analytic
    identities is by design — it errs only toward not-merging.
  * Affine: the exact 0/1 indicator at the drawn sizes; an indicator that
    is EMPTY at the drawn sizes abstains (it would be a degenerate zero
    that collides unequal programs — offset windows larger than the small
    random dims are the reproduced miscompile class).
  * Everything else — Derivative, Expectation, any future type — needs NO
    registration: the default is an opaque atom keyed by the node's
    name-sensitive structural fingerprint (refined_sort_key), driven
    entirely by structure(). Structurally equal nodes get equal atoms, so
    the coarsening invariant holds by construction for all types; the
    registrations above are precision upgrades that make cancellations
    through a type visible, not obligations.

Dim assignment: symbols sizing array axes draw small (2..5) with an
anti-constant guarantee (a later trial never repeats the trial-0 value, so
(n - c) * x cannot vanish at every trial); symbols reaching values only
through scalar weights draw from a ~2^40 range, killing the deterministic
name-pair collision class (Delta(h) vs Delta(v)). Division by an exact 0
residue abstains (no global retry machinery — abstention is always sound).

Incrementality: evaluated trial arrays are cached per node in a bounded
LRU (digests and abstentions are cached permanently on the node), so
fingerprinting a rebuilt parent costs O(parent) from its children's cached
arrays — the v1 lower-the-whole-subtree quadratic (which forced a 320-node
budget) is gone. Oversized ARRAYS (not trees) still abstain: literal dims
are honored, so a 512^3 constant is over budget while a 100k-node graph of
small tensors is fine.

Guarantee (atoms formulation): expressions equal as rational functions
over the atoms — variables ranging over their declared symmetric
subspaces — ALWAYS fingerprint equal (exact arithmetic has no order to
leak); unequal ones collide with probability <= deg/P per trial,
independently k times, plus ~2^-64 atom-birthday terms. Value identity
never feeds Tensor.__eq__/__hash__ (Tensor equality quotients outer edge
names; the fingerprint is name-sensitive, so it could not even satisfy
the hash/eq contract).
"""

import hashlib
from collections import OrderedDict
from fractions import Fraction
from functools import singledispatch
from typing import Any, Optional

import numpy as np
import sympy

from tensorgrad.structure import _children
from tensorgrad.tensor import Delta, Function, Product, Rename, Sum, Tensor, Variable, Zero

P = 524287  # Mersenne 2^19 - 1: pairwise products < 2^38, room for 2^24-term sums
K_TRIALS = 3
SEED = 0
_DIM_LO, _DIM_HI = 2, 5
_WIDE = 1 << 40
_MAX_ELEMS = 5**8  # abstain above this array size (guards literal dims too)
_MAX_STEP_TERMS = 1 << 24  # int64 overflow guard per einsum step

_FP_ATTR = "_szfp_v2"  # (support, digest, zero) or None — permanent, on the node
_SYMS_ATTR = "_szsyms_v2"  # (shape_syms, weight_syms) — permanent, on the node
_SIZE_ATTR = "_tsize_v1"

# Evaluated trial arrays per node: byte-bounded LRU holding
# (node, vals, nbytes) — the node ref pins the id against recycling;
# evicted entries recompute from children on demand. vals = tuple over
# trials of (edge_names, int64 array). During ONE _vals walk, freshly
# computed entries are pinned in a walk-local dict and merged afterwards,
# so eviction can never starve a pending parent (the review's livelock:
# sibling subtrees each larger than the cap re-evicting each other
# forever). A walk that outgrows the pin budget abstains at the root.
_ARRAYS: OrderedDict = OrderedDict()
_ARRAYS_BYTES = 0
_ARRAYS_CAP_BYTES = 256 << 20
_PIN_CAP = 150_000

_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class _Abstain(Exception):
    """This node opts out of the semantic tier (division by 0 mod P,
    oversized array, empty indicator, unresolvable scalar)."""


# ---------------------------------------------------------------------------
# Deterministic seeded randomness (blake2b, never python hash())
# ---------------------------------------------------------------------------


def _h(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p if isinstance(p, bytes) else repr(p).encode())
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big")


def _rand_tensor(shape: tuple, *key: Any) -> np.ndarray:
    rng = np.random.default_rng(_h(*key))
    return rng.integers(0, P, size=shape, dtype=np.int64)


# ---------------------------------------------------------------------------
# Dim assignment
# ---------------------------------------------------------------------------


def _small_dim(name: str, trial: int) -> int:
    span = _DIM_HI - _DIM_LO + 1
    v = _DIM_LO + _h(SEED, trial, "dim", name) % span
    if trial:
        # Anti-constant guarantee: never repeat the trial-0 value, so a
        # shape symbol cannot equal one constant across ALL trials.
        base = _DIM_LO + _h(SEED, 0, "dim", name) % span
        if v == base:
            v = _DIM_LO + (v - _DIM_LO + 1) % span
    return v


def _wide_dim(name: str, trial: int) -> int:
    return 2 + _h(SEED, trial, "dim-wide", name) % _WIDE


def _syms(t: Tensor) -> tuple[frozenset, frozenset]:
    """(shape symbols, weight symbols) of the subtree, cached per node.
    Shape symbols size array axes and must draw small; weight symbols reach
    values only through scalars and draw wide. The weight set is also the
    merge key's SUPPORT: terms scaling by different dim polynomials refuse
    to merge regardless of the draws."""
    hit = t.__dict__.get(_SYMS_ATTR)
    if hit is not None:
        return hit
    stack = [t]
    while stack:
        node = stack[-1]
        if node.__dict__.get(_SYMS_ATTR) is not None:
            stack.pop()
            continue
        kids = _children(node)
        missing = [c for c in kids if c.__dict__.get(_SYMS_ATTR) is None]
        if missing:
            stack.extend(missing)
            continue
        stack.pop()
        shape: set = set()
        weight: set = set()
        for expr in node.shape.values():
            shape.update(s.name for s in sympy.sympify(expr).free_symbols)
        if isinstance(node, Sum):
            for w in node.weights:
                weight.update(s.name for s in sympy.sympify(w).free_symbols)
        if isinstance(node, Delta) and node.order == 0:
            weight.update(s.name for s in sympy.sympify(node.size).free_symbols)
        # Affine row constants (duck-typed: Affine lives in an extras
        # module) interact with axis extents, so their symbols must draw
        # SMALL — a wide-drawn window size saturates every row and makes
        # differently-sized windows collide.
        for coeffs, const in getattr(node, "rows", ()) or ():
            shape.update(s.name for s in sympy.sympify(const).free_symbols)
        for coeffs, k, x in getattr(node, "range_rows", ()) or ():
            shape.update(s.name for s in sympy.sympify(k).free_symbols)
            shape.update(s.name for s in sympy.sympify(x).free_symbols)
        for c in kids:
            cs, cw = c.__dict__[_SYMS_ATTR]
            shape |= cs
            weight |= cw
        node.__dict__[_SYMS_ATTR] = (frozenset(shape), frozenset(weight))
    return t.__dict__[_SYMS_ATTR]


def _assign(expr: Any, node: Tensor, trial: int) -> dict:
    shape_syms, _ = _syms(node)
    return {
        s: (_small_dim(s.name, trial) if s.name in shape_syms else _wide_dim(s.name, trial))
        for s in sympy.sympify(expr).free_symbols
    }


def _dim_of(expr: Any, node: Tensor, trial: int) -> int:
    e = sympy.sympify(expr).subs(_assign(expr, node, trial))
    if not e.is_Integer or int(e) <= 0:
        raise _Abstain(f"unresolvable dim {expr}")
    return int(e)


def _scalar(w: Any, node: Tensor, trial: int) -> int:
    """Exact residue of a scalar weight (same vocabulary as the IR tier:
    ints/Fractions exact, floats and irrationals via their exact binary64
    Fraction, dim symbols under the trial's assignment).

    A NONZERO weight whose residue comes out 0 abstains: shape symbols draw
    from a 4-point domain, so a dim polynomial divisible by
    (n-2)(n-3)(n-4)(n-5) — or a literal multiple of P — evaluates to 0 at
    every trial while being nonzero at real dims. Trusting that 0 was a
    reproduced false-cancel; abstention costs only the structural fallback."""
    r = _scalar_raw(w, node, trial)
    if r == 0 and not (isinstance(w, (int, float, Fraction)) and w == 0) and sympy.sympify(w) != 0:
        raise _Abstain(f"nonzero weight with 0 residue: {w}")
    return r


def _scalar_raw(w: Any, node: Tensor, trial: int) -> int:
    if isinstance(w, int):
        return w % P
    if isinstance(w, Fraction):
        return _frac(w)
    if isinstance(w, float):
        return _frac(Fraction(w))
    e = sympy.sympify(w).subs(_assign(w, node, trial))
    if e.is_Integer:
        return int(e) % P
    if e.is_Rational:
        return _frac(Fraction(int(e.p), int(e.q)))
    if e.is_number and e.is_real:
        try:
            return _frac(Fraction(float(e.evalf(30))))
        except (ValueError, TypeError, OverflowError):
            pass
    raise _Abstain(f"unresolvable scalar {w}")


def _frac(fr: Fraction) -> int:
    d = fr.denominator % P
    if d == 0:
        raise _Abstain("denominator 0 mod P")
    return fr.numerator % P * pow(d, P - 2, P) % P


def _guard_size(dims: tuple) -> tuple:
    total = 1
    for d in dims:
        total *= d
    if total > _MAX_ELEMS:
        raise _Abstain(f"array too large: {dims}")
    return dims


def _align(edges: tuple, arr: np.ndarray, target: tuple) -> np.ndarray:
    if edges == target:
        return arr
    return np.transpose(arr, [edges.index(e) for e in target])


# ---------------------------------------------------------------------------
# Per-type value semantics. Each returns vals = tuple over trials of
# (edge_names, int64 array), or raises _Abstain.
# ---------------------------------------------------------------------------


@singledispatch
def _compute(t: Tensor, kv: list) -> tuple:
    """The DEFAULT semantics, driven entirely by structure(): any node type
    without a registered evaluator is an opaque atom — a seeded random
    tensor keyed by the node's name-sensitive structural fingerprint
    (refined_sort_key, canon-cached). Structurally equal nodes get equal
    atoms, so the coarsening invariant (structural merge => value merge)
    holds BY CONSTRUCTION for every type, including ones that don't exist
    yet. The registrations below are precision upgrades, not obligations:
    each one teaches a type its real field semantics so cancellations
    through it become visible — same architecture as the simplify rule
    catalog, where the default is always sound."""
    from tensorgrad.structure import refined_sort_key

    edges = tuple(sorted(t.shape.keys()))
    key = refined_sort_key(t)
    out = []
    for j in range(K_TRIALS):
        dims = _guard_size(tuple(_dim_of(t.shape[e], t, j) for e in edges))
        out.append((edges, _rand_tensor(dims, SEED, j, "node-atom", key, edges, dims)))
    return tuple(out)


@_compute.register
def _var(t: Variable, kv: list) -> tuple:
    edges = tuple(t.shape.keys())
    canon = tuple(sorted(edges))
    groups = [g for g in (getattr(t, "_symmetries", None) or set()) if len(g) > 1]
    if any(len(g) > 6 for g in groups):
        # Orbit sums above 720 permutations are too costly, and an
        # UNsymmetrized draw would split structurally-equal permuted pairs
        # (coarsening violation). Abstain: the structural key handles them.
        raise _Abstain("symmetry orbit > 6 axes")
    out = []
    for j in range(K_TRIALS):
        # Draw in SORTED-edge order and transpose back: Variable("x", i=n,
        # j=m) and Variable("x", j=m, i=n) are the same tensor and must
        # draw the same values (kwargs order is not identity).
        cdims = _guard_size(tuple(_dim_of(t.shape[e], t, j) for e in canon))
        arr = _align(canon, _rand_tensor(cdims, SEED, j, "var", t.name, canon, cdims), edges)
        dims = arr.shape
        for g in groups:
            # Orbit sum over the declared-symmetric axes: a uniform random
            # point of the symmetric subspace. Groups > 6 axes abstain from
            # symmetrization only (sound: the pair keeps two merge slots).
            import itertools

            axes = sorted(edges.index(e) for e in g)
            acc = np.zeros_like(arr)
            for perm in itertools.permutations(axes):
                order = list(range(arr.ndim))
                for src, dst in zip(axes, perm):
                    order[src] = dst
                acc = (acc + np.transpose(arr, order)) % P
            arr = acc
        out.append((edges, arr))
    return tuple(out)


@_compute.register
def _zero(t: Zero, kv: list) -> tuple:
    edges = tuple(t.shape.keys())
    return tuple(
        (edges, np.zeros(_guard_size(tuple(_dim_of(s, t, j) for s in t.shape.values())), dtype=np.int64))
        for j in range(K_TRIALS)
    )


@_compute.register
def _delta(t: Delta, kv: list) -> tuple:
    edges = tuple(t.shape.keys())
    out = []
    for j in range(K_TRIALS):
        if t.order == 0:
            # Scalar Delta(n) = n itself: the weight channel. The draw is
            # decided by THIS node's subtree (order-0 => wide); an enclosing
            # term using n as an axis size draws it small there — the mixed
            # draw under-merges trace-vs-weight forms of n, never miscompiles.
            out.append((edges, np.int64(_scalar(t.size, t, j)) * np.ones((), dtype=np.int64)))
            continue
        d = _dim_of(t.size, t, j)
        _guard_size((d,) * t.order)
        arr = np.zeros((d,) * t.order, dtype=np.int64)
        arr[tuple(np.arange(d) for _ in range(t.order))] = 1
        out.append((edges, arr))
    return tuple(out)


@_compute.register
def _rename(t: Rename, kv: list) -> tuple:
    (child,) = kv
    return tuple((tuple(t.mapping.get(e, e) for e in edges), arr) for edges, arr in child)


@_compute.register
def _sum(t: Sum, kv: list) -> tuple:
    edges = tuple(t.shape.keys())
    out = []
    zeroish = False
    for j in range(K_TRIALS):
        acc = None
        for w, term in zip(t.weights, kv):
            t_edges, arr = term[j]
            arr = _align(t_edges, arr, edges) * _scalar(w, t, j)
            acc = arr if acc is None else acc + arr
        arr = (acc if acc is not None else np.zeros((), dtype=np.int64)) % P
        zeroish = zeroish or (arr.size > 0 and not arr.any())
        out.append((edges, arr))
    if zeroish and len({_syms(term) for term in t.terms}) > 1:
        # An all-zero combination of terms with DIFFERENT symbol profiles
        # (e.g. Tr(I_n)*x - Tr(I_m)*x) can be a dim-draw coincidence, not an
        # identity — the small domain cannot certify it. Same-profile zeros
        # (x - x regrouped) are the feature and stay.
        raise _Abstain("zero sum over mixed symbol profiles")
    return tuple(out)


@_compute.register
def _product(t: Product, kv: list) -> tuple:
    out_edges = set(t.edges)
    target = tuple(t.shape.keys())
    out = []
    for j in range(K_TRIALS):
        if len(kv) > 64:
            raise _Abstain("product too wide for the pair-scan fold")
        items = [(list(term[j][0]), term[j][1] % P) for term in kv]
        if not items:
            out.append((target, np.ones((), dtype=np.int64)))
            continue
        while len(items) > 1:
            # Greedy connectivity: contract the most-connected pair first
            # (left-to-right folding builds outer products on long chains).
            best = max(
                ((len(set(items[i][0]) & set(items[k][0])), -i, -k) for i in range(len(items)) for k in range(i + 1, len(items))),
            )
            i, k = -best[1], -best[2]
            (ea, a), (eb, b) = items[i], items[k]
            rest = [items[m] for m in range(len(items)) if m not in (i, k)]
            needed = out_edges.union(*[set(e) for e, _ in rest]) if rest else out_edges
            contract = (set(ea) & set(eb)) - needed
            step_terms = 1
            for e in contract:
                step_terms *= a.shape[ea.index(e)]
            if step_terms > _MAX_STEP_TERMS:
                raise _Abstain("einsum step too large")
            letters: dict[str, str] = {}
            for e in ea + eb:
                if e not in letters:
                    if len(letters) >= len(_LETTERS):
                        raise _Abstain("too many product edges")
                    letters[e] = _LETTERS[len(letters)]
            res_edges = [e for e in ea if e not in contract] + [
                e for e in eb if e not in contract and e not in ea
            ]
            _guard_size(tuple((a.shape[ea.index(e)] if e in ea else b.shape[eb.index(e)]) for e in res_edges))
            eq = (
                "".join(letters[e] for e in ea)
                + ","
                + "".join(letters[e] for e in eb)
                + "->"
                + "".join(letters[e] for e in res_edges)
            )
            items = rest + [(res_edges, np.einsum(eq, a, b) % P)]
        edges, arr = items[0]
        out.append((target, _align(tuple(edges), arr, target)))
    return tuple(out)


@_compute.register
def _function(t: Function, kv: list) -> tuple:
    from tensorgrad.functions import _PowerFunction

    sig = t.signature
    if isinstance(sig, _PowerFunction) and isinstance(sig.k, int) and kv:
        # Integer powers are rational functions: evaluate exactly. Binary
        # exponentiation on the TRUE exponent (no Fermat reduction mod P-1,
        # so x^P and x stay distinct); negative k via the P-2 inverse,
        # abstaining on an exact 0 residue.
        (child,) = kv
        out = []
        for j in range(K_TRIALS):
            edges, arr = child[j]
            k = sig.k
            if k < 0:
                if not arr.all():
                    raise _Abstain("0 residue under negative power")
                arr = _pow_mod(arr, P - 2)
                k = -k
            out.append((edges, _pow_mod(arr, k)))
        return tuple(out)
    # Opaque atom: seeded random tensor keyed by the signature and the
    # VALUE-hashes of the inputs (not their structure), so consistency
    # propagates: f(A) == f(B) exactly when fp(A) == fp(B). Analytic
    # identities between atoms are invisible BY DESIGN (sound: under-merge).
    edges = tuple(sorted(t.shape.keys()))
    # Symbolic output sizes join the key so f with output size n and f with
    # output size m stay distinct atoms even when the small draws coincide.
    size_key = tuple(str(sympy.sympify(t.shape[e])) for e in edges)
    out = []
    for j in range(K_TRIALS):
        in_hashes = []
        for child, consumed in zip(kv, sig.inputs):
            c_edges, c_arr = child[j]
            # Consumed edges are anonymous ports (structure() quotients
            # their names): order them by name for a deterministic axis
            # correspondence but keep the NAMES out of the hash — only the
            # broadcast edges (output-visible) keep their names.
            eaten = tuple(sorted(e for e in c_edges if e in consumed))
            bcast = tuple(sorted(e for e in c_edges if e not in consumed))
            aligned = _align(c_edges, c_arr, eaten + bcast)
            in_hashes.append(
                _h(len(eaten), bcast, aligned.shape, np.ascontiguousarray(aligned).tobytes())
            )
        dims = _guard_size(tuple(_dim_of(t.shape[e], t, j) for e in edges))
        arr = _rand_tensor(dims, SEED, j, "fn-atom", _sig_key(sig), tuple(in_hashes), edges, size_key, dims)
        out.append((edges, arr))
    return tuple(out)


def _pow_mod(arr: np.ndarray, e: int) -> np.ndarray:
    result = np.ones_like(arr)
    base = arr % P
    while e:
        if e & 1:
            result = result * base % P
        base = base * base % P
        e >>= 1
    return result


def _sig_key(sig: Any) -> tuple:
    """Deterministic, address-free description of a FunctionSignature:
    class identity plus its simple attributes (sets sorted — never rely on
    frozenset iteration order)."""

    def norm(v: Any) -> Any:
        if isinstance(v, (frozenset, set)):
            return tuple(sorted(map(str, v)))
        if isinstance(v, dict):
            return tuple(sorted((str(k), norm(x)) for k, x in v.items()))
        if isinstance(v, (tuple, list)):
            return tuple(norm(x) for x in v)
        return repr(v)

    # sig.inputs holds the consumed-edge NAMES, which structure() treats as
    # anonymous ports — only their multiplicity is identity. sig.edges are
    # output names (identity-bearing) and stay via the attrs walk.
    attrs = tuple(
        (k, norm(tuple(len(es) for es in v) if k == "inputs" else v))
        for k, v in sorted(sig.__dict__.items())
        if not isinstance(v, Tensor)
    )
    return (type(sig).__module__, type(sig).__qualname__, attrs)


def _compute_affine(t: Any, kv: list) -> tuple:
    from tensorgrad.compiler.affine import indicator_tensor

    edges = tuple(t.shape.keys())
    shape_set, _ = _syms(t)
    # The +16 activity probe enlarges AXIS symbols only: a symbolic window
    # size draws small (it must be exercisable against the axes) but does
    # NOT grow with the tensor — bumping it too would disguise a finite
    # window as a structurally vacuous bound (tril).
    axis_syms = {s.name for e in t.shape.values() for s in sympy.sympify(e).free_symbols}
    out = []
    for j in range(K_TRIALS):

        def sub(x: Any, bump: int = 0, j: int = j) -> int:
            e = sympy.sympify(x)
            a = {
                s: (
                    _small_dim(s.name, j) + (bump if s.name in axis_syms else 0)
                    if s.name in shape_set
                    else _wide_dim(s.name, j)
                )
                for s in e.free_symbols
            }
            v = e.subs(a)
            if not v.is_Integer:
                raise _Abstain(f"affine row constant {x} unresolved")
            return int(v)

        dims = _guard_size(tuple(sub(t.shape[e]) for e in edges))
        pdims = tuple(sub(t.shape[e], 16) for e in edges)  # activity probe size
        grids, pgrids = np.indices(dims), np.indices(pdims)
        rows: list[tuple] = []
        for kind, coeffs, *consts in [("eq", c, k) for c, k in t.rows] + [
            ("range", c, k, x) for c, k, x in t.range_rows
        ]:
            cmap = {edges.index(e): sub(c) for e, c in coeffs.items()}
            acc = sum(c * grids[a] for a, c in cmap.items())
            pacc = sum(c * pgrids[a] for a, c in cmap.items())
            # Every bound must be EXERCISED at the drawn sizes, or provably
            # inert: a bound dormant at dims 2..5 but active at the +16
            # probe (a window wider than the drawn buffer) would leave its
            # constant invisible to the values — windows 0<=q-k<=6 and <=8
            # were bit-identical and cancelled (the reproduced false
            # merge). Bounds dormant at BOTH sizes are structurally vacuous
            # (tril's upper bound q-k <= n-1) and are safe to ignore.
            if kind == "eq":
                kd, kp = sub(consts[0]), sub(consts[0], 16)
                dormant_bad = (not (acc == kd).any() and (pacc == kp).any()) or (
                    not (acc != kd).any() and (pacc != kp).any()
                )
                rows.append(("eq", cmap, kd))
            else:
                kd, xd = sub(consts[0]), sub(consts[1])
                kp, xp = sub(consts[0], 16), sub(consts[1], 16)
                ed, ep = acc + kd, pacc + kp
                dormant_bad = (not (ed < 0).any() and (ep < 0).any()) or (
                    not (ed > xd - 1).any() and (ep > xp - 1).any()
                )
                rows.append(("range", cmap, kd, xd))
            if dormant_bad:
                raise _Abstain("affine bound dormant at drawn dims but active at real sizes")
        arr = np.asarray(indicator_tensor(list(dims), rows).numpy(), dtype=np.int64)
        if arr.size and not arr.any():
            # Empty at the drawn sizes but (typically) nonzero at real
            # sizes: a degenerate zero that collides unequal programs.
            raise _Abstain("affine indicator empty at drawn dims")
        out.append((edges, arr))
    return tuple(out)


_EXTRAS = False


def _ensure_extras() -> None:
    """Register handlers whose imports are heavy (Affine pulls the torch
    oracle) lazily, so plain `import tensorgrad` + default simplify stays
    light."""
    global _EXTRAS
    if _EXTRAS:
        return
    _EXTRAS = True
    try:
        from tensorgrad.compiler.affine import Affine

        _compute.register(Affine, _compute_affine)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The incremental driver + public API
# ---------------------------------------------------------------------------


def _abstained(t: Tensor) -> bool:
    return _FP_ATTR in t.__dict__ and t.__dict__[_FP_ATTR] is None


def _nbytes(vals: tuple) -> int:
    return sum(arr.nbytes for _, arr in vals)


def _lru_get(node: Tensor, pins: dict) -> Optional[tuple]:
    hit = pins.get(id(node))
    if hit is not None:
        return hit[1]
    entry = _ARRAYS.get(id(node))
    if entry is not None:
        _ARRAYS.move_to_end(id(node))
        return entry[1]
    return None


def _vals(t: Tensor) -> Optional[tuple]:
    """Trial arrays for `t`, computing (and caching) any missing subtree
    bottom-up from cached children — O(new nodes), the v2 point. Fresh
    results are pinned for the duration of the walk and merged into the
    byte-bounded LRU at the end."""
    global _ARRAYS_BYTES
    _ensure_extras()
    pins: dict = {}
    stack = [t]
    while stack:
        node = stack[-1]
        if id(node) in pins or id(node) in _ARRAYS or _abstained(node):
            stack.pop()
            continue
        if len(pins) > _PIN_CAP:
            t.__dict__[_FP_ATTR] = None
            return None
        if _compute.dispatch(type(node)) is _compute.dispatch(Tensor):
            # Default (structure-keyed atom): needs no children — compute
            # directly, so a huge opaque subtree costs one canon lookup,
            # and a failing island inside it cannot poison the atom.
            stack.pop()
            try:
                pins[id(node)] = (node, _compute(node, []))
            except Exception:
                node.__dict__[_FP_ATTR] = None
            continue
        kids = _children(node)
        missing = [c for c in kids if id(c) not in pins and id(c) not in _ARRAYS and not _abstained(c)]
        if missing:
            stack.extend(missing)
            continue
        stack.pop()
        kid_vals = []
        for c in kids:
            v = _lru_get(c, pins)
            if v is None:  # child abstained
                node.__dict__[_FP_ATTR] = None
                break
            kid_vals.append(v)
        else:
            try:
                vals = _compute(node, kid_vals)
            except Exception:
                node.__dict__[_FP_ATTR] = None
            else:
                pins[id(node)] = (node, vals)
    result = _lru_get(t, pins)
    for nid, (node, vals) in pins.items():
        if nid not in _ARRAYS:
            _ARRAYS[nid] = (node, vals, _nbytes(vals))
            _ARRAYS_BYTES += _ARRAYS[nid][2]
    while _ARRAYS_BYTES > _ARRAYS_CAP_BYTES and _ARRAYS:
        _, (_, _, nb) = _ARRAYS.popitem(last=False)
        _ARRAYS_BYTES -= nb
    return result


def tree_size(t: Tensor) -> int:
    """Cached tree-size (shared subtrees counted per occurrence). Kept for
    diagnostics; v2's incremental evaluation no longer needs a size budget."""
    hit = t.__dict__.get(_SIZE_ATTR)
    if hit is not None:
        return hit
    stack: list = [(t, False)]
    while stack:
        node, expanded = stack.pop()
        if node.__dict__.get(_SIZE_ATTR) is not None:
            continue
        kids = list(_children(node))
        if expanded:
            node.__dict__[_SIZE_ATTR] = 1 + sum(c.__dict__.get(_SIZE_ATTR, 1) for c in kids)
        else:
            stack.append((node, True))
            stack.extend((c, False) for c in kids if c.__dict__.get(_SIZE_ATTR) is None)
    return t.__dict__[_SIZE_ATTR]


def szfp(t: Tensor) -> Optional[tuple]:
    """Cached value fingerprint of `t` — (weight-symbol support, digest,
    is-zero) — or None when `t` abstains (which opts it out of semantic
    rewrites; the Sum merge falls back to the structural key)."""
    hit = t.__dict__.get(_FP_ATTR, _FP_ATTR)
    if hit is not _FP_ATTR:
        return hit
    try:
        vals = _vals(t)
    except Exception:
        vals = None
    if vals is None:
        t.__dict__[_FP_ATTR] = None
        return None
    parts = []
    zero = True
    for edges, arr in vals:
        order = tuple(sorted(edges))
        aligned = _align(edges, arr, order)
        parts += [order, aligned.shape, np.ascontiguousarray(aligned).tobytes()]
        zero = zero and not arr.any()
    fp = (_syms(t)[1], _h("szfp2", K_TRIALS, SEED, *parts), zero)
    t.__dict__[_FP_ATTR] = fp
    return fp


def is_zero(t: Tensor) -> Optional[bool]:
    """Cached exact-mod-P zero test; None when the node abstains. The zero
    bit is part of the szfp bundle — one evaluation serves both."""
    fp = szfp(t)
    return fp[2] if fp is not None else None
