"""Schwartz-Zippel numeric fingerprints on the compiler IR.

Probabilistic equality / zero testing of tensor expressions WITHOUT
expanding them: lower to the compiler IR (`lower_program`) and evaluate the
DAG bottom-up in EXACT arithmetic mod a prime P, at k independent trials.
Each trial uses

  * a random small dim-assignment (every sympy dim symbol -> 2..5), so
    dim-dependent scalars (Delta(n) = n, sum weights in n, ...) are
    distinguished from numeric constants across trials, and
  * seeded random integer tensors mod P for every Variable, keyed only by
    (seed, trial, var name, resolved dims) — NOT by program identity — so
    fingerprints agree across separately lowered programs.

Polynomial / rational structure (einsums, linear combinations, integer
powers, Fraction/float weights, affine constraint indicators, structural
constants) is evaluated exactly:

  * pairwise einsum steps in int64 with a mod-P reduction between steps
    (P < 2^19, so products < 2^38 and up to 2^24 summed terms per step fit
    in int64 with room to spare — guarded by an explicit assertion),
  * Fraction / float weights via modular inverse of the (exact binary)
    denominator,
  * integer pow(x, k) elementwise by modular exponentiation; k < 0 uses
    Fermat (x^(P-2) inverse); hitting an exact 0 residue triggers a retry
    of the whole trial with a fresh salt ("resample on 0").

Every non-polynomial node — MapNode exp/log/relu/tanh/erf/gt0/sign/abs/
equal, fractional pow, ReduceNode softmax/log_softmax/max/argmax,
GatherNode gather/one_hot — is NEVER evaluated. Its output is a seeded
random tensor keyed by (op, params, numeric fingerprint of its inputs), so
consistency propagates recursively: exp(A) == exp(B) exactly when the
fingerprints of A and B agree, and pow(exp(x), 2) == exp(x) * exp(x)
because the atom feeds back into the exactly-evaluated layer.

Guarantees (the "atoms formulation"):
  * SZ-sound for the multilinear + rational skeleton: if two expressions
    are equal as polynomials/rational functions over the atoms, the
    fingerprints ALWAYS agree; if they differ, they collide with
    probability <= deg/P per trial (~1e-5 here), independently k times.
  * False-negatives-only on transcendental identities: two formulations
    that are equal only through an analytic identity of the atoms (e.g.
    softmax(x) vs exp(x)/sum(exp(x)); abs(x) vs relu(x)+relu(-x)) get
    different atoms and hence different fingerprints BY DESIGN. This is
    the same class of blindness as any syntactic method.

Retries: division by zero mod P, non-positive resolved dims, or
non-square reshape totals abort the trial and re-run it with a new salt.
Because atom values and variable values are functions of *values*, two
semantically equal programs trigger retries identically, so the salt
sequence stays consistent. Fingerprints are comparable across separate
`numeric_fingerprint` calls with the same (k, seed) whenever the same salt
sequence succeeds (always, except adversarial persistent-zero division).

Public API:
  numeric_fingerprint(tensors, k=3, seed=0) -> list[int]
  equal_szfp(a, b, k=3, seed=0) -> bool
  is_zero_szfp(t, k=3, seed=0) -> bool
  verify_rewrite(before, after, k=3, seed=0) -> bool   (for #17)
"""

import hashlib
import string
from fractions import Fraction

import numpy as np
import sympy

from tensorgrad.compiler.ir import (
    ConstNode,
    EinsumNode,
    GatherNode,
    InputNode,
    LinearNode,
    MapNode,
    Node,
    ReduceNode,
    SDPAFwdNode,
    SDPABwdNode,
    toposort,
)
from tensorgrad.compiler.lower import lower_program

__all__ = ["numeric_fingerprint", "equal_szfp", "is_zero_szfp", "verify_rewrite", "P"]

# Mersenne prime 2^19 - 1. Pairwise einsum of residues: products < 2^38,
# and we allow at most 2^24 summed terms per output element per step, so
# intermediate int64 sums stay < 2^62 < 2^63. Exact, no overflow.
P = 524287
_MAX_SUM_TERMS = 1 << 24
_MAX_RETRIES = 500
_DIM_LO, _DIM_HI = 2, 5  # random dim assignments (inclusive)

_LETTERS = string.ascii_letters


class _Retry(Exception):
    """Abort the current trial and re-run with a fresh salt."""


# ---------------------------------------------------------------------------
# Deterministic hashing / seeded randomness (never python hash(); stable
# across processes and PYTHONHASHSEED values).
# ---------------------------------------------------------------------------


def _h(*parts) -> int:
    """64-bit deterministic hash of a tuple of printable/bytes parts."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        if isinstance(p, bytes):
            h.update(b"B")
            h.update(p)
        else:
            h.update(repr(p).encode())
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big")


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(_h(*key))


def _rand_tensor(shape, *key) -> np.ndarray:
    arr = _rng(*key).integers(0, P, size=tuple(shape), dtype=np.int64)
    return arr


def _vhash(arr: np.ndarray) -> int:
    """Numeric fingerprint of an evaluated array (value identity)."""
    return _h(arr.shape, np.ascontiguousarray(arr).tobytes())


# ---------------------------------------------------------------------------
# Exact scalars mod P
# ---------------------------------------------------------------------------


def _inv(a: int) -> int:
    a %= P
    if a == 0:
        raise _Retry("modular inverse of 0")
    return pow(a, P - 2, P)


def _fraction_residue(fr: Fraction) -> int:
    return fr.numerator % P * _inv(fr.denominator) % P


def _scalar_residue(w, assign: dict, ctx) -> int:
    """Exact residue of a scalar weight. Fractions/ints/floats are exact
    (floats via their exact binary Fraction); sympy expressions are
    substituted with the trial's dim assignment; anything irrational left
    over becomes a seeded random atom keyed by its canonical string."""
    if isinstance(w, int):
        return w % P
    if isinstance(w, Fraction):
        return _fraction_residue(w)
    if isinstance(w, float):
        return _fraction_residue(Fraction(w))
    e = sympy.sympify(w)
    if assign:
        e = e.subs(assign)
    if e.is_Integer:
        return int(e) % P
    if e.is_Rational:
        return _fraction_residue(Fraction(int(e.p), int(e.q)))
    if e.is_Float:
        return _fraction_residue(Fraction(float(e)))
    # Irrational scalar (e.g. sqrt(2), pi): treat as a transcendental atom.
    return _h(ctx, "scalar-atom", sympy.srepr(e)) % P


def _dim(d, assign: dict) -> int:
    """Resolve a (possibly symbolic) dim to a positive int, else retry."""
    if isinstance(d, (int, np.integer)):
        v = int(d)
    else:
        e = sympy.sympify(d).subs(assign)
        if not e.is_number:
            raise RuntimeError(f"Unresolved dim {d!r} after assignment {assign}")
        if not e.is_Integer:
            raise _Retry(f"non-integer dim {d} = {e}")
        v = int(e)
    if v < 1:
        raise _Retry(f"non-positive dim {d} = {v}")
    return v


def _node_syms(node: Node) -> set:
    """All sympy symbols this node's evaluation depends on."""
    syms = set()

    def add(x):
        try:
            e = sympy.sympify(x)
        except (sympy.SympifyError, TypeError, ValueError):
            return
        if hasattr(e, "free_symbols"):
            syms.update(e.free_symbols)

    for d in node.dims:
        add(d)
    if isinstance(node, EinsumNode):
        for d in node.wire_dims:
            add(d)
        add(node.weight)
        for coeffs, const in node.constraints:
            add(const)
            for _, c in coeffs:
                add(c)
    elif isinstance(node, ConstNode):
        for p in node.params:
            add(p)
    elif isinstance(node, LinearNode):
        for w in node.weights:
            if not isinstance(w, (int, float, Fraction)):
                add(w)
    return syms


# ---------------------------------------------------------------------------
# Exact einsum evaluation (pairwise, mod between steps)
# ---------------------------------------------------------------------------


def _contract(arrA, subsA, arrB, subsB, keep, wire_sz):
    """One einsum step: contract arrA (and optionally arrB) down to the
    wires in `keep`, mod P. Repeated wires within one operand are diagonals
    (np.einsum handles this natively)."""
    seen = dict.fromkeys(list(subsA) + (list(subsB) if subsB is not None else []))
    out = [w for w in seen if w in keep]
    n_sum = 1
    for w in seen:
        if w not in keep:
            n_sum *= wire_sz[w]
    if n_sum > _MAX_SUM_TERMS:
        raise RuntimeError(f"szfp: einsum step sums {n_sum} terms; exceeds exact int64 budget")
    if len(seen) > len(_LETTERS):
        raise RuntimeError("szfp: einsum step has more than 52 wires")
    let = {w: _LETTERS[i] for i, w in enumerate(seen)}
    lhs = "".join(let[w] for w in subsA)
    if subsB is not None:
        lhs += "," + "".join(let[w] for w in subsB)
    eq = lhs + "->" + "".join(let[w] for w in out)
    if subsB is None:
        res = np.einsum(eq, arrA)
    else:
        res = np.einsum(eq, arrA, arrB)
    return res % P, out


def _row_const_floors(nodes) -> dict:
    """Per-symbol-name dim floor: |concrete row constant| + 2 for every
    symbol sizing a wire of that row (offset rows like [i = o + 5] need the
    participating dims to exceed the offset to have any solution)."""
    floors: dict[str, int] = {}
    for n in nodes:
        for coeffs, const in getattr(n, "constraints", ()) or ():
            c = sympy.sympify(const)
            if not c.free_symbols and abs(int(c)) > 0:
                need = abs(int(c)) + 2
                for w, _ in coeffs:
                    d = sympy.sympify(n.wire_dims[w])
                    for s in d.free_symbols:
                        floors[s.name] = max(floors.get(s.name, 0), need)
    return floors


def _row_deficits(nodes, assign) -> dict:
    """Symbols whose dims must rise for every affine row to have a solution
    at `assign`. A row sum(c_w * i_w) == const with i_w in [0, size_w) is
    satisfiable only if const lies in the reachable interval; rows whose
    constant is SYMBOLIC (a concat part at offset `length`) cannot be
    floored up front — the needed size depends on the draw. When a row's
    constant falls outside the interval, lift the wires whose size symbols
    do not feed the constant (lifting a constant's own symbol chases its
    tail: [l == b - length] must widen buf, not length)."""
    need: dict[str, int] = {}
    for n in nodes:
        for coeffs, const in getattr(n, "constraints", ()) or ():
            cs = sympy.sympify(const)
            cval = cs.subs(assign)
            if not cval.is_number or not cval.is_integer:
                continue
            target = int(cval)
            lo, hi, symbolic = 0, 0, False
            for w, cf in coeffs:
                cfv = sympy.sympify(cf).subs(assign)
                if not cfv.is_number:
                    symbolic = True
                    break
                span = int(cfv) * (int(sympy.sympify(n.wire_dims[w]).subs(assign)) - 1)
                lo, hi = lo + min(0, span), hi + max(0, span)
            if symbolic or lo <= target <= hi:
                continue
            const_syms = {s.name for s in cs.free_symbols}
            for w, _ in coeffs:
                for s in sympy.sympify(n.wire_dims[w]).free_symbols:
                    if s.name not in const_syms:
                        need[s.name] = max(need.get(s.name, 0), abs(target) + 2)
    return need


def _draw_assign(nodes, syms, ctx) -> dict:
    """Random dims per symbol (keyed by NAME, independent of the program, so
    separately lowered expressions over the same symbols agree), redrawn
    with lifted floors until every affine row is interval-satisfiable. The
    empty-indicator retry in _eval_einsum remains the exactness backstop."""
    floors = _row_const_floors(nodes)
    assign: dict = {}
    for _ in range(4):
        assign = {
            s: (lo := max(_DIM_LO, floors.get(s.name, 0)))
            + _h(ctx, "dim", s.name, floors.get(s.name, 0)) % (max(_DIM_HI, lo + 3) - lo + 1)
            for s in syms
        }
        deficits = _row_deficits(nodes, assign)
        if not deficits:
            break
        for name, val in deficits.items():
            floors[name] = max(floors.get(name, 0), val)
    return assign


def _eval_einsum(node: EinsumNode, vals, assign, ctx) -> np.ndarray:
    wire_sz = {w: _dim(d, assign) for w, d in enumerate(node.wire_dims)}
    operands = [(vals[id(op)], list(subs)) for op, subs in zip(node.ops, node.in_subs)]

    # Affine constraint rows -> exact 0/1 indicator operands over their wires.
    for coeffs, const in node.constraints:
        wires = [w for w, _ in coeffs]
        cs = [sympy.sympify(c).subs(assign) for _, c in coeffs]
        cs = [Fraction(int(c.p), int(c.q)) if c.is_Rational else Fraction(int(c)) for c in cs]
        ce = sympy.sympify(const).subs(assign)
        cval = Fraction(int(ce.p), int(ce.q)) if ce.is_Rational else Fraction(int(ce))
        grids = np.indices([wire_sz[w] for w in wires])
        s = sum((c * g for c, g in zip(cs, grids)), start=np.zeros((), dtype=object))
        ind = (s == cval).astype(np.int64)
        if not ind.any():
            # An affine row with NO solution at these random dims (e.g. a
            # window offset larger than the randomly-drawn buffer) zeroes
            # this node and everything downstream — a degenerate value that
            # collides unequal programs (measured: consolidate merged a loss
            # with its unnormalized sibling and silently dropped a 1/384).
            # Real dims satisfy the row; redraw the trial.
            raise _Retry("empty affine indicator (unsatisfiable at random dims)")
        operands.append((ind, wires))

    # Make out_subs distinct: a repeated output wire is a diagonal embedding;
    # link the duplicates through explicit identity operands (mirrors lower.py).
    out_subs = list(node.out_subs)
    next_wire = len(node.wire_dims)
    seen = set()
    for pos, w in enumerate(out_subs):
        if w in seen:
            nw = next_wire
            next_wire += 1
            wire_sz[nw] = wire_sz[w]
            eye = np.eye(wire_sz[w], dtype=np.int64)
            operands.append((eye, [w, nw]))
            out_subs[pos] = nw
        else:
            seen.add(w)

    wres = _scalar_residue(node.weight, assign, ctx)
    out_shape = [wire_sz[w] for w in out_subs]

    if not operands:
        # Pure broadcast of the weight (e.g. a Ones vector: Delta order 1).
        return np.full(out_shape, wres, dtype=np.int64) if out_shape else np.array(wres, dtype=np.int64)

    def keep_after(i):
        k = set(out_subs)
        for _, s in operands[i + 1 :]:
            k |= set(s)
        return k

    acc, acc_subs = operands[0]
    acc, acc_subs = _contract(acc, acc_subs, None, None, keep_after(0), wire_sz)
    for i in range(1, len(operands)):
        arr, subs = operands[i]
        acc, acc_subs = _contract(acc, acc_subs, arr, subs, keep_after(i), wire_sz)

    # acc_subs is now a distinct subset of out_subs. Multiply the weight,
    # reorder to output order, and broadcast wires that never touched an
    # operand (pure-output wires).
    acc = acc * wres % P
    present = [w for w in out_subs if w in set(acc_subs)]
    if acc_subs != present:
        let = {w: _LETTERS[i] for i, w in enumerate(acc_subs)}
        acc = np.einsum(
            "".join(let[w] for w in acc_subs) + "->" + "".join(let[w] for w in present), acc
        )
    if len(present) != len(out_subs):
        present_set = set(present)
        acc = acc.reshape([wire_sz[w] if w in present_set else 1 for w in out_subs])
        acc = np.broadcast_to(acc, out_shape)
    return acc


# ---------------------------------------------------------------------------
# Node evaluation
# ---------------------------------------------------------------------------


def _eval_const(node: ConstNode, assign, ctx) -> np.ndarray:
    kind = node.kind
    if kind == "scalar":
        return np.array(_scalar_residue(node.params[0], assign, ctx), dtype=np.int64)
    if kind == "zero":
        return np.zeros([_dim(d, assign) for d in node.dims], dtype=np.int64)
    if kind == "delta":
        size, k = node.params
        n = _dim(size, assign)
        out = np.zeros([n] * k, dtype=np.int64)
        idx = np.arange(n)
        out[(idx,) * k] = 1
        return out
    if kind == "conv":
        # C[i, j, o] = 1 iff i == j + o (stride 1), matching the affine row
        # {input: 1, kernel: -1, output: -1} = 0 in lower.py.
        w_in, k_size, w_out = (_dim(d, assign) for d in node.params)
        i, j, o = np.indices((w_in, k_size, w_out))
        return (i == j + o).astype(np.int64)
    if kind == "reshape":
        sizes = [_dim(d, assign) for d in node.dims]
        total = 1
        for s in sizes:
            total *= s
        half = int(total**0.5)
        if half * half != total:
            # The reshape identity only exists when the (random) dims are
            # consistent; retry until the assignment satisfies it.
            raise _Retry(f"reshape total {total} not a perfect square")
        return np.eye(half, dtype=np.int64).reshape(sizes)
    raise NotImplementedError(f"szfp: const kind {kind!r}")


def _eval_pow_int(arr: np.ndarray, k: int) -> np.ndarray:
    """Elementwise x^k mod P, exact. Negative k via Fermat (x^(P-1) = 1);
    0^negative retries the trial ("resample on 0"). 0^0 = 1."""
    out = np.empty(arr.shape, dtype=np.int64)
    flat_in, flat_out = arr.reshape(-1), out.reshape(-1)
    e_nonzero = k % (P - 1)  # valid exponent for any nonzero residue
    for i, v in enumerate(flat_in.tolist()):
        v %= P
        if v == 0:
            if k < 0:
                raise _Retry("pow of 0 with negative exponent")
            flat_out[i] = 1 if k == 0 else 0
        else:
            flat_out[i] = pow(v, e_nonzero, P)
    return out


def _as_int_exponent(k):
    if isinstance(k, (int, np.integer)):
        return int(k)
    if isinstance(k, Fraction) and k.denominator == 1:
        return k.numerator
    if isinstance(k, sympy.Integer):
        return int(k)
    return None  # fractional -> atom


def _eval_map(node: MapNode, vals, assign, ctx) -> np.ndarray:
    aligned = []
    for opnd, perm in zip(node.ops, node.perms):
        arr = vals[id(opnd)]
        if perm != tuple(range(len(perm))):
            arr = np.transpose(arr, perm)
        aligned.append(arr)
    if node.op == "pow":
        k = _as_int_exponent(node.params[0])
        if k is not None:
            return _eval_pow_int(aligned[0], k)
    # Non-polynomial elementwise op (exp/log/relu/tanh/erf/gt0/sign/abs/
    # equal/fractional pow): a seeded random atom keyed by the op and the
    # numeric fingerprints of its (aligned) inputs. Recursive consistency.
    key = ("atom-map", node.op, repr(node.params), tuple(_vhash(a) for a in aligned))
    return _rand_tensor(aligned[0].shape, *key)


def _eval_node(node: Node, vals, assign, ctx) -> np.ndarray:
    if isinstance(node, InputNode):
        dims = tuple(_dim(d, assign) for d in node.dims)
        return _rand_tensor(dims, ctx, "var", node.var_name, dims)
    if isinstance(node, ConstNode):
        return _eval_const(node, assign, ctx)
    if isinstance(node, EinsumNode):
        return _eval_einsum(node, vals, assign, ctx)
    if isinstance(node, LinearNode):
        acc = np.zeros([_dim(d, assign) for d in node.dims], dtype=np.int64)
        for term, perm, w in zip(node.terms, node.perms, node.weights):
            arr = vals[id(term)]
            if perm != tuple(range(len(perm))):
                arr = np.transpose(arr, perm)
            acc = (acc + arr * _scalar_residue(w, assign, ctx)) % P
        return acc
    if isinstance(node, MapNode):
        return _eval_map(node, vals, assign, ctx)
    if isinstance(node, GatherNode):
        dims = tuple(_dim(d, assign) for d in node.dims)
        key = ("atom-gather", node.op, node.axis, tuple(_vhash(vals[id(o)]) for o in node.ops), dims)
        return _rand_tensor(dims, *key)
    if isinstance(node, ReduceNode):
        dims = tuple(_dim(d, assign) for d in node.dims)
        key = ("atom-reduce", node.op, node.axes, tuple(_vhash(vals[id(o)]) for o in node.ops), dims)
        return _rand_tensor(dims, *key)
    if isinstance(node, (SDPAFwdNode, SDPABwdNode)):
        dims = tuple(_dim(d, assign) for d in node.dims)
        key = ("atom-sdpa", type(node).__name__, node.scale, node.has_mask,
               getattr(node, "which", -1), node.perms, getattr(node, "res_perm", ()),
               tuple(_vhash(vals[id(o)]) for o in node.ops), dims)
        return _rand_tensor(dims, *key)
    raise NotImplementedError(f"szfp: no evaluation for {type(node).__name__}")


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------


def _eval_trial(outs, ctx):
    """Evaluate all outputs at one random point. Returns, per output,
    (sorted edge names, resolved dims, array with axes sorted by edge name)."""
    nodes = toposort([n for n, _ in outs])
    syms = set()
    for n in nodes:
        syms |= _node_syms(n)
    assign = _draw_assign(nodes, syms, ctx)
    vals: dict[int, np.ndarray] = {}
    for n in nodes:
        vals[id(n)] = _eval_node(n, vals, assign, ctx)
    results = []
    for node, order in outs:
        arr = vals[id(node)]
        if order:
            perm = sorted(range(len(order)), key=lambda a: order[a])
            arr = np.transpose(arr, perm)
        results.append((tuple(sorted(order)), arr.shape, np.ascontiguousarray(arr)))
    return results


def _evaluate(tensors, k, seed):
    """Lower and evaluate at k random points. Returns [trial][output] ->
    (sorted_edges, dims, array)."""
    _, outs = lower_program(list(tensors))
    trials = []
    for trial in range(k):
        for salt in range(_MAX_RETRIES):
            try:
                trials.append(_eval_trial(outs, (seed, trial, salt)))
                break
            except _Retry:
                continue
        else:
            raise RuntimeError(
                f"szfp: trial {trial} exceeded {_MAX_RETRIES} retries "
                "(persistent division by zero or unsatisfiable dim constraints)"
            )
    return trials


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def numeric_fingerprint(tensors, k: int = 3, seed: int = 0) -> list[int]:
    """One 64-bit fingerprint per tensor: the hash of its exact mod-P values
    at k seeded random evaluation points (random dims 2..5, random variable
    tensors, random atoms for non-polynomial ops), with output axes
    canonicalized by edge name. Equal fingerprints => the tensors agree at
    all k points => equal (in the atoms formulation) with high probability;
    fingerprints of semantically equal expressions are ALWAYS equal for the
    polynomial/rational skeleton."""
    trials = _evaluate(tensors, k, seed)
    fps = []
    for i in range(len(list(tensors))):
        parts = []
        for tvals in trials:
            edges, dims, arr = tvals[i]
            parts += [edges, dims, arr.tobytes()]
        fps.append(_h("szfp", k, seed, *parts))
    return fps


def equal_szfp(a, b, k: int = 3, seed: int = 0) -> bool:
    """Probabilistic equality test. True => a == b with high probability
    (error <= (deg/P)^k on the polynomial skeleton). False is definitive
    for the polynomial skeleton but may be a false negative when a and b
    are equal only through an analytic identity of non-polynomial atoms."""
    if a.shape != b.shape:
        return False
    fa, fb = numeric_fingerprint([a, b], k=k, seed=seed)
    return fa == fb


def is_zero_szfp(t, k: int = 3, seed: int = 0) -> bool:
    """Probabilistic zero test: True iff t evaluates to 0 mod P at all k
    random points. Detects cancellation without expansion."""
    trials = _evaluate([t], k, seed)
    return all(not tvals[0][2].any() for tvals in trials)


def verify_rewrite(before, after, k: int = 3, seed: int = 0) -> bool:
    """Rewrite-verification oracle for the factoring pass (#17): returns
    True when `after` is (with high probability) equivalent to `before`.
    A False from a purely algebraic rewrite means the rewrite is wrong;
    a False from a rewrite that folds atoms (e.g. exp/sum -> softmax) is
    the documented false-negative class and needs a whitelisted rule."""
    return equal_szfp(before, after, k=k, seed=seed)
