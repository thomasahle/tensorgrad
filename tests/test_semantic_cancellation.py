"""The semantic-cancellation tier (tensorgrad/fingerprint.py + the
sz_cancel knob in simplify_sum): value-zero terms drop and value-equal
term groups with cancelling weights collapse — the cancellations that
would otherwise require expanding exactly the right factors, discovered
WITHOUT expansion by exact evaluation mod P.

The headline: under the COMPILER preset (normalize_args — heavy syntactic
passes off), the softmax shift-invariance zero now falls out of ordinary
Sum simplification, which previously required either full default
simplify or the dedicated zerograd pass.
"""

import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler.runtime import normalize_args
from tensorgrad.tensor import Rename, Sum, Zero
from tensorgrad import fingerprint as fp


def _unwrap(t):
    while isinstance(t, Rename):
        t = t.tensor
    return t


def test_differently_grouped_equal_terms_cancel():
    """(x+y)@w and x@w + y@w are structurally different; the Counter merge
    cannot see them. With sz_cancel their difference collapses to Zero."""
    n, m = symbols("n m")
    x = Variable("x", b=n, d=m)
    y = Variable("y", b=n, d=m)
    w = Variable("w", d=m, o=m)
    grouped = (x + y) @ w
    expanded = Sum([x @ w, y @ w])
    diff = Sum([grouped, expanded], [1, -1])
    out = diff.simplify(normalize_args())
    assert isinstance(_unwrap(out), Zero), f"did not cancel: {out}"


def test_zero_valued_term_drops():
    """A term that is identically zero by algebra (x - x wrapped in a
    product) disappears from the sum without symbolic derivation."""
    n = symbols("n")
    x = Variable("x", i=n)
    w = Variable("w", i=n)
    zeroish = (x + (-1) * x) * w  # value-zero, structurally non-trivial
    keep = w * w
    out = Sum([zeroish, keep]).simplify(normalize_args())
    assert fp.is_zero(Sum([out, keep], [1, -1]))


def test_shift_invariance_zero_under_compiler_preset():
    """The zerograd motivating case, now falling out of plain simplify
    under the COMPILER preset: the attention key-bias gradient is zero by
    softmax shift invariance."""
    n, d = symbols("n d")
    q = Variable("q", seq=n, hs=d)
    k = Variable("k", key=n, hs=d)
    v = Variable("v", key=n, dv=d)
    bk = Variable("bk", hs=d)
    scores = F.dot(q, k + bk.rename(hs="hs"), dim="hs")
    loss = F.sum(F.dot(F.softmax(scores, dim="key"), v, dim="key"))
    g = loss.grad(bk).simplify(normalize_args())
    assert isinstance(_unwrap(g), Zero), f"shift-invariance zero not found: {type(g)}"


def test_symmetric_variable_transpose_cancels():
    """The value key must not under-merge what the structural key handles:
    for a declared-symmetric x, x - x.T cancels. Requires szfp to draw x
    from the symmetric subspace (InputNode.sym), the correct Schwartz-Zippel
    domain for a declared-symmetric variable."""
    n = symbols("n")
    x = Variable("x", i=n, j=n).with_symmetries("i j")
    xt = x.rename(i="j", j="i")
    assert fp.szfp(x) == fp.szfp(xt), "symmetric draw must make x and x.T fp-equal"
    out = Sum([x, xt], [1, -1]).simplify(normalize_args())
    assert isinstance(_unwrap(out), Zero)
    # and the asymmetric counterpart must NOT collapse
    y = Variable("y", i=n, j=n)
    out2 = Sum([y, y.rename(i="j", j="i")], [1, -1]).simplify(normalize_args())
    assert not isinstance(_unwrap(out2), Zero)


def test_value_equal_terms_merge_weights():
    """Non-cancelling value-equal terms merge into one slot with the summed
    weight (value-CSE, same trade as IR consolidation)."""
    n, m = symbols("n m")
    x = Variable("x", b=n, d=m)
    y = Variable("y", b=n, d=m)
    w = Variable("w", d=m, o=m)
    s = Sum([(x + y) @ w, Sum([x @ w, y @ w])], [1, 1]).simplify(normalize_args())
    ref = Sum([(x + y) @ w], [2]).simplify(normalize_args())
    assert fp.is_zero(Sum([s, ref], [1, -1]))


def test_non_cancelling_sums_untouched():
    n = symbols("n")
    x = Variable("x", i=n)
    y = Variable("y", i=n)
    out = Sum([x * x, y * y]).simplify(normalize_args())
    assert isinstance(out, Sum) and len(out.terms) == 2


def test_dim_symbols_do_not_collide():
    """Regression (review finding): dims draw from a tiny range, so two
    symbol NAMES can draw identical dims in all trials — deterministically.
    Delta(h)*y - Delta(v)*y is (h-v)*y, NOT zero; likewise Delta(k)*y vs
    5*y. Weight-only symbols must draw from a high-entropy range."""
    from tensorgrad.tensor import Delta

    n = symbols("n")
    y = Variable("y", i=n)
    # sweep many name pairs: at the old 2..5 draw, ~1/64 of pairs collided
    names = [f"s{i}" for i in range(20)] + list("abcdefghkuvwz")
    syms = symbols(" ".join(names))
    for s1, s2 in zip(syms, syms[1:]):
        out = Sum([Delta(s1) * y, Delta(s2) * y], [1, -1]).simplify(normalize_args())
        assert not isinstance(_unwrap(out), Zero), f"({s1})-({s2}) collapsed to Zero"
    for s in syms:
        out = Sum([Delta(s) * y, y], [1, -5]).simplify(normalize_args())
        assert not isinstance(_unwrap(out), Zero), f"({s})-5 collapsed to Zero"


def test_offset_window_term_not_dropped():
    """Regression (review finding): an affine indicator empty at the small
    random dims (offset >= the drawn buffer) must not fingerprint as zero —
    the mask term would silently vanish from every compiled Sum."""
    n, m = symbols("n m")
    x = Variable("x", length=m)
    y = Variable("y", seq=n)
    shifted = x @ F.window(length=m, seq=n, start=6)  # empty at dims 2..5
    out = Sum([shifted, y]).simplify(normalize_args())
    assert isinstance(_unwrap(out), Sum), f"mask term dropped: {out}"
    assert len(_unwrap(out).terms) == 2


def test_saturated_window_rows_do_not_collide():
    """Regression (v2 review): a range row WIDER than the drawn dims
    excludes nothing, so differently-sized windows (0<=q-k<=6 vs <=8) were
    bit-identical at every trial and cancelled. Every row must bite."""
    from tensorgrad.compiler.affine import affine_ineq

    n, w1, w2 = symbols("n w1 w2")
    a7 = affine_ineq({"q": 1, "k": -1}, 0, 6, q=n, k=n)
    a9 = affine_ineq({"q": 1, "k": -1}, 0, 8, q=n, k=n)
    out = Sum([a7, a9], [1, -1]).simplify(normalize_args())
    assert not isinstance(_unwrap(out), Zero), "different windows collapsed"
    # symbolic window sizes: row-constant symbols must draw small (shape
    # class), and saturation must abstain — never collide
    s1 = affine_ineq({"q": 1, "k": -1}, 0, w1, q=n, k=n)
    s2 = affine_ineq({"q": 1, "k": -1}, 0, w2, q=n, k=n)
    out2 = Sum([s1, s2], [1, -1]).simplify(normalize_args())
    assert not isinstance(_unwrap(out2), Zero), "symbolic windows collapsed"


def test_vanishing_weight_polynomial_does_not_cancel():
    """Regression (v2 review): shape symbols draw from a 4-point domain, so
    a weight divisible by (n-2)(n-3)(n-4)(n-5) is 0 at every trial while
    nonzero for all real n >= 6. A nonzero weight with 0 residue abstains."""
    n = symbols("n")
    x = Variable("X", i=n)
    y = Variable("Y", i=n)
    w = (n - 2) * (n - 3) * (n - 4) * (n - 5)
    out = Sum([x, y], [w, w]).simplify(normalize_args())
    assert not isinstance(_unwrap(out), Zero), "vanishing weight collapsed sum"


def test_variable_kwargs_order_is_not_identity():
    """Regression (v2 review): Variable('x', i=n, j=m) and
    Variable('x', j=m, i=n) are the same tensor and must fingerprint equal
    (draws keyed by sorted edges, not kwargs order)."""
    n, m = symbols("n m")
    a = Variable("x", i=n, j=m)
    b = Variable("x", j=m, i=n)
    assert a == b
    assert fp.szfp(a) == fp.szfp(b)
    out = Sum([a, b], [1, -1]).simplify(normalize_args())
    assert isinstance(_unwrap(out), Zero)


def test_function_output_size_is_identity():
    """Regression (v2 review, core structure() bug): f(x) with output size
    alpha and f(x) with output size beta were STRUCTURALLY equal (the
    output-port junction omitted size_key), so the plain structural merge
    combined genuinely different tensors — no fingerprints involved."""
    from tensorgrad.tensor import function

    n, alpha, beta = symbols("n alpha beta")
    x = Variable("x", i=n)
    fa = function("f", {"o": alpha}, (x, "i"))
    fb = function("f", {"o": beta}, (x, "i"))
    assert fa != fb, "different output sizes must not be structurally equal"
    assert fp.szfp(fa) != fp.szfp(fb)


def test_function_consumed_edge_names_are_anonymous():
    """Regression (v2 review): structure() quotients consumed input edge
    names to anonymous ports, so the fingerprint atom key must not leak
    them — name-fixed-isomorphic function applications fingerprint equal."""
    from tensorgrad.tensor import function

    n = symbols("n")
    xa = Variable("x", i=n).rename(i="a")
    xb = Variable("x", i=n).rename(i="b")
    fa = function("f", {"o": n}, (xa, "a"))
    fb = function("f", {"o": n}, (xb, "b"))
    assert fa == fb, "consumed-edge names are not identity"
    assert fp.szfp(fa) == fp.szfp(fb), "atom key leaks consumed-edge names"
    out = Sum([fa, fb], [1, -1]).simplify(normalize_args())
    assert isinstance(_unwrap(out), Zero)


def test_default_interactive_simplify_unaffected():
    """The knob is off by default: interactive simplify keeps its exact
    current behavior (no fingerprint work on plain .simplify())."""
    n, m = symbols("n m")
    x = Variable("x", b=n, d=m)
    y = Variable("y", b=n, d=m)
    w = Variable("w", d=m, o=m)
    diff = Sum([(x + y) @ w, Sum([x @ w, y @ w])], [1, -1])
    diff.simplify()  # default preset: knob off
    # no fingerprint work may happen on the default path (spot-check root)
    assert "_szfp_v2" not in diff.__dict__
