"""Tests for Variable value constraints (Variable.with_constraint).

The only constraint so far is "simplex": the entries along the constrained edge sum
to one (e.g. one-hot targets or probability distributions). simplify uses it in one
place: contracting the constrained edge against an all-ones vector (the order-1
Delta that F.sum produces) drops the variable, leaving all-ones on its other edges.

Constraints describe the VALUES a variable will hold, not the space it is optimized
over, so grad treats a constrained variable exactly like an unconstrained one.
"""

import pytest
import torch
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad.extras.evaluate import evaluate
from tensorgrad.tensor import Delta, Ones, Product, Variable

b, v, c = symbols("b v c")


def _one_hot(B, V, dtype=torch.float32):
    torch.manual_seed(0)
    yt = torch.zeros(B, V, dtype=dtype)
    yt[torch.arange(B), torch.randint(V, (B,))] = 1.0
    return yt.rename("b", "v")


# ---------------------------------------------------------------------------
# The core algebra rule: sum over the constrained edge = 1
# ---------------------------------------------------------------------------


def test_sum_over_constrained_edge():
    y = Variable("y", b, v).with_constraint("simplex", "v")
    assert F.sum(y, dim=("v",)).simplify() == Delta(b, "b")


def test_sum_order_one_variable_is_scalar_one():
    p = Variable("p", v).with_constraint("simplex", "v")
    # The scalar 1 is the empty product
    assert F.sum(p).simplify() == Product([])


def test_sum_over_all_edges():
    # sum_b sum_v y[b, v] = sum_b 1 = |b|, the order-0 Delta
    y = Variable("y", b, v).with_constraint("simplex", "v")
    assert F.sum(y).simplify() == Delta(b)


def test_sum_over_wrong_edge_does_not_fire():
    y = Variable("y", b, v).with_constraint("simplex", "v")
    s = F.sum(y, dim=("b",)).simplify()
    assert s.depends_on(y)
    assert s != Delta(v, "v")


def test_unconstrained_sum_does_not_fire():
    y = Variable("y", b, v)
    assert F.sum(y, dim=("v",)).simplify().depends_on(y)


def test_elementwise_factor_survives():
    # sum_v (y * s) is NOT sum_v y: the constrained edge is contracted with s's edge
    # through a hyperedge, not with a lone ones-vector, so nothing may cancel.
    y = Variable("y", b, v).with_constraint("simplex", "v")
    s = Variable("s", b, v)
    expr = F.sum(y * s, dim=("v",)).simplify()
    assert expr.depends_on(y) and expr.depends_on(s)


def test_scalar_factor_pulls_through():
    # (sum_v y[b,v]) * z[b] = z[b]
    y = Variable("y", b, v).with_constraint("simplex", "v")
    z = Variable("z", b)
    assert (F.sum(y, dim=("v",)) * z).simplify() == z


def test_two_constrained_variables():
    y1 = Variable("y1", b, v).with_constraint("simplex", "v")
    y2 = Variable("y2", b, v).with_constraint("simplex", "v")
    expr = F.sum(y1, dim=("v",)) * F.sum(y2, dim=("v",))
    assert expr.simplify() == Delta(b, "b")
    # Only the summed variable drops; the other one stays.
    expr2 = (F.sum(y1, dim=("v",)) * y2).simplify()
    assert not expr2.depends_on(y1) and expr2.depends_on(y2)


def test_constraint_survives_rename():
    y = Variable("y", b, v).with_constraint("simplex", "v")
    yr = y.rename(v="w", b="batch")
    assert F.sum(yr, dim=("w",)).simplify() == Delta(b, "batch")
    # The unconstrained edge, renamed, still does not fire.
    assert F.sum(yr, dim=("batch",)).simplify().depends_on(y)


def test_constraint_survives_substitute():
    t = Variable("t", b, v)
    expr = F.sum(t, dim=("v",))
    y = Variable("y", b, v).with_constraint("simplex", "v")
    assert expr.substitute(t, y).simplify() == Delta(b, "b")


# ---------------------------------------------------------------------------
# Structure: isomorphism, symmetries, validation
# ---------------------------------------------------------------------------


def test_constrained_not_isomorphic_to_unconstrained():
    y = Variable("y", b, v)
    yc = y.with_constraint("simplex", "v")
    assert y != yc
    assert yc == Variable("y", b, v).with_constraint("simplex", "v")
    assert "with_constraint" in repr(yc)


def test_with_symmetries_preserves_constraints():
    n = symbols("n")
    M = Variable("M", i=n, j=n).with_constraint("simplex", "i", "j").with_symmetries("i j")
    assert ("simplex", "i") in M._constraints and ("simplex", "j") in M._constraints
    assert F.sum(M, dim=("i",)).simplify() == Delta(n, "j")


def test_constraint_must_cover_symmetry_orbit():
    n = symbols("n")
    with pytest.raises(ValueError):
        Variable("M", i=n, j=n).with_symmetries("i j").with_constraint("simplex", "i")
    with pytest.raises(ValueError):
        Variable("M", i=n, j=n).with_constraint("simplex", "i").with_symmetries("i j")


def test_constraint_validation():
    y = Variable("y", b, v)
    with pytest.raises(ValueError):
        y.with_constraint("banana", "v")
    with pytest.raises(ValueError):
        y.with_constraint("simplex", "nope")
    with pytest.raises(ValueError):
        y.with_constraint("simplex")


def test_grad_is_unchanged_by_constraint():
    # Constraints describe values, not the optimization manifold: grad is untouched.
    y = Variable("y", b, v)
    yc = Variable("y", b, v).with_constraint("simplex", "v")
    new = {"b": "b2", "v": "v2"}
    assert yc.grad(yc, new_names=new).simplify() == y.grad(y, new_names=new).simplify()


# ---------------------------------------------------------------------------
# Numerics: the rewrite agrees with evaluation on actual simplex data
# ---------------------------------------------------------------------------


def test_evaluate_matches_rule():
    B, V = 3, 5
    y = Variable("y", b, v).with_constraint("simplex", "v")
    yt = _one_hot(B, V)
    expr = F.sum(y, dim=("v",))
    raw = evaluate(expr, {y: yt}, {b: B, v: V})
    simplified = evaluate(expr.simplify(), {y: yt}, {b: B, v: V})
    torch.testing.assert_close(raw.align_to("b").rename(None), torch.ones(B))
    torch.testing.assert_close(simplified.align_to("b").rename(None), torch.ones(B))


def test_compiled_program_drops_input():
    from tensorgrad.compiler import compile_to_callable

    B, V = 3, 5
    y = Variable("y", b, v).with_constraint("simplex", "v")
    fn = compile_to_callable(F.sum(y, dim=("v",)).simplify())
    assert "y" not in fn.input_names
    out = fn({}, {b: B, v: V})
    torch.testing.assert_close(out.align_to("b").rename(None), torch.ones(B))


# ---------------------------------------------------------------------------
# The pointwise pow split (functions.py) that lets the CE-softmax Hessian reach
# the ones-contraction: pow(Delta-joined product, k) distributes over the
# diagonally-joined groups. Check it against plain evaluation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [-1, -2, 2])
def test_pointwise_pow_split_numerics(k):
    x = Variable("x", b, v)
    e = F.exp(x)
    s = e * F.pow(F.sum(e, dim=("v",), keepdims=True), -1)  # primitives softmax
    expr = F.sum(F.pow(s, k), dim=("v",))
    B, V = 3, 4
    torch.manual_seed(1)
    xt = torch.randn(B, V).rename("b", "v")
    raw = evaluate(expr, {x: xt}, {b: B, v: V})
    split = evaluate(expr.simplify({"expand": True}), {x: xt}, {b: B, v: V})
    torch.testing.assert_close(
        raw.align_to("b").rename(None), split.align_to("b").rename(None)
    )
