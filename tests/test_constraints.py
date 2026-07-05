"""Tests for Variable value constraints (Variable.with_eq_constraint).

A constraint is an EQUATION written in tensorgrad itself: with_eq_constraint(lhs, rhs)
declares that lhs (an expression mentioning the variable) always equals rhs. simplify
replaces any subnetwork isomorphic to lhs by rhs. The classic example is the simplex
fact sum_v y[b,v] = 1 (one-hot targets, probability rows), which is what makes the
cross-entropy softmax Hessian come out target-free; but orthogonality, unit norms
etc. are the same mechanism.

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


def simplex(var: Variable, edge: str) -> Variable:
    """Entries along `edge` sum to one."""
    return var.with_eq_constraint(F.sum(var, dim=(edge,)), 1)


def _one_hot(B, V, dtype=torch.float32):
    torch.manual_seed(0)
    yt = torch.zeros(B, V, dtype=dtype)
    yt[torch.arange(B), torch.randint(V, (B,))] = 1.0
    return yt.rename("b", "v")


# ---------------------------------------------------------------------------
# The core algebra rule: sum over the constrained edge = 1
# ---------------------------------------------------------------------------


def test_sum_over_constrained_edge():
    y = simplex(Variable("y", b, v), "v")
    assert F.sum(y, dim=("v",)).simplify() == Delta(b, "b")


def test_sum_order_one_variable_is_scalar_one():
    p = simplex(Variable("p", v), "v")
    # The scalar 1 is the empty product
    assert F.sum(p).simplify() == Product([])


def test_sum_over_all_edges():
    # sum_b sum_v y[b, v] = sum_b 1 = |b|, the order-0 Delta
    y = simplex(Variable("y", b, v), "v")
    assert F.sum(y).simplify() == Delta(b)


def test_sum_over_wrong_edge_does_not_fire():
    y = simplex(Variable("y", b, v), "v")
    s = F.sum(y, dim=("b",)).simplify()
    assert s.depends_on(y)
    assert s != Delta(v, "v")


def test_unconstrained_sum_does_not_fire():
    y = Variable("y", b, v)
    assert F.sum(y, dim=("v",)).simplify().depends_on(y)


def test_elementwise_factor_survives():
    # sum_v (y * s) is NOT sum_v y: the constrained edge is contracted with s's edge
    # through a hyperedge, not with a lone ones-vector, so nothing may cancel.
    y = simplex(Variable("y", b, v), "v")
    s = Variable("s", b, v)
    expr = F.sum(y * s, dim=("v",)).simplify()
    assert expr.depends_on(y) and expr.depends_on(s)


def test_scalar_factor_pulls_through():
    # (sum_v y[b,v]) * z[b] = z[b]
    y = simplex(Variable("y", b, v), "v")
    z = Variable("z", b)
    assert (F.sum(y, dim=("v",)) * z).simplify() == z


def test_two_constrained_variables():
    y1 = simplex(Variable("y1", b, v), "v")
    y2 = simplex(Variable("y2", b, v), "v")
    expr = F.sum(y1, dim=("v",)) * F.sum(y2, dim=("v",))
    assert expr.simplify() == Delta(b, "b")
    # Only the summed variable drops; the other one stays.
    expr2 = (F.sum(y1, dim=("v",)) * y2).simplify()
    assert not expr2.depends_on(y1) and expr2.depends_on(y2)


def test_constraint_survives_rename():
    y = simplex(Variable("y", b, v), "v")
    yr = y.rename(v="w", b="batch")
    assert F.sum(yr, dim=("w",)).simplify() == Delta(b, "batch")
    # The unconstrained edge, renamed, still does not fire.
    assert F.sum(yr, dim=("batch",)).simplify().depends_on(y)


def test_constraint_survives_substitute():
    t = Variable("t", b, v)
    expr = F.sum(t, dim=("v",))
    y = simplex(Variable("y", b, v), "v")
    assert expr.substitute(t, y).simplify() == Delta(b, "b")


# ---------------------------------------------------------------------------
# Generality: other equations, same mechanism
# ---------------------------------------------------------------------------


def test_orthogonal_columns():
    # W^T W = I: projecting two vectors through W preserves their dot product.
    d, i = symbols("d i")
    W0 = Variable("W", d=d, i=i)
    W = W0.with_eq_constraint(W0 @ W0.rename(i="j"), Delta(i, "i", "j"))
    x = Variable("x", i=i)
    z = Variable("z", j=i)
    expr = ((W @ x) @ (W.rename(i="j") @ z)).simplify()
    assert not expr.depends_on(W)
    assert expr == (x.rename(i="j") @ z).simplify()
    # Numerics with an actual orthonormal matrix (columns orthonormal needs d >= i).
    D, I = 5, 3
    Wt = torch.linalg.qr(torch.randn(D, I)).Q.rename("d", "i")
    xt, zt = torch.randn(I).rename("i"), torch.randn(I).rename("j")
    raw = evaluate((W @ x) @ (W.rename(i="j") @ z), {W: Wt, x: xt, z: zt}, {d: D, i: I})
    torch.testing.assert_close(
        raw.rename(None), (xt.rename(None) * zt.rename(None)).sum(), rtol=1e-5, atol=1e-6
    )


def test_unit_norm():
    i = symbols("i")
    u0 = Variable("u", i)
    u = u0.with_eq_constraint(F.sum(u0 * u0, dim=("i",)), 1)
    assert F.sum(u * u, dim=("i",)).simplify() == Product([])
    # But a plain (unsquared) sum over u must NOT fire.
    assert F.sum(u, dim=("i",)).simplify().depends_on(u)


# ---------------------------------------------------------------------------
# Structure: isomorphism, symmetries, validation
# ---------------------------------------------------------------------------


def test_constrained_not_isomorphic_to_unconstrained():
    y = Variable("y", b, v)
    yc = simplex(y, "v")
    assert y != yc
    assert yc == simplex(Variable("y", b, v), "v")
    assert "with_eq_constraint" in repr(yc)


def test_with_symmetries_preserves_constraints():
    n = symbols("n")
    M0 = Variable("M", i=n, j=n)
    M = simplex(simplex(M0, "i"), "j").with_symmetries("i j")
    assert len(M._constraints) == 2  # one sum equation per edge
    assert F.sum(M, dim=("i",)).simplify() == Delta(n, "j")


def test_constraint_must_cover_symmetry_orbit():
    n = symbols("n")
    with pytest.raises(ValueError):
        simplex(Variable("M", i=n, j=n).with_symmetries("i j"), "i")
    with pytest.raises(ValueError):
        simplex(Variable("M", i=n, j=n), "i").with_symmetries("i j")


def test_constraint_validation():
    y = Variable("y", b, v)
    z = Variable("z", b)
    with pytest.raises(ValueError):  # lhs does not mention y
        y.with_eq_constraint(F.sum(z, dim=("b",)), 1)
    with pytest.raises(ValueError):  # free edges of lhs and rhs differ
        y.with_eq_constraint(F.sum(y, dim=("v",)), Ones(v))
    with pytest.raises(ValueError):  # rhs not smaller: no termination guarantee
        y.with_eq_constraint(F.sum(y, dim=("v",)), F.sum(Variable("w", b, v), dim=("v",)))


def test_grad_is_unchanged_by_constraint():
    # Constraints describe values, not the optimization manifold: grad is untouched.
    y = Variable("y", b, v)
    yc = simplex(Variable("y", b, v), "v")
    new = {"b": "b2", "v": "v2"}
    assert yc.grad(yc, new_names=new).simplify() == y.grad(y, new_names=new).simplify()


# ---------------------------------------------------------------------------
# Numerics: the rewrite agrees with evaluation on actual simplex data
# ---------------------------------------------------------------------------


def test_evaluate_matches_rule():
    B, V = 3, 5
    y = simplex(Variable("y", b, v), "v")
    yt = _one_hot(B, V)
    expr = F.sum(y, dim=("v",))
    raw = evaluate(expr, {y: yt}, {b: B, v: V})
    simplified = evaluate(expr.simplify(), {y: yt}, {b: B, v: V})
    torch.testing.assert_close(raw.align_to("b").rename(None), torch.ones(B))
    torch.testing.assert_close(simplified.align_to("b").rename(None), torch.ones(B))


def test_compiled_program_drops_input():
    from tensorgrad.compiler import compile_to_callable

    B, V = 3, 5
    y = simplex(Variable("y", b, v), "v")
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
