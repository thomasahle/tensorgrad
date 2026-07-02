"""Tests for Schwartz-Zippel numeric fingerprints (tensorgrad/compiler/szfp.py).

Acceptance table (task #19):
  (A) xW vs xW + 1e-6*yW        -> UNEQUAL (tiny float weights are exact mod P)
  (C) (x+c)W - xW - cW          -> ZERO    (cancellation without expansion)
  (D) Delta(i) vs constant 3    -> UNEQUAL (multi-dim-assignment trials)
  (E) (x+y)W vs xW + yW         -> EQUAL   (SZ soundness: identical polynomials)

Plus: soundness fuzz over testutils.generate_random_tensor_expression
(fingerprints never disagree with ground truth), recursive atom consistency,
and the documented false negative between primitive and fused softmax
gradients (different atoms by design).
"""

import random
import warnings

import pytest
import sympy
import torch

from tensorgrad import Delta, Variable
from tensorgrad import functions as F
from tensorgrad import testutils
from tensorgrad.tensor import Product, Sum
from tensorgrad.compiler.szfp import (
    equal_szfp,
    is_zero_szfp,
    numeric_fingerprint,
    verify_rewrite,
)

i, j = sympy.symbols("i j")
x = Variable("x", i)
y = Variable("y", i)
c = Variable("c", i)
g = Variable("g", i)
W = Variable("W", i, j)


# ---------------------------------------------------------------------------
# The four-case acceptance table
# ---------------------------------------------------------------------------


def test_case_a_tiny_perturbation_unequal():
    # 1e-6 is converted to its exact binary Fraction and reduced mod P, so a
    # perturbation no float evaluation could see is still a different residue.
    assert not equal_szfp(x @ W, x @ W + 1e-6 * (y @ W))


def test_case_c_cancellation_is_zero():
    # (x+c)W - xW - cW cancels without ever expanding the product.
    expr = (x + c) @ W - x @ W - c @ W
    assert is_zero_szfp(expr)
    assert equal_szfp((x + c) @ W, x @ W + c @ W)
    # And a non-cancelling variant is caught:
    assert not is_zero_szfp((x + c) @ W - x @ W)


def test_case_d_dim_dependent_scalar_unequal():
    # Delta(i) is the scalar i; with several random dim assignments (2..5)
    # some trial has i != 3, so it cannot be confused with the constant 3.
    assert not equal_szfp(Delta(i), Delta(3))
    assert equal_szfp(Delta(i), Delta(i))
    assert not is_zero_szfp(Delta(i) - Delta(3))
    assert is_zero_szfp(Delta(i) - Delta(i))


def test_case_e_distributivity_equal():
    assert equal_szfp((x + y) @ W, x @ W + y @ W)
    # ... and remains unequal to a genuinely different expression.
    assert not equal_szfp((x + y) @ W, x @ W)


# ---------------------------------------------------------------------------
# Exact-arithmetic details
# ---------------------------------------------------------------------------


def test_float_weights_are_exact_binary_fractions():
    from fractions import Fraction

    # 0.5 == Fraction(1, 2) exactly; both reduce to the same residue.
    assert equal_szfp(0.5 * x, Fraction(1, 2) * x)
    assert not equal_szfp(0.5 * x, Fraction(1, 3) * x)


def test_fraction_weights_modular_inverse():
    from fractions import Fraction

    expr = Fraction(1, 3) * x + Fraction(2, 3) * x
    assert equal_szfp(expr, x)
    assert is_zero_szfp(expr - x)


def test_integer_pow_and_modular_inverse():
    # pow(x, -1) is a modular inverse: x * (1/x) == 1 (an order-1 Ones).
    inv = F.pow(x, -1)
    prod = x * inv  # elementwise -> ones vector over i
    ones_i = Delta(i, "i")
    assert equal_szfp(prod, ones_i)
    # pow composition: (x^2)^-1 == x^-2
    assert equal_szfp(F.pow(F.pow(x, 2), -1), F.pow(x, -2))


def test_fingerprints_consistent_across_calls():
    # Variable/atom randomness is keyed by content, not program identity, so
    # the same tensor fingerprints identically alone or alongside others.
    fp_alone = numeric_fingerprint([x @ W])[0]
    fp_pair = numeric_fingerprint([x @ W, y @ W])
    assert fp_pair[0] == fp_alone
    assert fp_pair[1] != fp_alone
    assert isinstance(fp_alone, int)


# ---------------------------------------------------------------------------
# Atoms: recursive consistency and the documented false-negative class
# ---------------------------------------------------------------------------


def test_atom_recursive_consistency():
    e = F.exp(x)
    # The exp atom feeds back into the exactly-evaluated layer:
    assert equal_szfp(e * e, F.pow(e, 2))
    assert is_zero_szfp(F.relu(x) - F.relu(x))
    assert equal_szfp(F.softmax(x, ["i"]), F.softmax(x, ["i"]))
    # Different atoms stay different:
    assert not equal_szfp(F.exp(x), F.exp(y))
    assert not equal_szfp(F.relu(x), F.tanh(x))


def test_rational_identity_over_atoms():
    # The gradient of the *primitive* softmax loss equals the hand-written
    # s*(g - <g,s>) form. This is a pure rational-function identity over the
    # exp atoms (uses pow(-1), pow(-2), and heavy cancellation), so the
    # fingerprint proves it WITHOUT expansion and without knowing exp.
    e = F.exp(x)
    Z = e @ Delta(i, "i")
    s_prim = e * F.pow(Z, -1)
    grad = (s_prim @ g).grad(x).full_simplify()
    hand = (s_prim * (g - (g @ s_prim))).full_simplify()
    assert equal_szfp(grad, hand)
    assert verify_rewrite(grad, hand)


def test_softmax_primitive_vs_fused_false_negative():
    """DOCUMENTED FALSE NEGATIVE (by design of the atoms formulation).

    The fused gradient is expressed in softmax(x) atoms (ReduceNode), the
    primitive gradient in exp(x) and pow atoms. They are equal only through
    the *analytic* identity softmax(x) = exp(x)/sum(exp(x)), which the
    fingerprint deliberately does not know: each atom kind gets independent
    random values. So equal_szfp says UNEQUAL even though the expressions
    are semantically equal. This is the same blindness class as syntactic
    methods; #17 must whitelist atom-folding rewrites (exp/sum -> softmax)
    rather than rely on verify_rewrite for them.
    """
    s_fused = F.softmax(x, ["i"])
    e = F.exp(x)
    s_prim = e * F.pow(e @ Delta(i, "i"), -1)

    # Semantically equal, but different atoms -> unequal fingerprints:
    assert not equal_szfp(s_fused, s_prim)

    grad_fused = (s_fused @ g).grad(x).full_simplify()
    grad_prim = (s_prim @ g).grad(x).full_simplify()
    assert not equal_szfp(grad_fused, grad_prim)  # false negative, by design
    assert not verify_rewrite(grad_prim, grad_fused)  # -> needs a whitelisted rule in #17

    # Within each atom formulation the fingerprint is exact and stable:
    assert equal_szfp(grad_fused, (s_fused @ g).grad(x).full_simplify())
    assert equal_szfp(grad_prim, (s_prim @ g).grad(x).full_simplify())


# ---------------------------------------------------------------------------
# Soundness fuzz: fingerprints never disagree with ground truth
# ---------------------------------------------------------------------------


def _var_from_string_edges(name, edges):
    """testutils.generate_random_tensor_expression predates the sympy-Symbol
    Variable API; shim string edge lists to symbol kwargs (dim symbol = edge
    name, matching the shared a/b/c dims the generator assumes)."""
    return Variable(name, **{e: sympy.Symbol(e) for e in edges})


def _structural_shuffle(t, rng: random.Random):
    """A structurally different but semantically identical tensor: recursively
    permute Sum terms and Product factors."""
    if isinstance(t, Sum):
        idx = list(range(len(t.terms)))
        rng.shuffle(idx)
        return Sum([_structural_shuffle(t.terms[k], rng) for k in idx], [t.weights[k] for k in idx])
    if isinstance(t, Product):
        factors = [_structural_shuffle(f, rng) for f in t.factors]
        rng.shuffle(factors)
        return Product(factors)
    return t


def test_fuzz_soundness(monkeypatch):
    monkeypatch.setattr(testutils, "Variable", _var_from_string_edges)
    warnings.filterwarnings("ignore", message="Named tensors")

    n_pairs = 0
    for it in range(40):
        random.seed(it)
        torch.manual_seed(it)
        expr, tval, _variables = testutils.generate_random_tensor_expression(4)
        # Ground truth: with continuous random variable values, the torch
        # value is (exactly) zero iff the expression is the zero polynomial.
        truly_zero = bool((tval.rename(None) == 0).all())
        rng = random.Random(it)

        # Equal pairs: fingerprints of semantically equal forms MUST agree.
        assert equal_szfp(expr, expr.simplify())
        assert equal_szfp(expr, _structural_shuffle(expr, rng))
        assert equal_szfp(expr + expr, 2 * expr)
        assert is_zero_szfp(expr - expr)
        n_pairs += 4

        # Ground-truth-dependent pairs: never disagree with reality.
        assert is_zero_szfp(expr) == truly_zero
        assert equal_szfp(expr, 2 * expr) == truly_zero  # 2e == e iff e == 0
        n_pairs += 2
    assert n_pairs >= 200


# ---------------------------------------------------------------------------
# verify_rewrite API (for #17)
# ---------------------------------------------------------------------------


def test_verify_rewrite_api():
    assert verify_rewrite((x + y) @ W, x @ W + y @ W)
    assert verify_rewrite(x @ W, x @ W)
    assert not verify_rewrite(x @ W, y @ W)
    assert not verify_rewrite(x @ W, x @ W + 1e-6 * (y @ W))
    # Shape mismatch is an immediate False, not an error.
    assert not verify_rewrite(x, W)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
