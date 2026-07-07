"""Expectation.gaussian: the diagonal-Gaussian constructor.

The covariance tensor for z ~ N(mu, diag(std^2)) -- structural Deltas
wired through an order-3 copy tensor -- used to be an expert-only
construction; now a classmethod builds it. Contracts: identical resolution to
the manual construction, correct second moments vs closed form (including
broadcast std and 0-dim isotropic std), and loud errors on misuse.
"""

import pytest
import sympy
import torch

import tensorgrad.functions as F
from tensorgrad import Delta, Variable
from tensorgrad.extras.evaluate import evaluate
from tensorgrad.extras.expectation import Expectation

torch.set_num_threads(2)
b, l = sympy.symbols("b l")
B, L = 3, 4


def _vals(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, L, generator=g), torch.rand(B, L, generator=g) + 0.5)


def test_std_matches_manual_covariance_construction():
    z = Variable("z", b, l)
    mu = Variable("mu", b, l)
    s = Variable("s", b, l)
    # E[z (x) z'] under N(mu, diag(s^2)); second moment = mu mu' + diag(s^2)
    z2 = z * z.rename(b="b_")  # keeps l hadamard, batch outer -- fine for the test
    e_std = Expectation.gaussian(F.sum(z2), z, mu, std=s).full_simplify()
    # manual construction (what examples/vae.py used to do)
    s2 = (s * s).rename(b="bi", l="li")
    covar = s2 @ Delta(b, "b", "b2", "bi") @ Delta(l, "l", "l2", "li")
    e_man = Expectation(F.sum(z2), z, mu, covar, {"b": "b2", "l": "l2"}).full_simplify()
    muv, sv = _vals()
    r1 = evaluate(e_std, {mu: muv.rename("b", "l"), s: sv.rename("b", "l")})
    r2 = evaluate(e_man, {mu: muv.rename("b", "l"), s: sv.rename("b", "l")})
    torch.testing.assert_close(r1, r2)


def test_std_second_moment_closed_form():
    z = Variable("z", l)
    mu = Variable("mu", l)
    s = Variable("s", l)
    e = Expectation.gaussian(F.sum(z * z), z, mu, std=s).full_simplify()
    muv, sv = torch.randn(L), torch.rand(L) + 0.5
    out = evaluate(e, {mu: muv.rename("l"), s: sv.rename("l")})
    torch.testing.assert_close(out, (muv**2 + sv**2).sum())


def test_std_broadcasts_missing_edges_and_scalar():
    z = Variable("z", b, l)
    mu = Variable("mu", b, l)
    s_l = Variable("s", l)  # shared across the batch edge
    e = Expectation.gaussian(F.sum(z * z), z, mu, std=s_l).full_simplify()
    muv, _ = _vals(1)
    sv = torch.rand(L) + 0.5
    out = evaluate(e, {mu: muv.rename("b", "l"), s_l: sv.rename("l")})
    torch.testing.assert_close(out, (muv**2).sum() + B * (sv**2).sum())

    s0 = Variable("s0")  # 0-dim: isotropic
    e0 = Expectation.gaussian(F.sum(z * z), z, mu, std=s0).full_simplify()
    out0 = evaluate(e0, {mu: muv.rename("b", "l"), s0: torch.tensor(0.7)})
    torch.testing.assert_close(out0, (muv**2).sum() + B * L * 0.7**2)


def test_std_misuse_errors():
    z = Variable("z", l)
    with pytest.raises(ValueError, match="subset"):
        Expectation.gaussian(z, z, std=Variable("t", b))
