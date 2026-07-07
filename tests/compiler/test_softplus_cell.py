"""The softplus stabilization cell (task #53, forward half).

log(1 + exp(x)) overflows float32 at x ~ 89; the cell pattern-matches the
composition and emits torch.nn.functional.softplus. The DERIVATIVE's
sigmoid (exp/(1+exp), smeared through the derivative's delta-einsum) is a
separate pattern -- documented xfail below until the sigmoid cell lands.
"""

import pytest
import sympy
import torch

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable

torch.set_num_threads(2)
n = sympy.Symbol("n")
X = torch.tensor([-3.0, 0.0, 5.0, 100.0])


def test_softplus_fires_and_survives_overflow():
    x = Variable("x", n)
    prog = tg.compile(y=F.log(1 + F.exp(x)))
    out = prog(dims={n: 4}, x=X).y
    src = next(iter(prog._fn._specializations.values()))._source
    assert "softplus" in src
    torch.testing.assert_close(out.rename(None), torch.nn.functional.softplus(X))
    assert out.rename(None)[-1].item() == 100.0  # raw composition gives inf


def test_both_term_orders_match():
    x = Variable("x", n)
    prog = tg.compile(y=F.log(F.exp(x) + 1))
    out = prog(dims={n: 4}, x=X).y
    assert "softplus" in next(iter(prog._fn._specializations.values()))._source
    torch.testing.assert_close(out.rename(None), torch.nn.functional.softplus(X))


def test_gradient_correct_at_moderate_x():
    x = Variable("x", n)
    g = F.sum(F.log(1 + F.exp(x))).grad(x)
    prog = tg.compile(g=g)
    xm = torch.tensor([-3.0, 0.0, 5.0, 20.0])
    out = prog(dims={n: 4}, x=xm).g
    torch.testing.assert_close(out.rename(None), torch.sigmoid(xm), atol=1e-6, rtol=1e-6)


@pytest.mark.xfail(reason="derivative's sigmoid (exp/(1+exp) through the "
                   "derivative delta-einsum) not yet pattern-matched; "
                   "overflows at x ~ 89 -- the sigmoid cell is the follow-up")
def test_gradient_survives_overflow():
    x = Variable("x", n)
    g = F.sum(F.log(1 + F.exp(x))).grad(x)
    prog = tg.compile(g=g)
    out = prog(dims={n: 4}, x=X).g
    assert torch.isfinite(out.rename(None)).all()
