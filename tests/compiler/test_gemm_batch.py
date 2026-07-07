"""Shared-operand GEMM batching (compiler/gemm_batch.py).

Contracts: k matmuls reading the same operand collapse to one stacked
einsum plus free select views (values exact vs unbatched); the size
ceiling keeps memory-critical programs untouched; gradients through the
batched graph stay correct (the pass runs after consolidation on every
compiled program, so mingpt-family training exercises it constantly --
the demo gate in examples/ is the integration test).
"""

import sympy
import torch

import pytest

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import gemm_batch

torch.set_num_threads(2)


@pytest.fixture(autouse=True)
def _enable_batching():
    """The pass defaults OFF (measured regression on CPU eager; see the
    module); these tests exercise the machinery explicitly."""
    gemm_batch.GEMM_BATCHING = True
    yield
    gemm_batch.GEMM_BATCHING = False
n, d, f = sympy.symbols("n d f")
N, D, FF = 32, 16, 8


def _mk():
    g = torch.Generator().manual_seed(0)
    xv = torch.randn(N, D, generator=g)
    Ws = [torch.randn(D, FF, generator=g) for _ in range(3)]
    return xv, Ws


def test_three_projections_batch_to_one_gemm():
    x = Variable("x", n, d)
    Ws = [Variable(f"W{i}", d=d, f=f) for i in range(3)]
    prog = tg.compile(**{f"o{i}": x @ W for i, W in enumerate(Ws)})
    xv, Wv = _mk()
    feeds = {f"W{i}": Wv[i].rename("d", "f") for i in range(3)}
    out = prog(dims={n: N, d: D, f: FF}, x=xv, **feeds)
    src = next(iter(prog._fn._specializations.values()))._source
    assert "torch.stack" in src
    assert src.count("torch.mm(") + src.count("torch.bmm(") + src.count("torch.matmul(") <= 2
    for i in range(3):
        got = getattr(out, f"o{i}").align_to("n", "f").rename(None)
        torch.testing.assert_close(got, xv @ Wv[i])


def test_batched_values_match_unbatched_exactly():
    x = Variable("x", n, d)
    Ws = [Variable(f"V{i}", d=d, f=f) for i in range(3)]
    outs = {f"o{i}": F.sum(F.tanh(x @ W)) for i, W in enumerate(Ws)}
    xv, Wv = _mk()
    feeds = {f"V{i}": Wv[i].rename("d", "f") for i in range(3)}

    prog_on = tg.compile(**outs)
    r_on = prog_on(dims={n: N, d: D, f: FF}, x=xv, **feeds)
    gemm_batch.GEMM_BATCHING = False
    try:
        prog_off = tg.compile(**{k + "x": v for k, v in outs.items()})
        r_off = prog_off(dims={n: N, d: D, f: FF}, x=xv, **feeds)
    finally:
        gemm_batch.GEMM_BATCHING = True
    for i in range(3):
        torch.testing.assert_close(getattr(r_on, f"o{i}"), getattr(r_off, f"o{i}x"))


def test_size_ceiling_leaves_large_groups_alone():
    big = gemm_batch._SIZE_CEILING  # elements; 3 * (N*D) rows must exceed it
    NB = big  # (NB,) vectors: stacking 2 exceeds the ceiling
    m = sympy.Symbol("m")
    x = Variable("x", m)
    a = Variable("a", m)
    b = Variable("b", m)
    prog = tg.compile(p=F.sum(x * a), q=F.sum(x * b))  # not matmuls anyway
    # the real assertion: a group whose stacked size exceeds the ceiling
    # keeps separate GEMMs
    dbig = sympy.Symbol("dbig")
    y = Variable("y", n, dbig)
    W1 = Variable("Wa", dbig=dbig, f=f)
    W2 = Variable("Wb", dbig=dbig, f=f)
    prog2 = tg.compile(u=y @ W1, v=y @ W2)
    DB = gemm_batch._SIZE_CEILING // FF + 1  # each W: DB*FF > ceiling/2
    g = torch.Generator().manual_seed(1)
    yv = torch.randn(4, DB, generator=g)
    w1 = torch.randn(DB, FF, generator=g)
    w2 = torch.randn(DB, FF, generator=g)
    out = prog2(dims={n: 4, dbig: DB, f: FF}, y=yv, Wa=w1.rename("dbig", "f"), Wb=w2.rename("dbig", "f"))
    src = next(iter(prog2._fn._specializations.values()))._source
    assert "torch.stack" not in src
    torch.testing.assert_close(out.u.align_to("n", "f").rename(None), yv @ w1, rtol=1e-4, atol=1e-4)
