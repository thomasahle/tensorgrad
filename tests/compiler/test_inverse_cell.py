"""The matrix-inverse technology-mapping cell (compiler/cells.py).

F.inverse always evaluated in the interpreter; the cell gives it a compiler
lowering onto torch.linalg.inv. Contracts under test: the compiled kernel
matches torch and the interpreter's edge convention (same-name edges
cancel), broadcast batch edges ride along, and the Newton-step composition
H^-1 @ g -- the reason the cell exists -- solves a linear system.
"""

import sympy
import torch

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable

torch.set_num_threads(2)
n, d, b = sympy.symbols("n d b")
N = 9


def _spd(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(*shape, generator=g)
    return A @ A.transpose(-2, -1) + N * torch.eye(shape[-1])


def test_inverse_matches_torch():
    K = Variable("K", i=n, j=n)
    prog = tg.compile(out=F.inverse(K, {"i", "j"}))
    Kv = _spd(N, N)
    out = prog(dims={n: N}, K=Kv.rename("i", "j")).out
    # cancel convention: contracting same-name edges gives the identity, so
    # aligned (i, j) the values are the TRANSPOSED numeric inverse
    torch.testing.assert_close(
        out.align_to("i", "j").rename(None), torch.linalg.inv(Kv).T
    )


def test_inverse_broadcasts_over_batch():
    K = Variable("K", b=b, i=n, j=n)
    prog = tg.compile(out=F.inverse(K, {"i", "j"}))
    Kv = _spd(4, N, N, seed=1)
    out = prog(dims={n: N, b: 4}, K=Kv.rename("b", "i", "j")).out
    torch.testing.assert_close(
        out.align_to("b", "i", "j").rename(None),
        torch.linalg.inv(Kv).transpose(-2, -1),
    )


def test_inverse_contract_cancels():
    # K @ inverse(K) over the shared edges must be (numerically) a scalar
    # trace... no: contracting ONE shared edge yields the identity. This is
    # the convention the Newton step relies on.
    K = Variable("K", i=n, j=n)
    inv = F.inverse(K, {"i", "j"})
    ident = K @ inv.rename(i="k")  # contract j -> (i, k) should be eye
    prog = tg.compile(out=ident)
    Kv = _spd(N, N, seed=2)
    out = prog(dims={n: N}, K=Kv.rename("i", "j")).out
    torch.testing.assert_close(
        out.align_to("i", "k").rename(None), torch.eye(N), atol=1e-4, rtol=1e-4
    )


def test_newton_step_solves():
    # H^-1 @ g == solve(H, g): the composition the newton.py example runs.
    H = Variable("H", di=d, dj=d)
    g = Variable("g", di=d)
    prog = tg.compile(out=(F.inverse(H, {"di", "dj"}) @ g).rename(dj="di"))
    Hv = _spd(N, N, seed=3)
    gv = torch.randn(N, generator=torch.Generator().manual_seed(4))
    out = prog(dims={d: N}, H=Hv.rename("di", "dj"), g=gv.rename("di")).out
    torch.testing.assert_close(
        out.rename(None), torch.linalg.solve(Hv, gv), atol=1e-5, rtol=1e-5
    )
