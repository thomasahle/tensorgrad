"""The linalg technology-mapping peepholes (compiler/peepholes.py).

There is deliberately no F.solve in the language: users write
inverse(A) @ b and the compiler maps it onto torch.linalg.solve. Same for
log(det(A)) -> slogdet. Contracts: the emitted source uses the fused
kernel (and the dead inverse/det line is pruned), values match torch and
the interpreter's edge conventions in BOTH contraction orientations
(non-symmetric A pins them), and the float32 case that motivated slogdet
-- a kernel matrix whose raw det underflows to exactly 0 -- computes the
right log-determinant.
"""

import sympy
import torch

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable

torch.set_num_threads(2)
n = sympy.Symbol("n")
N = 24


def _mk(seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(N, N, generator=g) + N * torch.eye(N)  # well-conditioned, NOT symmetric
    b = torch.randn(N, generator=g)
    return A, b


def _src(prog):
    return next(iter(prog._fn._specializations.values()))._source


def test_solve_peephole_contract_first_edge():
    A = Variable("A", i=n, j=n)
    b = Variable("b", i=n)
    prog = tg.compile(x=F.inverse(A, {"i", "j"}) @ b)
    Av, bv = _mk(0)
    out = prog(dims={n: N}, A=Av.rename("i", "j"), b=bv.rename("i")).x
    src = _src(prog)
    assert "linalg.solve" in src and "linalg.inv" not in src
    torch.testing.assert_close(out.rename(None), torch.linalg.solve(Av, bv))


def test_solve_peephole_contract_second_edge():
    A = Variable("A", i=n, j=n)
    c = Variable("c", j=n)
    prog = tg.compile(x=F.inverse(A, {"i", "j"}) @ c)
    Av, cv = _mk(1)
    out = prog(dims={n: N}, A=Av.rename("i", "j"), c=cv.rename("j")).x
    src = _src(prog)
    assert "linalg.solve" in src and "linalg.inv" not in src
    torch.testing.assert_close(out.rename(None), torch.linalg.solve(Av.T, cv))


def test_solve_matches_interpreter():
    # The peephole must preserve the compiled-vs-interpreted contract.
    from tensorgrad.extras.evaluate import evaluate

    A = Variable("A", i=n, j=n)
    b = Variable("b", i=n)
    expr = F.inverse(A, {"i", "j"}) @ b
    Av, bv = _mk(2)
    ref = evaluate(expr.full_simplify(), {A: Av.rename("i", "j"), b: bv.rename("i")})
    out = tg.compile(x=expr)(dims={n: N}, A=Av.rename("i", "j"), b=bv.rename("i")).x
    torch.testing.assert_close(out.rename(None), ref.rename(None), atol=1e-4, rtol=1e-4)


def test_slogdet_peephole_survives_float32_underflow():
    K = Variable("K", i=n, j=n)
    prog = tg.compile(ld=F.log(F.det(K, {"i", "j"})))
    NN = 400
    Kv = 0.01 * torch.eye(NN) + 0.001 * torch.ones(NN, NN)
    assert torch.linalg.det(Kv).item() == 0.0  # the raw det path is dead here
    out = prog(dims={n: NN}, K=Kv.rename("i", "j")).ld
    src = _src(prog)
    assert "slogdet" in src
    torch.testing.assert_close(out, torch.linalg.slogdet(Kv)[1])


def test_matrix_rhs_not_rewritten_and_still_correct():
    # v1 scope is vector rhs; a matrix rhs keeps the inverse path and stays
    # correct -- misses degrade performance, never correctness.
    A = Variable("A", i=n, j=n)
    B = Variable("B", j=n, k=n)
    prog = tg.compile(x=F.inverse(A, {"i", "j"}) @ B)
    g = torch.Generator().manual_seed(3)
    Av = torch.randn(N, N, generator=g) + N * torch.eye(N)
    Bv = torch.randn(N, N, generator=g)
    out = prog(dims={n: N}, A=Av.rename("i", "j"), B=Bv.rename("j", "k")).x
    torch.testing.assert_close(
        out.align_to("i", "k").rename(None), torch.linalg.solve(Av.T, Bv),
        atol=1e-4, rtol=1e-4,
    )
