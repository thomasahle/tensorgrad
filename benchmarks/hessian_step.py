"""Hessian assembly of x^T A x at n=2048: tensorgrad vs torch.func vs jax.

The founding-benchmark pattern as a suite row. The Hessian of the
quadratic form is LITERALLY A + A^T; tensorgrad's

    H = f.grad(x).grad(x)

simplifies to that closed form and compiles to a two-statement program
that never reads x (tests/test_second_order.py pins the ceiling). Both
runtime-AD frameworks must instead replay autodiff -- forward-over-reverse,
n passes materializing the n x n Hessian element by element -- and no
backend fusion helps, because the backend is fusing the wrong algorithm.
Measured at n=2048 this is ~8x over torch.func and ~20x over jax.

The "step" recomputes the full Hessian from a fresh A view each time (the
assembly IS the workload; there is no training loop to hide it in).

Importing this file runs the correctness gate: tg's H equals
torch.func.hessian's to rtol 1e-4 (and jax.hessian's, when installed).

Stresses: second-derivative rewriting, closed-form recovery, and the
asymptotic argument of the paper's evaluation.
"""

import sys

import torch
from sympy import Symbol

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable

torch.set_num_threads(2)

BENCH_NAME = "hessian (x^T A x)"

N = 2048

_gen = torch.Generator().manual_seed(0)
AV = torch.randn(N, N, generator=_gen)
XV = torch.randn(N, generator=_gen)

# ------------------------------------------------------------------ tensorgrad
n = Symbol("n")
x = Variable("x", n)
A = Variable("A", i=n, j=n)
f = (x.rename(n="i") @ A) @ x.rename(n="j")
H = f.grad(x, {"n": "di"}).grad(x, {"n": "dj"})  # -> A + A^T, closed form


def make_tg_step():
    prog = tg.compile(H=H)
    # The compiled program's ONLY input is A: x cancelled out of the closed
    # form, so the input set itself states the theorem (prog.inputs == ['A']).
    assert prog.inputs == ["A"], prog.inputs

    def step():
        return prog(dims={n: N}, A=AV.rename("i", "j")).H

    return step


# ----------------------------------------------------------------------- torch
def _torch_f(xv):
    return xv @ AV @ xv


def make_torch_step():
    hess = torch.func.hessian(_torch_f)

    def step():
        with torch.enable_grad():
            return hess(XV)

    return step


# ------------------------------------------------------------------------- jax
def make_jax_step():
    import jax
    import jax.numpy as jnp

    Aj = jnp.asarray(AV.numpy())
    xj = jnp.asarray(XV.numpy())
    hess = jax.jit(jax.hessian(lambda v: v @ Aj @ v))

    def step():
        out = hess(xj)
        out.block_until_ready()
        return out

    return step


# -------------------------------------------------------------- correctness gate
def _gate():
    ref = AV + AV.T
    out = make_tg_step()()
    torch.testing.assert_close(out.align_to("di", "dj").rename(None), ref, rtol=1e-4, atol=1e-4)
    with torch.enable_grad():
        out_t = torch.func.hessian(_torch_f)(XV)
    torch.testing.assert_close(out_t, ref, rtol=1e-4, atol=1e-4)
    try:
        import jax  # noqa: F401
    except ImportError:
        return
    out_j = make_jax_step()()
    assert abs(float(out_j[0, 1]) - float(ref[0, 1])) < 1e-3


_gate()
print(f"[{BENCH_NAME}] correctness gate passed", file=sys.stderr)
