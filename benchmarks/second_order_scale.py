"""Scale sweep: tensorgrad vs torch.func on Hessians, N in {64, 256, 1024}."""

import sys
import time

import sympy
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(2)
import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable

n = sympy.Symbol("n")


def bench(f, budget_s=2.0):
    f()
    t0 = time.perf_counter()
    reps = 0
    while time.perf_counter() - t0 < budget_s:
        f()
        reps += 1
    return (time.perf_counter() - t0) / reps * 1e3


x = Variable("x", n)
A = Variable("A", i=n, j=n)
f_quad = (x.rename(n="i") @ A) @ x.rename(n="j")
H_quad = f_quad.grad(x, {"n": "di"}).grad(x, {"n": "dj"})
prog_quad = compile_to_callable(H_quad)

f_lse = F.log(F.sum(F.exp(x)))
H_lse = f_lse.grad(x, {"n": "di"}).grad(x, {"n": "dj"})
prog_lse = compile_to_callable(H_lse)

print(f"{'case':8s} {'N':>5s} {'tg ms':>9s} {'torch ms':>9s} {'speedup':>8s}")
for N in (64, 256, 1024):
    xv, Av = torch.randn(N), torch.randn(N, N)
    t_tg = bench(lambda: prog_quad({x: xv.rename("n"), A: Av.rename("i", "j")}, {n: N}))
    with torch.enable_grad():
        t_t = bench(lambda: torch.func.hessian(lambda xx: (xx @ Av @ xx))(xv))
    print(f"{'quad':8s} {N:5d} {t_tg:9.3f} {t_t:9.3f} {t_t / t_tg:7.1f}x")
for N in (64, 256, 1024):
    xv = torch.randn(N)
    t_tg = bench(lambda: prog_lse({x: xv.rename("n")}, {n: N}))
    with torch.enable_grad():
        t_t = bench(lambda: torch.func.hessian(lambda xx: torch.logsumexp(xx, 0))(xv))
    print(f"{'lse':8s} {N:5d} {t_tg:9.3f} {t_t:9.3f} {t_t / t_tg:7.1f}x")
