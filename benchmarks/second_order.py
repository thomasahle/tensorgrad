"""Second-order derivatives: tensorgrad vs torch.func (correctness + timing).

Cases chosen for known algebraic structure the compiler should FIND:
  quad:  f = x^T A x            -> H = A + A^T           (constant! no x-compute)
  lse:   f = logsumexp(x)       -> H = diag(p) - p p^T   (softmax structure)
  mlp:   f = sum(tanh(x W1) W2) -> dense but shareable
Each: correctness vs torch.func.hessian, then wall time and emitted-kernel
count for the tensorgrad program.
"""

import re
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
N = 64
DIMS = {n: N}


def kernels(prog):
    fn = next(iter(prog._specializations.values()))
    src = fn._source
    ops = re.findall(r"torch\.\w+\(|\.sum\(|\.argsort\(", src)
    return len(ops), src


def bench(f, reps=50):
    for _ in range(5):
        f()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


results = []

# ---- case 1: quadratic --------------------------------------------------
x = Variable("x", n)
A = Variable("A", i=n, j=n)
f_quad = F.dot(x.rename(n="i") @ A, x.rename(n="j"), dim="j") if False else (
    (x.rename(n="i") @ A) @ x.rename(n="j")
)
# f = sum_ij x_i A_ij x_j  (scalar)
g = f_quad.grad(x, {"n": "di"})
H = g.grad(x, {"n": "dj"})
prog = compile_to_callable(H)
xv = torch.randn(N)
Av = torch.randn(N, N)
with torch.enable_grad():
    Ht = torch.func.hessian(lambda xx: (xx @ Av @ xx))(xv)
out = prog({x: xv.rename("n"), A: Av.rename("i", "j")}, DIMS)
got = out.align_to("di", "dj").rename(None)
assert torch.allclose(got, Ht, atol=1e-4), (got - Ht).abs().max()
k, src = kernels(prog)
t_tg = bench(lambda: prog({x: xv.rename("n"), A: Av.rename("i", "j")}, DIMS))
with torch.enable_grad():
    t_torch = bench(lambda: torch.func.hessian(lambda xx: (xx @ Av @ xx))(xv))
results.append(("quad (H = A+A^T)", k, t_tg, t_torch))
print("=== quad generated source ===")
print(src)

# ---- case 2: logsumexp --------------------------------------------------
f_lse = F.log(F.sum(F.exp(x)))
H2 = f_lse.grad(x, {"n": "di"}).grad(x, {"n": "dj"})
prog2 = compile_to_callable(H2)
with torch.enable_grad():
    Ht2 = torch.func.hessian(lambda xx: torch.logsumexp(xx, 0))(xv)
out2 = prog2({x: xv.rename("n")}, DIMS)
got2 = out2.align_to("di", "dj").rename(None)
assert torch.allclose(got2, Ht2, atol=1e-4), (got2 - Ht2).abs().max()
k2, src2 = kernels(prog2)
t_tg2 = bench(lambda: prog2({x: xv.rename("n")}, DIMS))
with torch.enable_grad():
    t_torch2 = bench(lambda: torch.func.hessian(lambda xx: torch.logsumexp(xx, 0))(xv))
results.append(("lse (H = diag(p)-pp^T)", k2, t_tg2, t_torch2))
print("=== lse generated source ===")
print(src2)

# ---- case 3: 2-layer MLP ------------------------------------------------
m = sympy.Symbol("m")
M = 64
W1 = Variable("W1", n=n, m=m)
W2 = Variable("W2", m=m)
f_mlp = F.sum(F.tanh(x @ W1) * W2)
H3 = f_mlp.grad(x, {"n": "di"}).grad(x, {"n": "dj"})
prog3 = compile_to_callable(H3)
W1v, W2v = torch.randn(N, M) / N**0.5, torch.randn(M)
with torch.enable_grad():
    Ht3 = torch.func.hessian(lambda xx: (torch.tanh(xx @ W1v) * W2v).sum())(xv)
feed3 = {x: xv.rename("n"), W1: W1v.rename("n", "m"), W2: W2v.rename("m")}
out3 = prog3(dict(feed3), DIMS | {m: M})
got3 = out3.align_to("di", "dj").rename(None)
assert torch.allclose(got3, Ht3, atol=1e-4), (got3 - Ht3).abs().max()
k3, _ = kernels(prog3)
t_tg3 = bench(lambda: prog3(dict(feed3), DIMS | {m: M}))
with torch.enable_grad():
    t_torch3 = bench(lambda: torch.func.hessian(lambda xx: (torch.tanh(xx @ W1v) * W2v).sum())(xv))
results.append(("mlp tanh", k3, t_tg3, t_torch3))

print()
print(f"{'case':26s} {'tg kernels':>10s} {'tg ms':>8s} {'torch.func ms':>13s} {'speedup':>8s}")
for name, k, a, b in results:
    print(f"{name:26s} {k:10d} {a:8.3f} {b:13.3f} {b / a:7.1f}x")
