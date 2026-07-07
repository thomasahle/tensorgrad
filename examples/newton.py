"""Second-order optimization as ONE expression: Newton's method for
logistic regression, with the Hessian DERIVED symbolically.

Frameworks treat second-order training as exotic because runtime autodiff
must rebuild the Hessian every step -- d backward passes through a tape, or
forward-over-reverse with vmap gymnastics. Symbolically it is just two
applications of the same .grad():

    g = loss.grad(w)          # X^T (sigmoid(Xw) - y) / n  + lam w
    H = g.grad(w)             # X^T diag(s(1-s)) X / n     + lam I
    step = compile( w - inverse(H) @ g )

Nobody wrote those closed forms down: the compiler derives them from
log(1+exp(.)) by rewriting, then compiles loss, gradient, Hessian,
inverse and update into one straight-line program. The entire optimizer
is a single expression -- no learning rate, no schedule, no optimizer
state, no tuning. Each training "epoch" is one call, and convergence is
quadratic: |g|^2 squares its exponent every iteration until it hits
float precision (watch the printout: 1e-1, 1e-2, 1e-4, 1e-8, 1e-15).

The matrix inverse rides the same technology-mapping registry as the
fused attention/layernorm/gelu kernels (tensorgrad/compiler/cells.py):
F.inverse keeps its classic signature -- whose symbolic derivative is the
Matrix-Cookbook identity d(K^-1) = -K^-1 dK K^-1, unused here because we
differentiate BEFORE inverting -- and a ~30-line cell maps forward
applications onto torch.linalg.inv.

Even the metric is algebra: train accuracy is F.equal(F.gt0(Xw), y),
one more output of the same program.
"""

import time

import torch
from sympy import symbols

import tensorgrad.functions as F
import tensorgrad as tg
from tensorgrad import Variable

# No gradient tape, and for once not even a training loop to speak of.
torch.set_grad_enabled(False)
torch.set_num_threads(2)

# ----------------------------------------------------------------- config

N, D = 8192, 128  # examples, features
LAM = 1e-3  # ridge strength: strict convexity, bounded margins
MAX_ITERS = 10

n, d = symbols("n d")
DIMS = {n: N, d: D}

# ------------------------------------------------------------ the program
X = Variable("X", n, d)  # design matrix
y = Variable("y", n)  # 0/1 labels
w = Variable("w", d)  # the weights being optimized

z = X @ w  # margins (n,)
# Binary NLL written from primitives: log(1+exp(z)) - y z. The compiler
# differentiates THIS -- sigmoid, s(1-s), the whole GLM structure below is
# derived, not imported.
nll = F.sum(F.log(1 + F.exp(z)) - y * z) / N + (LAM / 2) * F.sum(w * w)

grad = nll.grad(w, {"d": "di"})  # (di,)
hess = grad.grad(w, {"d": "dj"})  # (di, dj) -- symbolic, closed form
newton = w - (F.inverse(hess, {"di", "dj"}) @ grad).rename(dj="d")

accuracy = F.sum(F.equal(F.gt0(z), y)) / N  # metric as algebra

# ----------------------------------------------------------------- driver


def main():
    step = tg.compile(w=newton, g2=F.sum(grad * grad), loss=nll, acc=accuracy, print_info=True)

    gen = torch.Generator().manual_seed(0)
    Xv = torch.randn(N, D, generator=gen)
    w_star = torch.randn(D, generator=gen) / D**0.5 * 3
    yv = (torch.sigmoid(Xv @ w_star) > torch.rand(N, generator=gen)).float()

    wv = torch.zeros(D)
    t0 = time.perf_counter()
    g2 = float("inf")
    for it in range(MAX_ITERS):
        out = step(dims=DIMS, X=Xv, y=yv, w=wv)
        wv, g2 = out.w, out.g2.item()
        print(f"newton step {it}:  loss {out.loss.item():.6f}  |grad|^2 {g2:.3e}  acc {out.acc.item():.3f}")
        if g2 < 1e-12:
            break
    per_step = (time.perf_counter() - t0) * 1e3 / (it + 1)
    converged = g2 < 1e-10
    print(
        f"converged to |grad|^2 = {g2:.1e} in {it + 1} steps "
        f"({per_step:.1f} ms/step, incl. compile) "
        f"({'PASS' if converged else 'FAIL'}, target 1e-10)"
    )


if __name__ == "__main__":
    main()
