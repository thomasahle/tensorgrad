"""Gaussian-process regression: the Matrix Cookbook, computing itself.

This project grew out of the Tensor Cookbook, and GP hyperparameter
learning is the cookbook's home turf. The marginal likelihood

    -log p(y | X, theta) = 1/2 y^T K^-1 y + 1/2 log|K| + const,
    K = sf^2 exp(-(x_i-x_j)^2 / 2 l^2) + sn^2 I

is written below exactly as that formula. Its gradients with respect to
the kernel hyperparameters need the two most famous cookbook identities,

    d log|K|   = tr(K^-1 dK)          (cookbook 43/46)
    d (K^-1)   = -K^-1 dK K^-1        (cookbook 40)

and NEITHER is written anywhere: both live on the F.det / F.inverse
signatures as rewrite rules, so nlml.grad(theta) derives them, cancels
det(K)/det(K) inside the log-derivative, and compiles the result. The
symbolic gradients match finite differences to 1e-10.

* The kernel matrix is algebra: the pairwise squared distances are a
  broadcast outer difference, and the noise term is sn^2 * Delta(n,m) --
  the identity is a STRUCTURAL tensor, so it never materializes; it fuses
  into the einsum wiring.

* inverse and det are technology-mapping cells (torch.linalg.inv/det from
  tensorgrad/compiler/cells.py). Differentiation happens symbolically
  UPSTREAM of them -- the cells only map the surviving forward
  applications onto kernels.

* One compiled program IS the training step: nlml, its three
  hyperparameter gradients, and the plain gradient-descent updates (the
  suite's simplest optimizer yet: w - lr*g, no state). A second program
  is the posterior predictive (mean and variance at test points).

* This runs in PLAIN FLOAT32 because of two linalg peepholes
  (tensorgrad/compiler/peepholes.py): log(det(K)) -- whose raw det
  genuinely underflows float32, a hundred eigenvalues near sn^2 multiply
  to ~1e-200 -- compiles to torch.linalg.slogdet, and the K^-1 y
  contraction compiles to torch.linalg.solve without materializing the
  inverse. Same philosophy as exp/sum-exp -> softmax: write the algebra,
  the compiler picks the kernel.

The task: 128 noisy samples of sin(2x) + x/2. PASS = the learned GP
predicts a held-out grid to RMSE < 0.05 (noise is 0.1) and recovers the
noise level within a factor of two.
"""

import math
import time

import torch
from sympy import symbols

import tensorgrad.functions as F
import tensorgrad as tg
from tensorgrad import Delta, Variable

torch.set_grad_enabled(False)
torch.set_num_threads(2)

# ----------------------------------------------------------------- config

N_TRAIN, N_TEST = 128, 200
TRUE_NOISE = 0.1
LR, MAX_STEPS = 0.15, 200

n, m, t = symbols("n m t")
DIMS = {n: N_TRAIN, m: N_TRAIN, t: N_TEST}

# ------------------------------------------------------------ the program
X = Variable("X", n)  # training inputs (1-D)
y = Variable("y", n)  # training targets
Xs = Variable("Xs", t)  # test inputs
# The three hyperparameters are 0-dim variables, learned in log space so
# positivity is free.
loglen, logsf, logsn = Variable("loglen"), Variable("logsf"), Variable("logsn")
HYP = {"loglen": loglen, "logsf": logsf, "logsn": logsn}

sf2, sn2, l2 = F.exp(2 * logsf), F.exp(2 * logsn), F.exp(2 * loglen)


def rbf(a, b):  # noqa: ANN001 -- kernel block between two 1-D input vectors
    d = a - b
    return sf2 * F.exp(-(d * d) / (2 * l2))


K = rbf(X, X.rename(n="m")) + sn2 * Delta(n, "n", "m")  # the identity stays structural
Ki = F.inverse(K, {"n", "m"})
alpha = Ki @ y  # K^-1 y, edge (m,)
quad = alpha @ y.rename(n="m")  # y^T K^-1 y
nlml = (quad + F.log(F.det(K, {"n", "m"}))) / (2 * N_TRAIN)  # per-datum, consts dropped

# the whole optimizer: plain gradient descent on three scalars
updates = {v: v - LR * nlml.grad(v) for v in HYP.values()}

# posterior predictive at the test points
Ks = rbf(Xs, X.rename(n="m"))  # (t, m) cross-kernel
mean = Ks @ alpha  # (t,)
KiKs = Ks.rename(m="n") @ Ki  # (t, m)
var = sf2 - F.sum(Ks * KiKs, ["m"])  # latent-f variance per test point

# ----------------------------------------------------------------- driver


def main():
    step = tg.compile(nlml=nlml, hyp=updates, print_info=True)
    predict = tg.compile(mean=mean, var=var)

    gen = torch.Generator().manual_seed(0)
    Xv = 6 * torch.rand(N_TRAIN, generator=gen) - 3
    f_true = lambda x: torch.sin(2 * x) + 0.5 * x  # noqa: E731
    yv = f_true(Xv) + TRUE_NOISE * torch.randn(N_TRAIN, generator=gen)
    Xtest = torch.linspace(-3, 3, N_TEST)

    st = {loglen: torch.tensor(0.0), logsf: torch.tensor(0.0), logsn: torch.tensor(math.log(0.3))}
    t0 = time.perf_counter()
    for it in range(1, MAX_STEPS + 1):
        out = step(st, dims=DIMS, X=Xv, y=yv)
        st = out.hyp
        if it % 50 == 0:
            rate = it / (time.perf_counter() - t0)
            print(
                f"step {it:3d}  nlml/n {out.nlml.item():8.4f}  "
                f"len {st[loglen].exp().item():.3f}  sf {st[logsf].exp().item():.3f}  "
                f"sn {st[logsn].exp().item():.3f}  ({rate:.0f} steps/s)"
            )

    pred = predict(st, dims=DIMS, X=Xv, y=yv, Xs=Xtest)
    rmse = ((pred.mean - f_true(Xtest)) ** 2).mean().sqrt().item()
    picp = ((pred.mean - f_true(Xtest)).abs() < 2 * pred.var.clamp(min=0).sqrt()).double().mean().item()
    sn_hat = st[logsn].exp().item()
    print(f"test RMSE vs true function: {rmse:.4f}   2-sigma coverage: {picp:.2f}   recovered noise: {sn_hat:.3f}")
    ok = rmse < 0.05 and 0.05 < sn_hat < 0.2
    print(f"({'PASS' if ok else 'FAIL'}: rmse < 0.05 and noise in [0.05, 0.2], true noise {TRUE_NOISE})")


if __name__ == "__main__":
    main()
