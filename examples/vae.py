"""A sample-free VAE, trained without an optimizer: symbolic expectations
+ derived Hessians = exact block-Newton.

Every VAE implementation contains two pieces of hand-derived math: the
reparametrization trick (because autodiff cannot differentiate through
sampling) and the closed-form Gaussian KL (because nobody wants to
estimate it). This file contains NEITHER -- and no sampling, no learning
rate, and no optimizer either.

* The ELBO is written as its definition, an expectation over the latent:

      ELBO = E_z[ log p(x|z) ] - E_z[ log q(z|x) - log p(z) ]

  with z ~ N(mu(x), diag(sigma^2)). The Expectation node
  (tensorgrad/extras/expectation.py) resolves every term by Stein's lemma
  during simplification. The textbook KL formula
  1/2 sum(mu^2 + sigma^2 - 2 log sigma - 1) is never written here: it is
  DERIVED (we verified the resolved algebra against Monte-Carlo to 1e-3
  and against the textbook form to machine precision). z never exists at
  runtime -- the only torch.randn in this file generates the dataset.

* TRAINING IS EXACT BLOCK-NEWTON, i.e. higher-order derivatives of a
  symbolically-resolved expectation. The resolved ELBO is quadratic in
  each weight block, so per block

      H = loss.grad(W).grad(W)          # d2/dW2 of a derived E[.]

  is exact, and one Newton solve lands on that block's conditional
  optimum. For a matrix block the 4-edge Hessian factors as a Kronecker
  product H = A (x) K (reconstruction separates over samples x outputs),
  and the factors are recovered FROM THE DERIVED H by plain Delta
  contractions:

      A*tr(K) = H @ Delta(e2,e2')    K*tr(A) = H @ Delta(e1,e1')
      tr(A)*tr(K) = H @ both         =>  inv(H) = sig * inv(At) (x) inv(Kt)

  so the whole solve is two small F.inverse calls (the technology-mapping
  cell examples/newton.py introduced). Cycling the four blocks
  (Gauss-Seidel) trains the model to the analytic optimum in ~5 rounds:
  no step size exists anywhere in this file.

* The posterior std is a per-coordinate PARAMETER, not a function of x:
  for a linear-Gaussian model the true posterior covariance is
  input-independent, and the derived gradient KNOWS it -- the compiled
  Newton update for sigma reads only (sigma, Wd); the compiler discovered
  the PPCA decoupling by cancellation.

The task: 1024 points on a 2-D gaussian manifold linearly embedded in 16
dimensions plus noise. A linear-decoder VAE is probabilistic PCA, so the
reconstruction has a KNOWN optimum: the rank-2 PCA floor. PASS = the
sample-free, optimizer-free VAE reaches it (ratio < 1.01).
"""

import time

import torch
from sympy import symbols

import tensorgrad.functions as F
import tensorgrad as tg
from tensorgrad import Delta, Variable
from tensorgrad.tensor import Tensor
from tensorgrad.extras.expectation import Expectation

# No tape, no sampling, no optimizer: gradients, Hessians and the ELBO's
# expectations are all symbolic.
torch.set_grad_enabled(False)
torch.set_num_threads(2)

# ----------------------------------------------------------------- config

N, D_X, LATENT = 1024, 16, 2
NOISE = 0.1  # observation noise of the true generative process
SIGMA_X2 = NOISE**2  # decoder's (fixed) observation variance
ROUNDS = 8

b, dx, l = symbols("b dx l")
DIMS = {b: N, dx: D_X, l: LATENT}
SYM = {"dx": dx, "l": l}

# ------------------------------------------------------------- the model
x = Variable("x", b, dx)  # the data (full batch: everything is deterministic)
z = Variable("z", b, l)  # the latent that only ever exists symbolically

We = Variable("We", dx=dx, l=l)  # encoder: mu(x) = x @ We
logs = Variable("logs", l=l)  # posterior log-std (input-independent, see above)
Wd = Variable("Wd", l=l, dx=dx)  # decoder weights
bd = Variable("bd", dx=dx)  # decoder bias
PARAMS = (We, Wd, bd, logs)

mu = x @ We
s = F.exp(logs)

# q(z|x) = N(mu, diag(s^2)) for the Expectation: mean tensor + covariance
# tensor covar[b,l,b2,l2] = delta(b,b2) delta(l,l2) s2[l], built from an
# order-3 copy tensor. Deltas stay structural -- nothing materializes.
covar = (s * s).rename(l="li") @ Delta(l, "l", "l2", "li") @ Delta(b, "b", "b2")


def E(t: Tensor) -> Tensor:
    """E_{z ~ q(z|x)}[t], resolved symbolically by Stein's lemma."""
    return Expectation(t, z, mu, covar, {"b": "b2", "l": "l2"}).full_simplify()


# The ELBO from its definition. Log-density algebra in, closed forms out.
err = x - (z @ Wd + bd)
zc = (z - mu) / s
log_px_z = -F.sum(err * err) / (2 * SIGMA_X2)  # log p(x|z), up to constants
log_pz = -F.sum(z * z) / 2  # standard-normal prior
log_qz = -F.sum(zc * zc) / 2 - N * F.sum(F.log(s))  # the posterior's own density

elbo = E(log_px_z) - (E(log_qz) - E(log_pz))  # = E[log p] - KL, all derived
loss = -elbo / N

# --------------------------------------------- training = derived Newton
# One exact Newton step per parameter block. grad(grad(E[.])) is the whole
# method; the Kronecker extraction below is the only "solver code".


def newton_matrix(W: Variable, e1: str, e2: str) -> Tensor:
    """Exact Newton update for a matrix block whose Hessian is a Kronecker
    product A (x) K -- recovered from the DERIVED 4-edge Hessian by Delta
    contractions, then inverted as two small matrix inverses."""
    g = loss.grad(W, {e1: f"{e1}1", e2: f"{e2}1"})
    H = g.grad(W, {e1: f"{e1}2", e2: f"{e2}2"})
    At = H @ Delta(SYM[e2], f"{e2}1", f"{e2}2")  # A * tr(K)
    Kt = H @ Delta(SYM[e1], f"{e1}1", f"{e1}2")  # K * tr(A)
    sig = At @ Delta(SYM[e1], f"{e1}1", f"{e1}2")  # tr(A) * tr(K)
    step = sig * (F.inverse(At, {f"{e1}1", f"{e1}2"}) @ g @ F.inverse(Kt, {f"{e2}1", f"{e2}2"}))
    return W - step.rename(**{f"{e1}2": e1, f"{e2}2": e2})


def newton_vector(w: Variable, e: str) -> Tensor:
    """Exact Newton update for a vector block (2-edge Hessian, one inverse)."""
    g = loss.grad(w, {e: f"{e}1"})
    H = g.grad(w, {e: f"{e}2"})
    return w - (F.inverse(H, {f"{e}1", f"{e}2"}) @ g).rename(**{f"{e}2": e})


NEWTON = {
    We: newton_matrix(We, "dx", "l"),
    Wd: newton_matrix(Wd, "l", "dx"),
    bd: newton_vector(bd, "dx"),
    logs: newton_vector(logs, "l"),  # convex in logs: Newton converges here too
}

# metric as algebra: reconstruction at the posterior mean
recon = x - (mu @ Wd + bd)
recon_mse = F.sum(recon * recon) / (N * D_X)

# ----------------------------------------------------------------- data


def make_data(gen):
    """s ~ N(0, I_2) embedded linearly in D_X dims + noise: exactly the
    probabilistic-PCA generative model, so the optimum is known."""
    A_true = torch.randn(LATENT, D_X, generator=gen)
    s_true = torch.randn(N, LATENT, generator=gen)
    return s_true @ A_true + NOISE * torch.randn(N, D_X, generator=gen)


def pca_floor(Xv):
    """The best possible affine rank-LATENT reconstruction (PCA)."""
    Xc = Xv - Xv.mean(0)
    U, S, V = torch.linalg.svd(Xc, full_matrices=False)
    lowrank = (U[:, :LATENT] * S[:LATENT]) @ V[:LATENT]
    return ((Xc - lowrank) ** 2).sum().item() / (N * D_X)


# ---------------------------------------------------------------- training


def main():
    # Four one-output programs, cycled Gauss-Seidel style: each block's
    # exact solve sees the others' fresh values. The state dict is keyed by
    # Variable and fed positionally, so each program takes what it needs.
    solvers = {v: tg.compile(o=expr) for v, expr in NEWTON.items()}
    score = tg.compile(mse=recon_mse, loss=loss)

    gen = torch.Generator().manual_seed(0)
    Xv = make_data(gen)
    floor = pca_floor(Xv)
    st = {
        x: Xv,
        We: 0.05 * torch.randn(D_X, LATENT, generator=gen),
        Wd: 0.05 * torch.randn(LATENT, D_X, generator=gen),
        bd: torch.zeros(D_X),
        logs: torch.zeros(LATENT),
    }

    t0 = time.perf_counter()
    ratio = float("inf")
    for it in range(1, ROUNDS + 1):
        for v in PARAMS:
            st[v] = solvers[v](st, dims=DIMS).o
        out = score(st, dims=DIMS)
        ratio = out.mse.item() / floor
        print(f"round {it}:  -elbo/N {out.loss.item():9.4f}  recon mse {out.mse.item():.5f}  ({ratio:.3f}x pca floor)")
        if ratio < 1.001:
            break
    total = time.perf_counter() - t0
    print(f"trained to the analytic optimum in {it} Newton rounds, {total:.1f}s total (incl. compile)")
    print(f"final recon mse / pca floor: {ratio:.4f} ({'PASS' if ratio < 1.01 else 'FAIL'}, target < 1.01)")


if __name__ == "__main__":
    main()
