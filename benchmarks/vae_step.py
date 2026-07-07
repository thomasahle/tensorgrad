"""Linear VAE training step: tensorgrad's EXACT step vs torch/jax reparam.

This benchmark compares different ALGORITHMS for the same model, and that
is the point. The model is the linear VAE of examples/vae.py (encoder
mu = x We, per-coordinate posterior std, linear decoder, fixed
observation noise) on the same fixed PPCA dataset (n=1024, d=16,
latent=2).

* tensorgrad's step is ONE Gauss-Seidel round of exact block-Newton: the
  ELBO's expectations are resolved symbolically (Stein's lemma; the
  closed-form KL is DERIVED), so the objective is deterministic and its
  per-block Hessians are exact -- four one-shot solves, no sampling, no
  learning rate. Five such rounds reach the analytic PPCA optimum.

* torch and jax CANNOT express that step: autodiff cannot differentiate
  through sampling, so they use the standard estimator -- the
  reparametrization trick (one Gaussian sample per datum) plus the
  HAND-CODED closed-form KL -- and an Adam update. Their per-step cost is
  lower; their steps are noisy first-order nudges rather than exact
  block solves, and they need hundreds of steps to approach the optimum
  the exact method reaches in five.

So read the per-step times together with the convergence gate, not as a
like-for-like kernel comparison.

Importing this file runs the gate: (1) tg's exact ELBO equals the
MONTE-CARLO EXPECTATION of the torch estimator's loss at shared initial
parameters (512 fresh samples, rel 2%) -- the KL terms agree exactly, the
reconstruction in expectation; (2) tg reaches 1.05x the PCA floor within
8 rounds; (3) the torch estimator's loss decreases over 50 Adam steps.

Stresses: Expectation resolution, grad-of-grad of resolved expectations,
the inverse cell, Kronecker-factor extraction by Delta contractions.
"""

import sys

import torch
from sympy import symbols

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Delta, Variable
from tensorgrad.extras.expectation import Expectation

torch.set_num_threads(2)

BENCH_NAME = "vae (exact vs reparam)"

N, D_X, LATENT = 1024, 16, 2
NOISE = 0.1
SIGMA_X2 = NOISE**2
ADAM_LR = 3e-3

# fixed data + shared initial parameters (identical across frameworks)
_gen = torch.Generator().manual_seed(0)
_A = torch.randn(LATENT, D_X, generator=_gen)
XV = (torch.randn(N, LATENT, generator=_gen) @ _A + NOISE * torch.randn(N, D_X, generator=_gen)).contiguous()
WE0 = 0.05 * torch.randn(D_X, LATENT, generator=_gen)
WD0 = 0.05 * torch.randn(LATENT, D_X, generator=_gen)
BD0 = torch.zeros(D_X)
LOGS0 = torch.zeros(LATENT)

# ------------------------------------------------------------------ tensorgrad
b, dx, l = symbols("b dx l")
DIMS = {b: N, dx: D_X, l: LATENT}
SYM = {"dx": dx, "l": l}

x = Variable("x", b, dx)
z = Variable("z", b, l)
We = Variable("We", dx=dx, l=l)
logs = Variable("logs", l=l)
Wd = Variable("Wd", l=l, dx=dx)
bd = Variable("bd", dx=dx)
PARAMS = (We, Wd, bd, logs)

mu = x @ We
s = F.exp(logs)


def _E(t):
    return Expectation.gaussian(t, z, mu, std=s).full_simplify()


err = x - (z @ Wd + bd)
zc = (z - mu) / s
elbo = _E(-F.sum(err * err) / (2 * SIGMA_X2)) - (
    _E(-F.sum(zc * zc) / 2 - N * F.sum(F.log(s))) - _E(-F.sum(z * z) / 2)
)
loss = -elbo / N


def _newton_matrix(W, e1, e2):
    g = loss.grad(W, {e1: f"{e1}1", e2: f"{e2}1"})
    H = g.grad(W, {e1: f"{e1}2", e2: f"{e2}2"})
    At = H @ Delta(SYM[e2], f"{e2}1", f"{e2}2")
    Kt = H @ Delta(SYM[e1], f"{e1}1", f"{e1}2")
    sig = At @ Delta(SYM[e1], f"{e1}1", f"{e1}2")
    step = sig * (F.inverse(At, {f"{e1}1", f"{e1}2"}) @ g @ F.inverse(Kt, {f"{e2}1", f"{e2}2"}))
    return W - step.rename(**{f"{e1}2": e1, f"{e2}2": e2})


def _newton_vector(w, e):
    g = loss.grad(w, {e: f"{e}1"})
    H = g.grad(w, {e: f"{e}2"})
    return w - (F.inverse(H, {f"{e}1", f"{e}2"}) @ g).rename(**{f"{e}2": e})


_NEWTON = {
    We: _newton_matrix(We, "dx", "l"),
    Wd: _newton_matrix(Wd, "l", "dx"),
    bd: _newton_vector(bd, "dx"),
    logs: _newton_vector(logs, "l"),
}
_recon = x - (mu @ Wd + bd)
_recon_mse = F.sum(_recon * _recon) / (N * D_X)


def _tg_state():
    return {x: XV, We: WE0.clone(), Wd: WD0.clone(), bd: BD0.clone(), logs: LOGS0.clone()}


def make_tg_step():
    solvers = {v: tg.compile(o=expr) for v, expr in _NEWTON.items()}
    st = _tg_state()

    def step():  # one Gauss-Seidel block-Newton round
        for v in PARAMS:
            st[v] = solvers[v](st, dims=DIMS).o

    return step


# ----------------------------------------------------------------------- torch
def _torch_params():
    return [WE0.clone().requires_grad_(True), WD0.clone().requires_grad_(True),
            BD0.clone().requires_grad_(True), LOGS0.clone().requires_grad_(True)]


def _torch_loss(params, xv, eps):
    we, wd, bdv, logsv = params
    muv = xv @ we
    sv = logsv.exp()
    zv = muv + sv * eps  # the reparametrization trick, hand-coded
    recon = ((xv - (zv @ wd + bdv)) ** 2).sum() / (2 * SIGMA_X2)
    kl = 0.5 * (muv**2 + sv**2 - 2 * logsv - 1).sum()  # closed-form KL, hand-coded
    return (recon + kl) / N


def make_torch_step():
    params = _torch_params()
    opt = torch.optim.Adam(params, lr=ADAM_LR)
    gen = torch.Generator().manual_seed(1)

    def step():
        eps = torch.randn(N, LATENT, generator=gen)
        with torch.enable_grad():
            lv = _torch_loss(params, XV, eps)
            opt.zero_grad(set_to_none=True)
            lv.backward()
            opt.step()

    return step


# ------------------------------------------------------------------------- jax
def make_jax_step():
    import jax
    import jax.numpy as jnp

    Xj = jnp.asarray(XV.numpy())
    p0 = {"we": jnp.asarray(WE0.numpy()), "wd": jnp.asarray(WD0.numpy()),
          "bd": jnp.asarray(BD0.numpy()), "logs": jnp.asarray(LOGS0.numpy())}

    def loss_fn(p, eps):
        muv = Xj @ p["we"]
        sv = jnp.exp(p["logs"])
        zv = muv + sv * eps
        recon = ((Xj - (zv @ p["wd"] + p["bd"])) ** 2).sum() / (2 * SIGMA_X2)
        kl = 0.5 * (muv**2 + sv**2 - 2 * p["logs"] - 1).sum()
        return (recon + kl) / N

    B1, B2, EPS_A = 0.9, 0.999, 1e-8

    @jax.jit
    def update(p, m, v, t, key):
        key, sub = jax.random.split(key)
        eps = jax.random.normal(sub, (N, LATENT))
        lv, g = jax.value_and_grad(loss_fn)(p, eps)
        m = jax.tree.map(lambda a, b: B1 * a + (1 - B1) * b, m, g)
        v = jax.tree.map(lambda a, b: B2 * a + (1 - B2) * b * b, v, g)
        c1, c2 = 1 / (1 - B1**t), 1 / (1 - B2**t)
        p = jax.tree.map(lambda w, a, bb: w - ADAM_LR * (c1 * a) / (jnp.sqrt(c2 * bb) + EPS_A), p, m, v)
        return p, m, v, lv, key

    state = {"p": p0, "m": jax.tree.map(jnp.zeros_like, p0),
             "v": jax.tree.map(jnp.zeros_like, p0), "t": 0, "key": jax.random.PRNGKey(1)}

    def step():
        state["t"] += 1
        state["p"], state["m"], state["v"], lv, state["key"] = update(
            state["p"], state["m"], state["v"], state["t"], state["key"]
        )
        lv.block_until_ready()

    return step


# -------------------------------------------------------------- correctness gate
def _gate():
    # (1) tg's exact ELBO == E[reparam estimator] at shared init (MC, rel 2%)
    prog = tg.compile(loss=loss)
    st = _tg_state()
    exact = prog(st, dims=DIMS).loss.item()
    gen = torch.Generator().manual_seed(7)
    params = _torch_params()
    with torch.no_grad():
        mc = sum(
            _torch_loss(params, XV, torch.randn(N, LATENT, generator=gen)).item() for _ in range(512)
        ) / 512
    assert abs(exact - mc) < 0.02 * abs(mc), (exact, mc)

    # (2) the exact method reaches the analytic optimum fast
    Xc = XV - XV.mean(0)
    U, S, V = torch.linalg.svd(Xc, full_matrices=False)
    floor = ((Xc - (U[:, :LATENT] * S[:LATENT]) @ V[:LATENT]) ** 2).sum().item() / (N * D_X)
    solvers = {v: tg.compile(o=expr) for v, expr in _NEWTON.items()}
    score = tg.compile(mse=_recon_mse)
    st = _tg_state()
    for _ in range(8):
        for v in PARAMS:
            st[v] = solvers[v](st, dims=DIMS).o
    assert score(st, dims=DIMS).mse.item() / floor < 1.05

    # (3) the torch estimator makes progress (sanity, not equivalence)
    params = _torch_params()
    opt = torch.optim.Adam(params, lr=ADAM_LR)
    gen2 = torch.Generator().manual_seed(2)

    def det_loss():  # deterministic eval at eps=0: recon-at-mean + KL
        with torch.no_grad():
            return _torch_loss(params, XV, torch.zeros(N, LATENT)).item()

    before = det_loss()
    for _ in range(50):
        eps = torch.randn(N, LATENT, generator=gen2)
        with torch.enable_grad():
            lv = _torch_loss(params, XV, eps)
            opt.zero_grad(set_to_none=True)
            lv.backward()
            opt.step()
    assert det_loss() < before, (det_loss(), before)


_gate()
print(f"[{BENCH_NAME}] correctness gate passed", file=sys.stderr)


def _report_time_to_target():
    """The honest metric for an algorithm comparison: wall time to reach
    1.05x the analytic PCA floor. Printed at import so every suite run
    carries it next to the per-step table."""
    import time as _time

    Xc = XV - XV.mean(0)
    U, S, V = torch.linalg.svd(Xc, full_matrices=False)
    floor = ((Xc - (U[:, :LATENT] * S[:LATENT]) @ V[:LATENT]) ** 2).sum().item() / (N * D_X)

    # tg: steady-state (programs pre-compiled), then timed rounds
    solvers = {v: tg.compile(o=expr) for v, expr in _NEWTON.items()}
    score = tg.compile(mse=_recon_mse)
    st = _tg_state()
    for v in PARAMS:  # warm: first call pays codegen
        st[v] = solvers[v](st, dims=DIMS).o
    st = _tg_state()
    t0 = _time.perf_counter()
    rounds = 0
    for rounds in range(1, 20):
        for v in PARAMS:
            st[v] = solvers[v](st, dims=DIMS).o
        if score(st, dims=DIMS).mse.item() / floor < 1.05:
            break
    t_tg = (_time.perf_counter() - t0) * 1e3

    # torch reparam+Adam: steps to the same target
    params = _torch_params()
    opt = torch.optim.Adam(params, lr=ADAM_LR)
    gen = torch.Generator().manual_seed(3)

    def _mse():
        with torch.no_grad():
            muv = XV @ params[0]
            return (((XV - (muv @ params[1] + params[2])) ** 2).sum() / (N * D_X)).item()

    t0 = _time.perf_counter()
    steps_t = 0
    for steps_t in range(1, 5001):
        eps = torch.randn(N, LATENT, generator=gen)
        with torch.enable_grad():
            lv = _torch_loss(params, XV, eps)
            opt.zero_grad(set_to_none=True)
            lv.backward()
            opt.step()
        if steps_t % 100 == 0 and _mse() / floor < 1.05:
            break
    t_th = (_time.perf_counter() - t0) * 1e3

    line = (f"[{BENCH_NAME}] time-to-1.05x-floor: tg exact {rounds} rounds / {t_tg:.1f} ms; "
            f"torch reparam {steps_t} steps / {t_th:.1f} ms")
    try:
        import jax
        import jax.numpy as jnp

        Xj = jnp.asarray(XV.numpy())
        p0 = {"we": jnp.asarray(WE0.numpy()), "wd": jnp.asarray(WD0.numpy()),
              "bd": jnp.asarray(BD0.numpy()), "logs": jnp.asarray(LOGS0.numpy())}

        def loss_fn(p, eps):
            muv = Xj @ p["we"]
            sv = jnp.exp(p["logs"])
            zv = muv + sv * eps
            recon = ((Xj - (zv @ p["wd"] + p["bd"])) ** 2).sum() / (2 * SIGMA_X2)
            kl = 0.5 * (muv**2 + sv**2 - 2 * p["logs"] - 1).sum()
            return (recon + kl) / N

        B1, B2, EPS_A = 0.9, 0.999, 1e-8

        @jax.jit
        def update(p, m, v, t, key):
            key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, (N, LATENT))
            lv, g = jax.value_and_grad(loss_fn)(p, eps)
            m = jax.tree.map(lambda a, b: B1 * a + (1 - B1) * b, m, g)
            v = jax.tree.map(lambda a, b: B2 * a + (1 - B2) * b * b, v, g)
            c1, c2 = 1 / (1 - B1**t), 1 / (1 - B2**t)
            p = jax.tree.map(lambda w, a, bb: w - ADAM_LR * (c1 * a) / (jnp.sqrt(c2 * bb) + EPS_A), p, m, v)
            return p, m, v, lv, key

        pj, mj, vj = p0, jax.tree.map(jnp.zeros_like, p0), jax.tree.map(jnp.zeros_like, p0)
        key = jax.random.PRNGKey(1)
        pj2, *_ = update(pj, mj, vj, 1, key)  # warm the jit
        jax.block_until_ready(pj2["we"])

        def _mse_j(p):
            muv = Xj @ p["we"]
            return float(((Xj - (muv @ p["wd"] + p["bd"])) ** 2).sum() / (N * D_X))

        t0 = _time.perf_counter()
        steps_j = 0
        for steps_j in range(1, 5001):
            pj, mj, vj, lv, key = update(pj, mj, vj, steps_j, key)
            if steps_j % 100 == 0 and _mse_j(pj) / floor < 1.05:
                break
        jax.block_until_ready(lv)
        t_jx = (_time.perf_counter() - t0) * 1e3
        line += f"; jax reparam {steps_j} steps / {t_jx:.1f} ms"
    except ImportError:
        pass
    print(line, file=sys.stderr)


_report_time_to_target()
