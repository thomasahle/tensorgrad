"""GP marginal-likelihood hyperparameter step: tensorgrad vs torch vs jax.

The step from examples/gp.py: RBF kernel over n=128 fixed 1-D inputs,

    nlml = ( y^T K^-1 y + log|K| ) / (2n),
    K    = sf^2 exp(-(x_i-x_j)^2 / 2 l^2) + sn^2 I,

three log-hyperparameters, plain gradient descent. tensorgrad DERIVES the
cookbook gradients (d log|K| = tr(K^-1 dK), d K^-1 = -K^-1 dK K^-1) from
the F.det / F.inverse rewrite rules and compiles nlml + 3 gradients + the
GD updates into ONE program; the linalg peepholes map K^-1 y onto
torch.linalg.solve and log det onto slogdet, so the whole thing runs in
plain float32. The torch reference writes the SAME nlml with
torch.linalg.solve/slogdet and lets autograd differentiate through them;
jax uses jit(value_and_grad) of the same formula. Nobody materializes an
inverse for the nlml itself.

Importing this file runs the correctness gate: from shared initial
hyperparameters, tg's nlml and all three gradients match torch's autograd
to rtol 1e-4 (and jax's, when jax is installed).

Stresses: derivative rules of det/inverse, the solve/slogdet peepholes,
scalar (0-dim) learned parameters, and a training step whose entire state
is three numbers.
"""

import math
import sys

import torch
from sympy import symbols

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Delta, Variable

torch.set_num_threads(2)

BENCH_NAME = "gp"

N = 128
LR = 0.15
INIT = {"loglen": 0.0, "logsf": 0.0, "logsn": math.log(0.3)}

# fixed data (identical across frameworks)
_gen = torch.Generator().manual_seed(0)
XV = (6 * torch.rand(N, generator=_gen) - 3).contiguous()
YV = (torch.sin(2 * XV) + 0.5 * XV + 0.1 * torch.randn(N, generator=_gen)).contiguous()

# ------------------------------------------------------------------ tensorgrad
n, m = symbols("n m")
X = Variable("X", n)
y = Variable("y", n)
loglen, logsf, logsn = Variable("loglen"), Variable("logsf"), Variable("logsn")
HYP = {"loglen": loglen, "logsf": logsf, "logsn": logsn}

_d = X - X.rename(n="m")
K = F.exp(2 * logsf) * F.exp(-(_d * _d) / (2 * F.exp(2 * loglen))) + F.exp(2 * logsn) * Delta(n, "n", "m")
quad = (F.inverse(K, {"n", "m"}) @ y) @ y.rename(n="m")
nlml = (quad + F.log(F.det(K, {"n", "m"}))) / (2 * N)
_updates = {v: v - LR * nlml.grad(v) for v in HYP.values()}
_grads = {f"g_{k}": nlml.grad(v) for k, v in HYP.items()}


def make_tg_step():
    prog = tg.compile(nlml=nlml, hyp=_updates)
    state = {v: torch.tensor(INIT[k]) for k, v in HYP.items()}

    def step():
        nonlocal state
        state = prog(state, dims={n: N, m: N}, X=XV, y=YV).hyp

    return step


# ----------------------------------------------------------------------- torch
def _torch_nlml(hyp, Xv, yv):
    d = Xv[:, None] - Xv[None, :]
    Km = torch.exp(2 * hyp[1]) * torch.exp(-(d * d) / (2 * torch.exp(2 * hyp[0])))
    Km = Km + torch.exp(2 * hyp[2]) * torch.eye(N)
    return (yv @ torch.linalg.solve(Km, yv) + torch.linalg.slogdet(Km)[1]) / (2 * N)


def make_torch_step():
    hyp = torch.tensor([INIT["loglen"], INIT["logsf"], INIT["logsn"]], requires_grad=True)

    def step():
        nonlocal hyp
        with torch.enable_grad():
            loss = _torch_nlml(hyp, XV, YV)
            (g,) = torch.autograd.grad(loss, hyp)
        hyp = (hyp.detach() - LR * g).requires_grad_(True)

    return step


# ------------------------------------------------------------------------- jax
def make_jax_step():
    import jax
    import jax.numpy as jnp

    Xj, yj = jnp.asarray(XV.numpy()), jnp.asarray(YV.numpy())

    def nlml_fn(hyp):
        d = Xj[:, None] - Xj[None, :]
        Km = jnp.exp(2 * hyp[1]) * jnp.exp(-(d * d) / (2 * jnp.exp(2 * hyp[0])))
        Km = Km + jnp.exp(2 * hyp[2]) * jnp.eye(N)
        return (yj @ jnp.linalg.solve(Km, yj) + jnp.linalg.slogdet(Km)[1]) / (2 * N)

    @jax.jit
    def update(hyp):
        val, g = jax.value_and_grad(nlml_fn)(hyp)
        return hyp - LR * g, val

    state = {"h": jnp.array([INIT["loglen"], INIT["logsf"], INIT["logsn"]])}

    def step():
        state["h"], val = update(state["h"])
        val.block_until_ready()

    return step


# -------------------------------------------------------------- correctness gate
def _gate():
    prog = tg.compile(nlml=nlml, **_grads)
    st = {v: torch.tensor(INIT[k]) for k, v in HYP.items()}
    out = prog(st, dims={n: N, m: N}, X=XV, y=YV)

    hyp = torch.tensor([INIT["loglen"], INIT["logsf"], INIT["logsn"]], requires_grad=True)
    with torch.enable_grad():
        loss = _torch_nlml(hyp, XV, YV)
        (g,) = torch.autograd.grad(loss, hyp)
    torch.testing.assert_close(out.nlml, loss.detach(), rtol=1e-4, atol=1e-5)
    for i, k in enumerate(["loglen", "logsf", "logsn"]):
        torch.testing.assert_close(getattr(out, f"g_{k}"), g[i], rtol=1e-4, atol=1e-5)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return
    Xj, yj = jnp.asarray(XV.numpy()), jnp.asarray(YV.numpy())

    def nlml_fn(h):
        d = Xj[:, None] - Xj[None, :]
        Km = jnp.exp(2 * h[1]) * jnp.exp(-(d * d) / (2 * jnp.exp(2 * h[0])))
        Km = Km + jnp.exp(2 * h[2]) * jnp.eye(N)
        return (yj @ jnp.linalg.solve(Km, yj) + jnp.linalg.slogdet(Km)[1]) / (2 * N)

    vj, gj = jax.value_and_grad(nlml_fn)(jnp.array([INIT["loglen"], INIT["logsf"], INIT["logsn"]]))
    assert abs(float(vj) - out.nlml.item()) < 1e-4 * max(1, abs(out.nlml.item()))
    for i, k in enumerate(["loglen", "logsf", "logsn"]):
        assert abs(float(gj[i]) - getattr(out, f"g_{k}").item()) < 1e-3, (k, float(gj[i]))


_gate()
print(f"[{BENCH_NAME}] correctness gate passed", file=sys.stderr)
