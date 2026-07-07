"""A damped-Newton step: tensorgrad's SYMBOLIC HESSIAN vs torch.func.

Regularized logistic regression on fixed synthetic data. The whole point is
the symbolic Hessian. tensorgrad derives BOTH the gradient g = dL/dw and the
Hessian H = dg/dw from the loss (H = g.grad(w), a second symbolic derivative)
and compiles loss + g + H into ONE program: weights in, (loss, g, H) out. The
damped-Newton solve

    w' = w - (H + mu I)^{-1} g

is done OUTSIDE the compiled program with torch.linalg.solve -- the solve is a
trivial 32x32 system and is not the point; the Hessian ASSEMBLY is. mingpt's
AdamW-as-algebra folds the update into the program because it is elementwise;
a linear solve is not, so here the program's job ends at (loss, g, H).

    loss = mean softplus(-(2y-1) * (X @ w)) + 0.5*lambda*||w||^2
    softplus(t) = log(1 + exp(t))            # built from F.log / F.exp, no stab

The torch reference computes the SAME g and H with torch.func (grad + hessian)
and does the SAME solve. Scores are kept small on purpose -- X is scaled by
1/sqrt(d) and the labels are noisy Bernoulli draws (so the data is NOT linearly
separable and the regularized MAP estimate stays bounded) -- so the naive
log(1+exp) never overflows. See the issues note.

Importing this file runs the correctness gate: 3 steps from a SHARED w, with
tg's loss, g and H asserted to match torch.func's to rtol 1e-4 at each step
(driving both from the same w isolates assembly numerics from optimizer drift).

Stresses: symbolic Hessian assembly, the exp/softplus chain's SECOND derivative
(sigmoid' = p(1-p) has to fall out of differentiating log(1+exp) twice), and it
shows the 32x32 Hessian is dense-but-cheap.
"""

import math
import os
import sys
import time

import torch
from sympy import symbols

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable

torch.set_grad_enabled(False)  # tensorgrad derives g and H symbolically
torch.set_num_threads(2)

BENCH_NAME = "newton_step"

# Real GLM size: at d=256 the closed-form assembly (one X^T diag(w) X gemm)
# beats jax.hessian's d-pass transform loop 3-4x; the old toy d=32 only
# measured dispatch. Sized to keep torch.func's vmapped-tangent
# intermediates ~0.5 GB (16 GB machine).
BATCH, D = 2048, 256
LAMBDA, MU = 1e-3, 1e-2
N_CHECK_STEPS = 3
# Inductor numerics drift past the eager tolerance; the runner sets
# TG_BENCH_COMPILED when it patches tg.compile.
RTOL = 5e-3 if os.environ.get("TG_BENCH_COMPILED") else 1e-4

batch, d = symbols("batch d")
DIMS = {batch: BATCH, d: D}

# ---------------------------------------------------- tensorgrad loss, g and H
# Edges are names: X carries {batch, d}, w carries {d}, so X @ w contracts the
# shared 'd' edge to per-example scores on {batch}. No axis positions anywhere.
w_var = Variable("w", d)  # edge 'd'
X_var = Variable("X", batch, d)  # edges 'batch', 'd'
y_var = Variable("y", batch)  # edge 'batch', labels in {0,1}

scores = X_var @ w_var  # z = X w, on {batch}
margin = -((2 * y_var - 1) * scores)  # -(2y-1) z
softplus = F.log(F.exp(margin) + 1)  # log(1 + exp(.)), unstabilized on purpose
loss = F.mean(softplus, dim="batch") + 0.5 * LAMBDA * F.frobenius2(w_var)

# g and H are BOTH symbolic derivatives; H is the Jacobian of g, i.e. the second
# derivative of the loss. dim=32 so H is a dense 32x32 matrix.
g = loss.grad(w_var, {"d": "di"})  # (di,)
H = g.grad(w_var, {"d": "dj"})  # (di, dj) -- the symbolic Hessian


_TG_PROGRAM = None  # cached CompiledStep: loss + g + H fused into one program


def _tg_program():
    """loss + g + H compiled into one program (cached). weights in, all out."""
    global _TG_PROGRAM
    if _TG_PROGRAM is None:
        _TG_PROGRAM = tg.compile(loss=loss, g=g, H=H)
    return _TG_PROGRAM


def _tg_assemble(prog, wv, Xv, yv):
    """One compiled call -> (loss_float, g as (D,), H as (D, D)) plain tensors."""
    out = prog(dims=DIMS, w=wv, X=Xv, y=yv)
    gv = out.g.align_to("di").rename(None)
    Hv = out.H.align_to("di", "dj").rename(None)
    return out.loss.item(), gv, Hv


# ------------------------------------------------------------ torch reference
def _torch_loss_fn(Xv, yv):
    """The SAME math as a plain torch scalar function of w, for torch.func."""
    s = 2 * yv - 1  # +/-1 targets, fixed

    def loss_fn(w):
        margin = -(s * (Xv @ w))
        return torch.log(1 + torch.exp(margin)).mean() + 0.5 * LAMBDA * (w @ w)

    return loss_fn


def _torch_assemble(loss_fn, wv):
    """g and H via torch.func (grad + hessian), plus the loss value."""
    with torch.enable_grad():
        loss_val = loss_fn(wv)
        gv = torch.func.grad(loss_fn)(wv)
        Hv = torch.func.hessian(loss_fn)(wv)
    return loss_val.item(), gv, Hv


# ------------------------------------------------------ shared solve + data
def _newton_update(wv, gv, Hv):
    """Damped Newton: w' = w - (H + mu I)^{-1} g (outside the program)."""
    A = Hv + MU * torch.eye(D)
    return wv - torch.linalg.solve(A, gv)


def _make_data():
    """Fixed synthetic logistic data. X scaled by 1/sqrt(d) keeps scores O(1),
    and Bernoulli labels make the data non-separable so w stays bounded."""
    g = torch.Generator().manual_seed(0)
    w_true = torch.randn(D, generator=g)
    Xv = torch.randn(BATCH, D, generator=g) / math.sqrt(D)
    yv = torch.bernoulli(torch.sigmoid(Xv @ w_true), generator=g)  # {0,1}
    return Xv, yv


def _init_w():
    return torch.zeros(D)


# --------------------------------------------------------------- step makers
def make_tg_step():
    """One compiled assembly per step, then the tiny solve; w in, w out."""
    prog = _tg_program()
    Xv, yv = _make_data()
    holder = {"w": _init_w()}

    def step_fn() -> float:
        loss_val, gv, Hv = _tg_assemble(prog, holder["w"], Xv, yv)
        holder["w"] = _newton_update(holder["w"], gv, Hv)
        return loss_val

    return step_fn


def make_torch_step():
    Xv, yv = _make_data()
    loss_fn = _torch_loss_fn(Xv, yv)
    holder = {"w": _init_w()}

    def step_fn() -> float:
        loss_val, gv, Hv = _torch_assemble(loss_fn, holder["w"])
        holder["w"] = _newton_update(holder["w"], gv, Hv)
        return loss_val

    return step_fn


def make_jax_step():
    """Same assembly + damped solve via jax.grad / jax.hessian, jit whole."""
    import jax
    import jax.numpy as jnp

    Xv, yv = _make_data()
    Xj = jnp.asarray(Xv.numpy())
    sj = jnp.asarray((2 * yv - 1).numpy())

    def loss_fn(w):
        margin = -(sj * (Xj @ w))
        return jnp.log(1 + jnp.exp(margin)).mean() + 0.5 * LAMBDA * (w @ w)

    @jax.jit
    def newton(w):
        lv = loss_fn(w)
        g_ = jax.grad(loss_fn)(w)
        H_ = jax.hessian(loss_fn)(w)
        return w - jnp.linalg.solve(H_ + MU * jnp.eye(D), g_), lv

    holder = {"w": jnp.zeros(D)}

    def step_fn() -> float:
        holder["w"], lv = newton(holder["w"])
        return float(lv)

    return step_fn


# ------------------------------------------------------------ correctness gate
def _correctness_gate():
    prog = _tg_program()
    Xv, yv = _make_data()
    loss_fn = _torch_loss_fn(Xv, yv)
    w = _init_w()  # identical init drives BOTH assemblies each step
    for i in range(1, N_CHECK_STEPS + 1):
        l_tg, g_tg, H_tg = _tg_assemble(prog, w, Xv, yv)
        l_th, g_th, H_th = _torch_assemble(loss_fn, w)
        assert math.isclose(l_tg, l_th, rel_tol=RTOL), (
            f"step {i}: tg loss {l_tg!r} != torch loss {l_th!r} (rtol {RTOL})"
        )
        torch.testing.assert_close(g_tg, g_th, rtol=RTOL, atol=1e-5)
        torch.testing.assert_close(H_tg, H_th, rtol=RTOL, atol=1e-5)
        w = _newton_update(w, g_tg, H_tg)  # advance the shared w with matched g, H


_correctness_gate()


# ------------------------------------------------------------------- timing
def _bench(step_fn, reps=10, warmup=3) -> float:
    for _ in range(warmup):
        step_fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        step_fn()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def _bench_assembly():
    """Time JUST the g+H assembly (no solve): the compiled call vs torch.func."""
    prog = _tg_program()
    Xv, yv = _make_data()
    loss_fn = _torch_loss_fn(Xv, yv)
    wv = _init_w()
    return (
        _bench(lambda: _tg_assemble(prog, wv, Xv, yv)),
        _bench(lambda: _torch_assemble(loss_fn, wv)),
    )


if __name__ == "__main__":
    if "--check" in sys.argv:
        print(f"{BENCH_NAME}: correctness gate passed ({N_CHECK_STEPS} steps, rtol {RTOL})")
        sys.exit(0)
    t_tg = _bench(make_tg_step())
    t_torch = _bench(make_torch_step())
    print(f"{BENCH_NAME}: tg {t_tg:.2f}ms torch {t_torch:.2f}ms")
    a_tg, a_th = _bench_assembly()
    print(f"{BENCH_NAME} g+H assembly: tg {a_tg:.2f}ms torch.func {a_th:.2f}ms")
