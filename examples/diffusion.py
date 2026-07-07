"""A denoising diffusion model (DDPM, Ho et al. 2020) as a tensorgrad program.

Where minGPT (examples/mingpt.py) is LayerNorm + GELU + AdamW, this file is
deliberately RMSNorm + SiLU + Lion -- a different norm (no mean subtraction,
no bias), a different activation (the sigmoid chain real diffusion nets
use), and a different optimizer (sign-based, half the moment state, no
bias-correction schedule). None of them exist anywhere in the framework:
each is a few lines of primitive algebra below, and the compiler derives
the backward passes and maps the update's sign() onto torch.sign the same
way it maps everything else.

The diffusion-specific structure:

* The forward corruption process is algebra INSIDE the compiled program.
  The host supplies clean points x0, gaussian noise eps and integer
  timesteps t; the noisy input

      x_t = sqrt(abar_t) * x0  +  sqrt(1 - abar_t) * eps

  is built by the program itself: the per-example schedule coefficients
  are the sparse contraction one_hot(t) @ table, so "index the noise
  schedule at each example's timestep" is minGPT's token-embedding
  paradigm applied to time -- the compiler maps it to index_select, and
  the dense indicator never exists.

* Time conditioning is a learned embedding, i.e. one more one_hot
  contraction. No sinusoidal encode-on-host, no concatenation gymnastics:
  the embedding enters the denoiser through its own projection and edge
  names do the bookkeeping.

* The noise schedule itself is fed as (steps,)-vectors computed once on
  the host: that's the SCHEDULE, not tensor compute.

* No autograd, no optimizer library. torch.set_grad_enabled(False) is
  global and permanent. The loss, one derived gradient per parameter, and
  the Lion update equations compile into ONE straight-line step program:
  weights in, weights out, the python loop is a dict reassignment.

* The ancestral SAMPLER is a second compiled program. One DDPM step

      x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-abar_t) * eps_hat) + sigma_t * z

  with its scalar coefficients selected from the schedule tables by a
  0-dim one_hot(t) contraction. The denoising loop is dict plumbing.

* The evaluation metric is algebra too: the data are 8 gaussians on a
  circle, and "how many generated samples landed within r of a true mode"
  is a distance contraction, an F.max over the mode edge and an F.gt0.

Running this file trains a 3-block residual-MLP denoiser on the
8-gaussians toy distribution and then draws 512 fresh samples by running
the compiled sampler for all 100 timesteps: >90% of them land within 0.35
of a true mode, covering all 8 modes.
"""

import math
import time

import torch
from sympy import symbols

import tensorgrad.functions as F
import tensorgrad as tg
from tensorgrad import Variable, typed
from tensorgrad.tensor import Tensor

# The punchline, same as minGPT's: no gradient tape, ever.
torch.set_grad_enabled(False)
torch.set_num_threads(2)

# ----------------------------------------------------------------- config

N_BLOCK, D_MODEL, D_T = 3, 128, 32
T_STEPS = 100  # diffusion timesteps
N_MODE, MODE_RADIUS, MODE_STD = 8, 2.0, 0.05
BATCH, MAX_STEPS = 256, 3000
LR, B1, B2, WD = 3e-4, 0.9, 0.99, 0.1  # Lion hyperparameters

batch, x, steps, dt, dm, d_mlp, mode = symbols("batch x steps dt dm d_mlp mode")
DIMS = {
    batch: BATCH,
    x: 2,
    steps: T_STEPS,
    dt: D_T,
    dm: D_MODEL,
    d_mlp: 4 * D_MODEL,
    mode: N_MODE,
}

params: dict[str, Variable] = {}


def param(name: str, **edges) -> Variable:
    return params.setdefault(name, Variable(name, **edges))


# -------------------------------------------------------------- the model
# Program inputs. x0/eps/t are the host's only per-example jobs (data
# sample, gaussian noise, uniform timestep); the schedule vectors are
# computed once at startup.
x0 = Variable("x0", batch, x)  # clean data points
eps = Variable("eps", batch, x)  # the noise the model learns to predict
t = Variable("t", batch)  # integer timesteps, one per example
sqrt_abar = Variable("sqrt_abar", steps)  # sqrt(abar_t)
sqrt_1mabar = Variable("sqrt_1mabar", steps)  # sqrt(1 - abar_t)


@typed
def rms_norm(h: Tensor[..., "dm"], name: str) -> Tensor[..., "dm"]:
    """RMSNorm (Zhang & Sennrich 2019): no mean subtraction, no bias --
    two lines of primitives, backward pass derived."""
    ms = F.mean(h * h, dim="dm", keepdims=True)
    return h / F.sqrt(ms + 1e-5) * param(name + ".g", dm=dm)


def silu(u: Tensor) -> Tensor:
    """SiLU / swish, the standard diffusion-net activation: x * sigmoid(x).
    A sigmoid chain, so its derivative exercises a different rewrite path
    than minGPT's tanh-based GELU."""
    return u * F.sigmoid(u)


@typed
def denoiser(xt: Tensor[..., "x"], temb: Tensor[..., "dt"]) -> Tensor[..., "x"]:
    """eps_hat(x_t, t): a residual MLP. The noisy point and the time
    embedding enter through their own projections and meet on the hidden
    edge 'dm' -- named edges make 'condition on t' a sum, not a concat.
    Polymorphic over batch: training conditions per-example, the sampler
    shares one timestep across the whole batch (a batch-free temb that
    simply broadcasts)."""
    h = xt @ param("in.w", x=x, dm=dm) + temb @ param("t.w", dt=dt, dm=dm) + param("in.b", dm=dm)
    for i in range(N_BLOCK):
        u = rms_norm(h, f"h{i}.ln") @ param(f"h{i}.w1", dm=dm, d_mlp=d_mlp) + param(f"h{i}.b1", d_mlp=d_mlp)
        h = h + silu(u) @ param(f"h{i}.w2", d_mlp=d_mlp, dm=dm)
    return rms_norm(h, "ln_f") @ param("out.w", dm=dm, x=x)


# The DDPM training objective. Everything after the inputs is algebra: the
# per-example schedule coefficients are one_hot contractions, corruption is
# elementwise, and the loss is a plain sum of squares.
t_hot = F.one_hot(t, steps)  # (batch, steps) sparse indicator
xt = (t_hot @ sqrt_abar) * x0 + (t_hot @ sqrt_1mabar) * eps
temb = t_hot @ param("temb", steps=steps, dt=dt)  # learned time embedding
resid = eps - denoiser(xt, temb)
loss = F.sum(resid * resid) / (BATCH * 2)

# --------------------------------------------------------- lion as algebra
# Lion (Chen et al. 2023): the update direction is the SIGN of an
# interpolated momentum -- one moment buffer instead of AdamW's two, and no
# bias-correction schedule, so the step program has no per-step scalar
# inputs at all. sign() is one more primitive the compiler maps to
# torch.sign; the optimizer is still just three lines of algebra fused
# into the same program as the loss and its gradients.

moments = {n: Variable(f"m.{n}", p.shape) for n, p in params.items()}


@typed
def lion(w: Tensor, g: Tensor, m: Tensor) -> tuple[Tensor, Tensor]:
    """One Lion step -> (w', m')."""
    update = F.sign(B1 * m + (1 - B1) * g)
    decay = 1 - LR * WD if w.order >= 2 else 1  # no decay on gains/biases
    return w * decay - LR * update, B2 * m + (1 - B2) * g


# ------------------------------------------------------------ the sampler
# One ancestral DDPM step as a compiled program. All samples share the
# timestep, so tv is a 0-dim input and each coefficient is the scalar
# contraction one_hot(tv) @ table (minGPT's decode_pos paradigm).

xt_s = Variable("xt", batch, x)  # current noisy sample
z = Variable("z", batch, x)  # fresh gaussian per step (host RNG)
tv = Variable("tv")  # the shared timestep, 0-dim
inv_sqrt_alpha = Variable("inv_sqrt_alpha", steps)  # 1/sqrt(alpha_t)
beta_over = Variable("beta_over", steps)  # beta_t/sqrt(1-abar_t)
sigma = Variable("sigma", steps)  # posterior std (0 at t=0)

tv_hot = F.one_hot(tv, steps)  # (steps,) one-hot
eps_hat = denoiser(xt_s, tv_hot @ params["temb"])
x_prev = (tv_hot @ inv_sqrt_alpha) * (xt_s - (tv_hot @ beta_over) * eps_hat) + (tv_hot @ sigma) * z

# -------------------------------------------------------- metric as algebra
# 8 gaussians on a circle. A generated batch scores by nearest-mode
# distance: an outer difference, a sum of squares over 'x', a max over the
# mode edge, and a step function. hits/coverage are two outputs of one
# compiled expression.

modes = Variable("modes", mode, x)  # the true mode centers, fed once
diff = xt_s - modes  # (batch, mode, x): broadcast outer difference
d2 = F.sum(diff * diff, ["x"])  # squared distance to every mode
near2 = -F.max(-d2, dim="mode")  # min over modes = -max of negation
in_mode = F.gt0(0.35**2 - near2)  # within radius 0.35 of some mode
nearest = F.one_hot(F.argmax(-d2, dim="mode"), mode)  # (batch, mode) indicator

# ----------------------------------------------------------------- data


def make_schedule():
    """Standard DDPM linear-beta schedule, computed once. Returns the
    (steps,) tables the two programs consume."""
    beta = torch.linspace(1e-4, 0.02, T_STEPS)
    alpha = 1 - beta
    abar = torch.cumprod(alpha, 0)
    return dict(
        sqrt_abar=abar.sqrt(),
        sqrt_1mabar=(1 - abar).sqrt(),
        inv_sqrt_alpha=alpha.rsqrt(),
        beta_over=beta / (1 - abar).sqrt(),
        sigma=torch.cat([torch.zeros(1), beta[1:].sqrt()]),  # no noise at t=0
    )


MODE_CENTERS = torch.stack(
    [
        MODE_RADIUS * torch.tensor([math.cos(2 * math.pi * k / N_MODE), math.sin(2 * math.pi * k / N_MODE)])
        for k in range(N_MODE)
    ]
)


def sample_data(n, gen):
    """The host's only data job: pick a mode, add gaussian jitter."""
    which = torch.randint(N_MODE, (n,), generator=gen)
    return MODE_CENTERS[which] + MODE_STD * torch.randn(n, 2, generator=gen)


def init_weights(gen):
    ws = {}
    for name, var in params.items():
        shape = [DIMS[s] for s in var.shape.values()]
        last = name.rsplit(".", 1)[-1]
        ws[var] = (
            torch.ones(shape)
            if last == "g"
            else torch.zeros(shape)
            if last[0] == "b"
            else 0.02 * torch.randn(shape, generator=gen)
        )
    return ws


# ---------------------------------------------------------------- training


def main():
    grads = tg.grad(loss, params)
    updates = {}
    for n, p in params.items():
        updates[p], updates[moments[n]] = lion(p, grads[n], moments[n])
    step = tg.compile(loss=loss, state=updates, print_info=True)
    sample = tg.compile(x=x_prev)
    score = tg.compile(hits=F.sum(in_mode), per_mode=F.sum(nearest, ["batch"]))

    gen = torch.Generator().manual_seed(0)
    state = init_weights(gen)
    state |= {moments[n]: torch.zeros_like(state[p]) for n, p in params.items()}
    tables = make_schedule()

    def draw_samples(n, seed):
        """Run the compiled sampler from pure noise down to t=0."""
        g = torch.Generator().manual_seed(seed)
        dims = DIMS | {batch: n}
        coefs = {k: tables[k] for k in ("inv_sqrt_alpha", "beta_over", "sigma")}
        pts = torch.randn(n, 2, generator=g)
        for tt in range(T_STEPS - 1, -1, -1):
            pts = sample(
                state, dims=dims, xt=pts, z=torch.randn(n, 2, generator=g), tv=float(tt), **coefs
            ).x
        return pts

    def evaluate(n=512, seed=1234):
        pts = draw_samples(n, seed)
        out = score(dims=DIMS | {batch: n}, xt=pts, modes=MODE_CENTERS)
        return out.hits.item() / n, int(out.per_mode.min().item())

    acc, t_start = 0.0, time.perf_counter()
    for it in range(1, MAX_STEPS + 1):
        out = step(
            state,
            dims=DIMS,
            x0=sample_data(BATCH, gen),
            eps=torch.randn(BATCH, 2, generator=gen),
            t=torch.randint(T_STEPS, (BATCH,), generator=gen).float(),
            sqrt_abar=tables["sqrt_abar"],
            sqrt_1mabar=tables["sqrt_1mabar"],
        )
        state = out.state
        if it == 1:
            t_start, warmup = time.perf_counter(), time.perf_counter() - t_start
            print(f"first step (planning + codegen): {warmup:.1f}s")
        if it % 500 == 0:
            acc, min_cover = evaluate()
            rate = (it - 1) / (time.perf_counter() - t_start)
            print(
                f"step {it:4d}  loss {out.loss.item():.4f}  "
                f"in-mode rate {acc:.3f}  min mode count {min_cover}  ({rate:.1f} steps/s)"
            )
            if acc >= 0.97:
                break
    print(f"final in-mode rate: {acc:.3f} ({'PASS' if acc > 0.9 else 'FAIL'}, target 0.9)")


if __name__ == "__main__":
    main()
