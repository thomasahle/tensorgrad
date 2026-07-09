"""One full CNN training step: tensorgrad vs torch autograd (correctness + timing).

Small conv net on synthetic 28x28 "images":

    conv1 (1->8, 5x5) -> relu -> mean-pool 2x2 -> conv2 (8->16, 5x5) -> relu
    -> global mean pool -> linear -> cross-entropy vs fixed random labels

The tensorgrad side is ONE compiled program: forward loss, all 6 parameter
gradients, and the SGD update w' = w - lr*g, mapping state -> new state.
2D convolution is two structural Convolution constants (C[i,j,k] = [i == j+k],
as in examples/mnist.py), so conv backward derives as the transposed
contraction with the same constants (scatter / transposed-conv pattern), and
the 2x2 mean-pool is a Reshape unflatten (h -> (ph, hp)) followed by a mean
over the fine edges -- an averaging contraction, no pooling primitive.

The torch reference is functional torch with autograd backward and the same
SGD math. Correctness gate at import: 3 steps from identical init on identical
data must give matching losses (rtol 1e-4).
"""

import math
import time

import torch
from sympy import symbols

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable

torch.set_num_threads(2)

BENCH_NAME = "conv_step"

# ----------------------------------------------------------------- config
BATCH, IMG_HW, KS, C1, C2, OUT = 64, 28, 5, 8, 16, 10
HW1 = IMG_HW - KS + 1  # 24 after conv1
HWP = HW1 // 2  # 12 after 2x2 mean-pool
HW2 = HWP - KS + 1  # 8 after conv2
LR = 0.05

batch, c0, h0, w0, ks = symbols("batch c0 h0 w0 ks")
c1, h1, w1, ph, pw, hp, wp = symbols("c1 h1 w1 ph pw hp wp")
c2, h2, w2, out = symbols("c2 h2 w2 out")
DIMS = {
    batch: BATCH,
    c0: 1,
    h0: IMG_HW,
    w0: IMG_HW,
    ks: KS,
    c1: C1,
    h1: HW1,
    w1: HW1,
    ph: HWP,
    pw: HWP,
    hp: 2,
    wp: 2,
    c2: C2,
    h2: HW2,
    w2: HW2,
    out: OUT,
}

params: dict[str, Variable] = {}


def param(name: str, **edges) -> Variable:
    return params.setdefault(name, Variable(name, **edges))


# -------------------------------------------------------------- the model
img = Variable("img", batch, c0, h0, w0)
labels = Variable("labels", batch)  # integer class ids carried as floats


def cnn(x):
    # conv1: the 2D convolution is two structural 3-tensors [i == j + k],
    # one per spatial axis; contracting the kernel over (c0, kh, kw) is
    # exactly cross-correlation, i.e. torch's conv2d.
    x = x @ F.Convolution(h0, kh=ks, h1=h1) @ F.Convolution(w0, kw=ks, w1=w1)
    x = x @ param("conv1.w", c0=c0, c1=c1, kh=ks, kw=ks) + param("conv1.b", c1=c1)
    x = F.relu(x)
    # mean-pool 2x2: unflatten h1 -> (ph, hp) and w1 -> (pw, wp) with the
    # structural Reshape identity, then average the fine 2x2 edges.
    x = x @ F.Reshape(h1, ph, hp) @ F.Reshape(w1, pw, wp)
    x = F.mean(x, dim=["hp", "wp"])
    # conv2, relu, global mean pool
    x = x @ F.Convolution(ph, kh2=ks, h2=h2) @ F.Convolution(pw, kw2=ks, w2=w2)
    x = x @ param("conv2.w", c1=c1, c2=c2, kh2=ks, kw2=ks) + param("conv2.b", c2=c2)
    x = F.relu(x)
    x = F.mean(x, dim=["h2", "w2"])  # (batch, c2)
    return x @ param("fc.w", c2=c2, out=out) + param("fc.b", out=out)


logits = cnn(img)  # (batch, out)
targets = F.one_hot(labels, out)  # integer labels stay integers
loss = F.mean(F.cross_entropy(logits, targets, dim="out"), dim="batch")

# One compiled program: loss, all gradients (one shared cotangent sweep),
# and the SGD update -- state in, new state out.
grads = tg.grad(loss, params)
updates = {p: p - LR * grads[n] for n, p in params.items()}
STEP = tg.compile(loss=loss, state=updates)

# ----------------------------------------------------------------- data
_gen = torch.Generator().manual_seed(0)
IMG = torch.randn(BATCH, 1, IMG_HW, IMG_HW, generator=_gen)
LABELS = torch.randint(0, OUT, (BATCH,), generator=_gen)


def torch_init(seed=1):
    """Torch-layout initial weights (conv2d/linear kaiming-uniform style).
    The single source of init for BOTH implementations."""
    g = torch.Generator().manual_seed(seed)

    def u(*shape, fan_in):
        bound = 1 / math.sqrt(fan_in)
        return (torch.rand(*shape, generator=g) * 2 - 1) * bound

    return {
        "conv1.w": u(C1, 1, KS, KS, fan_in=KS * KS),
        "conv1.b": u(C1, fan_in=KS * KS),
        "conv2.w": u(C2, C1, KS, KS, fan_in=C1 * KS * KS),
        "conv2.b": u(C2, fan_in=C1 * KS * KS),
        "fc.w": u(OUT, C2, fan_in=C2),
        "fc.b": u(OUT, fan_in=C2),
    }


def tg_state(w):
    """Copy the torch init into the tensorgrad state: name the axes and the
    runtime aligns them to each Variable's edge order."""
    names = {
        "conv1.w": ("c1", "c0", "kh", "kw"),
        "conv1.b": ("c1",),
        "conv2.w": ("c2", "c1", "kh2", "kw2"),
        "conv2.b": ("c2",),
        "fc.w": ("out", "c2"),
        "fc.b": ("out",),
    }
    return {params[k]: w[k].clone().rename(*names[k]) for k in params}


# ------------------------------------------------------------- step makers
def make_tg_step():
    state = tg_state(torch_init())
    img_named = IMG.rename("batch", "c0", "h0", "w0")
    labels_f = LABELS.float().rename("batch")

    def step_fn():
        nonlocal state
        out_ = STEP(state, dims=DIMS, img=img_named, labels=labels_f)
        state = out_.state  # the whole SGD update
        return out_.loss.item()

    return step_fn


class _ConvNet(torch.nn.Module):
    """The torch reference as an nn.Module, matching the suite's convention
    (mlp/gpt torch sides): under --torch-compile the harness compiles the
    MODULE forward and leaves backward to autograd and the update eager.
    (The previous purely-functional step -- backward + in-place SGD inside
    one function -- crashed dynamo's whole-function trace on this torch
    version, leaving the row unmeasurable.)"""

    def __init__(self, w: dict):
        super().__init__()
        self.p = torch.nn.ParameterDict(
            {k.replace(".", "_"): torch.nn.Parameter(v.clone()) for k, v in w.items()}
        )

    def forward(self, img):
        x = torch.nn.functional.conv2d(img, self.p["conv1_w"], self.p["conv1_b"])
        x = torch.relu(x)
        x = torch.nn.functional.avg_pool2d(x, 2)
        x = torch.nn.functional.conv2d(x, self.p["conv2_w"], self.p["conv2_b"])
        x = torch.relu(x)
        x = x.mean(dim=(2, 3))
        return torch.nn.functional.linear(x, self.p["fc_w"], self.p["fc_b"])


def make_torch_step():
    model = _ConvNet(torch_init())
    opt = torch.optim.SGD(model.parameters(), lr=LR)  # plain p -= LR * grad

    def step_fn():
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss_ = torch.nn.functional.cross_entropy(model(IMG), LABELS)
            loss_.backward()
        opt.step()
        return loss_.item()

    return step_fn


# ------------------------------------------------- correctness gate (import)
def _correctness_gate():
    tg_fn, th_fn = make_tg_step(), make_torch_step()
    for i in range(3):
        lt, lh = tg_fn(), th_fn()
        assert math.isclose(lt, lh, rel_tol=1e-4, abs_tol=1e-6), (
            f"step {i}: tensorgrad loss {lt} != torch loss {lh}"
        )


_correctness_gate()


# ----------------------------------------------------------------- timing
def _bench(step_fn, reps=10, warmup=3):
    for _ in range(warmup):
        step_fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        step_fn()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


if __name__ == "__main__":
    t_tg = _bench(make_tg_step())
    t_torch = _bench(make_torch_step())
    print(f"{BENCH_NAME}: tg {t_tg:.1f}ms torch {t_torch:.1f}ms")


def make_jax_step():
    """Architecture parity: conv(1->8,5x5) relu avgpool2 conv(8->16,5x5) relu
    globalmeanpool linear cross-entropy, plain SGD. Torch-layout weights, jit
    end-to-end."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    P = {k: jnp.asarray(v.numpy()) for k, v in torch_init().items()}
    imgj, labj = jnp.asarray(IMG.numpy()), jnp.asarray(LABELS.numpy())

    def conv(x, w, b):  # x:(N,Cin,H,W), w:(Cout,Cin,kh,kw) torch/OIHW layout
        y = lax.conv_general_dilated(x, w, (1, 1), "VALID",
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"))
        return y + b[None, :, None, None]

    def loss_fn(P):
        x = jax.nn.relu(conv(imgj, P["conv1.w"], P["conv1.b"]))
        n, c, h, w = x.shape
        x = x.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))  # avg-pool 2x2
        x = jax.nn.relu(conv(x, P["conv2.w"], P["conv2.b"]))
        x = x.mean((2, 3))  # global mean pool
        logits = x @ P["fc.w"].T + P["fc.b"]
        logp = jax.nn.log_softmax(logits, -1)
        return -jnp.take_along_axis(logp, labj[:, None], -1).mean()

    @jax.jit
    def update(P):
        lv, gr = jax.value_and_grad(loss_fn)(P)
        return jax.tree.map(lambda w, gw: w - LR * gw, P, gr), lv

    st = {"P": P}

    def step_fn() -> float:
        st["P"], lv = update(st["P"])
        return float(lv)

    return step_fn
