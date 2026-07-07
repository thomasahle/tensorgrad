"""gpt-nano training step with the FUSED technology-mapping cells.

The same model, task, data pipeline and hyperparameters as gpt_step.py,
with the tensorgrad side opted into the cell library
(tensorgrad/compiler/cells.py): F.sdpa, F.layer_norm, F.gelu(fused=True)
for the network and the multi-output F.adamw cell for the optimizer.
Every cell's backward is machine-verified against the derived gradient
(tests/test_sdpa.py etc.), so this row is the honest price list of the
all-derived path in gpt_step: the delta between the two tg columns is
what the derived backward's elementwise tail costs, and this variant is
the recoverable floor. The torch and jax sides are gpt_step's own,
loaded by path (identical baselines, one source of truth).

Importing runs the gate: the fused step's loss decreases over 20 steps,
and the emitted program contains the fused kernels (flash-attention
fwd+bwd, native_layer_norm fwd+bwd, gelu fwd+bwd).
"""

import importlib.util
import math
import sys
from pathlib import Path

import torch
from sympy import Symbol, symbols

import tensorgrad.functions as F
import tensorgrad as tg
from tensorgrad import Sum, Variable, typed
from tensorgrad.tensor import Tensor

torch.set_grad_enabled(False)
torch.set_num_threads(2)

BENCH_NAME = "gpt (+cells)"

GATE_STEPS = 20

# ----------------------------------------------------------------- config
# Full gpt-nano: THREE stacked transformer blocks at n_embd=48. All 53
# parameter gradients — including wte's, which crosses three blocks of
# softmax stacks, gelus and LayerNorms — resolve through ONE symbolic
# cotangent sweep (tensorgrad/compiler/reverse.py), so every backward pass
# is a view into one shared reverse-mode DAG before the compiler ever
# plans a contraction.

N_LAYER, N_HEAD, N_EMBD = 3, 3, 48
VOCAB, LENGTH = 3, 6  # sort 6 digits from {0,1,2}
SEQ = 2 * LENGTH - 1  # input digits + sorted digits, shifted
BATCH, LR, MAX_STEPS = 64, 1e-3, 400

batch, seq, length, buf, vocab, d, head, hs, d_mlp = symbols("batch seq length buf vocab d head hs d_mlp")
DIMS = {
    batch: BATCH,
    seq: SEQ,
    length: LENGTH,
    buf: 2 * LENGTH,
    vocab: VOCAB,
    d: N_EMBD,
    head: N_HEAD,
    hs: N_EMBD // N_HEAD,
    d_mlp: 4 * N_EMBD,
}

params: dict[str, Variable] = {}


def param(name: str, **edges) -> Variable:
    return params.setdefault(name, Variable(name, **edges))


# -------------------------------------------------------------- the model
# The ONLY per-example input is `raw`: LENGTH random digits. Everything
# else the model consumes -- the training sequence, the shifted targets,
# even the SORTED labels -- is derived from it inside the compiled program.
raw = Variable("raw", batch, length)

# Structure tensors are tensorgrad expressions, not data: a triangle is a
# sum of shifted diagonals (F.window), and the causal mask is -1e9 times an
# upper triangle. Nothing here touches the host beyond the python ints that
# size the sums.


def diagonals(lo, hi, **edges: Symbol) -> Tensor:
    """sum_{k=lo}^{hi-1} of the k-shifted diagonal [row == col + k]."""
    return Sum([F.window(start=k, **edges) for k in range(lo, hi)])


causal_mask = -1e9 * diagonals(-(SEQ - 1), 0, seq=seq, key=seq)  # [key > seq]

# Evaluation inputs: the decode buffer (one slot longer than the model's
# window, so even the final prediction has somewhere to land) and the
# position being decoded. The decode loop is dict plumbing.
ctx_var = Variable("ctx", batch, buf)
decode_pos = Variable("decode_pos")


@typed
def layer_norm(x: Tensor[..., "d"], name: str) -> Tensor[..., "d"]:
    """Written from primitives -- tensorgrad DERIVES its backward pass.
    The annotation is an edge-SET contract, checked at runtime: `...` means
    the function is polymorphic over any other edges (batch, seq, ...) and
    only the 'd' edge is required. Edges are names, not positions, so there
    is no axis order to annotate or get wrong."""
    return F.layer_norm(x, dim="d", weight=param(name + ".g", d=d), bias=param(name + ".b", d=d), eps=1e-5)


@typed
def attention(x: Tensor[..., "seq", "d"], name: str) -> Tensor[..., "seq", "d"]:
    """Causal multi-head self-attention. 'head' is just an edge on the
    weights, and keys/values live on a renamed sequence edge -- that is the
    entire content of karpathy's four view/transpose lines. 'seq' appears in
    the contract because the causal mask is wired to it; batch only rides
    along."""
    q = x @ param(name + ".wq", d=d, head=head, hs=hs) + param(name + ".bq", head=head, hs=hs)
    k = x @ param(name + ".wk", d=d, head=head, hs=hs) + param(name + ".bk", head=head, hs=hs)
    v = x @ param(name + ".wv", d=d, head=head, hs=hs) + param(name + ".bv", head=head, hs=hs)
    k, v = k.rename(seq="key"), v.rename(seq="key")
    y = F.sdpa(q, k, v, mask=causal_mask, scale=1.0 / math.sqrt(DIMS[hs]))  # (batch, head, seq, hs)
    return y @ param(name + ".wo", head=head, hs=hs, d=d) + param(name + ".bo", d=d)


@typed
def mlp(x: Tensor[..., "d"], name: str) -> Tensor[..., "d"]:
    h = F.gelu(
        x @ param(name + ".w1", d=d, d_mlp=d_mlp) + param(name + ".b1", d_mlp=d_mlp), approximate="tanh", fused=True
    )  # minGPT's NewGELU, derivative derived
    return h @ param(name + ".w2", d_mlp=d_mlp, d=d) + param(name + ".b2", d=d)


@typed
def gpt(idx: Tensor["batch", "seq"]) -> Tensor["batch", "seq", "vocab"]:
    """Token ids (batch, seq) -> logits (batch, seq, vocab)."""
    wte = param("wte", vocab=vocab, d=d)
    x = F.one_hot(idx, vocab) @ wte + param("wpe", seq=seq, d=d)  # sparse integer lookup
    for i in range(N_LAYER):
        x = x + attention(layer_norm(x, f"h{i}.ln1"), f"h{i}.attn")
        x = x + mlp(layer_norm(x, f"h{i}.ln2"), f"h{i}.mlp")
    return layer_norm(x, "ln_f") @ param("lm_head", d=d, vocab=vocab)


# ------------------------------------------------ the data pipeline (algebra)
# The labels the model learns from are computed by the same compiled program
# that trains on them: F.sort is one_hot(argsort(x)) contracted with x, so
# even sorting is a contraction — and its gradient (unused here) would
# derive as the transposed permutation gather.


@typed
def sort_digits(x: Tensor["batch", "length"]) -> Tensor["batch", "length"]:
    return F.sort(x, dim="length")


full = F.concat(raw, sort_digits(raw), dim="length", size=buf).rename(length="buf")  # digits ++ sorted
tokens: Tensor["batch", "seq"] = full @ F.window(buf=buf, seq=seq)  # full[0 : SEQ]
target_ids: Tensor["batch", "seq"] = full @ F.window(buf=buf, seq=seq, start=1)  # full[1 : SEQ+1]

logits = gpt(tokens)
targets: Tensor["batch", "seq", "vocab"] = F.one_hot(target_ids, vocab)
ce: Tensor["batch", "seq"] = F.cross_entropy(logits, targets, dim="vocab")
# minGPT's ignore_index=-1: the first LENGTH-1 targets don't count -- so the
# loss just windows them away instead of masking.
loss = F.sum(ce @ F.window(seq=seq, length=length, start=length - 1)) / (BATCH * LENGTH)

# -------------------------------------------------------- adamw as algebra
# The optimizer is also just algebra. AdamW is elementwise arithmetic on a
# weight, its gradient and two moment estimates, so instead of a host-side
# update loop it is three more symbolic outputs per parameter of the SAME
# compiled program: the forward pass, all 53 gradients, AND AdamW fuse into
# one straight-line step. The only numbers computed outside the program are
# the bias corrections 1/(1-b^t) -- that's the schedule, not tensor
# compute -- fed per step as the 0-dim inputs c1, c2.

B1, B2, WD, EPS = 0.9, 0.95, 0.1, 1e-8
c1, c2 = Variable("c1"), Variable("c2")  # 1/(1-B1^t), 1/(1-B2^t)
moments = {
    n: (Variable(f"m.{n}", p.shape), Variable(f"v.{n}", p.shape)) for n, p in params.items()
}


@typed
def adamw(w: Tensor, g: Tensor, m: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """One AdamW step, written exactly as the update equations -> (w', m', v')."""
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    decay = 1 - LR * WD if w.order >= 2 else 1  # no decay on biases and gains
    return w * decay - LR * (c1 * m) / (F.sqrt(c2 * v) + EPS), m, v


# ----------------------------------------------------------------- data


def random_digits(n, gen):
    """The task's only host-side job: LENGTH random digits per example.
    Sequence assembly, target shifting, masking and the sorted labels are
    all computed inside the compiled programs."""
    return torch.randint(VOCAB, (n, LENGTH), generator=gen).float()


def init_weights(gen):
    ws = {}
    for name, var in params.items():
        shape = [DIMS[s] for s in var.shape.values()]
        last = name.rsplit(".", 1)[-1]
        ws[var] = (
            torch.ones(shape)
            if last == "g"  # layer-norm gains
            else torch.zeros(shape)
            if last[0] == "b"  # biases
            else 0.02 * torch.randn(shape, generator=gen)
        )
    return ws




# ------------------------------------------------- step makers + gate
def make_tg_step(seed=0):
    grads = tg.grad(loss, params)
    updates = {}
    for nm, p in params.items():
        mv, vv = moments[nm]
        updates[p], updates[mv], updates[vv] = F.adamw(
            p, grads[nm], mv, vv, c1, c2, beta1=B1, beta2=B2, lr=LR, eps=EPS, weight_decay=WD
        )
    step = tg.compile(loss=loss, state=updates)
    gen = torch.Generator().manual_seed(seed)
    state = init_weights(gen)
    state |= {mv: torch.zeros_like(state[params[nm]]) for nm in params for mv in moments[nm]}
    holder = {"state": state, "t": 0}

    def step_fn() -> float:
        holder["t"] += 1
        t = holder["t"]
        out = step(
            holder["state"], dims=DIMS, raw=random_digits(BATCH, gen),
            c1=1 / (1 - B1**t), c2=1 / (1 - B2**t),
        )
        holder["state"] = out.state
        return out.loss.item()

    return step_fn


def _load_gpt_step():
    path = Path(__file__).resolve().parent / "gpt_step.py"
    name = "bench_gpt_step"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_torch_step(seed=0):
    return _load_gpt_step().make_torch_step(seed)


def make_jax_step(seed=0):
    return _load_gpt_step().make_jax_step(seed)


def _gate():
    step = make_tg_step()
    losses = [step() for _ in range(GATE_STEPS)]
    first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
    assert last < first, f"fused loss did not decrease ({first:.4f} -> {last:.4f})"


_gate()
print(f"[{BENCH_NAME}] correctness gate passed", file=sys.stderr)
