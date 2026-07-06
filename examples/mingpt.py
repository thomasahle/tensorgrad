"""minGPT (https://github.com/karpathy/minGPT) as a tensorgrad program.

What does a GPT look like when every tensor axis has a NAME and every gradient
is DERIVED symbolically instead of traced by autograd?

* No .view / .transpose / .reshape anywhere. Karpathy's

      k = k.view(B, T, nh, hs).transpose(1, 2)   # (B, nh, T, hs)

  is here just an edge named 'head' on the projection weights. The only
  axis bookkeeping in the whole model is semantic: attention keys/values get
  their sequence edge renamed 'seq' -> 'key', so scores naturally come out
  with (seq, key) edges.

* No autograd. torch.set_grad_enabled(False) below is global and permanent:
  loss.grad(p) builds a symbolic derivative, and compile_to_callable turns
  the loss, one gradient per parameter, AND the AdamW update into ONE fused
  straight-line program in which forward work is shared across outputs.
  PyTorch is used purely as a tensor library.

* No hand-written backward rules for composites. layer_norm below is three
  lines of mean/sqrt and its Jacobian is derived; softmax (and the
  log_softmax inside the loss) is fused and numerically stabilized by the
  compiler; gelu's tanh chain is differentiated symbolically too.

* Integer tokens stay integers: F.one_hot(tokens) is a sparse indicator
  tensor, and the embedding lookup is literally the contraction
  one_hot @ wte -- the compiler maps it to index_select and the DERIVED
  gradient to a scatter-add; the dense indicator never exists. Integer
  TARGETS stay integers the same way: one_hot builds the target
  distribution inside the program, and minGPT's ignore_index=-1 is a
  window that drops the ignored positions from the loss. No host-side
  encoding, no masking.

* The data pipeline and the metric are algebra too. The model's only
  per-example input is `raw`: LENGTH random digits. The training sequence,
  the shifted targets, and even the SORTED LABELS derive from it inside the
  compiled program -- counting sort is a one_hot contraction, a triangle
  contraction (cumulative counts) and step-function indicators. Greedy
  decode is argmax + a one_hot position-select, writing a prediction into
  the context is adding a one_hot outer product, and held-out exact-match
  is one more compiled expression. PyTorch's only remaining jobs: random
  number generation, building 0/1 structure constants, and 0-dim scalars.

* No optimizer library. AdamW is elementwise algebra on a weight, its
  gradient and two moment estimates, so it compiles INTO the step program:
  one call maps (weights, moments, batch) to (loss, new weights, new
  moments), and the python training loop is a dict reassignment. No
  in-place tensor mutation anywhere.

Running this file trains the full 3-block gpt-nano on karpathy's sorting
task ("given 6 digits, emit them sorted") to >90% held-out sequence
accuracy in ~15 seconds on CPU: ~5s to derive, plan and code-generate the
~1450-op fused step program (160 outputs: loss, 53 weights, 106 moments),
then training at ~28 steps/s eager — each step one call, weights in,
weights out. The gradients of one loss resolve JOINTLY through a single
symbolic cotangent sweep (tensorgrad/compiler/reverse.py), so all 53
backward passes share one interned reverse-mode DAG.
"""

import math
import time

import torch
from sympy import Symbol, symbols

import tensorgrad.functions as F
import tensorgrad as tg
from tensorgrad import Sum, Variable, typed
from tensorgrad.tensor import Tensor

# The punchline: no gradient tape, ever. All gradients in this file are
# symbolic expressions, derived from the loss and compiled ahead of time.
torch.set_grad_enabled(False)
torch.set_num_threads(2)

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
    x = x - F.mean(x, dim="d", keepdims=True)
    var = F.mean(x * x, dim="d", keepdims=True)
    return x / F.sqrt(var + 1e-5) * param(name + ".g", d=d) + param(name + ".b", d=d)


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
    att = F.dot(q, k, dim="hs") / math.sqrt(DIMS[hs])  # (batch, head, seq, key)
    att = F.softmax(att + causal_mask, dim="key")  # stabilized by the compiler
    y = F.dot(att, v, dim="key")  # (batch, head, seq, hs)
    return y @ param(name + ".wo", head=head, hs=hs, d=d) + param(name + ".bo", d=d)


@typed
def mlp(x: Tensor[..., "d"], name: str) -> Tensor[..., "d"]:
    h = F.gelu(
        x @ param(name + ".w1", d=d, d_mlp=d_mlp) + param(name + ".b1", d_mlp=d_mlp), approximate="tanh"
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


# ---------------------------------------------------------------- training


def main():
    # One symbolic cotangent sweep for all 53 gradients, then the whole
    # optimizer step as ONE pytree keyed by variable name: the result's
    # .state dict feeds straight back in, so the training loop is
    # `state = step(state, ...).state` -- no order bookkeeping anywhere.
    grads = tg.grad(loss, params)
    updates = {}
    for n, p in params.items():
        mv, vv = moments[n]
        updates[p], updates[mv], updates[vv] = adamw(p, grads[n], mv, vv)
    step = tg.compile(loss=loss, state=updates, print_info=True)
    # Evaluation is algebra too. One decode step: run the model on the
    # buffer's first `seq` slots, argmax over vocab, select the position
    # being decoded (a one_hot contraction), and write the prediction into
    # the NEXT slot (adding a one_hot outer product). One compiled program,
    # buffer in, buffer out; the decode loop is dict plumbing.
    eval_logits = gpt(ctx_var @ F.window(buf=buf, seq=seq))
    next_id = F.argmax(eval_logits, dim="vocab") @ F.one_hot(decode_pos, seq)
    decode = tg.compile(ctx=ctx_var + next_id * F.one_hot(decode_pos + 1, buf))
    seed_ctx = tg.compile(ctx=raw @ F.window(length=length, buf=buf))  # zero-padded embed
    # Exact-match accuracy: the decoded half of the buffer against the SAME
    # sort_digits program the training labels come from.
    decoded = ctx_var @ F.window(buf=buf, length=length, start=length)
    correct = F.equal(decoded, sort_digits(raw))  # (batch, length) of 0/1
    solved = F.gt0(F.sum(correct, ["length"]) - (LENGTH - 0.5))  # all six right
    score = tg.compile(hits=F.sum(solved))

    gen = torch.Generator().manual_seed(0)
    state = init_weights(gen)
    state |= {mv: torch.zeros_like(state[params[n]]) for n in params for mv in moments[n]}

    def held_out_accuracy(n=256, seed=1234):
        """Greedy autoregressive decode on fresh problems; exact-match rate."""
        g = torch.Generator().manual_seed(seed)
        inp = random_digits(n, g)
        dims = DIMS | {batch: n}
        ctx = seed_ctx(dims=dims, raw=inp).ctx
        for t in range(LENGTH - 1, SEQ):
            ctx = decode(state, dims=dims, ctx=ctx, decode_pos=float(t)).ctx
        return score(dims=dims, ctx=ctx, raw=inp).hits.item() / n

    acc, t_start = 0.0, time.perf_counter()
    for it in range(1, MAX_STEPS + 1):
        out = step(state, dims=DIMS, raw=random_digits(BATCH, gen),
                   c1=1 / (1 - B1**it), c2=1 / (1 - B2**it))
        state = out.state  # the whole optimizer step
        if it == 1:  # first call pays planning + codegen; time the rest
            t_start, warmup = time.perf_counter(), time.perf_counter() - t_start
            print(f"first step (planning + codegen): {warmup:.1f}s")
        if it % 50 == 0:
            acc = held_out_accuracy()
            rate = (it - 1) / (time.perf_counter() - t_start)
            print(f"step {it:4d}  loss {out.loss.item():.4f}  held-out acc {acc:.3f}  ({rate:.1f} steps/s)")
            if acc >= 0.99:
                break
    print(f"final held-out accuracy: {acc:.3f} ({'PASS' if acc > 0.9 else 'FAIL'}, target 0.9)")


if __name__ == "__main__":
    main()
