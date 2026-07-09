"""Definition folding: the catalog (cells declare their derived composition)
and, as later steps land, the fold engine (compiler/fold.py) and its
tg.compile integration. Template: test_softmax_fused.py -- tiny dims, fixed
seeds, gradient comparisons vs the derived form.
"""

import torch
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.compiler.cells import CELLS, Definition
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)

RTOL, ATOL = 1e-4, 1e-6


def test_gelu_definition_data():
    """Definition.derived and Definition.fused compute the same values --
    the rule's two sides agree (pure cells.py property, no engine)."""
    d = CELLS["gelu"].definition()
    assert isinstance(d, Definition) and d.cell == "gelu"

    i = symbols("i")
    x = Variable("x", i=i)
    params = {"approximate": "tanh"}
    torch.manual_seed(0)
    xv = torch.randn(11).rename("i")

    derived = d.derived([x], params)
    fused = d.fused([x], params)
    a = evaluate(derived, {x: xv}, {i: 11})
    b = evaluate(fused, {x: xv}, {i: 11})
    torch.testing.assert_close(
        a.align_to("i").rename(None), b.align_to("i").rename(None), rtol=RTOL, atol=ATOL
    )


def test_cells_without_definition_return_none():
    """Cells that declare no derived composition simply never fold."""
    for name, cell in CELLS.items():
        d = cell.definition()
        assert d is None or (isinstance(d, Definition) and d.cell == name)


# ---------------------------------------------------------------------------
# The engine, end-to-end through compile (gelu = the archetype cell)
# ---------------------------------------------------------------------------


def _gelu_mlp():
    b, d, m = symbols("b d m")
    x = Variable("x", b=b, d=d)
    w = Variable("w", d=d, m=m)
    bias = Variable("bias", m=m)
    h = F.gelu(x @ w + bias, approximate="tanh")
    loss = F.sum(h * h)
    torch.manual_seed(0)
    vals = {
        x: torch.randn(4, 5).rename("b", "d"),
        w: torch.randn(5, 6).rename("d", "m"),
        bias: torch.randn(6).rename("m"),
    }
    return loss, (w, bias), vals, {b: 4, d: 5, m: 6}


def _source(prog, vals, dims):
    prog(dict(vals), dict(dims))  # force specialization
    (spec,) = prog._specializations.values()
    return spec._source


def test_fold_gelu_mlp_fuses_and_matches_derived():
    """The derived tanh-gelu composition folds to the fused cell: emitted
    source uses the fused fwd+bwd kernels, and loss + gradients equal the
    unfolded program's (the safety-critical property)."""
    loss, (w, bias), vals, dims = _gelu_mlp()
    prog = compile_to_callable(loss, loss.grad(w), loss.grad(bias))
    prog0 = compile_to_callable(loss, loss.grad(w), loss.grad(bias), fold=False)
    assert prog.fold_fires == {"gelu": 1}
    assert prog0.fold_fires == {}

    src = _source(prog, vals, dims)
    assert "functional.gelu" in src and "gelu_backward" in src
    assert "torch.tanh" not in src  # no derived remnants of the folded site

    for a, c in zip(prog(vals, dims), prog0(vals, dims)):
        names = sorted(a.names)
        torch.testing.assert_close(
            a.align_to(*names).rename(None), c.align_to(*names).rename(None), rtol=RTOL, atol=ATOL
        )


def test_fold_near_miss_is_a_no_op():
    """A gelu-shaped tree with the WRONG constant (0.4 instead of 1/2) must
    not fold -- and must still compile correctly."""
    b, d = symbols("b d")
    x = Variable("x", b=b, d=d)
    from tensorgrad.functions import pow as tpow, tanh
    import math

    c = math.sqrt(2.0 / math.pi)
    almost = x * (1 + tanh(c * (x + 0.044715 * tpow(x, 3)))) * 0.4  # not gelu
    loss = F.sum(almost)
    prog = compile_to_callable(loss, loss.grad(x), loss.grad(x) + loss.grad(x))
    assert "gelu" not in prog.fold_fires
    torch.manual_seed(1)
    vals = {x: torch.randn(3, 4).rename("b", "d")}
    (lv, *_) = prog(vals, {b: 3, d: 4})
    # x*(1+tanh(...))*0.4 == 0.8 * gelu(x, "tanh")
    ref = 0.8 * torch.nn.functional.gelu(vals[x].rename(None), approximate="tanh").sum()
    torch.testing.assert_close(lv.rename(None), ref, rtol=RTOL, atol=ATOL)


def test_fold_flag_and_kwarg_disable():
    """fold=False and fold.FOLD=False both leave the program unfolded."""
    from tensorgrad.compiler import fold as fold_mod

    loss, (w, bias), vals, dims = _gelu_mlp()
    prog = compile_to_callable(loss, loss.grad(w), loss.grad(bias), fold=False)
    assert prog.fold_fires == {} and "torch.tanh" in _source(prog, vals, dims)

    old = fold_mod.FOLD
    fold_mod.FOLD = False
    try:
        prog2 = compile_to_callable(loss, loss.grad(w), loss.grad(bias))
        assert prog2.fold_fires == {}
    finally:
        fold_mod.FOLD = old


def test_fold_singleton_gradient_family_skips():
    """A lone Derivative output resolves through forward-mode step_derivative,
    where a fused cell would raise -- folding must skip the whole program and
    the program must still compile and run (guards the reverse.py singleton
    interaction)."""
    loss, (w, bias), vals, dims = _gelu_mlp()
    prog = compile_to_callable(loss, loss.grad(w))  # family of ONE
    assert prog.fold_fires == {"skipped:unresolved-gradient-family": 1}
    out = prog(vals, dims)
    assert len(out) == 2


def test_fold_interpreter_evaluate_agrees():
    """evaluate() of the folded tree equals evaluate() of the derived tree
    (the fused cell's eval oracle is the same value)."""
    from tensorgrad.compiler.fold import fold_program

    loss, _, vals, dims = _gelu_mlp()
    (folded,), fires = fold_program((loss,))
    assert fires.get("gelu") == 1
    a = evaluate(loss, dict(vals), dict(dims))
    c = evaluate(folded, dict(vals), dict(dims))
    torch.testing.assert_close(a.rename(None), c.rename(None), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# layer_norm / attention / adamw definitions on realistic structures
# ---------------------------------------------------------------------------


def _mini_block():
    """One multi-head pre-LN transformer block, hand-written from primitives
    (mingpt's idioms: renamed key edge, F.window causal mask, broadcast
    biases). Returns (out, params dict, symbol dims)."""
    import math
    from tensorgrad.tensor import Sum

    d, seq, head, hs, dmlp, batch = symbols("d seq head hs d_mlp batch")
    x = Variable("x", batch=batch, seq=seq, d=d)
    P: dict = {}

    def param(n, **e):
        return P.setdefault(n, Variable(n, **e))

    def layer_norm(z, nm):
        zc = z - F.mean(z, dim="d", keepdims=True)
        var = F.mean(zc * zc, dim="d", keepdims=True)
        return zc / F.sqrt(var + 1e-5) * param(nm + ".g", d=d) + param(nm + ".b", d=d)

    def attention(z, nm):
        q = z @ param(nm + ".wq", d=d, head=head, hs=hs)
        k = (z @ param(nm + ".wk", d=d, head=head, hs=hs)).rename(seq="key")
        v = (z @ param(nm + ".wv", d=d, head=head, hs=hs)).rename(seq="key")
        cmask = -1e9 * Sum([F.window(start=kk, seq=seq, key=seq) for kk in range(-2, 1)])
        att = F.softmax(F.dot(q, k, dim="hs") / math.sqrt(4) + cmask, dim="key")
        return F.dot(att, v, dim="key") @ param(nm + ".wo", head=head, hs=hs, d=d)

    def mlp(z, nm):
        h = F.gelu(z @ param(nm + ".w1", d=d, d_mlp=dmlp), approximate="tanh")
        return h @ param(nm + ".w2", d_mlp=dmlp, d=d)

    z = x + attention(layer_norm(x, "ln1"), "attn")
    z = z + mlp(layer_norm(z, "ln2"), "mlp")
    return z, P | {"x": x}, {batch: 2, seq: 5, d: 4, head: 2, hs: 4, dmlp: 6}


def test_fold_full_block_fire_counts():
    """One transformer block folds EXACTLY {sdpa:1, layer_norm:2, gelu:1} --
    pins the two-phase matcher, the DAG memo, the depth-robust extraction
    and the overlap rules (nested ln inside sdpa's q/k/v inputs survives)."""
    from tensorgrad.compiler.fold import fold_program

    z, _, _ = _mini_block()
    (_,), fires = fold_program((z,))
    assert fires == {"sdpa": 1, "layer_norm": 2, "gelu": 1}, fires


def test_fold_block_gradients_match_derived():
    """Compiled loss + gradients of the folded block equal the fold=False
    program's -- the safety-critical property, through the whole pipeline."""
    z, P, dims = _mini_block()
    loss = F.sum(z * z)
    wrt = [P["attn.wq"], P["ln1.g"], P["mlp.w1"]]
    prog = compile_to_callable(loss, *[loss.grad(p) for p in wrt])
    prog0 = compile_to_callable(loss, *[loss.grad(p) for p in wrt], fold=False)
    assert prog.fold_fires.get("sdpa") == 1 and prog.fold_fires.get("layer_norm") == 2

    torch.manual_seed(0)
    vals = {}
    for v in P.values():
        sizes = [ {"batch": 2, "seq": 5, "d": 4, "head": 2, "hs": 4, "d_mlp": 6}[e]
                  for e in v.edges ]
        vals[v] = (0.3 * torch.randn(*sizes)).rename(*v.edges)
    for a, c in zip(prog(vals, dims), prog0(vals, dims)):
        names = sorted(a.names)
        torch.testing.assert_close(
            a.align_to(*names).rename(None), c.align_to(*names).rename(None),
            rtol=1e-3, atol=1e-5,
        )


def test_fold_adamw_training_step():
    """A real (two-parameter) training step: the optimizer algebra folds to
    the multi-output adamw cell (w'/m'/v' rebuilt from ONE cell call via
    aliases), and the compiled updates equal the derived program's."""
    b, d, e = symbols("b d e")
    x = Variable("x", b=b, d=d)
    ws = {"w": Variable("w", d=d, e=e), "u": Variable("u", e=e)}
    loss = F.sum(F.gelu((x @ ws["w"]) * ws["u"], approximate="tanh"))
    B1, B2, LR, EPS = 0.9, 0.95, 1e-3, 1e-8
    c1 = Variable("c1")
    c2 = Variable("c2")
    outs = [loss]
    state = {}
    for n, w in ws.items():
        g = loss.grad(w)
        m = state[f"m.{n}"] = Variable(f"m.{n}", **{k: d if k == "d" else e for k in w.edges})
        v = state[f"v.{n}"] = Variable(f"v.{n}", **{k: d if k == "d" else e for k in w.edges})
        m2 = B1 * m + (1 - B1) * g
        v2 = B2 * v + (1 - B2) * g * g
        w2 = w * 0.9999 - LR * (c1 * m2) / (F.sqrt(c2 * v2) + EPS)
        outs += [w2, m2, v2]

    from tensorgrad.compiler.fold import fold_program

    _, fires = fold_program(tuple(outs))
    assert fires.get("adamw") == 2, fires

    prog = compile_to_callable(*outs)
    prog0 = compile_to_callable(*outs, fold=False)
    assert prog.fold_fires.get("adamw") == 2
    torch.manual_seed(1)
    dims = {b: 3, d: 4, e: 5}
    concrete = {"b": 3, "d": 4, "e": 5}
    vals = {x: torch.randn(3, 4).rename("b", "d"), c1: torch.tensor(1.5), c2: torch.tensor(1.2)}
    for var in list(ws.values()) + list(state.values()):
        sizes = [concrete[str(var.shape[k])] for k in var.edges]
        t = torch.randn(*sizes)
        if var.name.startswith("v."):
            t = t.abs()  # realistic: v is a running square
        vals[var] = t.rename(*var.edges)
    for a, c in zip(prog(vals, dims), prog0(vals, dims)):
        names = sorted(a.names)
        torch.testing.assert_close(
            a.align_to(*names).rename(None), c.align_to(*names).rename(None),
            rtol=1e-4, atol=1e-6,
        )


def test_fold_layer_norm_near_miss_unbiased_var():
    """An almost-layer-norm with the WRONG variance normalization must not
    fold (the value gate rejects; the program still compiles correctly)."""
    from tensorgrad.compiler.fold import fold_program

    d, seq = symbols("d seq")
    x = Variable("x", seq=seq, d=d)
    g = Variable("g", d=d)
    b = Variable("b", d=d)
    xc = x - F.mean(x, dim="d", keepdims=True)
    # wrong: mean over the OTHER axis -- shapes still work out via broadcast
    var = F.mean(xc * xc, dim="seq", keepdims=True)
    almost = xc / F.sqrt(var + 1e-5) * g + b
    (_,), fires = fold_program((almost,))
    assert "layer_norm" not in fires, fires
