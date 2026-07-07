"""F.sdpa: fused scaled-dot-product attention (technology-mapping primitive).

Forward IS softmax(scale*<q,k>_hs + mask) @_key v (machine-checked against
that definition); it compiles to torch's flash-attention CPU kernels for the
forward AND the reverse-mode backward, whose gradients are verified here
against torch.autograd. The backward is fused only in REVERSE mode, so the
gradient tests compile a FAMILY (as any training loop does).
"""

import sympy
import torch

import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate

torch.set_num_threads(2)
b, h, s, k, e = sympy.symbols("b h s k e")


def _rand(*shape):
    return torch.randn(*shape)


def test_forward_matches_definition_eager_and_compiled():
    torch.manual_seed(0)
    dims = {b: 2, h: 3, s: 5, k: 5, e: 4}
    q = Variable("q", b=b, h=h, seq=s, hs=e)
    kk = Variable("kk", b=b, h=h, key=k, hs=e)
    vv = Variable("vv", b=b, h=h, key=k, hs=e)
    out = F.sdpa(q, kk, vv, scale=0.5)
    assert set(out.edges) == {"b", "h", "seq", "hs"}
    qv, kv, vvv = _rand(2, 3, 5, 4), _rand(2, 3, 5, 4), _rand(2, 3, 5, 4)
    feed = {q: qv.rename("b", "h", "seq", "hs"), kk: kv.rename("b", "h", "key", "hs"),
            vv: vvv.rename("b", "h", "key", "hs")}
    ref = torch.nn.functional.scaled_dot_product_attention(qv, kv, vvv, scale=0.5)
    for got in (evaluate(out.simplify(), dict(feed), dims),
                compile_to_callable(out)(dict(feed), dims)):
        torch.testing.assert_close(got.align_to("b", "h", "seq", "hs").rename(None), ref)


def _grad_family(mask_var):
    torch.manual_seed(1)
    dims = {b: 2, h: 3, s: 5, e: 4}
    q = Variable("q", b=b, h=h, seq=s, hs=e)
    kk = Variable("kk", b=b, h=h, key=s, hs=e)
    vv = Variable("vv", b=b, h=h, key=s, hs=e)
    w = Variable("w", b=b, h=h, seq=s, hs=e)
    mv = torch.triu(torch.full((5, 5), -1e9), 1) if mask_var else None
    kwargs = {"scale": 0.5}
    feed = {q: _rand(2, 3, 5, 4).rename("b", "h", "seq", "hs"),
            kk: _rand(2, 3, 5, 4).rename("b", "h", "key", "hs"),
            vv: _rand(2, 3, 5, 4).rename("b", "h", "key", "hs"),
            w: _rand(2, 3, 5, 4).rename("b", "h", "seq", "hs")}
    if mask_var:
        m = Variable("m", seq=s, key=s)
        kwargs["mask"] = m
        feed[m] = mv.rename("seq", "key")
    loss = F.sum(F.sdpa(q, kk, vv, **kwargs) * w)
    prog = compile_to_callable(loss.grad(q), loss.grad(kk), loss.grad(vv))
    gq, gk, gv = prog(dict(feed), dims)
    qt, kt, vt = (feed[q].rename(None).clone().requires_grad_(True),
                  feed[kk].rename(None).clone().requires_grad_(True),
                  feed[vv].rename(None).clone().requires_grad_(True))
    with torch.enable_grad():
        o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, attn_mask=mv, scale=0.5)
        (o * feed[w].rename(None)).sum().backward()
    torch.testing.assert_close(gq.align_to("b", "h", "seq", "hs").rename(None), qt.grad)
    torch.testing.assert_close(gk.align_to("b", "h", "key", "hs").rename(None), kt.grad)
    torch.testing.assert_close(gv.align_to("b", "h", "key", "hs").rename(None), vt.grad)
    src = next(iter(prog._specializations.values()))._source
    assert "_scaled_dot_product_flash_attention_for_cpu" in src
    assert "_backward" in src  # fused backward emitted


def test_reverse_grads_match_autograd_no_mask():
    _grad_family(mask_var=False)


def test_reverse_grads_match_autograd_masked():
    _grad_family(mask_var=True)


def test_backward_deduped_one_call_per_site():
    """dq/dk/dv of one site share ONE fused backward call (which returns all
    three), not three separate recompute+backward calls."""
    torch.manual_seed(2)
    dims = {b: 2, h: 3, s: 5, e: 4}
    q = Variable("q", b=b, h=h, seq=s, hs=e)
    kk = Variable("kk", b=b, h=h, key=s, hs=e)
    vv = Variable("vv", b=b, h=h, key=s, hs=e)
    w = Variable("w", b=b, h=h, seq=s, hs=e)
    loss = F.sum(F.sdpa(q, kk, vv, scale=0.5) * w)
    prog = compile_to_callable(loss.grad(q), loss.grad(kk), loss.grad(vv))
    feed = {q: _rand(2, 3, 5, 4).rename("b", "h", "seq", "hs"),
            kk: _rand(2, 3, 5, 4).rename("b", "h", "key", "hs"),
            vv: _rand(2, 3, 5, 4).rename("b", "h", "key", "hs"),
            w: _rand(2, 3, 5, 4).rename("b", "h", "seq", "hs")}
    prog(dict(feed), dims)
    src = next(iter(prog._specializations.values()))._source
    assert src.count("_scaled_dot_product_flash_attention_for_cpu_backward") == 1
