"""one_hot identities as catalog rules (task #42).

sum_one_hot in simplify.py: contracting a one_hot's class edge with a ones
vector is the partition of unity sum_v [idx == v] = 1 (valid class ids —
the same precondition the gather/scatter lowering relies on). Makes CE
gradients/Hessians of IN-PROGRAM one_hot targets collapse exactly like
declared-simplex Variables (task #20), with no declaration.
"""

import torch
from sympy import symbols

torch.set_num_threads(2)

import tensorgrad.functions as F
from tensorgrad import Delta, Variable
from tensorgrad.compiler import compile_to_callable
from tensorgrad.extras.evaluate import evaluate
from tensorgrad.tensor import Sum


def _ce_setup():
    b, v = symbols("b v")
    idx = Variable("idx", b=b)
    x = Variable("x", b=b, v=v)
    loss = -F.sum(F.one_hot(idx, v) * F.log(F.softmax(x, dim="v"))) / Delta(b)
    return b, v, idx, x, loss


def test_sum_over_classes_collapses_to_ones():
    b, v = symbols("b v")
    idx = Variable("idx", b=b)
    s = F.sum(F.one_hot(idx, v), dim="v").simplify()
    assert not s.depends_on(idx)
    B, V = 4, 6
    torch.manual_seed(0)
    it = torch.randint(V, (B,)).float().rename("b")
    out = evaluate(s, {idx: it}, {b: B, v: V})
    assert torch.equal(out.rename(None), torch.ones(B))


def test_weighted_sum_does_not_fire():
    """sum_v w[v] * one_hot is a gather of w — NOT a partition of unity."""
    b, v = symbols("b v")
    idx, w = Variable("idx", b=b), Variable("w", v=v)
    s = F.sum(F.one_hot(idx, v) * w, dim="v").simplify()
    assert s.depends_on(idx)
    B, V = 4, 6
    torch.manual_seed(0)
    it = torch.randint(V, (B,)).float().rename("b")
    wt = torch.randn(V).rename("v")
    out = evaluate(s, {idx: it, w: wt}, {b: B, v: V})
    assert torch.allclose(out.rename(None), wt.rename(None)[it.rename(None).long()])


def test_ce_gradient_correct_with_rule_active():
    b, v, idx, x, loss = _ce_setup()
    g = loss.grad(x)
    prog = compile_to_callable(loss, g)
    B, V = 4, 6
    torch.manual_seed(0)
    it = torch.randint(V, (B,)).double().rename("b")
    xt = torch.randn(B, V, dtype=torch.float64).rename("b", "v")
    lv, gv = prog({idx: it, x: xt}, {b: B, v: V})
    xr = xt.rename(None).clone().requires_grad_(True)
    with torch.enable_grad():
        yr = torch.nn.functional.one_hot(it.rename(None).long(), V).double()
        lr = -(yr * torch.log_softmax(xr, dim=1)).sum() / B
        (gr,) = torch.autograd.grad(lr, xr)
    torch.testing.assert_close(lv.rename(None), lr.detach(), rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(
        gv.align_to("b", "v").rename(None), gr, rtol=1e-9, atol=1e-12
    )


def test_ce_hessian_is_target_free_and_correct():
    """The headline (#42): with IN-PROGRAM one_hot targets, the CE Hessian
    simplifies to a target-free form — previously only achievable by
    declaring a simplex constraint on a target Variable (#20)."""
    b, v, idx, x, loss = _ce_setup()
    H = loss.grad(x).grad(x, new_names={"b": "b2", "v": "v2"}).full_simplify()
    assert not H.depends_on(idx), "Hessian must not depend on the targets"
    assert isinstance(H, Sum) and len(H.terms) <= 3

    B, V = 3, 5
    torch.manual_seed(0)
    it = torch.randint(V, (B,)).double().rename("b")
    xt = torch.randn(B, V, dtype=torch.float64).rename("b", "v")
    prog = compile_to_callable(H)
    Hv = prog({x: xt}, {b: B, v: V}).align_to("b", "v", "b2", "v2").rename(None)
    yr = torch.nn.functional.one_hot(it.rename(None).long(), V).double()

    def torch_L(xf):
        return -(yr * torch.log_softmax(xf, dim=1)).sum() / B

    Href = torch.autograd.functional.hessian(torch_L, xt.rename(None))
    torch.testing.assert_close(Hv, Href, rtol=1e-9, atol=1e-12)
