"""tg.grad / tg.compile: structured (pytree) inputs and outputs."""

import pytest
import sympy
import torch

import tensorgrad as tg
import tensorgrad.functions as F
from tensorgrad import Variable
from tensorgrad.tensor import Derivative

torch.set_num_threads(2)

b, d = sympy.symbols("b d")


def _setup():
    x = Variable("x", b=b, d=d)
    params = {"w1": Variable("w1", d=d), "w2": Variable("w2", d=d)}
    loss = F.sum(F.softmax(x * params["w1"], dim="d") * params["w2"])
    return x, params, loss


def test_grad_mirrors_pytree():
    x, params, loss = _setup()
    grads = tg.grad(loss, params)
    assert set(grads) == {"w1", "w2"}
    assert all(type(g) is Derivative for g in grads.values())
    assert grads["w1"].x is params["w1"]
    nested = tg.grad(loss, {"a": [params["w1"]], "b": (params["w2"],)})
    assert type(nested["a"][0]) is Derivative and type(nested["b"][0]) is Derivative


def test_grad_rejects_nonscalar_and_nonvariable():
    x, params, loss = _setup()
    with pytest.raises(ValueError, match="scalar"):
        tg.grad(x, params)
    with pytest.raises(TypeError, match="expected Variable"):
        tg.grad(loss, {"bad": loss})


def test_compile_round_trip():
    x, params, loss = _setup()
    grads = tg.grad(loss, params)
    new_w = {n: params[n] - 0.1 * grads[n] for n in params}
    step = tg.compile(loss=loss, params=new_w)

    weights = {"w1": torch.randn(3), "w2": torch.randn(3)}
    xv = torch.randn(2, 3).rename("b", "d")
    out = step(weights, dims={b: 2, d: 3}, x=xv)
    assert set(out.params) == {"w1", "w2"}
    assert out.loss.shape == ()
    # feeding the result back: keyed by variable name, so the round trip
    # is a positional dict with no declarations or order bookkeeping
    out2 = step(out.params, dims={b: 2, d: 3}, x=xv)
    assert torch.isfinite(out2.loss)
    # matches the flat positional path exactly
    from tensorgrad.compiler import compile_to_callable

    flat = compile_to_callable(loss, new_w["w1"], new_w["w2"])
    ref = flat({x: xv, params["w1"]: weights["w1"].rename("d"),
                params["w2"]: weights["w2"].rename("d")}, {b: 2, d: 3})
    torch.testing.assert_close(out.loss.rename(None), ref[0].rename(None))
    torch.testing.assert_close(out.params["w1"].rename(None), ref[1].rename(None))


def test_compile_binding_errors_and_scalars():
    x, params, loss = _setup()
    step = tg.compile(loss=loss)
    with pytest.raises(KeyError, match="is not an input variable"):
        step(dims={b: 2, d: 3}, nope=torch.randn(1))
    # positional dicts keyed by name or Variable; python scalars auto-wrap
    c = Variable("c")
    loss2 = loss * c
    step2 = tg.compile(loss=loss2)
    out = step2({"x": torch.randn(2, 3)}, {params["w1"]: torch.randn(3)},
                dims={b: 2, d: 3}, w2=torch.randn(3), c=0.5)
    assert out.loss.shape == ()


def test_compile_needs_outputs():
    with pytest.raises(ValueError, match="at least one"):
        tg.compile()
