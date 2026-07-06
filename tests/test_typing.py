"""Tensor["a", "b"] edge-set annotations and the @typed decorator."""

import pytest
from sympy import symbols

import tensorgrad.functions as F
from tensorgrad import Tensor, Variable, typed
from tensorgrad.typing import EdgeSpec, EdgeTypeError

b, s, d, h = symbols("b s d h")


def test_class_getitem_forms():
    assert Tensor["batch", "seq"] == EdgeSpec(("batch", "seq"))
    assert Tensor["batch"] == EdgeSpec(("batch",))
    assert Tensor["batch seq"] == Tensor["batch", "seq"]  # space-separated
    assert Tensor["seq", "batch"] == Tensor["batch", "seq"]  # sets, not orders
    assert repr(Tensor["batch", "seq"]) == "Tensor['batch', 'seq']"


def test_class_getitem_rejects_bad_names():
    with pytest.raises(TypeError, match="distinct"):
        Tensor["batch", "batch"]
    with pytest.raises(TypeError, match="non-empty strings"):
        Tensor["batch", 3]


def test_typed_checks_return():
    @typed
    def good(x: Tensor["i", "j"]) -> Tensor["i", "j"]:
        return x * 2

    @typed
    def bad(x: Tensor["i", "j"]) -> Tensor["i", "k"]:
        return x * 2

    x = Variable("x", i=b, j=s)
    assert set(good(x).edges) == {"i", "j"}
    with pytest.raises(EdgeTypeError, match=r"return value.*expected edges \['i', 'k'\]"):
        bad(x)


def test_typed_checks_arguments():
    @typed
    def f(x: Tensor["i"], scale: float = 1.0) -> Tensor["i"]:
        return x * scale

    x = Variable("x", i=b)
    f(x)  # ok; the float parameter is ignored
    with pytest.raises(EdgeTypeError, match="argument 'x'.*missing.*extra"):
        f(Variable("y", j=s))


def test_typed_edge_order_is_irrelevant():
    @typed
    def f(x: Tensor["j", "i"]) -> Tensor["i", "j"]:
        return x

    f(Variable("x", i=b, j=s))  # both directions of the same set


def test_typed_through_real_algebra():
    @typed
    def attention_scores(q: Tensor["batch", "seq", "hs"], k: Tensor["batch", "key", "hs"]
                         ) -> Tensor["batch", "seq", "key"]:
        return F.dot(q, k, dim="hs")

    q = Variable("q", batch=b, seq=s, hs=h)
    k = Variable("k", batch=b, key=d, hs=h)
    out = attention_scores(q, k)
    assert set(out.edges) == {"batch", "seq", "key"}


def test_typed_rejects_non_tensor_result():
    @typed
    def f() -> Tensor["i"]:
        return 3  # pyright: ignore[reportReturnType]  # the runtime check is under test

    with pytest.raises(EdgeTypeError, match="got int"):
        f()


def test_untyped_annotations_pass_through():
    @typed
    def f(x: int, y) -> int:
        return x + y

    assert f(1, 2) == 3


def test_open_spec():
    assert repr(Tensor[..., "d"]) == "Tensor[..., 'd']"

    @typed
    def f(x: Tensor[..., "d"]) -> Tensor[..., "d"]:
        return x * 2

    f(Variable("x", d=d))  # exactly d
    f(Variable("x", batch=b, seq=s, d=d))  # d plus anything
    with pytest.raises(EdgeTypeError, match="including"):
        f(Variable("x", batch=b, seq=s))  # missing d

    @typed
    def any_tensor(x: Tensor[...]) -> Tensor[...]:
        return x

    any_tensor(Variable("x", i=b))
