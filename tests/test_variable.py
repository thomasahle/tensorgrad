import pytest
import networkx as nx
from sympy import symbols
from tensorgrad import Variable, Delta, Product, Zero


def test_variable_initialization():
    i, k = symbols("i k")
    v = Variable("x", i=i, j=i, k=k).with_symmetries("i j, k")
    assert v.name == "x"
    assert v.edges == {"i", "j", "k"}
    assert v.symmetries == {frozenset({"i", "j"}), frozenset({"k"})}


def test_variable_repr():
    i, k = symbols("i k")
    v = Variable("x", i=i, j=i, k=k).with_symmetries("i j, k")
    assert repr(v) == 'Variable("x", i, j, k).with_symmetries("i j, k")'


def test_variable_rename():
    i, k, a = symbols("i k a")
    v = Variable("x", i=i, j=i, k=k).with_symmetries("i j, k")
    v_renamed = v.rename(i="a")
    assert v_renamed.edges == {"a", "j", "k"}
    assert v_renamed.symmetries == {frozenset({"a", "j"}), frozenset({"k"})}


def test_variable_grad():
    i, j, k = symbols("i j k")
    v = Variable("x", i=i, j=j)
    grad_v = v.grad(v, {"i": "di", "j": "dj"})
    assert isinstance(grad_v, Product)
    assert len(grad_v.tensors) == 2
    assert all(isinstance(t, Delta) for t in grad_v.tensors)
    assert grad_v.tensors[0].edges == {"i", "di"}
    assert grad_v.tensors[1].edges == {"j", "dj"}

    w = Variable("y", k)
    grad_w = v.grad(w, {"k": "dk"})
    assert isinstance(grad_w, Zero)
    assert grad_w.edges == {"i", "j", "dk"}


def test_variable_depends_on():
    i, j, k = symbols("i j k")
    v = Variable("x", i=i, j=j)
    w = Variable("y", k=k)
    assert v.depends_on(v)
    assert not v.depends_on(w)


def test_variable_symmetries():
    i, k, l, a = symbols("i k l a")

    # We can't make symmetries for edges that don't have the same size
    with pytest.raises(ValueError):
        Variable("x", i=i, j=i, k=k, l=l).with_symmetries("i j k, l")

    v = Variable("x", i=i, j=i, k=i, l=l).with_symmetries("i j k, l")
    assert v.symmetries == {frozenset({"i", "j", "k"}), frozenset({"l"})}

    # Test that symmetries are preserved after renaming
    v_renamed = v.rename(i="a")
    assert v_renamed.symmetries == {frozenset({"a", "j", "k"}), frozenset({"l"})}


def test_variable_invalid_symmetries():
    i, j = symbols("i j")
    with pytest.raises(ValueError):
        Variable("x", i, j).with_symmetries("i j, k")  # 'k' is not in edges


def test_transpose_grad():
    i = symbols("i")
    X = Variable("X", i=i, j=i).with_symmetries("i j")
    Xt = X.rename(i="j", j="i")
    assert X == Xt
    assert (X + Xt).simplify() == (2 * X).simplify()
