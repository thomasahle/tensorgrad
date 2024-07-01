import pytest
import networkx as nx
from sympy import symbols
from tensorgrad import Variable, Copy, Product, Zero


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
    v_renamed = v.rename(i=a)
    assert v_renamed.edges == {"a", "j", "k"}
    assert v_renamed.original_edges == {"i", "j", "k"}
    assert v_renamed.symmetries == {frozenset({"i", "j"}), frozenset({"k"})}


def test_variable_grad():
    i, j, di, dj, k, dk = symbols("i j di dj k dk")
    v = Variable("x", i=i, j=j)
    grad_v = v.grad(v, {i: di, j: dj})
    assert isinstance(grad_v, Product)
    assert len(grad_v.tensors) == 2
    assert all(isinstance(t, Copy) for t in grad_v.tensors)
    assert grad_v.tensors[0].edges == {i, di}
    assert grad_v.tensors[1].edges == {j, dj}

    w = Variable("y", k=k)
    grad_w = v.grad(w, {k: dk})
    assert isinstance(grad_w, Zero)
    assert grad_w.edges == {"i", "j", "dk"}


def test_variable_structural_graph():
    i, k = symbols("i k")
    v = Variable("x", i=i, j=i, k=k).with_symmetries("i j, k")
    G, edges = v.structural_graph()
    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes) == 3  # root node + 2 edge groups
    assert G.nodes[0]["name"] == ("Variable", "x")
    assert {G.nodes[1]["name"], G.nodes[2]["name"]} == {
        ("Original Edge Name", "i j"),
        ("Original Edge Name", "k"),
    }


def test_variable_edge_equivalences():
    i, k, a, c = symbols("i k a c")
    v = Variable("x", i=i, j=i, k=k, orig={a: i, c: k}).with_symmetries("a b, c")
    equiv = list(v.edge_equivalences())
    print(equiv)
    assert ((v, ("Original", a)), (v, "i")) in equiv
    assert ((v, ("Original", a)), (v, "j")) in equiv
    assert ((v, ("Original", c)), (v, "k")) in equiv
    assert ((v, "i"), (v, "j")) in equiv


def test_variable_depends_on():
    i, j, k = symbols("i j k")
    v = Variable("x", i=i, j=j)
    w = Variable("y", k=k)
    assert v.depends_on(v)
    assert not v.depends_on(w)


def test_variable_symmetries():
    i, k, l, a = symbols("i k l a")
    v = Variable("x", i=i, j=i, k=k, l=l).with_symmetries("i j k, l")
    assert v.symmetries == {frozenset({"i", "j", "k"}), frozenset({"l"})}

    # Test that symmetries are preserved after renaming
    v_renamed = v.rename(i=a)
    assert v_renamed.symmetries == {frozenset({"i", "j", "k"}), frozenset({"l"})}


def test_variable_invalid_symmetries():
    i, j, k = symbols("i j k")
    with pytest.raises(ValueError):
        Variable("x", i=i, j=j).with_symmetries("i j, k")  # 'k' is not in edges


def test_transpose_grad():
    i = symbols("i")
    X = Variable("X", i=i, j=i).with_symmetries("i j")
    Xt = X.rename(i="j", j="i")
    assert X == Xt
    assert (X + Xt).simplify() == (2 * X).simplify()
