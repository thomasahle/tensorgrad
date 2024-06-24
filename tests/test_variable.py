import pytest
import networkx as nx
from tensorgrad import Variable, Copy, Product, Zero


def test_variable_initialization():
    v = Variable("x", ["i", "j", "k"], symmetries="i j, k")
    assert v.name == "x"
    assert v.edges == ["i", "j", "k"]
    assert v.original_edges == ["i", "j", "k"]
    assert v.o_symmetries == {frozenset(["i", "j"]), frozenset(["k"])}


def test_variable_repr():
    v = Variable("x", ["i", "j", "k"], symmetries="i j, k")
    assert repr(v) == 'Variable("x", "i, j, k", symmetries="i j, k")'


def test_variable_rename():
    v = Variable("x", ["i", "j", "k"], symmetries="i j, k")
    v_renamed = v.rename({"i": "a", "j": "b"})
    assert v_renamed.edges == ["a", "b", "k"]
    assert v_renamed.original_edges == ["i", "j", "k"]
    assert v_renamed.o_symmetries == {frozenset(["i", "j"]), frozenset(["k"])}


def test_variable_grad():
    v = Variable("x", ["i", "j"])
    grad_v = v.grad(v, ["di", "dj"])
    assert isinstance(grad_v, Product)
    assert len(grad_v.tensors) == 2
    assert all(isinstance(t, Copy) for t in grad_v.tensors)
    assert grad_v.tensors[0].edges == ["i", "di"]
    assert grad_v.tensors[1].edges == ["j", "dj"]

    w = Variable("y", ["k"])
    grad_w = v.grad(w, ["dk"])
    assert isinstance(grad_w, Zero)
    assert grad_w.edges == ["i", "j", "dk"]


def test_variable_structural_graph():
    v = Variable("x", ["i", "j", "k"], symmetries="i j, k")
    G, edges = v.structural_graph()
    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes) == 3  # root node + 2 edge groups
    assert G.nodes[0]["name"] == ("Variable", "x")
    assert {G.nodes[1]["name"], G.nodes[2]["name"]} == {
        ("Original Edge Name", "i j"),
        ("Original Edge Name", "k"),
    }
    assert edges == {"i": 1, "j": 1, "k": 2}


def test_variable_edge_equivalences():
    v = Variable("x", "i, j, k", orig="a, b, c", symmetries="a b, c")
    equiv = list(v.edge_equivalences())
    print(equiv)
    assert ((v, ("Original", "a")), (v, "i")) in equiv
    assert ((v, ("Original", "b")), (v, "j")) in equiv
    assert ((v, ("Original", "c")), (v, "k")) in equiv
    assert ((v, "i"), (v, "j")) in equiv


def test_variable_depends_on():
    v = Variable("x", ["i", "j"])
    w = Variable("y", ["k"])
    assert v.depends_on(v)
    assert not v.depends_on(w)


def test_variable_symmetries():
    v = Variable("x", ["i", "j", "k", "l"], symmetries="i j k, l")
    assert v.o_symmetries == {frozenset(["i", "j", "k"]), frozenset(["l"])}

    # Test that symmetries are preserved after renaming
    v_renamed = v.rename({"i": "a", "j": "b"})
    assert v_renamed.o_symmetries == {frozenset(["i", "j", "k"]), frozenset(["l"])}


def test_variable_invalid_symmetries():
    with pytest.raises(AssertionError):
        Variable("x", ["i", "j"], symmetries="i j, k")  # 'k' is not in edges


def test_transpose_grad():
    X = Variable("X", "i, j", symmetries="i j")
    Xt = X.rename({"i": "j", "j": "i"})
    assert X == Xt
    assert (X + Xt).simplify() == (2 * X).simplify()
