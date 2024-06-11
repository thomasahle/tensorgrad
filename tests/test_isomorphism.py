import torch
from tensorgrad import Variable
from tensorgrad.tensor import Copy, Product, Sum
from tensorgrad.testutils import assert_close, rand_values
import tensorgrad.functions as F


def test_hash_counterexample():
    A = Variable("A", ["i", "j"])
    B = Variable("B", ["j", "k"])
    C = Variable("C", ["k", "i"])

    x = (A @ B @ C) @ (A @ B @ C)
    y = A @ B @ C.rename({"i": "i2"}) @ A.rename({"i": "i2"}) @ B @ C
    x = x.simplify()
    y = y.simplify()

    assert x != y


def test_hash_counterexample2():
    #  B B      B     B
    # A D A    A \   / A
    # A D A vs A  D-D  A
    #  B B      B/   \B

    A = Variable("A", ["i", "j"])
    B = Variable("B", ["j", "k"])
    D = Variable("D", ["k", "k2", "l"])
    A2 = A.rename({"i": "i2"})
    B2 = B.rename({"k": "k2"})
    half1 = A @ B @ D @ B2 @ A2
    expr1 = half1 @ half1
    half2 = A @ B @ D @ B2 @ A
    expr2 = half2 @ half2

    assert expr1 != expr2


def test_6cycle_vs_two_3cycles():
    one_2cycle = Product(
        [
            Variable("A", ["i", "j"]),
            Variable("B", ["j", "k"]),
            Variable("C", ["k", "i2"]),
            Variable("D", ["i2", "j2"]),
            Variable("E", ["j2", "k2"]),
            Variable("F", ["k2", "i"]),
        ]
    )
    two_3cycles = Product(
        [
            Variable("A", ["i", "j"]),
            Variable("B", ["j", "k"]),
            Variable("C", ["k", "i"]),
            Variable("D", ["i2", "j2"]),
            Variable("E", ["j2", "k2"]),
            Variable("F", ["k2", "i2"]),
        ]
    )

    assert one_2cycle != two_3cycles


def test_4cycle_vs_two_2cycles():
    A = Variable("A", ["i", "j"])
    B = Variable("B", ["j", "k"])
    C = Variable("C", ["k", "l"])
    D = Variable("D", ["l", "i"])
    one_4cycle = Product([A, B, C, D])
    two_2cycles = Product([A, B.rename({"k": "i"}), C, D.rename({"i": "k"})])
    assert one_4cycle != two_2cycles


def test_symmetry():
    x = Variable("x", "i")
    expr = x @ x.rename({"i": "j"})
    assert expr.edges == ["i", "j"]
    name_i, name_j = expr.canonical_edge_names
    assert name_i == name_j

    expr = x @ Copy("i, j, k")
    assert expr.edges == ["j", "k"]
    name_j, name_k = expr.canonical_edge_names
    assert name_j == name_k


def test_example_from_softmax_hessian():
    # The two a are isomorphic, but they can't be added / subtracted, because the edges are different.
    #
    #    A   B       A   B
    #    |   ⅄   vs  |   ⅄
    #    i  j k      j  i k
    #
    A = Variable("A", "i")
    B = Variable("B", "l")
    graph = A @ B @ Copy("l, j, k")
    graph2 = graph.rename({"i": "j", "j": "i"})
    assert graph == graph2

    # These two are isomorphic, but they _can_ be added because of the symmetry.
    #
    #    A   B       A   B
    #    |   ⅄   vs  |   ⅄
    #    i  j k      i  k j
    #
    graph3 = graph.rename({"j": "k", "k": "j"})
    assert graph == graph3

    # The point is that when we include the graphs in bigger context, the (lack of) symmetry matters.
    variables = Product(
        [
            Variable("x", "i"),
            Variable("y", "j"),
            Variable("z", "k"),
        ]
    )

    assert graph @ variables != graph2 @ variables
    assert graph @ variables == graph3 @ variables


def test_symmetries():
    #    A   B
    #    |   ⅄
    #    i  j k
    A = Variable("A", "i")
    B = Variable("B", "l")
    graph = A @ B @ Copy("l, j, k")
    assert graph.symmetries == [{"i"}, {"j", "k"}]

    #    A   B      A   B
    #    |   ⅄   +  |   ⅄
    #    i  j k     i  k j
    graph2 = graph + graph.rename({"j": "k", "k": "j"})
    assert graph2.symmetries == [{"i"}, {"j", "k"}]

    #    A   B      A   B
    #    |   ⅄   +  |   ⅄
    #    i  j k     k  j i
    graph3 = graph + graph.rename({"i": "k", "k": "i"})
    assert graph3.symmetries == [{"i", "k"}, {"j"}]

    #    A   B      A   B      A   B
    #    |   ⅄   +  |   ⅄   +  |   ⅄
    #    i  j k     j  i k     k  j i
    graph4 = Sum([graph, graph.rename({"i": "j", "j": "i"}), graph.rename({"i": "k", "k": "i"})])
    assert graph4.symmetries == [{"i", "j", "k"}]


def test_symmetries_simplify_sum():
    #    A   B
    #    |   ⅄
    #    i  j k
    A = Variable("A", "i")
    B = Variable("B", "l")
    graph1 = A @ B @ Copy("l, j, k")
    #    A   B
    #    |   ⅄
    #    i  k j
    graph2 = graph1.rename({"j": "k", "k": "j"})

    # Flipping j and k we can still combine the two graphs.
    assert len((graph1 + graph2).simplify().tensors) == 1

    #    A   B
    #    |   ⅄
    #    k  j i
    graph3 = graph1.rename({"i": "k", "k": "i"})

    # Flipping i and k we can't combine the two graphs.
    assert len((graph1 + graph3).simplify().tensors) == 2

    #    A   B
    #    |   ⅄
    #    m  j k
    graph4 = graph1.rename({"i": "m"})
    # Even if graph1 and graph4 are isomorphic, we can't combine them.
    assert len((graph1 + graph4).simplify().tensors) == 2


def test_empty_graphs():
    empty_graph1 = Product([])
    empty_graph2 = Product([])
    assert empty_graph1 == empty_graph2


def test_single_variable_graphs():
    single_var1 = Product([Variable("A", ["i", "j"])])
    single_var2 = Product([Variable("B", ["i", "j"])])
    assert single_var1 != single_var2

    A = Variable("A", ["i", "j"])
    single_var1 = Product([A])
    single_var2 = Product([A.rename({"i": "i2"})])
    assert single_var1 == single_var2

    single_var3 = Product([A.rename({"i": "j", "j": "i"})])
    assert single_var1 == single_var3


def test_same_variable_names():
    A = Variable("A", ["i", "j"])
    B = Variable("B", ["j", "k"])
    C = Variable("C", ["k", "i"])
    graph1 = Product([A, B, C])
    graph2 = Product(
        [
            A.rename({"i": "i2", "j": "j2"}),
            B.rename({"k": "k2", "j": "j2"}),
            C.rename({"i": "i2", "k": "k2"}),
        ]
    )
    assert graph1 == graph2


def test_3cycle_with_selfloop():
    A = Variable("A", ["i", "j"])
    B = Variable("B", ["j", "k"])
    C = Variable("C", ["k", "i"])
    D = Variable("D", ["l", "m"])
    graph1 = Product([A, B, C, D])
    graph2 = Product([A, B, C, D.rename({"l": "n"})])
    assert graph1 == graph2


def test_disconnected_graphs():
    A = Variable("A", ["i", "j"])
    B = Variable("B", ["j", "k"])
    C = Variable("C", ["k", "i"])
    D = Variable("D", ["l", "m"])
    E = Variable("E", ["m", "n"])
    F = Variable("F", ["n", "l"])
    graph1 = Product([A, B, C, D, E, F])
    graph2 = Product(
        [
            A,
            B,
            C,
            D.rename({"l": "l2", "m": "m2"}),
            E.rename({"m": "m2", "n": "n2"}),
            F.rename({"n": "n2", "l": "l2"}),
        ]
    )
    assert graph1 == graph2


def test_different_number_of_variables():
    A = Variable("A", ["i", "j"])
    B = Variable("B", ["j", "k"])
    C = Variable("C", ["k", "i"])
    D = Variable("D", ["l", "m"])
    graph1 = Product([A, B, C])
    graph2 = Product([A, B, C, D])
    assert graph1 != graph2


def test_isomorphic_with_rename():
    A = Variable("A", ["i", "j"])
    B = Variable("B", ["j", "k"])
    C = Variable("C", ["k", "i"])
    graph1 = Product([A, B, C])
    graph2 = Product(
        [
            A.rename({"i": "x", "j": "y"}),
            B.rename({"j": "y", "k": "z"}),
            C.rename({"k": "z", "i": "x"}),
        ]
    )
    assert graph1 == graph2


def test_different_variables():
    e1 = Product([Variable("a", ["i"]), Copy(["j", "j1"])])
    e2 = Product([Variable("b", ["i"]), Copy(["j", "j1"])])
    assert e1 != e2
    e1 = Product([Variable("a", ["i"]), Copy(["i", "j", "j1"])])
    e2 = Product([Variable("b", ["i"]), Copy(["i", "j", "j1"])])
    assert e1 != e2
    x = Variable("x", ["a"], ["a"])
    z = Variable("z", ["a"], ["a"])
    e1 = x + z
    e2 = z + z
    assert e1 != e2


def test_broadcasted():
    e1 = Product(
        [
            Copy(["y_"]),
            Sum(
                [
                    Product([Variable("x", ["x"], ["x_"]), Copy(["y_"])]),
                    Product([Variable("y", ["y"], ["y_"]), Copy(["x_"])]),
                ],
                (1, 1),
            ),
        ]
    )
    e2 = Product(
        [
            Sum(
                [
                    Product([Variable("x", ["x"], ["x_"]), Copy(["y"])]),
                    Product([Variable("y", ["y"], ["y"]), Copy(["x_"])]),
                ],
                (1, 1),
            ),
            Copy(["y"]),
        ]
    )
    assert e1 == e2


def test_transpose_grad():
    x = Variable("X", "i, j")
    xt = x.rename({"j": "i", "i": "j"})
    res = xt.grad(x, ["a", "b"])
    expected = Copy("j, a") @ Copy("i, b")
    # The automorphism group here is more complicated than what we are allowing with "symmetries".
    # E.g. {i -> j, j -> i} is not allowed, but {i -> j, a -> b; j -> i, b -> a} _is_ allowed.
    # So the set of symmetries {{i, a}, {j, b}} is too weak. (Or is it?)
    # But (unless we support full automorphism groups) we should always give a conservative set
    # of symmetries here.
    assert expected.symmetries == [{"j", "a"}, {"i", "b"}]
    print(f"{res=}")
    print(f"{expected=}")
    assert res.is_isomorphic(expected, match_edges=True)


def test_symmetries_simplify_sum2():
    # Unfortunately it's not possible to make this test pass together with the preivous one, unless we majorly refactor the code...
    logits = Variable("logits", "C")
    expr = Sum(
        [
            Product(
                [
                    logits.rename({"C": "C__"}),
                    logits.rename({"C": "C_"}),
                ]
            ),
            Product(
                [
                    logits.rename({"C": "C_"}),
                    logits.rename({"C": "C__"}),
                ]
            ),
        ],
    )
    assert sorted(expr.tensors[0].edges) == sorted(expr.tensors[1].edges)
    print([hash(t) for t in expr.tensors])
    print([t.canonical_edge_names for t in expr.tensors])
    for t in expr.tensors:
        print([t.canonical_edge_names[t.edges.index(e)] for e in expr.edges])
    assert len({hash(t) for t in expr.tensors}) == 1
    assert expr.symmetries == [{"C_", "C__"}]
    for t in expr.tensors:
        assert t.symmetries == [{"C_", "C__"}]
    assert len(expr.simplify().tensors) == 1
