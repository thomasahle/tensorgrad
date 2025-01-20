from sympy import symbols

from tensorgrad.tensor import Variable, Sum, Product
from tensorgrad.serializers.to_latex import to_latex, Rename


def test_variable_indexed():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    latex = to_latex(X)  # index-based by default
    assert latex == "X_{i,j}"


def test_variable_index_free_no_transpose():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    X.original_edges = (i, j)
    latex = to_latex(X, index_free=True)
    assert latex == "X"


def test_variable_index_free_transpose():
    i, j = symbols("i j")
    X = Variable("X", j, i)
    X.original_edges = (i, j)
    latex = to_latex(X, index_free=True)
    assert latex == "X^T"


def test_sum_indexed():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    Y = Variable("Y", i, j)
    expr = Sum([X, Y], weights=[2, -1])
    latex = to_latex(expr)  # indexed
    assert latex == "2 X_{i,j} - Y_{i,j}"


def test_sum_index_free():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    Y = Variable("Y", i, j)
    expr = Sum([X, Y], weights=[1, 1])
    latex = to_latex(expr, index_free=True)
    assert latex == "X + Y"


def test_product_indexed():
    i, j, k = symbols("i j k")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    expr = Product([A, B])
    latex = to_latex(expr)
    assert latex == "A_{i,j} B_{j,k}"


def test_product_index_free():
    i, j, k = symbols("i j k")
    A = Variable("A", i, j)
    B = Variable("B", j, k)
    A.original_edges = (i, j)
    B.original_edges = (j, k)
    expr = Product([A, B])
    latex = to_latex(expr, index_free=True)
    assert latex == "A B"


def test_rename_indexed():
    i, j = symbols("i j")
    X = Variable("X", i, j)
    # rename i->k, j->l
    rename_expr = Rename(X, {"i": "k", "j": "l"})
    # By default we have a placeholder implementation
    latex = to_latex(rename_expr)
    assert "X_{k,l}" in latex
