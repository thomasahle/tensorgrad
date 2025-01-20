from sympy import symbols

from tensorgrad.tensor import Copy, Variable
import tensorgrad.functions as F


def test_interaction_with_copy():
    i = symbols("i")
    x0 = Variable("x", i)
    x1 = x0.rename(i="i_")
    c0 = Copy(i, "i")
    c1 = Copy(i, "i_")
    assert (x0 @ c0).simplify() == (x1 @ c1).simplify()


def test_interaction_with_copy2():
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i_=i)
    assert (x.rename(i="i_") @ y).simplify() == (x @ Copy(i, "i", "i_") @ y).simplify()
    assert (x @ y).simplify() != (x @ Copy(i, "i", "i_") @ y).simplify()


def test_rename_with_function():
    i = symbols("i")
    x = Variable("x", i)
    a = (F.relu(x) @ Copy(i, "i")).simplify()
    b = (F.relu(x.rename(i="i_")) @ Copy(i, "i_")).simplify()
    c = (F.relu(x).rename(i="i_") @ Copy(i, "i_")).simplify()
    print(a.graph_to_string())
    print(b.graph_to_string())
    print(c.graph_to_string())
    assert b == c
    assert a == b
    assert a == c
