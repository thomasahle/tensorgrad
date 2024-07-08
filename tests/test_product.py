from sympy import symbols
from tensorgrad import Variable, Product


def test_components():
    i = symbols("i")
    V = Variable("V", i)
    t = Product([V, V])
    assert t.components() == [t]
