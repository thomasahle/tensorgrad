from sympy import symbols
import tensorgrad.functions as F
from tensorgrad import Delta, Ones, Product, Variable


def test_combine_powers():
    i = symbols("i")
    out = F._PowerFunction._combine_powers(
        [
            F.pow(Delta(i), k=2),
            Delta(i),
        ]
    )
    assert F.pow(Delta(i), k=3) in out

    out = F._PowerFunction._combine_powers(
        [
            F.pow(Delta(i), k=2),
            F.pow(Delta(i), k=2),
        ]
    )
    assert F.pow(Delta(i), k=4) in out

    out = F._PowerFunction._combine_powers(
        [
            F.pow(Delta(i), k=5),
            F.pow(Delta(i), k=-2),
        ]
    )
    assert F.pow(Delta(i), k=3) in out

    out = F._PowerFunction._combine_powers(
        [
            F.pow(Delta(i), k=5),
            F.pow(Delta(i), k=-2),
            Delta(i),
        ]
    )
    assert F.pow(Delta(i), k=4) in out


def test_combine_large_powers():
    # There used to be an arbitrary bail-out for powers > 5.
    i = symbols("i")
    xs = F.sum(Variable("x", i))  # scalar
    assert (F.pow(xs, 6) * F.pow(xs, 2)).simplify() == F.pow(xs, 8).simplify()


def test_softmax_bookkeeping():
    # s = e / Z with e = exp(x) and Z = sum(e). Multiplying back by Z should
    # cancel the pow(Z, -1) factor: s * Z -> e.
    i = symbols("i")
    x = Variable("x", i)
    e = F.exp(x)
    Z = F.sum(e)
    s = e * F.pow(Z, -1)
    assert (s * Z).simplify() == e.simplify()


def test_cancel_through_hyperedge():
    # (x * y) * pow(x, -1) -> y, where * is the elementwise (Hadamard) product,
    # so the occurrences of x are connected through a copy hyperedge.
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    assert ((x * y) * F.pow(x, -1)).simplify() == y
    assert (F.pow(x, -1) * x).simplify() == Ones(i=i).simplify()


def test_combine_through_hyperedge():
    i = symbols("i")
    x = Variable("x", i)
    assert (x * x * x).simplify() == F.pow(x, 3).simplify()
    assert (F.pow(x, 2) * F.pow(x, 3)).simplify() == F.pow(x, 5).simplify()


def test_combine_direct_contraction():
    # x @ x = sum_e x_e x_e, where the two occurrences share a contracted edge
    # directly (no copy tensor in between). Canonical form: sum(pow(x, 2)).
    i = symbols("i")
    x = Variable("x", i)
    assert (x @ x).simplify() == F.sum(F.pow(x, 2)).simplify()


def test_fraction_of_fraction():
    i = symbols("i")
    x = Variable("x", i)
    # 1 / (1 / x) = x
    assert F.pow(F.pow(x, -1), -1).simplify() == x

    a = F.sum(Variable("a", i))
    b = F.sum(Variable("b", i))
    c = F.sum(Variable("c", i))
    # (a/b) / (c/b) = a/c
    lhs = (a * F.pow(b, -1)) * F.pow(c * F.pow(b, -1), -1)
    assert lhs.simplify() == (a * F.pow(c, -1)).simplify()
    # (a/b) * inv(a/b) = 1
    frac = a * F.pow(b, -1)
    assert (frac * F.pow(frac, -1)).simplify() == Product([])
    # (x/y) * (y/x) = 1 elementwise
    y = Variable("y", i)
    assert ((x * F.pow(y, -1)) * (y * F.pow(x, -1))).simplify() == Ones(i=i).simplify()
