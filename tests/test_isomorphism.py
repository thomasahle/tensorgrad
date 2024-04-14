import torch
from tensorgrad import Variable
from tests.utils import assert_close, rand_values


def test_hash_counterexample():
    A = Variable("A", ["i", "j"])
    B = Variable("B", ["j", "k"])
    C = Variable("C", ["k", "i"])

    x = (A @ B @ C) @ (A @ B @ C)
    y = A @ B @ C.rename({"i": "i2"}) @ A.rename({"i": "i2"}) @ B @ C
    x = x.simplify()
    y = y.simplify()

    ts = rand_values([A, B, C], i=3, j=3, k=3, i2=3)
    print(f"{ts[A]=}")
    print(f"{ts[B]=}")
    print(f"{ts[C]=}")

    a, b, c = ts[A].rename(None), ts[B].rename(None), ts[C].rename(None)
    a, b, c = ts[A].rename(None), ts[B].rename(None), ts[C].rename(None)
    print(torch.trace(a @ b @ c) ** 2)
    print(torch.trace(a @ b @ c @ a @ b @ c))
    print((A @ B @ C).evaluate(ts) ** 2)
    print(x.evaluate(ts), y.evaluate(ts))
    print(x.simplify().evaluate(ts), y.simplify().evaluate(ts))
    assert_close(x.evaluate(ts), y.evaluate(ts))

    assert hash(x) == hash(y)
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

    # Oddly enough, even though the two graphs are not isomorphic, they still have the same value...
    ts = rand_values([A, B, D], i=3, j=3, k=3, k2=3, i2=3, l=3)
    print(expr1.evaluate(ts), expr2.evaluate(ts))
    assert_close(expr1.evaluate(ts), expr2.evaluate(ts))

    assert hash(expr1) == hash(expr2)
    assert expr1 != expr2
