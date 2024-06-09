from tensorgrad import Variable, Product


def test_components():
    V = Variable("V", ["i"])
    t = Product([V, V])
    assert t.components() == [t]
