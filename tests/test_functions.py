import torch

from tensor import Variable
import functions as F


def assert_close(a, b):
    assert set(a.names) == set(b.names)
    a = a.align_to(*b.names)
    torch.testing.assert_close(a.rename(None), b.rename(None))


def test_frobenius2():
    t = torch.randn(2, 3, 4, names=("a", "b", "c"))
    v = Variable("t", ["a", "b", "c"])
    frob = F.frobenius2(v)
    res = frob.evaluate({v: t})
    expected = (t * t).sum()
    assert_close(res, expected)


def test_diag():
    v = Variable("v", ["a"])
    mat = F.diag(v, ["a", "b"])
    t = torch.randn(2, names=("a",))
    res = mat.evaluate({v: t})
    expected = torch.diag(t.rename(None)).rename("a", "b")
    assert_close(res, expected)


def test_einsum():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["j", "k"])
    c = Variable("c", ["k", "l"])
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(3, 4, names=("j", "k"))
    t_c = torch.randn(4, 5, names=("k", "l"))

    # Test basic einsum
    res = F.einsum([a, b], ["i", "k"]).evaluate({a: t_a, b: t_b})
    expected = torch.einsum("ij,jk->ik", t_a.rename(None), t_b.rename(None)).rename("i", "k")
    assert_close(res, expected)

    # Test einsum with multiple tensors
    res = F.einsum([a, b, c], ["i", "l"]).evaluate({a: t_a, b: t_b, c: t_c})
    expected = torch.einsum("ij,jk,kl->il", t_a.rename(None), t_b.rename(None), t_c.rename(None)).rename(
        "i", "l"
    )
    assert_close(res, expected)


def test_kronecker():
    a = Variable("a", ["i", "j"])
    b = Variable("b", ["k", "l"])
    t_a = torch.randn(2, 3, names=("i", "j"))
    t_b = torch.randn(4, 5, names=("k", "l"))
    result = F.kronecker(a, b).evaluate({a: t_a, b: t_b})
    expected = (
        torch.kron(t_a.rename(None), t_b.rename(None))
        .reshape(2, 4, 3, 5)
        .rename("i", "k", "j", "l")
        .align_to("i", "j", "k", "l")
    )
    torch.testing.assert_close(result.rename(None), expected.rename(None))


def test_sum():
    a = Variable("a", ["i", "j"])
    t_a = torch.randn(2, 3, names=("i", "j"))

    # Test sum over one dimension
    result = F.sum(a, ["i"]).evaluate({a: t_a})
    expected = t_a.sum(dim="i")
    torch.testing.assert_close(result.rename(None), expected.rename(None))

    # Test sum over multiple dimensions
    result = F.sum(a, ["i", "j"]).evaluate({a: t_a})
    expected = t_a.sum(dim=("i", "j"))
    torch.testing.assert_close(result.rename(None), expected.rename(None))


def test_pow():
    a = Variable("a", ["i", "j"])
    t_a = torch.randn(2, 3, names=("i", "j")).abs()
    result = F.pow(a, -1).evaluate({a: t_a})
    expected = torch.pow(t_a.rename(None), -1).rename("i", "j")
    assert_close(result, expected)


def test_log():
    a = Variable("a", ["i", "j"])
    t_a = torch.randn(2, 3, names=("i", "j")).abs()
    print("loga", F.log(a))
    result = F.log(a).evaluate({a: t_a})
    expected = torch.log(t_a.rename(None)).rename("i", "j")
    assert_close(result, expected)


def test_log_grad():
    v = Variable("v", ["i"])
    t_v = torch.randn(3, names=("i",)).abs()
    print("logv", F.log(v))
    assert F.log(v).edges == ["i"]
    print("grad logv", F.log(v).grad(v))
    jacobian = F.log(v).grad(v).simplify()
    print(f"{jacobian=}")
    assert set(jacobian.edges) == {"i", "i_"}
    result = jacobian.evaluate({v: t_v})
    expected = torch.diag(torch.pow(t_v.rename(None), -1)).rename("i", "i_")
    assert_close(result, expected)


# Sum(
#     [
#         Product(
#             [
#                 Product(
#                     [
#                         Function(pow(-1), [], [(Variable(v, ["i"], ["i_"]),)]),
#                         Product([Copy(["i", "i_", "i__"])]),
#                     ]
#                 ),
#                 Derivative(Variable(v, ["i"], ["i__"]), Variable(v, ["i"], ["i"]), ["i_"]),
#             ]
#         )
#     ],
#     [1],
# )


def test_exp():
    a = Variable("a", ["i", "j"])
    t_a = torch.randn(2, 3, names=("i", "j"))
    result = F.exp(a).evaluate({a: t_a})
    expected = torch.exp(t_a.rename(None)).rename("i", "j")
    assert_close(result, expected)
