from tensorgrad.tensor import Copy


def test_copy_loop():
    expr = Copy("i, j") @ Copy("i, j")
    # Really this should be the constant `i`, like Sum(Copy(), [i]).
    # assert expr.simplify() == Copy()

    # The best way to ensure we don't lose information is to just not contract self-loops
    assert expr.simplify() == Copy("i") @ Copy("i")

    # Concrete example from https://stats.stackexchange.com/questions/589669/gaussian-fourth-moment-formulas
    #    (
    #        [
    #            Product(
    #                [
    #                    Variable("A", ["j_", "j1"], orig=["j", "j1"]),
    #                    Copy(["i", "k"], link=Variable("X", ["i", "j"])),
    #                    Copy(["j", "l"], link=Variable("X", ["i", "j"])),
    #                ]
    #            ),
    #            Product([Copy(["i", "k"]), Copy(["j1", "l"])]),
    #        ],
    #    )
    #    Product([Copy(["k"]), Variable("A", ["j_", "j"], orig=["j", "j1"])])
    #
    # Note that the Copy's also have some linked Variables.
    # This can be important when it comes to evaluating the expression.
    # But the copy simplification happens in Product.simplify, and does that even know about linked variables?
    # And how do we choose which of the linked variables to keep?
    # Maybe the best solution is to just keep the loop alive...


def test_copy_double():
    expr = Copy("p0, p2, p0_") @ Copy("p0, p0_")
    assert expr.simplify() == Copy("p2")


def test_copy_trace():
    expr = Copy("p, p1") @ Copy("p, p1")
    expected = Copy("p") @ Copy("p")
    assert expr.simplify() == expected.simplify()
