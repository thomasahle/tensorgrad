from sympy import symbols
import tensorgrad.functions as F
from tensorgrad import Delta


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
