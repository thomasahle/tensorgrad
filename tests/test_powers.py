from sympy import symbols
import tensorgrad.functions as F
from tensorgrad import Copy


def test_combine_powers():
    i = symbols("i")
    out = F._PowerFunction._combine_powers(
        [
            F.pow(Copy(i), k=2),
            Copy(i),
        ]
    )
    assert F.pow(Copy(i), k=3) in out

    out = F._PowerFunction._combine_powers(
        [
            F.pow(Copy(i), k=2),
            F.pow(Copy(i), k=2),
        ]
    )
    assert F.pow(Copy(i), k=4) in out

    out = F._PowerFunction._combine_powers(
        [
            F.pow(Copy(i), k=5),
            F.pow(Copy(i), k=-2),
        ]
    )
    assert F.pow(Copy(i), k=3) in out

    out = F._PowerFunction._combine_powers(
        [
            F.pow(Copy(i), k=5),
            F.pow(Copy(i), k=-2),
            Copy(i),
        ]
    )
    assert F.pow(Copy(i), k=4) in out
