from .tensor import (
    Tensor,  # noqa: F401
    Function,  # noqa: F401
    Zero,  # noqa: F401
    Product,  # noqa: F401
    Sum,  # noqa: F401
    Variable,  # noqa: F401
    Delta,  # noqa: F401
    Ones,  # noqa: F401
    Derivative,  # noqa: F401
    function,  # noqa: F401
)
from typing import TYPE_CHECKING, Any
from .typing import typed  # noqa: F401
from .extras.expectation import Expectation  # noqa: F401
from .extras.book_layout import to_book_tikz  # noqa: F401
from .functions import frobenius2, kronecker, diag, sum, log, pow, trace  # noqa: F401
from sympy import symbols  # noqa: F401

# `compile`, `grad`, `Output` live in the AOT compiler, which imports torch.
# They are loaded LAZILY (PEP 562) so `import tensorgrad` -- and the diagram /
# book-layout path in particular -- stays torch-free: important for lightweight
# CLI startup and for running the layout engine in the browser (Pyodide, which
# has no torch wheel). Accessing tensorgrad.compile/grad/Output, or
# `from tensorgrad import compile`, transparently pulls them in on first use.
if TYPE_CHECKING:  # keep the names visible to type checkers
    from .compiler.runtime import Output, compile, grad  # noqa: F401

_LAZY_COMPILER = {"Output", "compile", "grad"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_COMPILER:
        from .compiler import runtime

        vals = {n: getattr(runtime, n) for n in _LAZY_COMPILER}
        globals().update(vals)
        return vals[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The expression-problem split added a tensorgrad/grad.py SUBMODULE, and the
# import system binds it as an attribute here during package init (simplify
# imports it) -- shadowing the lazy user-facing `grad` FUNCTION above, since
# PEP 562 __getattr__ only fires for MISSING attributes. Drop the binding:
# `import tensorgrad.grad` still works through sys.modules, and attribute
# access falls through to the lazy compiler export as documented.
globals().pop("grad", None)
