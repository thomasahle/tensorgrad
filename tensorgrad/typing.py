"""Edge-set annotations: Tensor["batch", "seq"] and the @typed decorator.

Tensorgrad tensors have NAMED edges and no axis order, so a tensor "type" is
just its edge-name set. That makes annotations both simpler and stronger than
positional shape types (jaxtyping et al.): there is no order to get wrong,
and conformance is exact set equality.

    from tensorgrad import Tensor, Variable, typed

    @typed
    def attention(x: Tensor["batch", "seq", "d"], name: str
                  ) -> Tensor["batch", "head", "seq", "hs"]:
        ...

@typed checks every annotated Tensor parameter on the way in and the return
value on the way out; a mismatch raises EdgeTypeError naming the function,
the argument, and the expected vs actual edge sets. Non-Tensor annotations
and unannotated parameters are ignored, so the decorator can be applied to
mixed-signature functions. Checking is skipped entirely under `python -O`
(the wrapper collapses to the raw function).

Static checkers cannot verify edge propagation through tensor algebra
(that would need einsum-aware inference); the annotations type-check as
opaque values and are enforced at runtime only.
"""

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from tensorgrad.tensor import Tensor

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["EdgeSpec", "EdgeTypeError", "typed"]


class EdgeTypeError(TypeError):
    """A tensor's edge set does not match its Tensor[...] annotation."""


class EdgeSpec:
    """The value of Tensor["a", "b"]: an edge-name set contract.

    `open=True` (written Tensor[..., "d"]) means the named edges are
    REQUIRED but any other edges may be present — the batch-polymorphic
    contract of a function that only touches specific edges."""

    __slots__ = ("edges", "open")

    def __init__(self, edges: tuple[str, ...], open: bool = False):
        self.edges = edges
        self.open = open

    @classmethod
    def of(cls, item: Any) -> "EdgeSpec":
        """Build from a __class_getitem__ subscript: Tensor["a", "b"],
        Tensor["a"], the space-separated Tensor["a b"], or the open form
        Tensor[..., "d"] (an Ellipsis anywhere marks the spec open)."""
        if isinstance(item, tuple):
            parts = item
        else:
            parts = (item,)
        open_spec = False
        names: list[str] = []
        for p in parts:
            if p is Ellipsis:
                open_spec = True
            elif isinstance(p, str) and p:
                names.extend(p.split())
            else:
                raise TypeError(f"Tensor[...] edge names must be non-empty strings, got {p!r}")
        if len(set(names)) != len(names):
            raise TypeError(f"Tensor[...] edge names must be distinct, got {tuple(names)}")
        return cls(tuple(names), open=open_spec)

    def check(self, value: "Tensor", where: str) -> None:
        from tensorgrad.tensor import Tensor

        if not isinstance(value, Tensor):
            raise EdgeTypeError(f"{where}: expected a Tensor with edges {set(self.edges)}, got {type(value).__name__}")
        have, want = set(value.edges), set(self.edges)
        if self.open:
            if want <= have:
                return
            raise EdgeTypeError(f"{where}: expected edges including {sorted(want)}, got {sorted(have)}")
        if have != want:
            missing, extra = want - have, have - want
            detail = "".join(
                f", {kind} {sorted(names)}" for kind, names in (("missing", missing), ("extra", extra)) if names
            )
            raise EdgeTypeError(f"{where}: expected edges {sorted(want)}, got {sorted(have)}{detail}")

    def __repr__(self) -> str:
        parts = (["..."] if self.open else []) + list(map(repr, self.edges))
        return f"Tensor[{', '.join(parts)}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EdgeSpec) and set(self.edges) == set(other.edges) and self.open == other.open

    def __hash__(self) -> int:
        return hash((frozenset(self.edges), self.open))


def _specs(func: Callable[..., Any]) -> dict[str, EdgeSpec]:
    """The function's EdgeSpec annotations, evaluating string annotations
    (`from __future__ import annotations` turns them all into strings)."""
    out = {}
    for name, ann in getattr(func, "__annotations__", {}).items():
        if isinstance(ann, str):
            try:
                ann = eval(ann, getattr(func, "__globals__", {}))  # noqa: S307
            except Exception:
                continue
        if isinstance(ann, EdgeSpec):
            out[name] = ann
    return out


def typed(func: F) -> F:
    """Enforce Tensor[...] edge-set annotations at call boundaries."""
    if not __debug__:
        return func
    specs = _specs(func)
    if not specs:
        return func
    sig = inspect.signature(func)
    ret = specs.pop("return", None)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        for name, spec in specs.items():
            if name in bound.arguments:
                spec.check(bound.arguments[name], f"{func.__qualname__}(): argument '{name}'")
        result = func(*args, **kwargs)
        if ret is not None:
            ret.check(result, f"{func.__qualname__}(): return value")
        return result

    # functools.wraps preserves the signature, but the wrapper's static type
    # (a _Wrapped) can't be reconciled with the TypeVar F.
    return wrapper  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
