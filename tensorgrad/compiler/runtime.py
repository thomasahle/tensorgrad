"""Runtime for compiled tensorgrad programs.

compile_to_callable(*tensors) lowers all outputs into one shared IR DAG at
compile time. The returned callable specializes lazily per shape signature
(concrete dims): the first call with a given set of sizes generates and
exec's straight-line torch code; subsequent calls hit a dict lookup and run
the generated function directly.

API-compatible with tensorgrad.extras.to_pytorch.compile_to_callable:
    f = compile_to_callable(loss, *grads, verbose=False)
    loss_val, *grad_vals = f({x: x_t, y: y_t, ...}, {batch: 64, ...})
Input tensors are torch *named* tensors; outputs are named tensors too.
"""

from collections import OrderedDict

import sympy
import torch

from tensorgrad.tensor import Derivative, Tensor, Variable, compile_simplify_args
from tensorgrad.compiler.lower import lower_program
from tensorgrad.compiler.codegen_torch import TorchCodegen

# Max number of cached shape/dtype specializations per program (LRU-evicted).
SPECIALIZATION_CACHE_SIZE = 32


def _contains_derivative(tensors) -> bool:
    """Iterative DAG walk (shared visited-set across all outputs)."""
    seen: set[int] = set()
    stack = list(tensors)
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Derivative):
            return True
        stack.extend(getattr(t, "factors", ()) or ())
        stack.extend(getattr(t, "terms", ()) or ())
        stack.extend(getattr(t, "inputs", ()) or ())
        if (inner := getattr(t, "tensor", None)) is not None:
            stack.append(inner)
    return False


class CompiledProgram:
    def __init__(
        self,
        tensors: tuple[Tensor, ...],
        verbose: bool = False,
        torch_compile: bool = False,
        simplify: bool | str = "auto",
    ):
        # Derivative nodes can't be lowered; resolve them here so users can
        # write compile_to_callable(loss, *[loss.grad(p) for p in params])
        # directly. One shared args dict = one shared memo across ALL outputs,
        # which preserves cross-output subtree sharing (better than per-output
        # simplify_for_compile calls). "auto" simplifies only when a Derivative
        # is present, keeping behavior identical for pre-simplified inputs.
        if simplify == "auto":
            simplify = _contains_derivative(tensors)
        if simplify:
            shared_args = compile_simplify_args()
            tensors = tuple(t.simplify_for_compile(shared_args) for t in tensors)
        self.tensors = tensors
        self.verbose = verbose
        self.torch_compile = torch_compile
        self.builder, self.outputs = lower_program(list(tensors))
        self.codegen = TorchCodegen(self.builder, self.outputs)
        self.input_names = self.codegen.input_names  # sorted variable names
        self.vars = [self.builder.input_vars[n] for n in self.input_names]
        # All symbols we need concrete values for
        self.symbols: set[sympy.Symbol] = set()
        for var in self.vars:
            for s in var.shape.values():
                if isinstance(s, sympy.Symbol):
                    self.symbols.add(s)
        for t in tensors:
            for s in t.shape.values():
                if isinstance(s, sympy.Symbol):
                    self.symbols.add(s)
        # LRU cache of shape/dtype specializations (bounded: a training loop
        # sweeping many batch sizes must not accumulate compiled code forever).
        self._specializations: OrderedDict[tuple, object] = OrderedDict()

    # -----------------------------------------------------------------

    def _resolve_dims(self, values, shapes):
        dims = dict(shapes) if shapes else {}
        for v, t in values.items():
            if not isinstance(v, Variable):
                continue
            for e, size in zip(v.shape.keys(), self._aligned(v, t).shape):
                sym = v.shape[e]
                if isinstance(sym, sympy.Symbol):
                    if sym in dims and dims[sym] != size:
                        raise ValueError(f"Conflicting size for {sym}: {dims[sym]} vs {size}")
                    dims.setdefault(sym, size)
        missing = {s for s in self.symbols if s not in dims}
        if missing:
            raise ValueError(f"Missing dimension values for symbols: {missing}")
        return dims

    def _aligned(self, var: Variable, t: torch.Tensor) -> torch.Tensor:
        """Align a named tensor to the variable's canonical edge order."""
        want = tuple(var.edges)
        if t.names == want:
            return t
        if all(n is None for n in t.names):
            return t  # unnamed: trust the caller's order
        return t.align_to(*want)

    def __call__(self, values: dict, shapes: dict = None):
        dims = self._resolve_dims(values, shapes)

        args = []
        for var in self.vars:
            t = values.get(var)
            if t is None:
                raise KeyError(f"No value provided for variable {var.name}")
            args.append(self._aligned(var, t).rename(None))

        # The specialization's constants are built in the inputs' (promoted)
        # floating dtype, so bf16/fp64 inputs never meet fp32 constants.
        dtype = None
        for a in args:
            if a.is_floating_point():
                dtype = a.dtype if dtype is None else torch.promote_types(dtype, a.dtype)
        if dtype is None:
            dtype = torch.get_default_dtype()

        key = tuple(sorted((s.name, v) for s, v in dims.items()))
        if dtype != torch.get_default_dtype():
            key = key + (str(dtype),)
        fn = self._specializations.get(key)
        if fn is None:
            fn = self.codegen.specialize(dims, verbose=self.verbose, dtype=dtype)
            if self.torch_compile:
                # Straight-line torch calls, no eval, no named tensors — but
                # affine view rewrites query .stride()/.storage_offset() at
                # runtime, which dynamo cannot trace fullgraph (returns
                # non-Tensor ints). fullgraph=False graph-breaks around them
                # and fuses the rest; measured at/below autograd either way.
                fn = torch.compile(fn, fullgraph=False, dynamic=False)
            self._specializations[key] = fn
            if len(self._specializations) > SPECIALIZATION_CACHE_SIZE:
                self._specializations.popitem(last=False)
        else:
            self._specializations.move_to_end(key)

        with torch.no_grad():
            outs = fn(*args)

        wrapped = []
        for out, (node, edge_order), t in zip(outs, self.outputs, self.tensors):
            named = out.refine_names(*edge_order)
            # API parity with the old backends: if an output has the same edge
            # set as an input variable but different order (common for
            # gradients), align it to the variable's order.
            for var in self.vars:
                if set(edge_order) == set(var.edges) and list(edge_order) != list(var.edges):
                    named = named.align_to(*var.edges)
                    break
            wrapped.append(named)
        if len(wrapped) == 1:
            return wrapped[0]
        return tuple(wrapped)


def compile_to_callable(
    *tensors: Tensor,
    verbose: bool = False,
    torch_compile: bool = False,
    simplify: bool | str = "auto",
):
    """Compile one or more tensorgrad tensors into a fast callable.

    Derivative nodes are resolved automatically (see simplify_for_compile), so
    gradients can be passed raw:

        step = compile_to_callable(loss, *[loss.grad(p) for p in params])

    Returns f(values: dict[Variable, torch.Tensor], shapes: dict[Symbol, int])
    -> named torch.Tensor or tuple of them (one per input tensor).
    """
    return CompiledProgram(
        tuple(tensors), verbose=verbose, torch_compile=torch_compile, simplify=simplify
    )
