"""Runtime for compiled tensorgrad programs.

compile_to_callable(*tensors) lowers all outputs into one shared IR DAG at
compile time. The returned callable specializes lazily per shape signature
(concrete dims): the first call with a given set of sizes generates and
exec's straight-line torch code; subsequent calls hit a dict lookup and run
the generated function directly.

Usage:
    f = compile_to_callable(loss, *grads, verbose=False)
    loss_val, *grad_vals = f({x: x_t, y: y_t, ...}, {batch: 64, ...})
Input tensors are torch *named* tensors; outputs are named tensors too.
"""

from collections import OrderedDict
import time

import sympy
from typing import Any, Optional, Union

import torch

from tensorgrad.tensor import Derivative, Tensor, Variable
from tensorgrad.compiler.lower import lower_program
from tensorgrad.compiler.codegen_torch import TorchCodegen

# Max number of cached shape/dtype specializations per program (LRU-evicted).
SPECIALIZATION_CACHE_SIZE = 32


def normalize_args() -> dict:
    """The simplify() argument preset for the compiler's normalize stage:
    resolve Derivative nodes and derivative signatures, leave the algebra to
    the IR passes, memoize across shared subtrees. simplify() keeps its memo
    inside the args dict, so passing ONE dict instance to several simplify()
    calls shares the memo — preserving subtree sharing between a loss and its
    gradients (full_simplify would be exponential in depth on deep models;
    this stays linear)."""
    return {
        "grad_steps": float("inf"),
        "expand_functions": False,
        "combine_products": False,
        "sum_combine_terms": False,
        "factor_components": False,
        "memoize": True,
    }


class CompiledProgram:
    def __init__(
        self,
        tensors: tuple[Tensor, ...],
        verbose: bool = False,
        torch_compile: bool = False,
        simplify: bool = True,
    ):
        # Normalization is the compiler's first stage: resolve Derivative nodes
        # and derivative-signature Functions (which lowering cannot handle),
        # with the algebra otherwise left to the IR passes. Cheap and
        # idempotent on already-simplified input. simplify=False is a developer
        # escape hatch for compiling a raw structure verbatim (codegen tests).
        # The caller's edge ORDER is part of the output contract. Normalizing
        # may return an interned isomorphic twin whose edges come out in a
        # different order (which twin depends on what was built first — the
        # hash-cons table is process-global), so record the declared orders
        # before simplifying.
        self._declared_edges = [tuple(t.edges) for t in tensors]
        if simplify:
            # Joint reverse-mode resolution: a family of Derivative outputs
            # sharing one scalar base resolves through ONE cotangent sweep,
            # so all gradients reference the same interned subexpressions
            # (see compiler/reverse.py). Unrecognized shapes pass through to
            # the independent chain-rule path below.
            from tensorgrad.compiler.reverse import resolve_shared_gradients

            tensors = resolve_shared_gradients(tensors)
            shared_args = normalize_args()
            tensors = tuple(t.simplify(shared_args) for t in tensors)
        self.tensors = tensors
        self.verbose = verbose
        self.torch_compile = torch_compile
        # True once any specialization had to give up on fullgraph=True and
        # recompile with fullgraph=False (see _torch_compile). Inspectable by
        # tests and diagnostics.
        self.used_fullgraph_fallback = False
        self.builder, self.outputs = lower_program(list(tensors))
        self.codegen = TorchCodegen(self.builder, self.outputs)
        self.input_names = self.codegen.input_names  # sorted variable names
        self.vars = [self.builder.input_vars[n] for n in self.input_names]
        # All symbols we need concrete values for. Collected from EVERY IR
        # node, not just inputs and outputs: a size can be internal-only
        # (e.g. a window into a larger buffer runs the model on `seq` slots
        # while the program's inputs and outputs are all `buf`-sized).
        from tensorgrad.compiler.ir import toposort as _toposort

        self.symbols: set[sympy.Symbol] = set()
        for node in _toposort([n for n, _ in self.outputs]):
            for dim in node.dims:
                for s in sympy.sympify(dim).free_symbols:
                    if isinstance(s, sympy.Symbol):
                        self.symbols.add(s)
        for t in tensors:
            for s in t.shape.values():
                if isinstance(s, sympy.Symbol):
                    self.symbols.add(s)
        # LRU cache of shape/dtype specializations (bounded: a training loop
        # sweeping many batch sizes must not accumulate compiled code forever).
        # key -> generated specialization function (Any: exec-produced callables)
        self._specializations: "OrderedDict[tuple, Any]" = OrderedDict()

    # -----------------------------------------------------------------

    def _torch_compile(self, fn):
        """torch.compile with fullgraph=True, guarded by a lazy fallback.

        Codegen emits spec-time integer strides (codegen_torch.STATIC_STRIDES),
        so the straight-line program — no eval, no named tensors, no torch ops
        returning python ints — traces as ONE dynamo graph and Inductor can
        fuse across the whole program. Dynamo only raises at the first CALL,
        so the guard wraps invocation: if it still refuses (e.g. the
        STATIC_STRIDES=False runtime-stride emission, or an op outside
        dynamo's support), recompile that specialization with fullgraph=False
        (graph breaks around the offending op, fuses the rest) and record the
        event in self.used_fullgraph_fallback."""
        import torch._dynamo

        state = {"fn": torch.compile(fn, fullgraph=True, dynamic=False)}

        def wrapper(*args):
            try:
                return state["fn"](*args)
            except torch._dynamo.exc.TorchDynamoException:  # pyright: ignore[reportAttributeAccessIssue]  # dynamo has no stubs
                self.used_fullgraph_fallback = True
                state["fn"] = torch.compile(fn, fullgraph=False, dynamic=False)
                return state["fn"](*args)

        wrapper._source = fn._source  # keep the generated source inspectable
        return wrapper

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

    def __call__(
        self, values: dict, shapes: "Optional[dict]" = None
    ) -> "tuple[torch.Tensor, ...] | torch.Tensor":
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
                fn = self._torch_compile(fn)
            self._specializations[key] = fn
            if len(self._specializations) > SPECIALIZATION_CACHE_SIZE:
                self._specializations.popitem(last=False)
        else:
            self._specializations.move_to_end(key)

        with torch.no_grad():
            outs = fn(*args)

        wrapped = []
        for out, (node, edge_order), declared in zip(outs, self.outputs, self._declared_edges):
            named = out.refine_names(*edge_order)
            # API parity with the old backends: if an output has the same edge
            # set as an input variable but different order (common for
            # gradients, whose declared expression order is the derivative
            # machinery's, not the variable's), align it to the variable's
            # order. When several variables share the edge set (tied
            # embeddings: wte (vocab,d) vs lm_head (d,vocab)), an ORDERED
            # match wins — the output must not adopt another variable's order.
            # When no variable claims the edge set (or several tie
            # ambiguously), fall back to the edge order the caller's
            # expression declared: normalization may have swapped to an
            # interned isomorphic twin whose edges come out in another order.
            tie_orders = {tuple(var.edges) for var in self.vars if set(edge_order) == set(var.edges)}
            if tuple(edge_order) not in tie_orders and len(tie_orders) == 1:
                named = named.align_to(*next(iter(tie_orders)))
            elif (
                (not tie_orders or len(tie_orders) > 1)
                and set(declared) == set(edge_order)
                and tuple(edge_order) != declared
            ):
                named = named.align_to(*declared)
            wrapped.append(named)
        if len(wrapped) == 1:
            return wrapped[0]
        return tuple(wrapped)


def compile_to_callable(
    *tensors: Tensor,
    verbose: bool = False,
    torch_compile: bool = False,
    simplify: bool = True,
    print_info: bool = False,
) -> CompiledProgram:
    """Compile one or more tensorgrad tensors into a fast callable.

    Inputs are normalized automatically (Derivative nodes and derivative
    signatures resolved; see normalize_args), so gradients can be
    passed raw:

        step = compile_to_callable(loss, *[loss.grad(p) for p in params])

    Returns f(values: dict[Variable, torch.Tensor], shapes: dict[Symbol, int])
    -> named torch.Tensor or tuple of them (one per input tensor).
    """
    t0 = time.perf_counter()
    program = CompiledProgram(tuple(tensors), verbose=verbose, torch_compile=torch_compile, simplify=simplify)
    if print_info:
        # Just some stats: how many tensor ops in the compiled program?
        from tensorgrad.compiler.ir import ConstNode, InputNode, toposort

        n_ops = sum(
            not isinstance(n, (InputNode, ConstNode)) for n in toposort([n for n, _ in program.outputs])
        )
        print(
            f"compiled loss + {len(tensors)} gradients + adamw ({len(program.outputs)} "
            f"outputs) into one program of {n_ops} tensor ops "
            f"({time.perf_counter() - t0:.1f}s)"
        )
    return program

# ---------------------------------------------------------------------------
# Structured compile: named, pytree-shaped inputs and outputs
# ---------------------------------------------------------------------------
# compile_to_callable is positional: N tensors in, an N-tuple out, and the
# caller mirrors the order (the fragile part of every training loop). The
# layer below is pure sugar over it — nothing new is compiled:
#
#     grads = tg.grad(loss, params)                    # dict in, dict out
#     step = tg.compile(loss=loss, state=updates)      # Tensor or pytree each
#     out = step(out.state, dims=DIMS, raw=digits)     # dicts/keywords by name
#     loss_val = out.loss                              # out.state mirrors its dict
#
# Binding rules for step(...): keywords name input Variables; positional
# dicts are feed dicts keyed by Variable or by variable name (so a state
# pytree keyed by variable names round-trips with no declarations); dims=
# are the symbol sizes. Python scalars are wrapped with torch.as_tensor,
# so schedules like c1=1/(1-b**t) feed directly.


Pytree = Union[Tensor, dict, list, tuple]


def _flatten(tree: Any, path: str = "") -> list[tuple[str, Any]]:
    """Deterministic (path, leaf) pairs; dicts by insertion order."""
    if isinstance(tree, dict):
        return [p for k, v in tree.items() for p in _flatten(v, f"{path}.{k}")]
    if isinstance(tree, (list, tuple)):
        return [p for i, v in enumerate(tree) for p in _flatten(v, f"{path}[{i}]")]
    return [(path, tree)]


def _unflatten(tree: Any, leaves: list, pos: int = 0) -> tuple[Any, int]:
    if isinstance(tree, dict):
        out = {}
        for k, v in tree.items():
            out[k], pos = _unflatten(v, leaves, pos)
        return out, pos
    if isinstance(tree, (list, tuple)):
        vals = []
        for v in tree:
            r, pos = _unflatten(v, leaves, pos)
            vals.append(r)
        return type(tree)(vals), pos
    return leaves[pos], pos + 1


def grad(loss: Tensor, params: Pytree) -> Any:
    """Gradients of a scalar loss w.r.t. a pytree of Variables: the same
    pytree of (lazy) Derivative nodes. Compiling several of them together
    resolves the whole family through one shared cotangent sweep
    (tensorgrad/compiler/reverse.py)."""
    if loss.order != 0:
        raise ValueError(f"grad expects a scalar loss, got edges {set(loss.edges)}")
    flat = _flatten(params)
    for path, p in flat:
        if not isinstance(p, Variable):
            raise TypeError(f"grad: params{path} is {type(p).__name__}, expected Variable")
    grads = [Derivative(loss, p) for _, p in flat]
    tree, _ = _unflatten(params, grads)
    return tree


class Output:
    """Compiled-step results: one attribute per output keyword, each shaped
    like the pytree it was declared with."""

    def __init__(self, **groups):
        self.__dict__.update(groups)

    def __getattr__(self, name: str) -> Any:
        # Attributes are set dynamically from the compile-time output
        # keywords; this stub tells static checkers any name is fair game.
        raise AttributeError(f"no output named {name!r}; available: {sorted(self.__dict__)}")

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={type(v).__name__}" for k, v in self.__dict__.items())
        return f"Output({parts})"


class CompiledStep:
    def __init__(self, outputs: dict[str, Pytree], torch_compile: bool, print_info: bool = False):
        self._out_trees = dict(outputs)
        flat: list[Tensor] = []
        self._out_slices: dict[str, tuple[int, int]] = {}
        for name, tree in outputs.items():
            leaves = [t for _, t in _flatten(tree)]
            for t in leaves:
                if not isinstance(t, Tensor):
                    raise TypeError(f"compile: output {name!r} contains {type(t).__name__}, expected Tensor")
            self._out_slices[name] = (len(flat), len(flat) + len(leaves))
            flat.extend(leaves)
        self._fn = compile_to_callable(*flat, torch_compile=torch_compile, print_info=print_info)
        self._by_name = {v.name: v for v in self._fn.vars}

    def _add(self, feed: dict, var, value) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.get_default_dtype())
        feed[var] = value

    def __call__(self, *dicts: dict, dims: Optional[dict] = None, **kw) -> Output:
        feed: dict[Variable, torch.Tensor] = {}
        for d in dicts:
            for k, v in d.items():
                var = self._by_name.get(k) if isinstance(k, str) else k
                if var is None:
                    raise KeyError(f"unknown input variable name {k!r}")
                self._add(feed, var, v)
        for name, value in kw.items():
            if name not in self._by_name:
                known = sorted(self._by_name)
                raise KeyError(f"{name!r} is not an input variable; known: {known}")
            self._add(feed, self._by_name[name], value)
        outs = self._fn(feed, dims)
        if not isinstance(outs, tuple):
            outs = (outs,)
        groups = {}
        for name, tree in self._out_trees.items():
            lo, hi = self._out_slices[name]
            groups[name], _ = _unflatten(tree, list(outs[lo:hi]))
        return Output(**groups)


def compile(
    torch_compile: bool = False,
    print_info: bool = False,
    **outputs: Pytree,
) -> CompiledStep:
    """Compile named outputs (Tensors or pytrees of Tensors) into one fused
    program. Returns a callable whose result carries one attribute per
    output keyword, shaped like the declared pytree.

    Feeding binds by variable name (keywords or name-keyed dicts), so a
    state pytree KEYED BY VARIABLE NAMES round-trips with no declarations:
    `out = step(out.state, dims=DIMS, raw=batch)` — no order bookkeeping.

    (`torch_compile` and `print_info` are reserved keywords; every other
    keyword names an output.)"""
    if not outputs:
        raise ValueError("compile needs at least one named output")
    return CompiledStep(outputs, torch_compile, print_info)
