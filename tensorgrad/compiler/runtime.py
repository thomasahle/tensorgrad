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
        fold: bool = True,
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
        self.fold_fires: dict[str, int] = {}
        if simplify:
            # Definition folding runs FIRST, on the pristine trees: it needs
            # the un-normalized compositions (normalization flattens e.g. the
            # softmax@v boundary), and it must precede gradient resolution so
            # the reverse sweep differentiates THROUGH the fused cells' VJPs.
            # Value-gated with fallback; see compiler/fold.py.
            originals = tensors
            # Zero-gradient pruning FIRST, on the pristine trees: gradients
            # that are provably zero by symmetry (szfp-certified on the
            # unfolded algebra — fused cells are opaque atoms) compile to
            # Zero, deleting their whole cotangent chains. Must precede
            # folding for provability; see compiler/zerograd.py.
            from tensorgrad.compiler.zerograd import prune_zero_gradients

            tensors = prune_zero_gradients(tensors)
            originals = tensors
            if fold:
                from tensorgrad.compiler.fold import fold_program

                tensors, self.fold_fires = fold_program(tensors, verbose=verbose)
            # Joint reverse-mode resolution: a family of Derivative outputs
            # sharing one scalar base resolves through ONE cotangent sweep,
            # so all gradients reference the same interned subexpressions
            # (see compiler/reverse.py). Unrecognized shapes pass through to
            # the independent chain-rule path below.
            from tensorgrad.compiler.reverse import resolve_shared_gradients

            try:
                tensors = resolve_shared_gradients(tensors)
                shared_args = normalize_args()
                tensors = tuple(t.simplify(shared_args) for t in tensors)
            except Exception:
                # Fused cells are reverse-mode only; if a gradient escaped
                # the reverse sweep into forward-mode resolution (fold's
                # preflight is a heuristic), retry the whole normalization
                # from the UN-folded originals -- never fail a compile that
                # worked before folding existed.
                if not self.fold_fires:
                    raise
                self.fold_fires = {"skipped:gradient-resolution-fallback": 1}
                tensors = resolve_shared_gradients(originals)
                tensors = tuple(t.simplify(normalize_args()) for t in tensors)
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
        # Per-output wrap plans, hoisted out of __call__ (they are static,
        # and the tie-break scan over all variables cost ~8ms/call on a
        # 160-output program). API parity with the old backends: an output
        # whose edge SET matches an input variable adopts that variable's
        # edge order (gradients declare the derivative machinery's order) —
        # an ORDERED match wins over adopting another variable's order (tied
        # embeddings: wte (vocab,d) vs lm_head (d,vocab)). With no unique
        # variable claim, fall back to the caller's declared expression
        # order: normalization may have swapped in an interned isomorphic
        # twin whose edges come out in another order.
        self._wrap_plans: list[tuple[Optional[tuple[int, ...]], tuple[str, ...]]] = []
        # Last-call fast path (see __call__): (shapes-dict ref, shapes copy,
        # per-var (var, names, shape, dtype, unnamed), specialized fn).
        self._call_plan: Optional[tuple] = None
        # Identity map from last call's named output views to their unnamed
        # bases (#65a): round-tripped state tensors skip aten::rename. Holds
        # (view, base) pairs — strong refs pin the views' ids; lookups verify
        # `is` identity so id reuse can never alias.
        self._last_out_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for (node, edge_order), declared in zip(self.outputs, self._declared_edges):
            tie_orders = {tuple(v.edges) for v in self.vars if set(edge_order) == set(v.edges)}
            if tuple(edge_order) not in tie_orders and len(tie_orders) == 1:
                target = next(iter(tie_orders))
            elif (
                (not tie_orders or len(tie_orders) > 1)
                and set(declared) == set(edge_order)
                and tuple(edge_order) != declared
            ):
                target = tuple(declared)
            else:
                target = tuple(edge_order)
            perm = None if target == tuple(edge_order) else tuple(edge_order.index(e) for e in target)
            self._wrap_plans.append((perm, target))
        # LRU cache of shape/dtype specializations (bounded: a training loop
        # sweeping many batch sizes must not accumulate compiled code forever).
        # key -> generated specialization function (Any: exec-produced callables)
        self._specializations: "OrderedDict[tuple, Any]" = OrderedDict()

    # -----------------------------------------------------------------

    def _torch_compile(self, fn: Any) -> Any:
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

        # Dynamo's shared entry frame re-specializes per compiled callable
        # (guarded on fn.__code__), so a process that compiles many programs
        # -- the benchmark suite compiles 15+ -- trips the default recompile
        # limit (8), which under fullgraph=True RAISES instead of falling
        # back. Raise the ceiling; the per-program fallback below still
        # catches a genuine overflow.
        cfg = torch._dynamo.config
        cfg.cache_size_limit = max(getattr(cfg, "cache_size_limit", 8), 128)
        if hasattr(cfg, "recompile_limit"):
            cfg.recompile_limit = max(cfg.recompile_limit, 128)

        # freezing=True unlocks Inductor's constant-folding + MKLDNN weight
        # prepacking. Safe for our callables BY CONSTRUCTION: they are pure
        # functions whose weights/data/scalars are all per-call arguments
        # (verified: outputs respond to changed data and scalars exactly like
        # the unfrozen build; 10-step training trajectories match to float
        # rounding), so only the hoisted spec-time constants can fold. This
        # is an advantage of the functional design that nn.Module TRAINING
        # cannot use (freezing would inline its in-place-updated parameters).
        # Measured: mlp step 2.88 -> 2.07ms (a CPU-wall row).
        opts: dict[str, Any] = {"options": {"freezing": True}}
        state = {"fn": torch.compile(fn, fullgraph=True, dynamic=False, **opts)}

        def wrapper(*args: Any) -> Any:
            try:
                return state["fn"](*args)
            except torch._dynamo.exc.TorchDynamoException:  # pyright: ignore[reportAttributeAccessIssue]  # dynamo has no stubs
                self.used_fullgraph_fallback = True
                state["fn"] = torch.compile(fn, fullgraph=False, dynamic=False, **opts)
                return state["fn"](*args)

        wrapper._source = fn._source  # type: ignore[attr-defined]  # keep the generated source inspectable
        return wrapper

    def _resolve_dims(self, values: dict, shapes: Optional[dict]) -> dict:
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
        # Fast path: in a training loop every call carries the same dims
        # dict, and per variable the same names/shape/dtype — which implies
        # the same specialization, the same alignment decisions and the same
        # promoted dtype as the recorded previous call, so all of that
        # bookkeeping (sympy-keyed dims resolution, alignment probing,
        # dtype promotion, spec lookup) can be skipped. Measured 225us per
        # call on the diffusion step — 30% of the whole step at that scale.
        # Any mismatch falls through to the general path, which re-records.
        plan = self._call_plan
        # (contents comparison, not identity: the caller may mutate its dict)
        if plan is not None and plan[1] == shapes:
            # Identity-cached views (#65a): a training loop feeds our own
            # last outputs straight back as the next state. The strong-ref
            # map from each named output view to its unnamed base (strong
            # refs also pin ids, so no id-reuse aliasing) lets round-tripped
            # tensors skip aten::rename — measured ~0.5us x ~40 renames plus
            # the alias churn per step on the sub-ms rows.
            out_map = self._last_out_map
            args = []
            ok = True
            for var, want_names, want_shape, want_dtype, unnamed in plan[2]:
                t = values.get(var)
                if (
                    t is None
                    or t.names != want_names
                    or t.shape != want_shape
                    or t.dtype is not want_dtype
                ):
                    ok = False
                    break
                if unnamed:
                    args.append(t)
                else:
                    hit = out_map.get(id(t))
                    if hit is not None and hit[0] is t:
                        args.append(hit[1])
                    else:
                        args.append(t.rename(None))
            if ok:
                with torch.no_grad():
                    outs = plan[3](*args)
                wrapped = []
                new_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                for out, (perm, names_t) in zip(outs, self._wrap_plans):
                    base = out if perm is None else out.permute(perm)
                    w = base.refine_names(*names_t)
                    # Key by id but STORE the view too: the `is` check at
                    # lookup makes id reuse (a dropped output's id recycled
                    # by an unrelated tensor) impossible to confuse — the
                    # bug class shows up as wrong-shape einsum errors.
                    new_map[id(w)] = (w, base)
                    wrapped.append(w)
                self._last_out_map = new_map
                return wrapped[0] if len(wrapped) == 1 else tuple(wrapped)

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
            fn = self.codegen.specialize(dims, verbose=self.verbose, dtype=dtype,
                                         for_inductor=self.torch_compile)
            if self.torch_compile:
                fn = self._torch_compile(fn)
            self._specializations[key] = fn
            if len(self._specializations) > SPECIALIZATION_CACHE_SIZE:
                self._specializations.popitem(last=False)
        else:
            self._specializations.move_to_end(key)

        with torch.no_grad():
            outs = fn(*args)

        # Record the fast-path plan — only when no input needed align_to
        # (then the fast path's raw rename(None) is exactly what this call
        # did). names/shape/dtype per var pin the specialization choice.
        entries = []
        plain = True
        for var in self.vars:
            t = values[var]
            unnamed = all(n is None for n in t.names)
            if not unnamed and t.names != tuple(var.edges):
                plain = False
                break
            entries.append((var, t.names, t.shape, t.dtype, unnamed))
        self._call_plan = (None, dict(shapes) if shapes else None, entries, fn) if plain else None

        wrapped = []
        for out, (perm, names_t) in zip(outs, self._wrap_plans):
            if perm is not None:
                out = out.permute(perm)
            wrapped.append(out.refine_names(*names_t))
        if len(wrapped) == 1:
            return wrapped[0]
        return tuple(wrapped)


def compile_to_callable(
    *tensors: Tensor,
    verbose: bool = False,
    torch_compile: bool = False,
    simplify: bool = True,
    fold: bool = True,
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
    program = CompiledProgram(
        tuple(tensors), verbose=verbose, torch_compile=torch_compile, simplify=simplify, fold=fold
    )
    if print_info:
        # Just some stats: how many tensor ops in the compiled program?
        from tensorgrad.compiler.ir import ConstNode, InputNode, toposort

        n_ops = sum(
            not isinstance(n, (InputNode, ConstNode)) for n in toposort([n for n, _ in program.outputs])
        )
        print(
            f"compiled {len(program.outputs)} outputs into one program of "
            f"{n_ops} tensor ops ({time.perf_counter() - t0:.1f}s)"
        )
        if program.fold_fires:
            print("folded: " + ", ".join(f"{k} x{v}" for k, v in sorted(program.fold_fires.items())))
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

    def __init__(self, **groups: Any):
        self.__dict__.update(groups)

    def __getattr__(self, name: str) -> Any:
        # Attributes are set dynamically from the compile-time output
        # keywords; this stub tells static checkers any name is fair game.
        raise AttributeError(f"no output named {name!r}; available: {sorted(self.__dict__)}")

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={type(v).__name__}" for k, v in self.__dict__.items())
        return f"Output({parts})"


class CompiledStep:
    def __init__(
        self, outputs: dict[str, Pytree], torch_compile: bool, print_info: bool = False, fold: bool = True
    ):
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
        self._fn = compile_to_callable(*flat, torch_compile=torch_compile, print_info=print_info, fold=fold)
        self._by_name = {v.name: v for v in self._fn.vars}

    @property
    def inputs(self) -> list[str]:
        """The program's input variable names. Smaller than you might
        expect is normal: partial derivatives legitimately drop variables
        by cancellation (an input set is a statement of conditional
        independence, not an oversight)."""
        return sorted(self._by_name)

    def _add(self, feed: dict, var: Any, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.get_default_dtype())
        feed[var] = value

    def __call__(self, *dicts: dict, dims: Optional[dict] = None, **kw: Any) -> Output:
        feed: dict[Variable, torch.Tensor] = {}
        for d in dicts:
            # Positional dicts are BULK state (keyed by Variable or by name)
            # and may carry more than this program consumes — extra entries
            # are skipped, so one state dict can feed several programs.
            # Keywords below stay strict (a typo there should fail loudly).
            for k, v in d.items():
                var = self._by_name.get(k) if isinstance(k, str) else k
                if var is not None and (not isinstance(var, Variable) or var.name in self._by_name):
                    self._add(feed, var, v)
        for name, value in kw.items():
            if name not in self._by_name:
                raise KeyError(
                    f"{name!r} is not an input of this program; inputs: {self.inputs}. "
                    f"Note that derived programs may not depend on every variable "
                    f"(partial derivatives drop inputs by cancellation). Keyword "
                    f"feeds are strict to catch typos; pass bulk state as a "
                    f"positional dict, where extra entries are skipped."
                )
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
    fold: bool = True,
    **outputs: Pytree,
) -> CompiledStep:
    """Compile named outputs (Tensors or pytrees of Tensors) into one fused
    program. Returns a callable whose result carries one attribute per
    output keyword, shaped like the declared pytree.

    Feeding binds by variable name (keywords or name-keyed dicts), so a
    state pytree KEYED BY VARIABLE NAMES round-trips with no declarations:
    `out = step(out.state, dims=DIMS, raw=batch)` — no order bookkeeping.

    (`torch_compile`, `print_info` and `fold` are reserved keywords; every
    other keyword names an output. `fold=False` disables definition folding
    — the value-gated rewrite of derived gelu/layer-norm/attention
    compositions onto the fused cells; see compiler/fold.py.)"""
    if not outputs:
        raise ValueError("compile needs at least one named output")
    return CompiledStep(outputs, torch_compile, print_info, fold)
