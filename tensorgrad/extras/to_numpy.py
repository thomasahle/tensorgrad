from functools import singledispatchmethod
import textwrap
import numpy as np
import sympy
import math
import sparse

import opt_einsum as oe
import torch

from tensorgrad import functions as F
from tensorgrad.tensor import (
    Rename,
    Tensor,
    Variable,
    Product,
    Sum,
    Delta,
    Zero,
    Derivative,
    Function,
)


def compile_to_callable(*tensors: Tensor, verbose=False, torch_compile=False):
    """
    Build a Python callable returning NumPy arrays:
        f(values, shapes=None) -> np.ndarray or tuple[np.ndarray, ...]

    where:
      - `values` is a dict {Variable: np.ndarray}.
      - `shapes` is an optional dict {sympy.Symbol: int} with dimension sizes.
      - The code generator inspects all Symbols in the given Tensors, sorts them by
        symbol name, and uses them in the generated function signature.

    The generated function `_generated_forward(...)` will be something like:

        def _generated_forward(batch, c0, w0, h0, _var_1234, _var_5678, ...):
            ...
            return result

    This function is then executed via `eval(...)`.
    """
    context = CodegenContext()
    final_var_names = [context._emit_tensor(t) for t in tensors]
    script = "\n".join(context.lines)

    function_name = "_generated_forward"

    # Sort the symbol names so we have a deterministic argument order
    all_symbols_sorted = sorted(context.seen_symbols, key=lambda s: s.name)

    # Gather placeholders for each Variable
    var_placeholders = [f"_var_{id(v)}" for v in context.seen_variables]

    # The final signature is: def _generated_forward(<dims>, <vars>):
    # e.g. def _generated_forward(batch, w0, out, _var_140661515588336, _var_140661515588464, ...):
    symbol_args = [f"{context.declare_dimension(s)}" for s in all_symbols_sorted]
    all_args = symbol_args + var_placeholders
    signature = ", ".join(all_args)

    out_type = "np.ndarray" if len(tensors) == 1 else "tuple[np.ndarray, ...]"

    # Build the generated function as a string
    wrapped_code = f"""\
def {function_name}({signature}) -> {out_type}:
{textwrap.indent(script, "    ")}
    return {', '.join(final_var_names)}
"""
    # Maybe later we can support numba
    # if torch_compile:
    #     wrapped_code = "@numba.jit()\n" + wrapped_code

    # TODO: How can we move work that doesn't depend on the inputs out of the function?

    if verbose:
        print()
        print("\n".join(f"{i+1:3}: {line}" for i, line in enumerate(wrapped_code.split("\n"))))

    # Prepare a namespace for exec
    ns = {}
    exec(
        wrapped_code,
        {
            "np": np,
            "oe": oe,
            "math": math,
            "sparse": sparse,
        },
        ns,
    )
    generated_forward = ns[function_name]

    # Keep track of how many times the inner function is called
    n_calls = 0

    def forward_with_values(values: dict[Variable, torch.Tensor], shapes: dict[sympy.Symbol, int] = None):
        """
        Invoke the generated function `_generated_forward(...)` by passing
        dimension sizes in sorted order (by symbol name), followed by the
        variable arrays from `values`.
        """
        if shapes is None:
            shapes = {}

        # Populate shapes dict from the provided arrays
        for v, arr in values.items():
            if not isinstance(v, Variable):
                continue
            for dim_symbol, size in zip(v.shape.keys(), arr.shape):
                # `dim_symbol` is a sympy.Symbol for that dimension
                if dim_symbol not in shapes:
                    shapes[dim_symbol] = size
                elif shapes[dim_symbol] != size:
                    raise ValueError(
                        f"Conflicting size for symbol {dim_symbol}: "
                        f"got both {shapes[dim_symbol]} and {size}"
                    )

        # Check we got the right dimension arguments
        missing = set(all_symbols_sorted) - set(shapes.keys())
        if missing:
            raise ValueError(f"Missing dimension values for symbols: {missing}")

        # Build the local namespace
        local_ns = {}
        local_ns[function_name] = generated_forward

        # Insert dimension sizes
        for sym in all_symbols_sorted:
            local_ns[sym.name] = shapes[sym]

        # Insert variable arrays
        for var in context.seen_variables:
            placeholder_name = f"_var_{id(var)}"
            if var not in values:
                raise KeyError(f"No value provided for variable {var.name}")
            local_ns[placeholder_name] = values[var].rename(None).cpu().numpy()

        # Evaluate the function call:
        call_args = [s.name for s in all_symbols_sorted] + [f"_var_{id(v)}" for v in context.seen_variables]
        call_expr = f"{function_name}({', '.join(call_args)})"

        nonlocal n_calls
        if verbose and n_calls == 0:
            print(f"{call_expr}")
        n_calls += 1

        outputs = eval(call_expr, {}, local_ns)

        # If there's only one output, wrap it in a tuple for uniformity
        if len(tensors) == 1:
            outputs = (outputs,)

        # Wrap back in pytorch named tensors
        wrapped = []
        for numpy_array, original_tensor in zip(outputs, tensors):
            if not isinstance(numpy_array, np.ndarray):
                numpy_array = np.array(numpy_array)
            wrapped.append(torch.from_numpy(numpy_array).refine_names(*original_tensor.edges))

        # Return as a tuple or a single array
        if len(wrapped) == 1:
            return wrapped[0]
        return tuple(wrapped)

    return forward_with_values


class CodegenContext:
    """
    Maintains the lines of generated Python code and caches expressions.
    """

    def __init__(self):
        self.lines = []
        self.cache = {}
        self.seen_variables = set()
        self.seen_symbols = {}
        self.seen_names = set()
        self.var_types = {}

    def emit(self, line: str):
        self.lines.append(line)

    def fresh_name(self, base: str):
        """Generate a fresh variable name each time it is called."""
        name = base
        i = 1
        while name in self.seen_names:
            name = f"{base}_{i}"
            i += 1
        self.seen_names.add(name)
        return name

    def declare_dimension(self, symbol):
        """
        Return a string naming this dimension: either:
          - str(symbol) if it's an integer
          - symbol.name (with uniqueness checking) if it's a Sympy Symbol
        """
        if isinstance(symbol, int):
            return str(symbol)
        if isinstance(symbol, sympy.Symbol):
            if symbol not in self.seen_symbols:
                # Ensure we don't collide with e.g. loop variables
                name = self.fresh_name(symbol.name)
                self.seen_symbols[symbol] = name
            return self.seen_symbols[symbol]
        raise TypeError(f"Expected int or sympy.Symbol, got {symbol} (type={type(symbol)})")

    def set_var_type(self, var_name, var_type):
        self.var_types[var_name] = var_type

    def get_var_type(self, var_name):
        return self.var_types.get(var_name)

    def _emit_tensor(self, tensor: Tensor) -> str:
        """
        Return the name of a local variable in the code that holds `tensor`.
        If we've already generated code for an isomorphic version of `tensor`,
        return the existing name; otherwise generate new code.
        """
        if (val := self.cache.get(tensor)) is not None:
            other, name = val
            # Attempt to see if the current `tensor` is a permutation of `other`
            mapping = next(tensor.isomorphisms(other), None)
            if mapping is not None:
                # Build a permutation to reorder the existing name
                other_edges = list(other.edges)
                perm = [other_edges.index(mapping[e]) for e in tensor.edges]
                if perm != list(range(tensor.order)):
                    return f"{name}.transpose({perm})"
                return name

        var_name = self._emit_tensor_impl(tensor)
        self.cache[tensor] = (tensor, var_name)
        return var_name

    @singledispatchmethod
    def _emit_tensor_impl(self, tensor: Tensor) -> str:
        raise NotImplementedError(f"No codegen handler for {type(tensor)}")

    @_emit_tensor_impl.register
    def _(self, tensor: Variable) -> str:
        self.seen_variables.add(tensor)
        var_name = self.fresh_name(f"var_{tensor.name}")
        placeholder_name = f"_var_{id(tensor)}"
        self.emit(f"{var_name} = {placeholder_name}  # shape: {', '.join(tensor.edges)}")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Zero) -> str:
        var_name = self.fresh_name("zero_")
        shape_dims = [self.declare_dimension(t.shape[e]) for e in t.edges]
        shape_str = ", ".join(shape_dims)
        self.emit(f"{var_name} = np.zeros(({shape_str},), dtype=np.float32)  # edges: {', '.join(t.edges)}")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Delta) -> str:
        var_name = self.fresh_name("delta_")
        edges = list(t.edges)
        order = len(edges)
        size = self.declare_dimension(t._size)

        if order == 0:
            # Scalar
            self.emit(f"{var_name} = np.array({size}, dtype=np.float32)")
        elif order == 1:
            self.emit(f"{var_name} = np.ones(({size},), dtype=np.float32)  # {', '.join(edges)}")
        elif order == 2:
            self.emit(f"{var_name} = np.eye({size}, dtype=np.float32)  # {', '.join(edges)}")
        else:
            shape_str = ", ".join([size] * order)
            self.emit(f"{var_name} = np.zeros(({shape_str},), dtype=np.float32)")
            loopvar = self.fresh_name("i")
            code = textwrap.dedent(f"""\
            for {loopvar} in range({size}):
                {var_name}[{','.join([loopvar]*order)}] = 1.0
            """)
            self.emit(code)
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: F.Convolution) -> str:
        var_name = self.fresh_name("conv_")
        edges = list(t.edges)
        assert len(edges) == 3, "Convolution must have 3 edges"

        w_in = self.declare_dimension(t.shape[t.input_name])
        k_size = self.declare_dimension(t.shape[t.kernel_name])
        w_out = self.declare_dimension(t.shape[t.output_name])

        coords = self.fresh_name("coords")
        data = self.fresh_name("data")
        k = self.fresh_name("k")
        j = self.fresh_name("j")
        self.emit(
            textwrap.dedent(f"""\
            {coords} = np.array([[{k}+{j}, {j}, {k}]
                               for {j} in range({k_size})
                               for {k} in range({w_out})
                               if {j} + {k} < {w_in}])
            {data} = np.ones({coords}.shape[0], dtype=np.float32)
            {var_name} = sparse.COO({coords}.T, {data}, shape=({w_in}, {k_size}, {w_out}))
            """)
        )
        self.set_var_type(var_name, sparse.COO)
        return var_name

        # self.emit(f"{var_name} = np.zeros(({w_in}, {k_size}, {w_out}), dtype=np.float32)")
        # loop_k = self.fresh_name("k")
        # loop_j = self.fresh_name("j")
        # code = textwrap.dedent(f"""\
        # for {loop_k} in range({w_out}):
        #     for {loop_j} in range({k_size}):
        #         {var_name}[{loop_k}+{loop_j}, {loop_j}, {loop_k}] = 1.0
        # """)
        # self.emit(code)
        # return var_name

    @_emit_tensor_impl.register
    def _(self, t: F.Reshape) -> str:
        """
        Example code for some reshape operation.
        """
        var_name = self.fresh_name("reshape_")
        edges = list(t.edges)
        shape_dims = [self.declare_dimension(t.shape[e]) for e in edges]

        tmp_prod = self.fresh_name("prod_")
        tmp_half = self.fresh_name("half_")
        tmp_base = self.fresh_name("base_")

        shape_list_str = "[" + ",".join(shape_dims) + "]"
        self.emit(f"{tmp_prod} = 1")
        self.emit(f"for _sz in {shape_list_str}:")
        self.emit(f"    {tmp_prod} *= _sz")
        self.emit(f"{tmp_half} = int(math.isqrt({tmp_prod}))")
        self.emit(f"if {tmp_half}*{tmp_half} != {tmp_prod}:")
        self.emit(f"    raise ValueError('Not a perfect square: '+str({tmp_prod}))")
        self.emit(f"{tmp_base} = np.eye({tmp_half}, dtype=np.float32)")
        # Reshape to final shape
        shape_call = ", ".join(shape_dims)
        self.emit(f"{var_name} = {tmp_base}.reshape({shape_call})  # edges: {', '.join(t.edges)}")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Sum) -> str:
        var_name = self.fresh_name("sum_")
        # For a sum of terms, we start from zero of the correct shape
        shape_dims = [self.declare_dimension(t.shape[e]) for e in t.edges]
        shape_str = ", ".join(shape_dims)

        self.emit(f"{var_name} = np.zeros(({shape_str},), dtype=np.float32)  # edges: {', '.join(t.edges)}")

        for w, st in zip(t.weights, t.terms):
            c_name = self._emit_tensor(st)
            c_name += permute_code(st, t)  # if we need a transpose
            if w == 1:
                code = f"{var_name} += {c_name}"
            elif w == -1:
                code = f"{var_name} -= {c_name}"
            else:
                code = f"{var_name} += {float(w)} * {c_name}"
            if self.get_var_type(var_name) == sparse.COO:
                code += ".todense()"
            self.emit(code)

        return var_name

    @_emit_tensor_impl.register
    def _(self, prod: Product) -> str:
        """
        Emit code for a Product, optionally using opt_einsum or manual loops.
        """
        var_name = self.fresh_name("prod_")

        # Handle edge case: empty product
        if not prod.factors:
            self.emit(f"{var_name} = np.array(1.0, dtype=np.float32)  # empty product")
            return var_name

        # Single tensor product
        if len(prod.factors) == 1:
            single_name = self._emit_tensor(prod.factors[0])
            self.emit(f"{var_name} = {single_name}  # single tensor product")
            return var_name

        factors, eq = prod._to_einsum_eq()
        op_list = ", ".join(self._emit_tensor(ft) for ft in factors)

        self.emit(f"{var_name} = oe.contract('{eq}', {op_list}, optimize='auto')")

        self.emit(f"if isinstance({var_name}, sparse.COO):")
        self.emit(f"    {var_name} = {var_name}.todense()")
        # if all(self.get_var_type(c) is sparse.COO for c in op_list):
        #     self.emit("# Expecting a sparse result")
        #     self.set_var_type(var_name, sparse.COO)
        # else:
        #     self.emit("# Expecting a dense result")
        #     self.emit(f"assert isinstance({var_name}, np.ndarray)")

        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Function) -> str:
        signature = t.signature
        name = "".join(c for c in signature.name if c.isalnum())
        var_name = self.fresh_name(f"fn_{name}")
        child_names = [self._emit_tensor(inp) for inp in t.inputs]

        if isinstance(signature, F._PowerFunction):
            code_str = f"np.power({child_names[0]}, {signature.k})"
            code_str += permute_code(t.inputs[0], t)
        elif signature.name in ("exp", "log"):
            code_str = f"np.{signature.name}({child_names[0]})"
            code_str += permute_code(t.inputs[0], t)
        elif signature.name == "relu":
            code_str = f"np.maximum(0, {child_names[0]})"
            code_str += permute_code(t.inputs[0], t)
        elif signature.name == "gt0":
            code_str = f"({child_names[0]} >= 0).astype(np.float32)"
            code_str += permute_code(t.inputs[0], t)
        elif signature.name == "argmax":
            edges = list(t.inputs[0].edges)
            dim = edges.index(signature.dim)
            code_str = f"np.argmax({child_names[0]}, axis={dim})"
            del edges[dim]
            perm = [edges.index(e) for e in t.edges]
            if perm != list(range(t.order)):
                code_str += f".transpose({', '.join(map(str, perm))})"
            assert not t.shape_out, "argmax should have no output dims"
        else:
            raise NotImplementedError(f"Unknown function {t.signature}")

        self.emit(f"{var_name} = {code_str}  # edges: {', '.join(t.edges)}")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Rename) -> str:
        var_name = self._emit_tensor(t.tensor)
        edges = list(t.tensor.edges)
        inverse_mapping = {e: o for o, e in t.mapping.items()}
        perm = [edges.index(inverse_mapping.get(e, e)) for e in edges]
        if perm != list(range(t.order)):
            self.emit(f"{var_name} = {var_name}.transpose({', '.join(map(str, perm))})")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Derivative) -> str:
        raise NotImplementedError(f"Derivative not implemented in codegen. Please simplify first: {t=}")


def permute_code(source: Tensor, target: Tensor) -> str:
    """
    Return a string for `.transpose(...)` in NumPy if needed, to reorder
    'source' edges into 'target' edges.
    """
    source_edges = list(source.edges)
    perm = [source_edges.index(e) for e in target.edges]
    if perm == list(range(len(perm))):
        return ""
    return f".transpose({perm})"
