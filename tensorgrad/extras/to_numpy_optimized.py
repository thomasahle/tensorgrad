from functools import singledispatchmethod, lru_cache
from collections import defaultdict
import textwrap
import numpy as np
import sympy
import math
import sparse
import scipy.sparse

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


@lru_cache(maxsize=128)
def compute_permutation(source_edges: tuple, target_edges: tuple) -> tuple:
    """Cache permutation computations"""
    return tuple(source_edges.index(e) for e in target_edges)


def compile_to_callable(*tensors: Tensor, verbose=False, torch_compile=False, dtype=np.float32):
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
    context = CodegenContext(dtype=dtype)
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
            "scipy": scipy,
        },
        ns,
    )
    generated_forward = ns[function_name]

    # Pre-compute argument positions for efficiency
    arg_positions = {sym: i for i, sym in enumerate(all_symbols_sorted)}
    var_positions = {var: i + len(all_symbols_sorted) for i, var in enumerate(context.seen_variables)}

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
            for dim_symbol, size in zip(v.shape.values(), arr.shape):
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

        # Build args list efficiently using pre-computed positions
        args = [None] * (len(all_symbols_sorted) + len(context.seen_variables))
        
        # Insert dimension sizes
        for sym in all_symbols_sorted:
            args[arg_positions[sym]] = shapes[sym]
        
        # Insert variable arrays (with optimization to avoid unnecessary conversions)
        for var in context.seen_variables:
            if var not in values:
                raise KeyError(f"No value provided for variable {var.name}")
            arr = values[var]
            # Only convert if necessary
            if hasattr(arr, 'numpy'):  # It's a tensor
                arr = arr.rename(None).cpu().numpy()
            args[var_positions[var]] = arr

        nonlocal n_calls
        if verbose and n_calls == 0:
            call_expr = f"{function_name}({', '.join(symbol_args + var_placeholders)})"
            print(f"{call_expr}")
        n_calls += 1

        # Direct function call instead of eval
        outputs = generated_forward(*args)

        # If there's only one output, wrap it in a tuple for uniformity
        if len(tensors) == 1:
            outputs = (outputs,)

        # Wrap back in pytorch named tensors
        wrapped = []
        for numpy_array, original_tensor in zip(outputs, tensors):
            if not isinstance(numpy_array, np.ndarray):
                # Handle sparse arrays
                if hasattr(numpy_array, 'todense'):
                    numpy_array = numpy_array.todense()
                else:
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

    def __init__(self, dtype=np.float32):
        self.lines = []
        self.cache = {}
        self.seen_variables = set()
        self.seen_symbols = {}
        self.name_counters = defaultdict(int)
        self.var_types = {}
        self.dtype = dtype
        # Extract just the dtype name
        if hasattr(dtype, '__name__'):
            self.dtype_str = f"np.{dtype.__name__}"
        else:
            # Handle numpy dtype instances
            self.dtype_str = f"np.{np.dtype(dtype).name}"

    def emit(self, line: str):
        self.lines.append(line)

    def fresh_name(self, base: str):
        """Generate a fresh variable name efficiently using counters."""
        count = self.name_counters[base]
        self.name_counters[base] += 1
        return f"{base}_{count}" if count > 0 else base

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
        # Track symbols from variable shapes
        for symbol in tensor.shape.values():
            if isinstance(symbol, sympy.Symbol):
                self.declare_dimension(symbol)
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Zero) -> str:
        var_name = self.fresh_name("zero")
        shape_dims = [self.declare_dimension(t.shape[e]) for e in t.edges]
        shape_str = ", ".join(shape_dims)
        self.emit(f"{var_name} = np.zeros(({shape_str},), dtype={self.dtype_str})  # edges: {', '.join(t.edges)}")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Delta) -> str:
        var_name = self.fresh_name("delta")
        edges = list(t.edges)
        order = len(edges)
        size = self.declare_dimension(t._size)

        if order == 0:
            # Scalar
            self.emit(f"{var_name} = np.array({size}, dtype={self.dtype_str})")
        elif order == 1:
            self.emit(f"{var_name} = np.ones(({size},), dtype={self.dtype_str})  # {', '.join(edges)}")
        elif order == 2:
            self.emit(f"{var_name} = np.eye({size}, dtype={self.dtype_str})  # {', '.join(edges)}")
        else:
            shape_str = ", ".join([size] * order)
            self.emit(f"{var_name} = np.zeros(({shape_str},), dtype={self.dtype_str})")
            loopvar = self.fresh_name("i")
            code = textwrap.dedent(f"""\
            for {loopvar} in range({size}):
                {var_name}[{','.join([loopvar]*order)}] = 1.0
            """)
            self.emit(code)
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: F.Convolution) -> str:
        var_name = self.fresh_name("conv")
        edges = list(t.edges)
        assert len(edges) == 3, "Convolution must have 3 edges"

        w_in = self.declare_dimension(t.shape[t.input_name])
        k_size = self.declare_dimension(t.shape[t.kernel_name])
        w_out = self.declare_dimension(t.shape[t.output_name])

        # Generate sparse convolution - same as original
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
            {data} = np.ones({coords}.shape[0], dtype={self.dtype_str})
            {var_name} = sparse.COO({coords}.T, {data}, shape=({w_in}, {k_size}, {w_out}))
            """)
        )
        self.set_var_type(var_name, sparse.COO)
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: F.Reshape) -> str:
        """
        Example code for some reshape operation.
        """
        var_name = self.fresh_name("reshape")
        edges = list(t.edges)
        shape_dims = [self.declare_dimension(t.shape[e]) for e in edges]

        tmp_prod = self.fresh_name("prod")
        tmp_half = self.fresh_name("half")
        tmp_base = self.fresh_name("base")

        shape_list_str = "[" + ",".join(shape_dims) + "]"
        self.emit(f"{tmp_prod} = 1")
        self.emit(f"for _sz in {shape_list_str}:")
        self.emit(f"    {tmp_prod} *= _sz")
        self.emit(f"{tmp_half} = int(math.isqrt({tmp_prod}))")
        self.emit(f"if {tmp_half}*{tmp_half} != {tmp_prod}:")
        self.emit(f"    raise ValueError('Not a perfect square: '+str({tmp_prod}))")
        self.emit(f"{tmp_base} = np.eye({tmp_half}, dtype={self.dtype_str})")
        # Reshape to final shape
        shape_call = ", ".join(shape_dims)
        self.emit(f"{var_name} = {tmp_base}.reshape({shape_call})  # edges: {', '.join(t.edges)}")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Sum) -> str:
        var_name = self.fresh_name("sum")
        
        # Check if all terms are sparse to decide on strategy
        sparse_terms = []
        for st in t.terms:
            if isinstance(st, (Delta, F.Convolution)):
                sparse_terms.append(True)
            elif (cached := self.cache.get(st)) is not None:
                _, cached_name = cached
                sparse_terms.append(self.get_var_type(cached_name) == sparse.COO)
            else:
                sparse_terms.append(False)
        
        all_sparse = all(sparse_terms) and len(t.terms) > 1
        
        if all_sparse:
            # Use sparse addition which can be more efficient
            self.emit(f"{var_name} = None")
            for i, (w, st) in enumerate(zip(t.weights, t.terms)):
                c_name = self._emit_tensor(st)
                c_name += permute_code(st, t)
                
                if i == 0:
                    if w == 1:
                        self.emit(f"{var_name} = {c_name}")
                    elif w == -1:
                        self.emit(f"{var_name} = -{c_name}")
                    else:
                        self.emit(f"{var_name} = {float(w)} * {c_name}")
                else:
                    if w == 1:
                        self.emit(f"{var_name} = {var_name} + {c_name}")
                    elif w == -1:
                        self.emit(f"{var_name} = {var_name} - {c_name}")
                    else:
                        op = "+" if w > 0 else "-"
                        self.emit(f"{var_name} = {var_name} {op} {abs(float(w))} * {c_name}")
            self.set_var_type(var_name, sparse.COO)
        else:
            # Dense implementation
            shape_dims = [self.declare_dimension(t.shape[e]) for e in t.edges]
            shape_str = ", ".join(shape_dims)
            self.emit(f"{var_name} = np.zeros(({shape_str},), dtype={self.dtype_str})  # edges: {', '.join(t.edges)}")

            for w, st in zip(t.weights, t.terms):
                c_name = self._emit_tensor(st)
                c_name += permute_code(st, t)
                if w == 1:
                    code = f"{var_name} += {c_name}"
                elif w == -1:
                    code = f"{var_name} -= {c_name}"
                else:
                    code = f"{var_name} += {float(w)} * {c_name}"
                if self.get_var_type(c_name) == sparse.COO:
                    code += ".todense()"
                self.emit(code)

        return var_name

    @_emit_tensor_impl.register
    def _(self, prod: Product) -> str:
        """
        Emit code for a Product, optionally using opt_einsum or manual loops.
        """
        var_name = self.fresh_name("prod")

        # Handle edge case: empty product
        if not prod.factors:
            self.emit(f"{var_name} = np.array(1.0, dtype={self.dtype_str})  # empty product")
            return var_name

        # Single tensor product
        if len(prod.factors) == 1:
            single_name = self._emit_tensor(prod.factors[0])
            self.emit(f"{var_name} = {single_name}  # single tensor product")
            return var_name

        # Check for simple matrix multiplication pattern
        if len(prod.factors) == 2:
            f1, f2 = prod.factors
            if (f1.order == 2 and f2.order == 2 and 
                len(set(f1.edges) & set(f2.edges)) == 1):
                # Use np.matmul which is often faster than einsum for matrix multiply
                name1 = self._emit_tensor(f1)
                name2 = self._emit_tensor(f2)
                # Need to ensure correct axis alignment
                common_edge = list(set(f1.edges) & set(f2.edges))[0]
                f1_edges = list(f1.edges)
                f2_edges = list(f2.edges)
                
                # Transpose if needed to put common edge last for f1, first for f2
                if f1_edges.index(common_edge) == 0:
                    name1 += ".T"
                if f2_edges.index(common_edge) == 1:
                    name2 += ".T"
                
                self.emit(f"{var_name} = np.matmul({name1}, {name2})")
                
                # Handle output transpose if needed
                expected_edges = [e for e in f1.edges if e != common_edge] + [e for e in f2.edges if e != common_edge]
                if list(prod.edges) != expected_edges:
                    perm = [expected_edges.index(e) for e in prod.edges]
                    self.emit(f"{var_name} = {var_name}.transpose({perm})")
                
                return var_name

        # General einsum case
        factors, eq = prod._to_einsum_eq()
        op_list = ", ".join(self._emit_tensor(ft) for ft in factors)

        self.emit(f"{var_name} = oe.contract('{eq}', {op_list}, optimize='auto')")

        self.emit(f"if isinstance({var_name}, sparse.COO):")
        self.emit(f"    {var_name} = {var_name}.todense()")

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
            code_str = f"({child_names[0]} >= 0).astype({self.dtype_str})"
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
    perm = compute_permutation(tuple(source.edges), tuple(target.edges))
    if perm == tuple(range(len(perm))):
        return ""
    return f".transpose({perm})"