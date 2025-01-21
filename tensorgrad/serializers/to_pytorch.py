from functools import singledispatchmethod
import textwrap
import torch
import sympy

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


class CodegenContext:
    """
    Holds:
      - lines of Python code
      - a cache from Tensor -> local var name
      - a set of encountered Variables
      - a set of encountered Sympy Symbols
    """

    def __init__(self):
        self.lines = []
        self.cache = {}
        self.seen_variables = set()
        self.seen_symbols = set()
        self.seen_names = set()

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
          - str(symbol) if it's an int
          - symbol.name (and record symbol in self.seen_symbols) if it's a Sympy Symbol
        """
        if isinstance(symbol, int):
            # A literal integer
            return str(symbol)
        if isinstance(symbol, sympy.Symbol):
            self.seen_symbols.add(symbol)
            return symbol.name  # e.g. "batch"
        raise TypeError(f"Expected int or sympy.Symbol, got {symbol} (type={type(symbol)})")

    def _emit_tensor(self, tensor: Tensor) -> str:
        """Return the name of a local variable that holds `tensor`."""
        if (val := self.cache.get(tensor)) is not None:
            # We might find an old version of the tensor, but it could have
            # a different permutation / naming than we now expect.
            # Find the isomorphic representative that we matched
            other, name = val
            # Find the mapping from "tensor's" naming to "other's" naming
            mapping = next(tensor.isomorphisms(other), None)
            # Turn into permutation
            other_names = list(other.edges)
            perm = [other_names.index(mapping[e]) for e in tensor.edges]
            if perm != list(range(tensor.order)):
                name += f".permute({', '.join(map(str,perm))})"
            return name

        var_name = self._emit_tensor_impl(tensor)
        self.cache[tensor] = (tensor, var_name)
        return var_name

    @singledispatchmethod
    def _emit_tensor_impl(self, tensor: Tensor) -> str:
        """Base implementation for tensor emission."""
        raise NotImplementedError(f"Don't know how to emit code for {type(tensor)}")

    @_emit_tensor_impl.register
    def _(self, tensor: Variable) -> str:
        self.seen_variables.add(tensor)
        var_name = self.fresh_name(f"var_{tensor.name}")
        placeholder_name = f"_var_{id(tensor)}"
        self.emit(f"{var_name} = {placeholder_name}")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Zero) -> str:
        var_name = self.fresh_name("zero_")
        shape_dims = [self.declare_dimension(t.shape[e]) for e in t.edges]
        self.emit(f"{var_name} = torch.zeros([{','.join(shape_dims)}])  # {', '.join(t.edges)}")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Delta) -> str:
        var_name = self.fresh_name("copy_")
        edges = list(t.edges)
        order = len(edges)
        shape_dims = [self.declare_dimension(t.shape[e]) for e in edges]

        if order == 0:
            size_str = self.declare_dimension(t.size)
            self.emit(f"{var_name} = torch.tensor({size_str}, dtype=torch.float32)")
        elif order == 1:
            self.emit(
                f"{var_name} = torch.ones([{shape_dims[0]}], dtype=torch.float32)  # {', '.join(edges)}"
            )
        elif order == 2:
            self.emit(f"{var_name} = torch.eye({shape_dims[0]}, dtype=torch.float32)  # {', '.join(edges)}")
        else:
            dims_str = "[" + ",".join(shape_dims) + "]"
            self.emit(f"{var_name} = torch.zeros({dims_str}, dtype=torch.float32)  # {', '.join(edges)}")
            loopvar = self.fresh_name("i")
            code = textwrap.dedent(f"""\
                for {loopvar} in range({shape_dims[0]}):
                    {var_name}[{','.join([loopvar]*order)}] = 1
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

        self.emit(f"{var_name} = torch.zeros([{w_in}, {k_size}, {w_out}])  # {', '.join(edges)}")

        loop_k = self.fresh_name("k")
        loop_j = self.fresh_name("j")
        code = textwrap.dedent(f"""\
        for {loop_k} in range({w_out}):
            for {loop_j} in range({k_size}):
                {var_name}[{loop_k}+{loop_j}, {loop_j}, {loop_k}] = 1
        """)
        self.emit(code)
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: F.Reshape) -> str:
        var_name = self.fresh_name("reshape_")
        edges = list(t.edges)
        shape_dims = [self.declare_dimension(t.shape[e]) for e in edges]

        tmp_prod = self.fresh_name("prod")
        tmp_half = self.fresh_name("half")
        tmp_base = self.fresh_name("base")

        shape_list_str = "[" + ",".join(shape_dims) + "]"
        # rename_str = f".rename(*{tuple(edges)})" if edges else ""

        self.emit(f"{tmp_prod} = 1")
        self.emit(f"for _sz in {shape_list_str}:\n    {tmp_prod} *= _sz")
        self.emit(f"{tmp_half} = int(math.isqrt({tmp_prod}))")
        self.emit(
            f"if {tmp_half} * {tmp_half} != {tmp_prod}: raise ValueError('Not a perfect square: '+str({tmp_prod}))"
        )
        self.emit(f"{tmp_base} = torch.eye({tmp_half})")
        # context.emit(f"{var_name} = {tmp_base}.reshape({','.join(shape_dims)}){rename_str}")
        self.emit(f"{var_name} = {tmp_base}.reshape({','.join(shape_dims)})")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Sum) -> str:
        var_name = self.fresh_name("sum_")
        self.emit(f"{var_name} = 0")

        for w, st in zip(t.weights, t.tensors):
            c_name = self._emit_tensor(st)
            if w == 1:
                self.emit(f"{var_name} += {c_name}")
            elif w == -1:
                self.emit(f"{var_name} -= {c_name}")
            else:
                self.emit(f"{var_name} += {float(w)} * {c_name}")

        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Product) -> str:
        """Emit code for a Product tensor."""
        var_name = self.fresh_name("prod_")

        # Handle empty product
        if not t.tensors:
            self.emit(f"{var_name} = torch.tensor(1.0)  # empty product")
            return var_name

        # Handle single tensor case
        if len(t.tensors) == 1:
            single_name = self._emit_tensor(t.tensors[0])
            self.emit(f"{var_name} = {single_name}  # single tensor product")
            return var_name

        self.emit(f"# Product of {len(t.tensors)} tensors")

        edge_numbers = {e: i for i, e in enumerate({e for child in t.tensors for e in child.edges})}

        parts = []
        for inner in t.tensors:
            torch_var = self._emit_tensor(inner)
            parts.append(f"{torch_var}")
            parts.append(str([edge_numbers[e] for e in inner.edges]))

        parts.append(str([edge_numbers[e] for e in t.edges]))
        self.emit(f"{var_name} = torch.einsum({', '.join(parts)})")

        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Function) -> str:
        signature = t.signature
        name = "".join(c for c in signature.name if c.isalnum())
        var_name = self.fresh_name(f"fn_{name}")
        child_names = [self._emit_tensor(inp) for inp in t.inputs]

        if isinstance(signature, F._PowerFunction):
            self.emit(f"{var_name} = torch.pow({child_names[0]}, {signature.k})")
        elif signature.name in ("exp", "log", "relu"):
            self.emit(f"{var_name} = torch.{signature.name}({child_names[0]})")
        elif signature.name == "gt0":
            self.emit(f"{var_name} = ({child_names[0]} >= 0).float()")
        elif signature.name == "argmax":
            dim = list(t.inputs[0].edges).index(signature.dim)
            self.emit(f"{var_name} = {child_names[0]}.argmax(dim={dim})")
            assert not t.shape_out, "argmax should have no output dims"
        else:
            raise NotImplementedError(f"Don't know how to emit code for {t.signature}")

        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Rename) -> str:
        var_name = self._emit_tensor(t.tensor)
        edges = list(t.edges)
        inverse_mapping = {e: o for o, e in t.mapping}
        perm = [edges.index(inverse_mapping.get(e, e)) for e in edges]
        if perm != list(range(t.order)):
            self.emit(f"{var_name} = {var_name}.permute({', '.join(map(str, perm))})")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Derivative) -> str:
        raise NotImplementedError(f"Derivative not implemented in codegen, please simplify first. {t=}")


def compile_to_callable(*tensors: Tensor, verbose=False, torch_compile=True):
    """
    Build a Python callable:  f(values, *symbol_args) -> torch.Tensor
      where 'symbol_args' are the integer dimension values for each distinct Sympy.Symbol
      in sorted order by symbol.name, and 'values' is a dict {Variable: torch.Tensor}.

    The generated function `_generated_forward(...)` will have arguments like:
        def _generated_forward(batch, c0, w0, h0, _var_1234, _var_5678, ...):
            ...
            return result
    """
    context = CodegenContext()
    final_var_names = [context._emit_tensor(t) for t in tensors]
    script = "\n".join(context.lines)

    function_name = "_generated_forward"

    # Sort the symbol names so we have a deterministic argument order
    all_symbols_sorted = sorted(context.seen_symbols, key=lambda s: s.name)
    # e.g. if we have sympy.Symbol("batch"), sympy.Symbol("w0"), etc.

    # Gather placeholders for each Variable
    var_placeholders = [f"_var_{id(v)}:torch.Tensor" for v in context.seen_variables]

    # The final signature is: def _generated_forward(<dims>, <vars>):
    # e.g. def _generated_forward(batch, w0, out, _var_140661515588336, _var_140661515588464, ...):
    symbol_args = [f"{s.name}:int" for s in all_symbols_sorted]
    all_args = symbol_args + var_placeholders
    signature = ", ".join(all_args)

    out_type = "torch.Tensor" if len(tensors) == 1 else "tuple[torch.Tensor]"

    # Use a raw triple-quoted string with f-string substitution, no dedent
    wrapped_code = f"""\
def {function_name}({signature}) -> {out_type}:
{textwrap.indent(script, "    ")}
    return {', '.join(final_var_names)}
"""
    if torch_compile:
        wrapped_code = "@torch.compile()\n" + wrapped_code

    if verbose:
        print()
        print("\n".join(f"{i+1:3}: {line}" for i, line in enumerate(wrapped_code.split("\n"))))

    ns = {}
    exec(wrapped_code, {"torch": torch}, ns)
    generated_forward = ns[function_name]

    def forward_with_values(values: dict[Variable, torch.Tensor], shapes: dict[sympy.Symbol, int]):
        """
        The user must pass dimension sizes in the same order as sorted(context.seen_symbols).
        For example: forward_with_values({x: x_val}, 64, 28, 10, ...)
        """
        # Check we got the right number of dimension arguments
        if not set(all_symbols_sorted).issubset(shapes.keys()):
            diff = set(all_symbols_sorted) - set(shapes.keys())
            raise ValueError(f"Missing dimension values for symbols: {diff}")

        # Add variables to namespace
        local_ns = {}
        local_ns[function_name] = generated_forward
        for sym, val in shapes.items():
            local_ns[sym.name] = val
        for var in context.seen_variables:
            placeholder_name = f"_var_{id(var)}"
            if var not in values:
                raise KeyError(f"No value provided for variable {var.name}")
            # Ensure the torch tensors follow the order of the variable edges.
            # The code will assume that this is always the case.
            # print(f"{values[var].names=}, {var.edges=} {var.orig=}")
            # local_ns[placeholder_name] = values[var].align_to(*(var.orig[e] for e in var.edges))
            local_ns[placeholder_name] = values[var].align_to(*var.edges)

        # Now call `_generated_forward(batch, w0, out, _var_..., _var_...)`
        call_args = [s.name for s in all_symbols_sorted] + [
            f"_var_{id(v)}.rename(None)" for v in context.seen_variables
        ]
        call_expr = f"{function_name}({', '.join(call_args)})"
        outputs = eval(call_expr, {}, local_ns)

        # If there's only one output, we pack it in a tuple
        if len(tensors) == 1:
            outputs = (outputs,)

        return {tensor: output.refine_names(*tensor.edges) for tensor, output in zip(tensors, outputs)}

    return forward_with_values
