from collections import defaultdict
from functools import singledispatchmethod
import textwrap
import torch
import sympy
from string import ascii_letters

import opt_einsum as oe
from tensorgrad.extras._convolution import conv_einsum_dispatch


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
    symbol_args = [f"{context.declare_dimension(s)}:int" for s in all_symbols_sorted]
    all_args = symbol_args + var_placeholders
    signature = ", ".join(all_args)

    out_type = "torch.Tensor" if len(tensors) == 1 else "tuple[torch.Tensor]"

    # Use a raw triple-quoted string with f-string substitution, no dedent
    wrapped_code = f"""\
@torch.no_grad()
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
    exec(
        wrapped_code,
        {
            "torch": torch,
            "oe": oe,
            "conv_einsum_dispatch": conv_einsum_dispatch,
            # "sparse": sparse, "ctg": ctg,
        },
        ns,
    )
    generated_forward = ns[function_name]

    # Keep track of how many times the inner function is called
    n_calls = 0

    def forward_with_values(values: dict[Variable, torch.Tensor], shapes: dict[sympy.Symbol, int] = None):
        """
        The user must pass dimension sizes in the same order as sorted(context.seen_symbols).
        For example: forward_with_values({x: x_val}, 64, 28, 10, ...)
        """
        # Load sizes of variables into the dimensions dictionary and check consistency
        # This is like the logic in tensor.evaluate(...)
        if shapes is None:
            shapes = {}
        for v, t in values.items():
            if not isinstance(v, Variable):
                continue
            for e, ts in zip(t.names, t.shape):
                vs = v.shape[e]  # symbolic dimension
                if vs not in shapes:
                    shapes[vs] = ts
                elif shapes[vs] != ts:
                    raise ValueError(f"Conflicting size for dim {e}")

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
            local_ns[placeholder_name] = values[var].align_to(*var.edges).rename(None)

        # Now call `_generated_forward(batch, w0, out, _var_..., _var_...)`
        call_args = [s.name for s in all_symbols_sorted] + [f"_var_{id(v)}" for v in context.seen_variables]
        call_expr = f"{function_name}({', '.join(call_args)})"

        nonlocal n_calls
        if verbose and n_calls == 0:
            print(f"{call_expr}")
        n_calls += 1

        outputs = eval(call_expr, {}, local_ns)

        # If there's only one output, we pack it in a tuple
        if len(tensors) == 1:
            outputs = (outputs,)

        # If people call compile(...) with just one tensor, they also assume to get back just one tensor
        res = []
        for tensor, output in zip(tensors, outputs):
            named_output = output.refine_names(*tensor.edges)
            # If this output has the same shape as an input variable but different edge order,
            # align it to match the variable (common for gradients after simplification)
            for var in context.seen_variables:
                if set(tensor.edges) == set(var.edges) and list(tensor.edges) != list(var.edges):
                    named_output = named_output.align_to(*var.edges)
                    break
            res.append(named_output)
        if len(res) == 1:
            return res[0]
        return tuple(res)

    return forward_with_values


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
        self.seen_symbols = {}
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
            if symbol not in self.seen_symbols:
                # Make sure the symbol name doesn't clash with e.g. an loop iterator (i)
                name = self.fresh_name(symbol.name)
                self.seen_symbols[symbol] = name
            return self.seen_symbols[symbol]
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
        self.emit(f"{var_name} = {placeholder_name}  # ({', '.join(tensor.edges)})")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Zero) -> str:
        var_name = self.fresh_name("zero_")
        shape_dims = [self.declare_dimension(t.shape[e]) for e in t.edges]
        self.emit(f"{var_name} = torch.zeros([{','.join(shape_dims)}])  # {', '.join(t.edges)}")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Delta) -> str:
        var_name = self.fresh_name("delta_")
        edges = list(t.edges)
        order = len(edges)
        size = self.declare_dimension(t._size)

        if order == 0:
            size_str = self.declare_dimension(t.size)
            self.emit(f"{var_name} = torch.tensor({size_str}, dtype=torch.float32)")
        elif order == 1:
            self.emit(f"{var_name} = torch.ones([{size}], dtype=torch.float32)  # {', '.join(edges)}")
        elif order == 2:
            self.emit(f"{var_name} = torch.eye({size}, dtype=torch.float32)  # {', '.join(edges)}")
        else:
            shape_str = ", ".join([size] * order)
            method = "dense"
            if method == "dense":
                self.emit(
                    f"{var_name} = torch.zeros([{shape_str}], dtype=torch.float32)  # {', '.join(edges)}"
                )
                loopvar = self.fresh_name("i")
                self.emit(
                    textwrap.dedent(f"""\
                    for {loopvar} in range({size}):
                        {var_name}[{','.join([loopvar]*order)}] = 1
                """)
                )
            elif method == "sparse.COO":
                self.emit(
                    textwrap.dedent(f"""
                    diagonal_indices = torch.arange({size}, dtype=torch.int64)
                    coords = torch.stack([diagonal_indices] * {order})  # Shape will be (order, size)
                    data = torch.ones({size}, dtype=torch.float32)
                    {var_name} = sparse.COO(coords.numpy(), data.numpy(), shape=({shape_str}))  # {', '.join(edges)}
                """).strip()
                )
            elif method == "sparse.CSR":
                indices_str = ", ".join(["diagonal_indices"] * order)
                self.emit(
                    textwrap.dedent(f"""
                    diagonal_indices = torch.arange({size}, dtype=torch.int64)
                    indices = torch.stack([{indices_str}])  # Shape will be (order, nnz)
                    values = torch.ones({size}, dtype=torch.float32)
                    {var_name} = torch.sparse_coo_tensor(indices, values, size=({shape_str}))  # {', '.join(edges)}
                """).strip()
                )
            elif method == "sparse.CSC":
                self.emit(
                    textwrap.dedent(f"""
                    indices = torch.arange({size}, dtype=torch.int64)
                    values = torch.ones({size}, dtype=torch.float32)
                    {var_name} = torch.sparse_csr_tensor(
                        crow_indices=torch.arange({size} + 1, dtype=torch.int64),
                        col_indices=indices,
                        values=values,
                        size=({shape_str})
                    )
                """).strip()
                )

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
        self.emit(f"{var_name} = 0  # ({', '.join(t.edges)})")

        for w, st in zip(t.weights, t.terms):
            c_name = self._emit_tensor(st)
            # We need to permute the tensor to match the parent's order
            c_name += permute(st, t)
            if w == 1:
                self.emit(f"{var_name} += {c_name}")
            elif w == -1:
                self.emit(f"{var_name} -= {c_name}")
            else:
                self.emit(f"{var_name} += {float(w)} * {c_name}")

        return var_name

    @_emit_tensor_impl.register
    def _(self, prod: Product) -> str:
        """Emit code for a Product tensor."""
        var_name = self.fresh_name("prod_")
        # Handle empty product
        if not prod.factors:
            self.emit(f"{var_name} = torch.tensor(1.0)  # empty product")
            return var_name

        # Handle single tensor case
        if len(prod.factors) == 1:
            single_name = self._emit_tensor(prod.factors[0])
            self.emit(f"{var_name} = {single_name}  # single tensor product")
            return var_name

        self.emit(f"# Product of {len(prod.factors)} tensors")

        method = "einsum_opt"
        if method == "einsum_opt":
            # Einsum opt requires single letter names
            name_to_idx = defaultdict(lambda: ascii_letters[len(name_to_idx)])

            # Compress Delta tensors
            # contracted_edges = set.union(*[set(t.edges) for t in t.tensors]) - t.edges
            factor_names, factors = [], []
            for ft in prod.factors:
                if isinstance(ft, F.Convolution):
                    factors.append(ft)
                    shape = [
                        self.declare_dimension(ft.shape[e])
                        for e in (ft.input_name, ft.kernel_name, ft.output_name)
                    ]
                    factor_names.append(f"({shape[0]}, {shape[1]}, {shape[2]})")
                elif isinstance(ft, Delta) and ft.order >= 1:
                    output_edges = ft.edges & prod.edges
                    input_edges = ft.edges - output_edges
                    # However, einsum doesn't support duplicated indices in the output, like i->ii,
                    # So we need to recreate some Delta tensors manually
                    if len(input_edges) >= 1:
                        e0 = list(input_edges)[0]
                        if len(output_edges) <= 1:
                            for e in ft.edges:
                                name_to_idx[e] = name_to_idx[e0]
                        if len(output_edges) > 1:
                            for e in input_edges:
                                name_to_idx[e] = name_to_idx[e0]
                            compressed_delta = Delta(ft._size, e0, *output_edges)
                            factors.append(compressed_delta)
                            factor_names.append(self._emit_tensor(compressed_delta))
                    else:
                        factors.append(ft)
                        factor_names.append(self._emit_tensor(ft))
                else:
                    factors.append(ft)
                    factor_names.append(self._emit_tensor(ft))

            operands = self.fresh_name("operands_")
            self.emit(f"{operands} = [{', '.join(factor_names)}]")

            equation = ",".join("".join(name_to_idx[e] for e in ft.edges) for ft in factors)

            equation += "->" + "".join(name_to_idx[e] for e in prod.edges)
            shape_strs = [", ".join(self.declare_dimension(ft.shape[e]) for e in ft.edges) for ft in factors]
            shape_str = "[" + "], [".join(shape_strs) + "]"
            path_name = self.fresh_name("path_")
            self.emit(f"{path_name} = oe.contract_expression('{equation}', {shape_str}).contraction_list")

            # TODO: Be smarter about converting Delta tensors to indices,
            # and use somethin glike the following to manually contract tensors.
            # But if the tensor is a Convolution, we can use
            # other.unfold(0, K, 1) instead.

            ids, eq_name = self.fresh_name("ids_"), self.fresh_name("eq_")
            args = self.fresh_name("args_")
            self.emit(f"for {ids}, _, {eq_name}, _, _ in {path_name}:")
            self.emit(f"    {args} = [{operands}.pop(i) for i in sorted({ids}, reverse=True)]")
            self.emit(
                f"    if len({args}) == 2 and (isinstance({args}[0], tuple) or isinstance({args}[1], tuple)):"
            )
            self.emit(f"        {operands}.append(conv_einsum_dispatch({eq_name}, {args}[0], {args}[1]))")
            self.emit(f"    else:")  # noqa: F541
            self.emit(f"        {operands}.append(torch.einsum({eq_name}, *{args}))")
            self.emit(f"{var_name} = {operands}[0]")

            return var_name

        if method == "pytorch":
            edge_numbers = {e: i for i, e in enumerate({e for child in prod.factors for e in child.edges})}

            parts = []
            for inner in prod.factors:
                torch_var = self._emit_tensor(inner)
                parts.append(
                    f"{torch_var}, {[edge_numbers[e] for e in inner.edges]},  # ({', '.join(inner.edges)})"
                )

            parts.append(f"{[edge_numbers[e] for e in prod.edges]}  # ({', '.join(prod.edges)})")
            inner = "\n    ".join(parts)  # python3.11 f-strings can't have backslash
            self.emit(f"{var_name} = torch.einsum(\n    {inner}\n    )")
        elif method == "cotengra":
            # Instead of using edge numbers, we'll use the edge names directly since cotengra supports hashable objects
            parts = []
            arrays = []
            inputs = []

            for inner in prod.factors:
                torch_var = self._emit_tensor(inner)
                arrays.append(torch_var)
                inputs.append(tuple(inner.edges))
                parts.append(f"# {torch_var}: ({', '.join(inner.edges)})")

            # Add comments showing the structure for debugging
            self.emit("\n".join(parts))

            arrays_str = ", ".join(arrays)
            inputs_str = repr(inputs)
            output_str = repr(tuple(prod.edges))

            # Use array_contract since we have arbitrary hashable indices
            self.emit(f"{var_name} = ctg.array_contract(")
            self.emit(f"    arrays=[{arrays_str}],")
            self.emit(f"    inputs={inputs_str},")
            self.emit(f"    output={output_str},")
            self.emit("    optimize='auto'")
            self.emit(")")

        return var_name

    def _emit_torch_einsum(
        self, var_names: list[str], ts: list[Tensor], edge_map: dict[str, str], eq: str
    ) -> str:
        pass
        # edge_numbers = {e: i for i, e in enumerate({e for child in t.tensors for e in child.edges})}

        # parts = []
        # for inner in t.tensors:
        #     torch_var = self._emit_tensor(inner)
        #     parts.append(
        #         f"{torch_var}, {[edge_numbers[e] for e in inner.edges]},  # ({', '.join(inner.edges)})"
        #     )

        # parts.append(f"{[edge_numbers[e] for e in t.edges]}  # ({', '.join(t.edges)})")
        # inner = "\n    ".join(parts)  # python3.11 f-strings can't have backslash
        # self.emit(f"{var_name} = torch.einsum(\n    {inner}\n    )")

    @_emit_tensor_impl.register
    def _(self, t: Function) -> str:
        signature = t.signature
        name = "".join(c for c in signature.name if c.isalnum())
        var_name = self.fresh_name(f"fn_{name}")
        child_names = [self._emit_tensor(inp) for inp in t.inputs]

        if isinstance(signature, F._PowerFunction):
            fun_expr = f"torch.pow({child_names[0]}, {signature.k}){permute(t.inputs[0], t)}"
        elif isinstance(signature, F._LogFunction):
            fun_expr = f"torch.log({child_names[0]}){permute(t.inputs[0], t)}"
        elif signature.name in ("exp", "relu", "sign", "abs"):
            fun_expr = f"torch.{signature.name}({child_names[0]}){permute(t.inputs[0], t)}"
        elif signature.name == "gt0":
            fun_expr = f"({child_names[0]} >= 0).float(){permute(t.inputs[0], t)}"
        elif signature.name == "argmax":
            edges = list(t.inputs[0].edges)
            dim = edges.index(signature.dim)
            fun_expr = f"{child_names[0]}.argmax(dim={dim}).float()"  # Convert to float for compatibility
            del edges[dim]
            perm = [edges.index(e) for e in t.edges]
            if perm != list(range(t.order)):
                fun_expr += f".permute({', '.join(map(str, perm))})"
            assert not t.shape_out, "argmax should have no output dims"
        elif signature.name == "equal":
            # Direct equality comparison - much more efficient than 1 - |sign(x - y)|
            fun_expr = f"({child_names[0]} == {child_names[1]}).float(){permute(t.inputs[0], t)}"
        elif signature.name == "softmax":
            # Softmax over specified dimensions
            edges = list(t.inputs[0].edges)
            dims_to_softmax = list(signature.inputs[0])  # The dimensions to apply softmax over
            dim_indices = [edges.index(d) for d in dims_to_softmax]
            if len(dim_indices) == 1:
                fun_expr = f"torch.nn.functional.softmax({child_names[0]}, dim={dim_indices[0]}){permute(t.inputs[0], t)}"
            else:
                # For multi-dimensional softmax, flatten, apply softmax, then reshape
                # This matches the evaluation in evaluate.py
                raise NotImplementedError("Multi-dimensional softmax not yet supported in code generation")
        else:
            raise NotImplementedError(f"Don't know how to emit code for {t.signature}")

        self.emit(f"{var_name} = {fun_expr}  # ({', '.join(t.edges)})")

        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Rename) -> str:
        var_name = self._emit_tensor(t.tensor)
        edges = list(t.tensor.edges)
        inverse_mapping = {e: o for o, e in t.mapping.items()}
        perm = [edges.index(inverse_mapping.get(e, e)) for e in edges]
        if perm != list(range(t.order)):
            self.emit(f"{var_name} = {var_name}.permute({', '.join(map(str, perm))})")
        return var_name

    @_emit_tensor_impl.register
    def _(self, t: Derivative) -> str:
        raise NotImplementedError(f"Derivative not implemented in codegen, please simplify first. {t=}")


def permute(tensor: Tensor, parent: Tensor) -> str:
    """Return a string that permutes the tensor to match the parent's order."""
    edges = list(tensor.edges)
    perm = [edges.index(e) for e in parent.edges]
    if perm == list(range(parent.order)):
        return ""
    return f".permute({', '.join(map(str, perm))})"
