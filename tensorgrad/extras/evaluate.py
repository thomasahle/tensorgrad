from collections import Counter
from functools import singledispatch, singledispatchmethod
import math
from types import SimpleNamespace
from typing import Any, Sequence, cast
import torch
from sympy import Symbol
from tensorgrad import Tensor, Variable, Derivative, Function, Product
from tensorgrad.functions import (
    _DeterminantFunction,
    _EqualFunction,
    _LogFunction,
    _PowerFunction,
    _RenameFunction,
    _ScaleFunction,
    _MatrixInverseFunction,
    _SimpleFunction,
    _ArgMaxFunction,
    _ArgSortFunction,
    _OneHotFunction,
    _MaxGradFunction,
    _MaxFunction,
    _ZeroFunction,
    Convolution,
    Reshape,
)
from tensorgrad.tensor import Delta, FunctionSignature, Rename, Sum, Zero
from tensorgrad.compiler.cells import CELLS, _FusedFunction, _FusedVJP
from tensorgrad.utils import KeyStoringDict


def evaluate(
    tensor: Tensor,
    values: dict[Variable, torch.Tensor],
    dims: dict[Symbol, int] | None = None,
) -> torch.Tensor:
    return Context(values, dims).evaluate(tensor)


class Context:
    def __init__(
        self,
        values: dict[Variable, torch.Tensor],
        dims: dict[Symbol, int] | None = None,
    ):
        self.values = KeyStoringDict(values)
        self.dims = dims if dims is not None else {}

        # Load sizes of variables into the dimensions dictionary and check consistency
        for v, t in values.items():
            if not isinstance(v, Variable):
                continue
            for e, ts in zip(t.names, t.shape):
                vs = v.shape[e]  # symbolic dimension
                if vs not in self.dims:
                    self.dims[vs] = ts
                elif self.dims[vs] != ts:
                    raise ValueError(f"Conflicting size for dim {e}")

    def evaluate(self, tensor: Tensor) -> torch.Tensor:
        # See if we have already evaluated a tensor isomorphic to this one
        if (other_tensor := self.values.get_with_key(tensor)) is not None:
            # Find the isomorphic representative that we matched
            other, torch_tensor = other_tensor
            assert torch_tensor.names == tuple(other.edges)
            mapping = next(other.isomorphisms(tensor), None)
            assert mapping is not None  # get_with_key matched, so an isomorphism exists
            res = torch_tensor.rename(**mapping).align_to(*tensor.edges)

            if __debug__:
                expected = self._evaluate(tensor)
                # We guarantee that inner_evaluate returns the edges in the same order as tensor.edges,
                # and res has had the order forced on it by align_to.
                assert expected.names == res.names, f"{expected.names=} {res.names=} {tensor.edges=}"
                torch.testing.assert_close(res.rename(None), expected.rename(None))

            return res

        res = self._evaluate(tensor)

        if __debug__:
            assert not torch.isnan(res.rename(None)).any(), f"Got NaN in result in {tensor}"
            # We guarantee that inner_evaluate returns the edges in the same order as tensor.edges
            assert res.names == tuple(tensor.edges), f"Expected {tensor.edges=} but got {res.names=}"
            assert all(
                res.size(k) == self.dims[v]  # type: ignore[call-overload]  # named dim
                for k, v in tensor.shape.items()
                if v in self.dims
            )

        # If the output defined a dimension size we didn't know about, store it
        # Note this is not a guaranteed metohd, since it creates a dependency on
        # the order of evaluation.
        for k, v in tensor.shape.items():
            if v not in self.dims:
                self.dims[v] = res.size(k)  # type: ignore[call-overload]  # named dim

        self.values[tensor] = res
        return res

    @singledispatchmethod
    def _evaluate(self, tensor: Tensor) -> torch.Tensor:
        """
        The inner implementation of tensor evaluation.

        Subclasses should override this to define the actual evaluation logic.

        Args:
            values: A dictionary mapping variable tensors to their values.
            dims: A dictionary mapping edge names to their dimensions.

        Returns:
            The result of evaluating this tensor.
        """
        raise NotImplementedError(f"Cannot evaluate {type(tensor)}")

    @_evaluate.register
    def _(self, var: Variable) -> torch.Tensor:
        # Mostly this won't be called, since the Variable will be picked up by the caching.
        # However, the __debug__ section of evaluate calls this directly, so we need to handle it.
        res = self.values.get(var)
        if res is None:
            raise ValueError(f"Missing value for {var}, got {self.values.keys()=}")
        return res

    @_evaluate.register
    def _(self, rename: Rename) -> torch.Tensor:
        res = self.evaluate(rename.tensor)
        if rename.mapping:
            # If mapping is empty, the rename would fail
            res = res.rename(**rename.mapping)
        return res

    @_evaluate.register
    def _(self, delta: Delta) -> torch.Tensor:
        size = self.dims[delta.size]
        if not delta.edges:
            # Return float to match the dtype of every other evaluation path
            # (and the compiler backend, which folds order-0 deltas into weights).
            return torch.tensor(float(size))
        copy = torch.zeros([size] * delta.order)
        for idx in range(size):
            copy[(idx,) * len(delta.edges)] = 1
        return copy.rename(*delta.edges)

    @_evaluate.register
    def _(self, zero: Zero) -> torch.Tensor:
        return torch.zeros([self.dims[s] for s in zero.shape.values()]).rename(*zero.edges)

    @_evaluate.register
    def _(self, fn: Function) -> torch.Tensor:
        xvals = [self.evaluate(t) for t in fn.inputs]
        res = evaluate_function(fn.signature, *xvals)
        # We require the signature eval to match the names, but not necessarily the order
        assert set(res.names) == fn.edges
        return res.align_to(*fn.edges)

    @_evaluate.register
    def _(self, deriv: Derivative) -> torch.Tensor:
        # We could use numerical differentiation here...  But it would potentially require quite a lot of
        # evaluations, since we need to evaluate the tensor in all directions.
        raise ValueError("Derivative tensors cannot be evaluated directly. Please use simplify() first.")

    @_evaluate.register
    def _(self, prod: Product) -> torch.Tensor:
        # TODO: Keep track of how many contractions we made
        # extras["contractions"] = extras.get("contractions", 0) + len(prod.contractions)
        if not prod.factors:
            return torch.tensor(1.0)
        # Order-0 Deltas are scalar factors equal to their size. They have no edges, so
        # the einsum/copy-merging machinery below would silently drop them. Pull them out
        # and multiply the result at the end.
        scalar = 1
        factors = []
        for t in prod.factors:
            if isinstance(t, Delta) and not t.edges:
                scalar *= self.dims[t.size]
            else:
                factors.append(t)
        if not factors:
            return torch.tensor(float(scalar))
        # We use "operator" einsum interface, which doesn't require single letter names.
        # e.g.  einsum('i,i', b, b)  =  einsum(b, [0], b, [0])
        # and   einsum('ij,jk->ik', b, c)  =  einsum(b, [0, 1], c, [1, 2], [0, 2])
        edge_numbers = {e: i for i, e in enumerate({e for t in factors for e in t.edges})}

        # Optimize by skipping Delta tensors when possible
        # We equate their edges to avoid materializing large identity matrices
        merge_copies = True
        # We can't remove copies, if that's all we have
        if all(isinstance(t, Delta) for t in factors):
            merge_copies = False

        # Check if any output edge is only provided by Deltas
        # If so, we can't skip those Deltas
        if merge_copies:
            non_delta_edges = {e for t in factors if not isinstance(t, Delta) for e in t.edges}
            for e in prod.edges:
                if e not in non_delta_edges:
                    # Output edge only provided by Deltas - can't merge
                    merge_copies = False
                    break

        # We can make this more efficient by removing Delta tensors.
        # Equate all edges of each Delta to the same index
        next_delta_idx = max(edge_numbers.values()) + 1 if edge_numbers else 0
        if merge_copies:
            for t in factors:
                if isinstance(t, Delta):
                    # Use a fresh index for this Delta's edges
                    for e in t.edges:
                        edge_numbers[e] = next_delta_idx
                    next_delta_idx += 1

        # Build output specification - may have repeated indices after Delta merging
        output_indices = [edge_numbers[e] for e in prod.edges]

        # Check if we have repeated indices in output (einsum doesn't support this)
        has_repeated_output = len(set(output_indices)) != len(output_indices)

        if has_repeated_output and merge_copies:
            # Need to deduplicate output, run einsum, then expand with diagonal
            # Track which output positions share the same index
            dedup_output: list[int] = []
            seen = {}
            for idx in output_indices:
                if idx not in seen:
                    seen[idx] = len(dedup_output)
                    dedup_output.append(idx)

            # Build parts for einsum with deduplicated output
            parts = []
            for t in factors:
                if not isinstance(t, Delta):
                    torch_tensor = self.evaluate(t)
                    parts.append(torch_tensor.rename(None))
                    parts.append([edge_numbers[e] for e in torch_tensor.names])
            parts.append(dedup_output)

            # Run einsum with deduplicated output
            result = torch.einsum(*parts)

            # Now expand result to have repeated dimensions using diagonal
            # Map from original output index -> position in deduplicated output
            expansion_spec = []
            for orig_idx in output_indices:
                expansion_spec.append(seen[orig_idx])

            # Build einsum to expand: for each repeated index, add an identity matrix
            if expansion_spec != list(range(len(dedup_output))):
                # Need to actually expand
                expand_parts = [result, list(range(len(dedup_output)))]
                next_idx = len(dedup_output)
                final_output = []
                idx_to_final = {}  # Maps dedup idx to final output idx

                for dedup_idx in expansion_spec:
                    if dedup_idx not in idx_to_final:
                        # First use of this dedup index in output - use it directly
                        idx_to_final[dedup_idx] = dedup_idx
                        final_output.append(dedup_idx)
                    else:
                        # Repeated use - add identity matrix to create diagonal
                        size = result.shape[dedup_idx]
                        identity = torch.eye(size)
                        expand_parts.extend([identity, [dedup_idx, next_idx]])
                        final_output.append(next_idx)
                        next_idx += 1

                expand_parts.append(final_output)
                result = torch.einsum(*expand_parts)

            if scalar != 1:
                result = result * scalar
            return result.rename(*prod.edges)
        else:
            # No repeated output or not merging - use standard path
            parts = []
            for t in factors:
                if not merge_copies or not isinstance(t, Delta):
                    torch_tensor = self.evaluate(t)
                    parts.append(torch_tensor.rename(None))
                    parts.append([edge_numbers[e] for e in torch_tensor.names])
            parts.append(output_indices)
            result = torch.einsum(*parts)
            if scalar != 1:
                result = result * scalar
            return result.rename(*prod.edges)

    @_evaluate.register
    def _(self, sum_: Sum) -> torch.Tensor:
        values = [self.evaluate(t).align_to(*sum_.edges) for t in sum_.terms]
        return sum(float(w) * v for w, v in zip(sum_.weights, values))

    ################################################################################
    # Function classes
    ################################################################################

    @_evaluate.register
    def _(self, conv: Convolution) -> torch.Tensor:
        w_in = self.dims.get(conv.shape[conv.input_name])
        k_size = self.dims.get(conv.shape[conv.kernel_name])
        w_out = self.dims.get(conv.shape[conv.output_name])

        # We only need 2/3 of the input sizes to be given
        if Counter([w_in, k_size, w_out])[None] >= 2:
            raise ValueError(f"Convolution expects >= 2 of {conv.shape.keys()} to be given")
        if w_in is None:
            assert w_out is not None and k_size is not None  # at most one None, see above
            w_in = w_out + k_size - 1
        elif k_size is None:
            assert w_out is not None
            k_size = w_in - w_out + 1
        elif w_out is None:
            w_out = w_in - k_size + 1
        elif w_out != w_in - k_size + 1:
            raise ValueError(f"{w_out=} != {w_in=} - {k_size=} + 1")

        # Make a tensor T, such that T[i,j,k] = 1 iff i=j+k
        res = torch.zeros(w_in, k_size, w_out)
        for k in range(w_out):
            for j in range(k_size):
                res[k + j, j, k] = 1
        return res.rename(conv.input_name, conv.kernel_name, conv.output_name)

    @_evaluate.register
    def _(self, reshape: Reshape) -> torch.Tensor:
        if not set(reshape.shape.values()).issubset(self.dims.keys()):
            diff = reshape.shape.values() - self.dims.keys()
            raise ValueError(f"Dims {diff} not supplied to Reshape")
        sizes = [self.dims[s] for s in reshape.shape.values()]
        full = math.prod(sizes)
        half = int(math.sqrt(full))
        if half**2 != full:
            raise ValueError(f"{sizes=} must multiply to a square number")
        return torch.eye(half).reshape(*sizes).rename(*reshape.edges)


@singledispatch
def evaluate_function(func: FunctionSignature, *xs: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(f"Cannot evaluate {func}")


@evaluate_function.register
def _(func: _ScaleFunction, *xs: torch.Tensor) -> torch.Tensor:
    # (cast: alpha is a plain numeric scalar; numbers.Number has no static operators)
    return cast(float, func.alpha) * evaluate_function(func.inner, *xs)


def _(func: _DeterminantFunction, x: torch.Tensor) -> torch.Tensor:
    (dims,) = func.inputs
    new_names = [n for n in x.names if n not in dims]  # Names after the determinant
    return torch.linalg.det(x.rename(None)).rename(*new_names)


@evaluate_function.register
def _(func: _MatrixInverseFunction, x: torch.Tensor) -> torch.Tensor:
    if not set(func.edges).issubset(x.names):
        raise ValueError(f"Input {x.names} didn't have all edges {func.edges}")
    d1, d2 = func.edges
    # torch.inverse assumes matrix dimensions are at the end.
    # We swap d1 and d2 for z1, so it's the edges with
    # the same name that cancel, and not the opposite name.
    z = x.align_to(..., d2, d1)
    z = x.align_to(..., d1, d2).rename(None)
    out_names = x.align_to(..., d2, d1).names
    return torch.inverse(z).rename(*out_names).align_to(*x.names)


@evaluate_function.register
def _(func: _ArgMaxFunction, x: torch.Tensor) -> torch.Tensor:
    i = x.names.index(func.dim)
    names = list(x.names)
    names.pop(i)
    return torch.argmax(x.rename(None), dim=i).rename(*names)


def _sdpa_forward(
    func: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """The definition: softmax(scale*<q,k>_hs + mask) @_key v, as plain torch
    over named tensors. Batch = every shared edge other than seq/key/hs."""
    batch = sorted(func.batch)
    q = q.align_to(*batch, func.seq, func.hs).rename(None)
    k = k.align_to(*batch, func.key, func.hs).rename(None)
    v = v.align_to(*batch, func.key, func.hs).rename(None)
    bshape = q.shape[:-2]
    S, K, E = q.shape[-2], k.shape[-2], q.shape[-1]
    qb, kb, vb = q.reshape(-1, S, E), k.reshape(-1, K, E), v.reshape(-1, K, E)
    scores = func.scale * torch.bmm(qb, kb.transpose(-2, -1))  # (B, S, K)
    if mask is not None:
        m = mask.align_to(func.seq, func.key).rename(None)
        scores = scores + m
    att = torch.softmax(scores, dim=-1)
    out = torch.bmm(att, vb).reshape(*bshape, S, E)
    return out.rename(*batch, func.seq, func.hs)


@evaluate_function.register
def _(func: _FusedFunction, *xs: torch.Tensor) -> torch.Tensor:
    return CELLS[func.cell_name].eval_fwd(func.params, xs, getattr(func, "out_idx", 0))


@evaluate_function.register
def _(func: _FusedVJP, *xs: torch.Tensor) -> torch.Tensor:
    # xs = the cell's VJP inputs (originals + cotangent u); the cell knows u's
    # position (e.g. before an attention mask), so hand it the whole list.
    return CELLS[func.cell_name].eval_bwd(func.params, func.which, xs)


# The fused-cell oracles (tensorgrad/compiler/cells.py dispatches here through
# the generic _FusedFunction / _FusedVJP registrations above). `params` is the
# cell's argument dict; a SimpleNamespace lets the reference forwards read it
# as `func.batch` / `func.seq` / ... unchanged.


def _eval_sdpa_fwd(params: dict[str, Any], inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    f = SimpleNamespace(**params)
    q, k, v = inputs[0], inputs[1], inputs[2]
    mask = inputs[3] if f.has_mask else None
    return _sdpa_forward(f, q, k, v, mask)


def _eval_sdpa_bwd(params: dict[str, Any], which: int, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """The reverse VJP d(input `which`) given cotangent u (inputs[3]), via
    autograd on the reference attention. The oracle for the fused backward."""
    fwd = SimpleNamespace(**params)
    q, k, v, u = inputs[0], inputs[1], inputs[2], inputs[3]
    mask = inputs[4] if fwd.has_mask else None
    raws = [t.rename(None).detach().clone().requires_grad_(True) for t in (q, k, v)]
    named = [raws[j].rename(*inputs[j].names) for j in range(3)]
    with torch.enable_grad():
        out = _sdpa_forward(fwd, named[0], named[1], named[2], mask)
        u_al = u.align_to(*out.names).rename(None)
        (g,) = torch.autograd.grad(out.rename(None), raws[which], grad_outputs=u_al)
    return g.rename(*inputs[which].names)


def _layer_norm_forward(
    func: Any, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """The definition: weight*(x-mean)/sqrt(var+eps)+bias over the `dim` axis
    (biased variance), as plain torch over named tensors. An INDEPENDENT
    oracle (not torch's fused native_layer_norm kernel). Batch = every edge
    of x other than dim."""
    batch = sorted(func.batch)
    xv = x.align_to(*batch, func.dim).rename(None)
    w = weight.align_to(func.dim).rename(None)
    b = bias.align_to(func.dim).rename(None)
    mean = xv.mean(dim=-1, keepdim=True)
    var = xv.var(dim=-1, unbiased=False, keepdim=True)
    normed = (xv - mean) / torch.sqrt(var + func.eps)
    out = normed * w + b
    return out.rename(*batch, func.dim)


def _eval_layer_norm_fwd(params: dict[str, Any], inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    return _layer_norm_forward(SimpleNamespace(**params), inputs[0], inputs[1], inputs[2])


def _eval_layer_norm_bwd(params: dict[str, Any], which: int, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    fwd = SimpleNamespace(**params)
    x, weight, bias, u = inputs[0], inputs[1], inputs[2], inputs[3]
    raws = [t.rename(None).detach().clone().requires_grad_(True) for t in (x, weight, bias)]
    named = [raws[j].rename(*inputs[j].names) for j in range(3)]
    with torch.enable_grad():
        out = _layer_norm_forward(fwd, named[0], named[1], named[2])
        u_al = u.align_to(*out.names).rename(None)
        (g,) = torch.autograd.grad(out.rename(None), raws[which], grad_outputs=u_al)
    return g.rename(*inputs[which].names)


@evaluate_function.register
def _(func: _ArgSortFunction, x: torch.Tensor) -> torch.Tensor:
    i = x.names.index(func.dim)
    names = x.names
    return torch.argsort(x.rename(None), dim=i).to(x.dtype).rename(*names)


@evaluate_function.register
def _(func: _MaxGradFunction, x: torch.Tensor) -> torch.Tensor:
    (dims,) = func.inputs
    adim = [x.names.index(e) for e in dims]
    x, names = x.rename(None), x.names
    max_vals = x.amax(dim=adim, keepdim=True)
    mask = (x == max_vals).float()
    res = mask / mask.sum(dim=adim, keepdim=True).clamp(min=1.0)
    return res.rename(*names)


@evaluate_function.register
def _(func: _MaxFunction, x: torch.Tensor) -> torch.Tensor:
    (dims,) = func.inputs
    return torch.amax(
        x.rename(None),
        dim=[x.names.index(e) for e in dims],
        keepdim=False,
    ).rename(*(n for n in x.names if n not in dims))


@evaluate_function.register
def _(func: _PowerFunction, x: torch.Tensor) -> torch.Tensor:
    if func.k < 0 and not x.is_floating_point():
        # Integer tensors cannot take negative powers; float tensors keep
        # their dtype (hardcoding float32 here silently downcast float64
        # evaluations, poisoning mixed-dtype einsums downstream).
        x = x.to(torch.get_default_dtype())
    return torch.pow(x, float(func.k))


@evaluate_function.register
def _(func: _LogFunction, x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


@evaluate_function.register
def _(func: _SimpleFunction, x: torch.Tensor) -> torch.Tensor:
    if func.name == "exp":
        return torch.exp(x)
    if func.name == "sign":
        return torch.sign(x)
    if func.name == "relu":
        return torch.relu(x)
    if func.name == "abs":
        return torch.abs(x)
    if func.name == "tanh":
        return torch.tanh(x)
    if func.name == "erf":
        return torch.erf(x)
    if func.name == "gt0":
        return torch.where(x.rename(None) > 0, 1.0, 0.0).rename(*x.names)
    raise NotImplementedError(f"Cannot evaluate simple function {func.name!r}")


@evaluate_function.register
def _(func: _EqualFunction, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Evaluate element-wise equality comparison."""
    return (x.rename(None) == y.rename(None)).float().rename(*x.names)


@evaluate_function.register
def _(func: _OneHotFunction, idx: torch.Tensor, size_carrier: torch.Tensor) -> torch.Tensor:
    """out[eq_edge, *idx_edges] = 1.0 where idx == eq_edge index. The second
    input only carries the number of classes (its single edge's size)."""
    num_classes = size_carrier.size(func.dim)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]  # named dim
    flat = idx.rename(None).long().reshape(1, -1)
    # default dtype, not float32: a hardcoded dtype poisons float64
    # evaluations downstream (mixed-dtype einsum/mm errors).
    onehot = (flat == torch.arange(num_classes).unsqueeze(1)).to(torch.get_default_dtype())
    return onehot.reshape((num_classes,) + tuple(idx.shape)).rename(func.eq_edge, *idx.names)


@evaluate_function.register
def _(func: _RenameFunction, x: torch.Tensor) -> torch.Tensor:
    # (was `func.inner.eval(x)`; FunctionSignature never had an `eval` method,
    # so this path could only raise AttributeError before)
    return evaluate_function(func.inner, x).rename(**func.renames)


@evaluate_function.register
def _(func: _ZeroFunction, x: torch.Tensor) -> torch.Tensor:
    # Like any function, we have to support broadcasted inputs, so we detect
    # which names are in x, which are not consumed in self.inputs
    broadcasted = [e for e in x.names if e not in func.inputs[0]]
    return torch.zeros(
        size=[x.size(o) for o in broadcasted + list(func.new_edges.values())],  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]  # named dim
        names=broadcasted + list(func.new_edges.keys()),
    )


@evaluate_function.register
def _(func: _DeterminantFunction, x: torch.Tensor) -> torch.Tensor:
    (dims,) = func.inputs
    new_names = [n for n in x.names if n not in dims]  # Names after the determinant
    return torch.linalg.det(x.rename(None)).rename(*new_names)


# Affine (compiler) registers its evaluate oracle HERE rather than at its own
# import time: affine importing evaluate would cycle, since evaluate imports the
# compiler. Safe now -- Context and every registration above are defined.
from tensorgrad.compiler.affine import _register_evaluate_oracle  # noqa: E402

_register_evaluate_oracle()
