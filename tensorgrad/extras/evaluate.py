from collections import Counter
from functools import singledispatch, singledispatchmethod
import math
import torch
from sympy import Symbol
from tensorgrad import Tensor, Variable, Derivative, Function, Product
from tensorgrad.functions import (
    _DeterminantFunction,
    _PowerFunction,
    _RenameFunction,
    _ScaleFunction,
    _MatrixInverseFunction,
    _SimpleFunction,
    _ArgMaxFunction,
    _MaxGradFunction,
    _MaxFunction,
    _SoftmaxFunction,
    _ZeroFunction,
    Convolution,
    Reshape,
)
from tensorgrad.tensor import Delta, FunctionSignature, Rename, Sum, Zero
from tensorgrad.utils import KeyStoringDict


def evaluate(
    tensor: Tensor,
    values: dict[Variable, torch.Tensor],
    dims: dict[Symbol, int] | None = None,
):
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
            assert all(res.size(k) == self.dims[v] for k, v in tensor.shape.items() if v in self.dims)

        # If the output defined a dimension size we didn't know about, store it
        # Note this is not a guaranteed metohd, since it creates a dependency on
        # the order of evaluation.
        for k, v in tensor.shape.items():
            if v not in self.dims:
                self.dims[v] = res.size(k)

        self.values[tensor] = res
        return res

    @singledispatchmethod
    def _evaluate(self, tensor: Tensor):
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
    def _(self, var: Variable):
        # Mostly this won't be called, since the Variable will be picked up by the caching.
        # However, the __debug__ section of evaluate calls this directly, so we need to handle it.
        res = self.values.get(var)
        if res is None:
            raise ValueError(f"Missing value for {var}, got {self.values.keys()=}")
        return res

    @_evaluate.register
    def _(self, rename: Rename):
        res = self.evaluate(rename.tensor)
        if rename.mapping:
            # If mapping is empty, the rename would fail
            res = res.rename(**rename.mapping)
        return res

    @_evaluate.register
    def _(self, delta: Delta):
        size = self.dims[delta.size]
        if not delta.edges:
            return torch.tensor(size)
        copy = torch.zeros([size] * delta.order)
        for idx in range(size):
            copy[(idx,) * len(delta.edges)] = 1
        return copy.rename(*delta.edges)

    @_evaluate.register
    def _(self, zero: Zero):
        return torch.zeros([self.dims[s] for s in zero.shape.values()]).rename(*zero.edges)

    @_evaluate.register
    def _(self, fn: Function):
        xvals = [self.evaluate(t) for t in fn.inputs]
        res = evaluate_function(fn.signature, *xvals)
        # We require the signature eval to match the names, but not necessarily the order
        assert set(res.names) == fn.edges
        return res.align_to(*fn.edges)

    @_evaluate.register
    def _(self, deriv: Derivative):
        # We could use numerical differentiation here...  But it would potentially require quite a lot of
        # evaluations, since we need to evaluate the tensor in all directions.
        raise ValueError("Derivative tensors cannot be evaluated directly. Please use simplify() first.")

    @_evaluate.register
    def _(self, prod: Product):
        # TODO: Keep track of how many contractions we made
        # extras["contractions"] = extras.get("contractions", 0) + len(prod.contractions)
        if not prod.tensors:
            return torch.tensor(1.0)
        # We use "operator" einsum interface, which doesn't require single letter names.
        # e.g.  einsum('i,i', b, b)  =  einsum(b, [0], b, [0])
        # and   einsum('ij,jk->ik', b, c)  =  einsum(b, [0, 1], c, [1, 2], [0, 2])
        edge_numbers = {e: i for i, e in enumerate({e for t in prod.tensors for e in t.edges})}

        # TODO: Merging copies is currently broken, because einsum doesn't allow us
        # to use the same index twice in the output.
        merge_copies = False
        # We can't remove copies, if that's all we have
        if all(isinstance(t, Delta) for t in prod.tensors):
            merge_copies = False
        # We can't merge when there are repeated edges in the output, due to einsum
        # limmitations
        if len(set(edge_numbers[e] for e in prod.edges)) != len(prod.edges):
            merge_copies = False

        # We can make this more efficient by removing Delta tensors.
        if merge_copies:
            for t in prod.tensors:
                if isinstance(t, Delta):
                    i0 = len(edge_numbers)
                    for i, e in enumerate(t.edges):
                        edge_numbers[e] = i0
        parts = []
        for t in prod.tensors:
            if not merge_copies or not isinstance(t, Delta):
                torch_tensor = self.evaluate(t)
                parts.append(torch_tensor.rename(None))
                parts.append([edge_numbers[e] for e in torch_tensor.names])
        parts.append([edge_numbers[e] for e in prod.edges])
        return torch.einsum(*parts).rename(*prod.edges)

    @_evaluate.register
    def _(self, sum_: Sum):
        values = [self.evaluate(t).align_to(*sum_.edges) for t in sum_.tensors]
        return sum(float(w) * v for w, v in zip(sum_.weights, values))

    ################################################################################
    # Function classes
    ################################################################################

    @_evaluate.register
    def _(self, conv: Convolution):
        w_in = self.dims.get(conv.shape[conv.input_name])
        k_size = self.dims.get(conv.shape[conv.kernel_name])
        w_out = self.dims.get(conv.shape[conv.output_name])

        # We only need 2/3 of the input sizes to be given
        if Counter([w_in, k_size, w_out])[None] >= 2:
            raise ValueError(f"Convolution expects >= 2 of {conv.shape.keys()} to be given")
        if w_in is None:
            w_in = w_out + k_size - 1
        elif k_size is None:
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
    def _(self, reshape: Reshape):
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
def _(func: _ScaleFunction, *xs: torch.Tensor):
    return func.alpha * evaluate_function(func.inner, *xs)


def _(func: _DeterminantFunction, x: torch.Tensor):
    (dims,) = func.inputs
    new_names = [n for n in x.names if n not in dims]  # Names after the determinant
    return torch.linalg.det(x.rename(None)).rename(*new_names)


@evaluate_function.register
def _(func: _MatrixInverseFunction, x: torch.Tensor) -> torch.Tensor:
    if not func.edges.issubset(x.names):
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
def _(func: _SimpleFunction, x: torch.Tensor) -> torch.Tensor:
    return func._eval_fn(x)


@evaluate_function.register
def _(func: _ArgMaxFunction, x: torch.Tensor) -> torch.Tensor:
    i = x.names.index(func.dim)
    names = list(x.names)
    names.pop(i)
    return torch.argmax(x.rename(None), dim=i).rename(*names)


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
    if func.k < 0:
        x = x.to(torch.float)
    return torch.pow(x, float(func.k))


@evaluate_function.register
def _(func: _SimpleFunction, x: torch.Tensor) -> torch.Tensor:
    if func.name == "exp":
        return torch.exp(x)
    if func.name == "log":
        return torch.log(x)
    if func.name == "sign":
        return torch.sign(x)
    if func.name == "relu":
        return torch.relu(x)
    if func.name == "abs":
        return torch.abs(x)
    if func.name == "gt0":
        return torch.where(x.rename(None) > 0, 1.0, 0.0).rename(*x.names)


@evaluate_function.register
def _(func: _SoftmaxFunction, x: torch.Tensor) -> torch.Tensor:
    (dims,) = func.inputs
    sizes = [x.size(d) for d in dims]
    names = [d for d in x.names if d not in dims]
    other_sizes = [x.size(n) for n in names]
    # Softmax doesn't support named dimensions, so we have to rename them to None.
    # We move the affected dimensions to the front, then flatten them.
    y = x.align_to(*dims, *names).rename(None).flatten(start_dim=0, end_dim=len(dims) - 1)
    return y.softmax(dim=0).reshape(sizes + other_sizes).rename(*dims, *names).align_to(*x.names)


@evaluate_function.register
def _(func: _RenameFunction, x: torch.Tensor) -> torch.Tensor:
    return func.inner.eval(x).rename(**func.renames)


@evaluate_function.register
def _(func: _ZeroFunction, x: torch.Tensor) -> torch.Tensor:
    # Like any function, we have to support broadcasted inputs, so we detect
    # which names are in x, which are not consumed in self.inputs
    broadcasted = [e for e in x.names if e not in func.inputs[0]]
    return torch.zeros(
        size=[x.size(o) for o in broadcasted + list(func.new_edges.values())],
        names=broadcasted + list(func.new_edges.keys()),
    )


@evaluate_function.register
def _(func: _DeterminantFunction, x: torch.Tensor) -> torch.Tensor:
    (dims,) = func.inputs
    new_names = [n for n in x.names if n not in dims]  # Names after the determinant
    return torch.linalg.det(x.rename(None)).rename(*new_names)
