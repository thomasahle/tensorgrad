from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import prod
from typing import Any, Callable, Iterable, Optional
from abc import ABC

import torch

# TODO:
# - Code generation (e.g. Triton)
# - Prettier printing
# - Use a real algorithm to check isomorphism, like vf2++
# - Maybe don't use isomorphic equality as the default equality for tensors, since it leads to some annoying footguns.
# - Taking the derivative with respect to multiple variables at the same time (full backprop)
#   Maybe we can still take derivatives individually, and just use the isomorphic hashing to avoid recomputions?
# X Redo the function class using FunctionInfo instead of subclasses.
# - Also support broadcasting better, since simple two input functions like Cross Entropy is currently basically not possible.
# - Introduce a function object, that's not a tensor, but which creates a tensor when called. This makes it easier to define
#   operators, such as taylor expansion, which feeds the function some specific inputs.
# More simplification rules:
# - Support for specific functions and their simplification rules (pow(-1) cancelation, etc)
# X Elementwise functions that get broadcasted, like 1/sum_x(exp(x)) can be commuted with broadcasting
# - Optional "expand" setting that expands the expression to a sum of products
# - Optional "function expand" that converts e.g. "softmax" into it's components
# Smaller things:
# - We don't need weights in Sum. We can just use a Consant([]) tensor with a weight and add it to a product.
# - Stuff from https://en.wikipedia.org/wiki/Penrose_graphical_notation
#   - Symmetrization/Antisymmetrization
#   - Matrix inverses


class Tensor(ABC):
    edges = []
    tensors = []

    @property
    def rank(self):
        return len(self.edges)

    def grad(self, x: "Variable", new_names: Optional[list[str]]) -> "Tensor":
        """Take the derivative with respect to the variable x.
        Pushes the derivative one step through the tensor.
        If you want to push it all the way through, use simplify."""
        raise NotImplementedError

    def rename(self, kwargs: dict[str, str]):
        self._check_rename(kwargs)
        """Renames *the free edges* of the tensor.
        The inner edges may get renamed if necessary to avoid clashes with the new names."""
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def simplify(self, args: dict[str, Any] = None):
        """Apply various simplification rules.
        May rename things internally, but should never change any free edges.
        """
        return self

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor],
        *,
        dims: dict[str, int] | None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        raise NotImplementedError

    def edge_equivalences(self) -> list[tuple[tuple["Tensor", str], tuple["Tensor", str]]]:
        """Return a list of tuples of equivalent edges."""
        return []

    def __add__(self, other):
        return Sum([self, other])

    def __sub__(self, other):
        return Sum([self, other], [1, -1])

    def __neg__(self):
        return Sum([self], [-1])

    def __matmul__(self, other):
        return Product([self, other])

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        """Contract self and other, but use a 3d-identity to keep the shared edges free."""
        if isinstance(other, int) or isinstance(other, float):
            return Sum([self], [other])
        # Element-wise (Hadamard) product is easy to implement using Copy tensors
        # These are the edges we multiply over
        shared_edges = set(self.edges) & set(other.edges)
        (t0, t1), (rename0, rename1) = make_distinct(self, other, used_names=shared_edges)
        return Product([t0, t1] + [Copy([e, rename0[e], rename1[e]]) for e in shared_edges])

    def __truediv__(self, other):
        # Avoid circular import
        from tensorgrad.functions import pow

        return self * pow(other, -1)

    def __eq__(self, other):
        # Note: we use approximate isomorphic equality, using Weisfeiler Leman hash
        # TODO: Use a real isomorphism test
        # FIXME: Sometimes we need tensors to be the same, even with different edge values, such
        # as when we are trying to reuse values already computed.
        # Other times we want them to be different if the external edges are different, since it
        # is important to the further operations that are going to happen on these edges.
        # There's more: It is not even enough that the edges are the same on the surface,
        # it also matters on where they attach further inside the tensor.
        return hash(self) == hash(other) and self.edges == other.edges

    def is_isomorphic(self, other):
        # TODO: Also return a mapping/rename of the outer edges that can be used to transform an
        # torch tensor computed with one naming scheme into the other.
        return hash(self) == hash(other)

    def _check_evaluate(
        self,
        values: dict["Variable", torch.tensor],
        dims: dict[str, int] | None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        if extras is None:
            extras = {}
        if "edge_dims" not in extras:
            extras["edge_dims"] = {}
        if id(self) not in extras["edge_dims"]:
            shapes = {v: dict(zip(t.names, t.shape)) for v, t in values.items() if isinstance(v, Variable)}
            extras["edge_dims"] |= compute_edge_dims(self, shapes, extra_dims=dims)
        return extras

    def _check_rename(self, kwargs: dict[str, str]):
        """Check that the renaming is valid, and return the renaming dictionary."""
        if len({kwargs.get(e, e) for e in self.edges}) != len(self.edges):
            raise ValueError(f"Renamed an edge to an existing edge name. {self.edges=} {kwargs=}")
        # Restrict renaming to free edges
        return {new: old for new, old in kwargs.items() if new in self.edges}

    def _check_grad(self, x: "Variable", new_names: Optional[list[str]] = None) -> list[str]:
        """When taking a derivative with respect to a variable, we need some edge names for the derivative.
        If new_names is not given, we generate names based on the edge names of x.
        However, we need to make sure they don't clash with any names already present in the tensor.
        If new_names is already given, we just check to make sure they are not already present in the tensor.
        """
        # It shouldn't be necessary for the edges to be unused in the entire tensor, just in the free edges.
        if new_names is not None:
            assert len(x.edges) == len(new_names)
            if used := set(new_names) & set(self.edges):
                raise ValueError(f"{used} are already present in tensor, {self}")
            return new_names
        new_edges, _rename = unused_edge_names(x.edges, self.edges, suffix="_")
        return new_edges

    def _check_simplify(self, args: dict[str, Any] | None = None):
        if args is None:
            args = {}
        # args["grad_steps"] allows us to control how far we propagate the derivative.
        args.setdefault("grad_steps", float("inf"))
        args.setdefault("sum_combine_terms", True)
        return args


################################################################################
# Variable
################################################################################


class Variable(Tensor):
    def __init__(self, name, edges: Iterable[str], surface_edges=None):
        self.name = name
        # The original edges are saved so evaluation can happen with the original
        # edge names given to the variable, without caring about what renaming we
        # might have done in the meantime.
        self.original_edges = list(edges)
        self.edges = self.original_edges if surface_edges is None else surface_edges

    def grad(self, x: "Variable", new_names: Optional[list[str]] = None):
        new_names = self._check_grad(x, new_names)
        # We compare using name here, rather than id (x is self), since we might have
        # changed due to renaming.
        if x.name == self.name:
            # Note: This is the product of n identity matrices, which is exactly the identity tensor.
            return Product(Copy([e, new], link=self) for e, new in zip(self.edges, new_names))
        return Zero(self.edges + new_names)

    def __repr__(self):
        return f"Variable('{self.name}', {self.original_edges}, {self.edges})"

    def __hash__(self):
        return hash(("Variable", self.name, len(self.edges)))

    def __eq__(self, other):
        # FIXME: Sometimes we need variables to be the same, even with different names, such
        # as when using the value of a variable that has been given to us by the user.
        # But other times we can't just declare them equal, and e.g. join them in a sum,
        # such as if the different names are used to indicate an outer product.
        return self.name == other.name and len(self.edges) == len(other.edges)

    def simplify(self, args: dict[str, Any] = None):
        return self

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)  # Checks only free edges are in kwargs
        return Variable(
            self.name,
            self.original_edges,
            [kwargs.get(e, e) for e in self.edges],
        )

    def edge_equivalences(self):
        # Since the user gives edge dimensions in terms of variables, it's important to keep track
        # of renamed edge names.
        for e1, e2 in zip(self.original_edges, self.edges):
            yield (self, e1), (self, e2)

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor | Callable],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        extras = self._check_evaluate(values, dims, extras)
        if self not in values:
            raise ValueError(f"Missing value for {self}, got {values}")
        # For now we just assume positional consistency
        if values[self].names != tuple(self.original_edges):
            raise ValueError(f"Edge names {values[self].names} don't match original names of {self}")
        return values[self].rename(*self.edges)


################################################################################
# Constants
################################################################################


class Constant(Tensor, ABC):
    def __init__(self, edges: Iterable[str], link: Variable | None = None):
        """A constant tensor with the given edges.
        The link is a variable that this tensor is associated with, and will be used to compute edge dimensions"""
        super().__init__()
        self.edges = list(edges)
        self.link = link

    def __repr__(self) -> str:
        extra = ""
        if self.link is not None:
            extra = f", link={self.link}"
        return f"{type(self).__name__}({self.edges}{extra})"

    def __hash__(self):
        return hash((type(self).__name__, len(self.edges)))

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        link = self.link.rename(kwargs) if self.link is not None else None
        c = type(self)([kwargs.get(e, e) for e in self.edges], link=link)
        assert set(c.edges) == {kwargs.get(e, e) for e in self.edges}
        return c

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = self._check_grad(x, new_names)
        return Zero(self.edges + new_names)

    def edge_equivalences(self) -> list[tuple[tuple["Tensor", str], tuple["Tensor", str]]]:
        if self.link is not None:
            for e in self.link.edges:
                if e in self.edges:
                    yield (self, e), (self.link, e)

    def _check_evaluate(self, values, dims=None, extras=None):
        extras = super()._check_evaluate(values, dims, extras)
        for e in self.edges:
            if e not in extras["edge_dims"][id(self)]:
                raise ValueError(f"Missing edge dimension for {e}.")
        return extras


class Copy(Constant):
    """The "Copy" tensor is defined by C_{i,j,k} = 1 if i == j == k, else 0
    Or alternatively as Δₙ = ∑ₓ (eₓ)⊗ⁿ, where are e_i are basis vectors
    For rank 2, this is the identity matrix. For higher rank, the identity tensor is
    the product of n identity matrices (rank 2 Copy's).
    """

    def edge_equivalences(self):
        yield from super().edge_equivalences()
        for e in self.edges[1:]:
            yield (self, self.edges[0]), (self, e)

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        if not self.edges:
            return torch.tensor(1.0)
        if len(self.edges) > 1:
            print(f"Warning: Evaluating rank {len(self.edges)} Copy tensor. This is probably a mistake.")
        extras = self._check_evaluate(values, dims, extras)
        edge_dims = extras["edge_dims"][id(self)]
        shape = [edge_dims[e] for e in self.edges if e in edge_dims]
        assert len(shape) == len(self.edges) and len(set(shape)) == 1
        copy = torch.zeros(shape)
        for idx in range(shape[0]):
            copy[(idx,) * len(self.edges)] = 1
        values[self] = copy.rename(*self.edges)
        return values[self]


class Zero(Constant):
    """Matrix such that Z_{i,j,k} = 0 for all i, j, k"""

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        if not self.edges:
            return torch.tensor(0.0)
        extras = self._check_evaluate(values, dims, extras)
        values[self] = torch.zeros([extras["edge_dims"][id(self)][e] for e in self.edges]).rename(*self.edges)
        return values[self]


def Ones(edges: list[str], link=None) -> Tensor:
    """Matrix such that O_{i,j,k} = 1 for all i, j, k"""
    return Product([Copy([e], link=link) for e in edges])


################################################################################
# Function
################################################################################


@dataclass
class FunctionInfo:
    name: str
    eval: Callable[[list[torch.tensor]], torch.tensor] = None
    derivative: Callable[[int, list[str]], Tensor] = None
    # Could add simplification rules here


class Function(Tensor):
    # A function that takes one or more tensors as input and produces a new tensor as output.
    # Examples:

    # # Matrix multiplication:
    # Say x and y are matrices with shapes (b, i, j) and (b, j, k).
    # We might define f = Function("matmul", [], (x, "j"), (y, "j"))
    # This function takes in the j-edge from each matrix, and doesn't produce any new edges.
    # The shared edge b is super imposed...
    # All in all we end up with a tensor of shape (b, i, k) as we would expect.

    # # Unfold
    # Say we have a tensor x with shape (b, c, w, h) and we want to unfold it to do convolution.
    # We might define f = Function("unfold", ["patch", "i"], (x, "w", "h"))
    # This function takes in the w and h edges from x, and produces a number of patches, each with dimension i
    # The new tensor will have shape (b, c, patches, i)
    # One will typically permute to (b, patches, c, i) and flatten the last two dimensions to (b, patches, j)
    # Then one can apply a linear function to get (b, patches, k) before finally reshaping to (b, c, patches, l)
    # and unfolding to (b, c, w, h) again.

    # # Cross entropy
    # This is an example of a function that takes two inputs.
    # f = Function("ce", ["ce"], (x, "logits"), (y, "probs")])
    # as before, x and y will typically have some shared edges, like (b, logits) and (b, prob).
    # So the output will have shape (b, ce).

    # # Multi-query Dot Product Attention
    # Suppose we have tensors: q with shape (b, n_q, d), k with shape (b, seq, d), and v with shape (b, seq, d_out)
    # first we form (b, n_q, seq) then we apply softmax to get (b, n_q, prob) then multiply by v to get (b, n_q, d_out)
    # We can define a function f = Function("attention", ["d_out"], (q, "d"), (k, "seq", "d"), (v, "seq", "d_out")).

    # Of course, many of these functions could more easily be written as a combination of simpler functions.

    def __init__(
        self,
        fn_info: FunctionInfo | str,
        edges_out: Iterable[str],
        *inputs: Iterable[tuple[Tensor | str]],
        orig_edges_out: list[str] = None,
        orig_edges_in: list[tuple[str]] = None,
        orig_edges_ts: list[tuple[str]] = None,
    ):
        self.edges_out = list(edges_out)
        self.inputs = list(inputs)
        self.tensors = [t for t, *_ in self.inputs]

        # We need to keep track of the original edges of the function, since we might rename them.
        self.orig_edges_out = self.edges_out if orig_edges_out is None else list(orig_edges_out)
        assert len(self.edges_out) == len(self.orig_edges_out)
        self.orig_edges_in = [es for _, *es in inputs] if orig_edges_in is None else list(orig_edges_in)
        self.orig_edges_ts = [t.edges for t, *_ in inputs] if orig_edges_ts is None else list(orig_edges_ts)
        assert len(self.inputs) == len(self.orig_edges_ts) == len(self.orig_edges_in)
        self.orig_edges = self.orig_edges_out + [
            e for (ets, ein) in zip(self.orig_edges_ts, self.orig_edges_in) for e in ets if e not in ein
        ]

        # The actual edges of the function is the "output" edges,
        # plus any edges from the inputs that are not "input edges"
        self.edges = edges_out[:]
        shared_edges = set()
        for t, *input_edges in self.inputs:
            # Check input edges are present in tensor
            for e in input_edges:
                if e not in t.edges:
                    raise ValueError(f"Edge {e} is not present in input tensor {t}")
            # Check broadcasted edges are not already present in the function
            for e in [e for e in t.edges if e not in input_edges]:
                if e in edges_out:
                    raise ValueError(f"Edge {e} is both in edge_out and a broadcasted edge of an input.")
                if e in shared_edges:
                    raise ValueError(
                        f"Edge {e} is already used by another input tensor. Please rename."
                        + " If you need broadcasting, use a copy tensor."
                    )
                shared_edges.add(e)
                self.edges.append(e)

        if isinstance(fn_info, str):
            self.fn_info = FunctionInfo(
                fn_info,
                NotImplementedError,
                (
                    lambda i, new_edges, *ts: Function(
                        f"D_{i}{fn_info}",
                        # It is commonly assumed that FunctionInfo doesn't know about renaming that has
                        # happened to a function since it was created. In this particular case of the "dummy fi"
                        # we actually do know the renaming, but we will still stick with the original names for consistency.
                        self.orig_edges_out + new_edges,
                        *[(t, *oes) for t, oes in zip(ts, self.orig_edges_in)],
                    )
                ),
            )
        else:
            self.fn_info = fn_info

    #################
    # The following methods need to be implemented by subclasses

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        renamed_inputs = []
        for t, *inner_edges in self.inputs:
            rename = {e: kwargs.get(e, e) for e in self.edges if e not in inner_edges}
            renamed_inputs.append((t.rename(rename), *inner_edges))
        res = Function(
            self.fn_info,
            [kwargs.get(e, e) for e in self.edges_out],
            *renamed_inputs,
            orig_edges_out=self.orig_edges_out,
            orig_edges_in=self.orig_edges_in,
            orig_edges_ts=self.orig_edges_ts,
        )
        assert set(res.edges) == {kwargs.get(e, e) for e in self.edges}
        return res

    def simplify(self, args: dict[str, Any] = None):
        args = self._check_simplify(args)
        new_inputs = [(t.simplify(args=args), *es) for t, *es in self.inputs]

        # Broadcasting can be pulled out of the function.
        pulled_out = []
        new_inputs2 = []
        orig_edges_ts = []
        for (t, *es), orig_edges in zip(new_inputs, self.orig_edges_ts):
            if isinstance(t, Product):
                new_prod = []
                for u in t.tensors:
                    if isinstance(u, Copy) and u.rank == 1 and u.edges[0] in t.edges and u.edges[0] not in es:
                        pulled_out.append(u)
                    else:
                        new_prod.append(u)
                new_inputs2.append((Product(new_prod), *es))
            else:
                new_inputs2.append((t, *es))
            # We have to remove the pulled out edges from original_edges_ts
            t_new, *_ = new_inputs2[-1]
            rename = dict(zip(t.edges, orig_edges))
            orig_edges_ts.append([rename[e] for e in t_new.edges])
        new_inputs = new_inputs2

        # - If the function is multiplicative, we can factor any tensors with no input_edges.
        # But the only relevant multiplicative function is probably pow(), so maybe it can just
        # define it's own simplification rule...

        res = Function(
            self.fn_info,
            self.edges_out,
            *new_inputs,
            orig_edges_out=self.orig_edges_out,
            orig_edges_in=self.orig_edges_in,
            orig_edges_ts=orig_edges_ts,
        )
        if pulled_out:
            res = Product([res] + pulled_out)
        assert set(res.edges) == set(self.edges), "Free edges should be preserved"
        return res

    def edge_equivalences(self):
        # We can't really say anything about the edges of the function itself (self.edges_out),
        # but at least we can say something about the broadcasted edges.
        for t, *inner_edges in self.inputs:
            yield from t.edge_equivalences()
            for e in t.edges:
                if e not in inner_edges:
                    yield (t, e), (self, e)
        # We could maybe also say that input edges with the same name are equivalent?

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = self._check_grad(x, new_names)
        # We sum over each input function, just like the normal chain rule:
        # d/dx f(g₁(x), …, gₖ(x)) = Σᵢ₌₁ᵏ (d/dx gᵢ(x)) Dᵢf(g₁(x), …, gₖ(x))

        # D_i adds a new output edge to the function, which is contracted with
        # the normal output edge of the tensor. So we need to make sure this doesn't
        # clash with an existing output edge of f.
        parts = []
        for i, (t, *input_edges) in enumerate(self.inputs):
            # We need "connection" edges for each edge in input_edges. Mostly we could just use the same name
            # but they need to avoid clashing with "new_names" and the output edges of the tensor.

            avoid = set(new_names) | set(self.edges) | set(self.orig_edges)
            connection_edges, rename = unused_edge_names(input_edges, avoid)

            # Take the derivative of the function and the inner function (chain rule)
            part_a = self.fn_info.derivative(i, connection_edges, *[t for t, *_ in self.inputs])
            # We need to rename the output to the current names used by the function, which may have changed since
            # the fn_info.derivative was created.
            part_a = part_a.rename(dict(zip(self.orig_edges, self.edges)))
            assert set(part_a.edges) == set(self.edges + connection_edges)
            part_b = Derivative(t.rename(rename), x, new_names)

            # The "external edges" are going to be joined with copy tensors, resulting in joint broadcasting.
            # So we need to rename them in both the function and the inner. Like we do in __mul__.
            avoid |= set(connection_edges)
            broadcasted_edges = [e for e in t.edges if e not in input_edges]
            assert set(broadcasted_edges) == set(t.rename(rename).edges) - set(connection_edges)
            # We could nearly use `make_distinct` here, but we don't want to make the connection_edges distinct,
            # so we just do it manually.
            ext_a, rename_a = unused_edge_names(broadcasted_edges, avoid, suffix="_0")
            avoid |= set(ext_a)
            ext_b, rename_b = unused_edge_names(broadcasted_edges, avoid, suffix="_1")
            joins = [Copy([e, rename_a[e], rename_b[e]]) for e in broadcasted_edges]
            parts.append(Product([part_a.rename(rename_a), part_b.rename(rename_b)] + joins))

        res = Sum(parts)
        assert set(res.edges) == set(self.edges + new_names), f"{res.edges} != {self.edges + new_names}"
        return res

    def __hash__(self):
        return hash(
            ("Function", self.fn_info.name, len(self.edges_out))
            # In contrast to Sum and Product, for Functions the order _does_ matter
            + tuple(map(hash, [t for t, *es in self.inputs]))
            + tuple(map(len, self.inputs))
        )

    def __repr__(self):
        extras = []
        if self.edges_out != self.orig_edges_out:
            extras.append(f"orig_edges_out={self.orig_edges_out}")
        if self.orig_edges_in != [es for _, *es in self.inputs]:
            extras.append(f"orig_edges_in={self.orig_edges_in}")
        if self.orig_edges_ts != self.orig_edges_ts:
            extras.append(f"orig_edges_ts={self.orig_edges_ts}")
        extras = ", ".join(extras)
        if extras:
            extras = ", " + extras
        return (
            f"Function('{self.fn_info.name}', {self.edges_out}, {', '.join(map(repr, self.inputs))}{extras})"
        )

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        extras = self._check_evaluate(values, dims, extras)
        if self in values:
            return values[self]
        inner_values = []
        for (t, *input_edges), oes in zip(self.inputs, self.orig_edges_ts):
            inner_value = t.evaluate(values, extras=extras)
            assert inner_value.names == tuple(t.edges), f"Expected {t.edges}, got {inner_value.names}"
            # After evaluation we need to rename the output edges to the original names of the function,
            # as they are expected by fn_info.eval.
            inner_values.append(inner_value.rename(*oes))
        out = self.fn_info.eval(*inner_values)
        # After evaluation we need to rename the output edges back to their current values.
        assert out.names == tuple(self.orig_edges)
        out = out.rename(*self.edges)
        values[self] = out
        assert out.names == tuple(self.edges)
        return out


class Derivative(Tensor):
    def __init__(self, tensor: Tensor, x: Variable, new_names: Optional[list[str]] = None):
        self.tensor = tensor
        self.x = x
        # _check_grad makes sure the new_names are not already present in self.edges.
        # But we haven't set self.edges yet, so we call it on tensor instead of self.
        self.new_names = tensor._check_grad(x, new_names)
        self.edges = tensor.edges + self.new_names

    def simplify(self, args: dict[str, Any] = None):
        args = self._check_simplify(args)
        if args["grad_steps"] == 0:
            # If grad_steps is 0, we pass the simplify through the derivative.
            res = Derivative(self.tensor.simplify(args), self.x, self.new_names)
        else:
            args["grad_steps"] -= 1
            # Have to call simplify twice to avoid an infinite loop when stacking multiple derivatives.
            res = self.tensor.simplify(args).grad(self.x, self.new_names).simplify(args)
        assert set(res.edges) == set(self.edges), f"Edges changed from {self.edges} to {res.edges}"
        return res

    def grad(self, x: Variable, new_names: list[str] | None) -> Tensor:
        new_names = self._check_grad(x, new_names)
        # Recall grad is about pushing a derivative through a function,
        # so we apply it on the inside of the derivative, not the outside.
        res = Derivative(Derivative(self.tensor, x, new_names), self.x, self.new_names)
        assert set(res.edges) == set(self.edges + new_names)
        return res

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        # The free edges of Derivative are both the free edges of self.tensor and the new_names.
        # This is the only place where we need to rename the "internal edges" of the tensor.
        res = Derivative(
            self.tensor.rename(kwargs),
            self.x,
            [kwargs.get(e, e) for e in self.new_names],
        )
        assert set(res.edges) == {kwargs.get(e, e) for e in self.edges}
        return res

    def __repr__(self):
        return f"Derivative({self.tensor}, {self.x}, {self.new_names})"

    def __hash__(self):
        return hash(("Derivative", hash(self.tensor), hash(self.x), len(self.new_names)))

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        # We could use numerical differentiation here...  But it would potentially require quite a lot of
        # evaluations, since we need to evaluate the tensor in all directions.
        raise ValueError("Derivative tensors cannot be evaluated directly. Please use simplify() first.")

    def edge_equivalences(self):
        # Not really needed since we don't expect to be evaluated
        yield from self.tensor.edge_equivalences()


################################################################################
# Product
################################################################################


class Product(Tensor):
    def __init__(self, tensors: Iterable[Tensor]):
        self.tensors = list(tensors)
        self.edges = []
        self.contractions = []
        cnt_edges = Counter(sum(((t.edges for t in self.tensors)), []))
        for edge, cnt in cnt_edges.most_common():
            if cnt == 1:
                self.edges.append(edge)
            elif cnt == 2:
                self.contractions.append(edge)
            else:
                raise ValueError(
                    f"edge {edge} had multiplicity {cnt}. Use an identity tensor to combine multiple edges."
                )

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        # The main challenge is that renaming the free edges of the tensor can cause clashes with the internal/contracted edges.
        # For each tensor, it must rename *its free edges* to avoid *free edges of other tensors*, to prevent a clash,
        # *unless* the edge is a contraction, in which case it must rename to avoid *all edges of other tensors*.
        # Also, it's important for all these edges to rename in a consistent way.
        # They need to avoid:
        # - Hitting the free edges (non contractions)
        # - Hitting the new names
        # - Hitting the other contractions
        # It's OK if they hit:
        # - the non-contractions that *are* being renamed (but that could be a bit unnecessary work...)
        # - Their own name (so we can't just give it a list of *all* names of the tensors)
        all_edges = {e for t in self.tensors for e in t.edges}
        contractions = all_edges - set(self.edges)
        avoid = {kwargs.get(e, e) for e in self.edges} | set(self.edges)
        _new_edges, rename = unused_edge_names(contractions, avoid)
        # It's safe to add the kwargs to rename, since self._check_rename restricts kwargs to only
        # contain keys that are in self.edges.
        rename |= kwargs
        res = Product([t.rename(rename) for t in self.tensors])
        assert set(res.edges) == {kwargs.get(e, e) for e in self.edges}
        return res

    def _avoid_internal_edges(self, names_to_avoid: set[str]) -> "Product":
        """Rename internal edges to avoid clashes with the names_to_avoid"""
        # TODO: This method is being overly harsh. It could probably learn something from the rename method.
        if overlap := set(self.edges) & set(names_to_avoid):
            raise ValueError(f"Don't use this method to rename free edges: {overlap}")
        used_names = set(names_to_avoid)
        for t in self.tensors:
            # Try to avoid unnecessary renaming other inner edges. Not strictly necessary. Is it even a good idea?
            used_names |= set(t.edges)
        inner_names = {e for t in self.tensors for e in t.edges if e not in self.edges}
        _new_names, rename = unused_edge_names(inner_names, used_names)
        res = Product([t.rename(rename) for t in self.tensors])
        assert res == self, "Renaming shouldn't change the hash/equality"
        assert set(res.edges) == set(self.edges), "Free edges should be preserved"
        return res

    @staticmethod
    def merge(products: list["Product"]) -> "Product":
        """Rename internal edges (multiplicity > 1) to be distinct between each group."""
        # The union of free edges from each product contains the free edges and "level 1" inner edges.
        used_edges = {e for p in products for e in p.edges}
        res = []
        for p in products:
            # Maybe this could also be expressed in terms of avoid_internal_edges
            inner_edges = {e for t in p.tensors for e in t.edges if e not in p.edges}
            new_names, rename = unused_edge_names(inner_edges, used_edges)
            for t in p.tensors:
                res.append(t.rename(rename))
            used_edges |= set(new_names)  # Later renames should not clash with this one
        return Product(res)

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = self._check_grad(x, new_names)
        # Since we are adding new edges to an internal tensor in the product, we need to make sure
        # none of the other tensors in the product have edges that clash with these new edges.
        new_prod = self._avoid_internal_edges(new_names)
        res = Sum(
            [
                Product(new_prod.tensors[:i] + [Derivative(t, x, new_names)] + new_prod.tensors[i + 1 :])
                for i, t in enumerate(new_prod.tensors)
            ]
        )
        assert set(res.edges) == set(self.edges + new_names)
        return res

    def __repr__(self):
        return f"Product({self.tensors})"

    def __hash__(self):
        # We really want a perfect Graph Isomorphism hash. But alas that doesn't exist.
        # So we use "Weisfeiler Leman" and hope everything is fine and good.
        # Just be aware that two cycles of length 3 will get the same hash as a single cycle of length 6,
        # *if* all the nodes themselves have the same hash.
        tensors = sorted(self.tensors, key=hash)
        neighbors = [[t2 for t2 in tensors if set(t1.edges) & set(t2.edges)] for t1 in tensors]
        tensor_hashes = [hash(t) for t in tensors]
        # print(f"{tensor_hashes=}")
        for _ in range(3):
            tensor_hashes = [
                # We always sort the neighbors to make the hash order independent
                hash((h,) + tuple(hash(t2) for t2 in ns))
                for h, ns in zip(tensor_hashes, neighbors)
            ]
            # print(f"{tensor_hashes=}")
        return hash(("Product",) + tuple(sorted(tensor_hashes)))

    def edge_equivalences(self):
        pairs = defaultdict(list)
        for t in self.tensors:
            yield from t.edge_equivalences()
            for e in t.edges:
                pairs[e].append(t)
        for e, ts in pairs.items():
            if len(ts) == 1:
                yield (self, e), (ts[0], e)
            else:
                t1, t2 = ts
                yield (t1, e), (t2, e)

    def evaluate(
        self,
        values: dict["Variable", torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        extras = self._check_evaluate(values, dims, extras)
        if self in values:
            # FIXME: This is a hack that assumes edge ordering is consistent between isomorphic tensors.
            return values[self].rename(*self.edges)
        # TODO: We can make this more efficient by removing Copy tensors.
        # Keep track of how many contractions we made
        extras["contractions"] = extras.get("contractions", 0) + len(self.contractions)
        # We use "operator" einsum interface, which doesn't require single letter names.
        edge_numbers = {e: i for i, e in enumerate({e for t in self.tensors for e in t.edges})}
        parts = []
        for t in self.tensors:
            parts.append(t.evaluate(values, extras=extras).rename(None))
            parts.append([edge_numbers[e] for e in t.edges])
        parts.append([edge_numbers[e] for e in self.edges])
        print(f"{parts=}")
        out = torch.einsum(*parts).rename(*self.edges)
        print(f"{out=}")
        assert out.names == tuple(self.edges)
        values[self] = out
        return out

    def simplify(self, args: dict[str, Any] = None):
        args = self._check_simplify(args)
        tensors = [t.simplify(args=args) for t in self.tensors]
        # If any tensor in a product is 0, so is the whole product
        if any(isinstance(t, Zero) for t in tensors):
            return Zero(self.edges)

        # Decide whether to push products through sums. This can be useful to simplify networks for
        # presentation, but it can also blow up the number of terms. So we make it optional.
        if args.get("distributed_products", False):
            raise NotImplementedError("Not implemented yet")

        # We can do a "small" kind of distributed products, which is handling children that are single sums
        # Also, if a child is a sum with a single element, we can pull the weight up.
        # In general, we can pull out the least common multiple of the weights of the children.

        if single_sums := [t for t in tensors if isinstance(t, Sum) and len(t.tensors) == 1]:
            tensors = [t if t not in single_sums else t.tensors[0] for t in tensors]
            weight = prod(t.weights[0] for t in single_sums)
        else:
            weight = 1

        # Combine nested products. Note that not combining products may be useful to keep the
        # "natural" contraction order. This can speed up evaluation, since sub-tensors can be reused.
        if args.get("associative_products", True):
            sub_products = [t if isinstance(t, Product) else Product([t]) for t in tensors]
            children = Product.merge(sub_products).tensors
        else:
            children = tensors

        # Simplify Copy Tensors
        while True:
            for e in [e for t in children for e in t.edges]:
                ts = [t for t in children if e in t.edges]
                if len(ts) == 1:
                    continue
                t1, t2 = ts
                # Merge connected copy tensors into one
                if isinstance(t1, Copy) and isinstance(t2, Copy):
                    new = Copy(list(set(t1.edges + t2.edges) - {e}))
                    # Can't use children.remove(t1) since we've overloaded equality to mean isomorphic...
                    children = [t for t in children if t is not t1]
                    children = [t for t in children if t is not t2]
                    children.append(new)
                    break
                # Remove identity matrices
                if isinstance(t1, Copy) and len(t1.edges) == 2:
                    t1, t2 = t2, t1
                if isinstance(t2, Copy) and len(t2.edges) == 2:
                    (other_edge,) = set(t2.edges) - {e}
                    # Don't create self loops
                    if other_edge in t1.edges:
                        continue
                    rename = {e: other_edge}
                    children = [t for t in children if t is not t1]
                    children = [t for t in children if t is not t2]
                    children.append(t1.rename(rename))
                    break
            else:
                # If we didn't find any identity matrices to remove, we are done.
                break

        # Remove empty Copy's (they are just the constant 1)
        children = [t for t in children if not (isinstance(t, Copy) and t.rank == 0)]

        # Base cases
        if not children:
            # The only issue here is that we're throwing away edge names.
            res = Copy([])
        if len(children) == 1:
            res = children[0]
        else:
            res = Product(children)

        if weight != 1:
            res = Sum([res], [weight])

        assert set(res.edges) == set(self.edges), f"Edges changed from {self.edges} to {res.edges}"
        return res


################################################################################
# Sum
################################################################################


class Sum(Tensor):
    def __init__(self, tensors: Iterable[Tensor], weights: list[int] = None):
        tensors = list(tensors)
        # Broadcasting means we always upgrade to the super set of edges
        edges = {e for t in tensors for e in t.edges}
        self.tensors = []
        for t in tensors:
            missing = list(edges - set(t.edges))
            # Note: don't broadcast if the tensor is already full, since that would create new
            # Ones([]) objects after simplification is supposed to have completed.
            self.tensors.append(t @ Ones(missing) if missing else t)
        self.edges = list(edges)
        self.weights = [1] * len(tensors) if weights is None else weights
        assert len(tensors) == len(self.weights)

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        res = Sum([t.rename(kwargs) for t in self.tensors], self.weights)
        assert set(res.edges) == {kwargs.get(e, e) for e in self.edges}
        return res

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = self._check_grad(x, new_names)
        res = Sum([Derivative(t, x, new_names) for t in self.tensors], self.weights)
        assert set(res.edges) == set(
            self.edges + new_names
        ), f"Edges changed from {self.edges} to {res.edges}"
        return res

    def simplify(self, args: dict[str, Any] = None):
        args = self._check_simplify(args)
        if args["sum_combine_terms"]:
            # Identify tensors with multiplicity and combine them
            ws_tensors = defaultdict(int)
            for w, t in zip(self.weights, self.tensors):
                t = t.simplify(args=args)
                if isinstance(t, Sum):
                    for w1, t1 in zip(t.weights, t.tensors):
                        ws_tensors[t1] += w * w1
                else:
                    ws_tensors[t] += w
            # Remove zero tensors or zero weights.
            # Note: This won't change the shape of the tensor, since all summands have been broadcasted.
            ws_tensors = [(w, t) for t, w in ws_tensors.items()]
        else:
            ws_tensors = [(w, t.simplify(args=args)) for w, t in zip(self.weights, self.tensors)]
        ws_tensors = [(w, t) for w, t in ws_tensors if w != 0 and not isinstance(t, Zero)]
        # Base case. Here we can't just return Zero([]), since that would change the signature of the tensor.
        if not ws_tensors:
            return Zero(self.edges)
        weights, tensors = zip(*ws_tensors)
        # If there is just one tensor with weight 1, we don't need LinearComb
        if weights == (1,):
            return tensors[0]
        res = Sum(tensors, weights)
        assert set(res.edges) == set(self.edges), f"Edges changed from {self.edges} to {res.edges}"
        return res

    def __repr__(self):
        return f"Sum({self.tensors}, {self.weights})"

    def __hash__(self):
        return hash(("Sum",) + tuple(sorted(map(hash, zip(self.tensors, self.weights)))))

    def edge_equivalences(self):
        for t in self.tensors:
            yield from t.edge_equivalences()
            for e in t.edges:
                yield (t, e), (self, e)

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        extras = self._check_evaluate(values, dims, extras)
        if self in values:
            return values[self]
        values[self] = sum(
            w * t.evaluate(values, extras=extras).align_to(*self.edges)
            for w, t in zip(self.weights, self.tensors)
        )
        return values[self]


################################################################################
# Some useful functions
################################################################################


def unused_edge_names(edges: Iterable[str], used_names: Iterable[str], suffix: str = ""):
    """Given a list of edges, return a list of new names that are not in used_names.
    They also won't be translated to the same names.
    Also append suffix to the new names.
    Also returns a dictionary that can be used to rename the edges in the original list.
    """
    used_names = set(used_names)
    new_edges = []
    rename = {}
    for e in edges:
        orig = e
        e += suffix
        while e in used_names:
            e += "_"
        new_edges.append(e)
        used_names.add(e)
        if e != orig:
            rename[orig] = e
    return new_edges, rename


def make_distinct(*tensors: list["Tensor"], used_names=None) -> list["Tensor"]:
    """Makes sure all tensors have distinct edges.
    Optionally takes used_names, an extra set of names to avoid.
    suffix is an optional string to append to the new names.
    """
    # Copy the set, so we don't modify the input
    used_names = set() if used_names is None else set(used_names)
    # No reason to rename the names that are unique
    cnt = Counter(e for t in tensors for e in t.edges)
    unique_names = {e for e, c in cnt.items() if c == 1}
    # Unless the names are already used, then we can't save them
    unique_names -= used_names
    used_names |= unique_names
    res = []
    renames = []
    for i, t in enumerate(tensors):
        edges = set(t.edges) - unique_names
        new_names, rename = unused_edge_names(edges, used_names, suffix=f"_{i}")
        # Make sure all t.edges are present in rename
        for e in unique_names:
            rename[e] = e
        used_names |= set(new_names)
        res.append(t.rename(rename))
        renames.append(rename)
    return res, renames


def join_dicts(all_shapes: Iterable[dict[str, int]]) -> dict[str, int]:
    shapes = {}
    for s in all_shapes:
        for k, v in s.items():
            if k in shapes:
                if shapes[k] != v:
                    raise ValueError(f"Shapes mismatch: {shapes[k]} != {v}")
        shapes |= s
    return shapes


def compute_edge_dims(
    root: "Tensor",
    shapes: dict["Variable", dict[str, int]],
    extra_dims: dict[str, int] | None = None,
) -> dict[int, dict[str, int]]:
    graph = defaultdict(list)
    v_ids = defaultdict(list)
    for (t1, e1), (t2, e2) in root.edge_equivalences():
        # Make keys using ids, since we want to distinguish between isomorphic tensors
        key1 = (id(t1), e1)
        key2 = (id(t2), e2)
        graph[key1].append(key2)
        graph[key2].append(key1)
        # Make sure we have the updated variable ids. The same variable can have multiple ids,
        # as it has been renamed differently throughout the graph.
        for t in [t1, t2]:
            if isinstance(t, Variable):
                v_ids[t].append(id(t))
    # Add the initial known values
    res = defaultdict(dict)
    queue = deque()
    for v, edges in shapes.items():
        for e, size in edges.items():
            if v not in v_ids:
                print(f"Info: Variable {v} was not found in graph {v_ids}.")
                continue
            for v_id in v_ids[v]:
                res[v_id][e] = size
                queue.append((v_id, e, size))
    if extra_dims is not None:
        for e, size in extra_dims.items():
            res[id(root)][e] = size
            queue.append((id(root), e, size))
    # BFS to propagate the values
    while queue:
        id_t1, e, size = queue.popleft()
        for id_t2, e2 in graph[(id_t1, e)]:
            if e2 not in res[id_t2]:
                res[id_t2][e2] = size
                queue.append((id_t2, e2, size))
            elif (size2 := res[id_t2][e2]) != size:
                raise ValueError(f"Size mismatch: {size} != {size2}")
    return res
