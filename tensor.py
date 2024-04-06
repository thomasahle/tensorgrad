from collections import Counter, defaultdict, deque
from typing import Any, Callable, Iterable, Optional, Union
from abc import ABC, abstractmethod

import torch

# TODO:
# - Evaluation of functions
# - Support for specific functions and their simplification rules (cross entropy, etc)
# - Code generation? (e.g. Triton)
# - Prettier printing
# - Stuff from https://en.wikipedia.org/wiki/Penrose_graphical_notation
#   - Symmetrization/Antisymmetrization


def all_edges(tensor):
    edges = set(tensor.edges)
    for t in getattr(tensor, "tensors", []):
        edges |= all_edges(t)
    return edges


def unused_edge_names(edges, used_names, suffix=""):
    used_names = set(used_names)
    new_edges = []
    rename = {}
    for e in edges:
        orig = e
        if e in used_names:
            e += suffix
        while e in used_names:
            e += "_"
        new_edges.append(e)
        used_names.add(e)
        if e != orig:
            rename[orig] = e
    return new_edges, rename


def make_distinct(*tensors: list["Tensor"], preserve_free=True, used_names=None, suffix="") -> list["Tensor"]:
    """Makes sure all tensors have distinct edges. *All the way down*.
    If preserve_free is True, we only rename the edges that are not free in any of the tensors.
    Optionally takes used_names, an extra set of names to avoid.
    suffix is an optional string to append to the new names.
    """
    if used_names is None:
        used_names = set()
    else:
        # Copy the set, so we don't modify the input
        used_names = set(used_names)
    if preserve_free:
        # We reserve the free edges, so other edges are not renamed to clash with them.
        for t in tensors:
            used_names |= set(t.edges)
    res = []
    renames = []
    for t in tensors:
        edges = all_edges(t)
        if preserve_free:
            edges -= set(t.edges)
        new_names, rename = unused_edge_names(edges, used_names, suffix=suffix)
        used_names |= set(new_names)
        res.append(t.rename(rename))
        renames.append(rename)
    return res, renames


def ensure_edges_unused(tensor: "Tensor", x: "Variable", new_names: Optional[list[str]] = None) -> list[str]:
    if new_names is not None:
        assert len(x.edges) == len(new_names)
        used = set(new_names) & set(tensor.edges)
        if used:
            raise ValueError(f"{used} are already present in tensor")
        return new_names
    all_tensor_edges = all_edges(tensor)
    new_edges, _rename = unused_edge_names(x.edges, all_tensor_edges)
    return new_edges


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
) -> dict[tuple["Tensor", str], int]:
    """
    Compute the edge dimensions for all tensors in the graph, starting from the root tensor.

    The function performs a depth-first search to establish parent-child relationships and iteratively
    computes edge dimensions based on the available information from variables and parent tensors.
    It assumes that each tensor's `update_edge_dims` method is implemented correctly.
    """
    # To avoid issues with colliding names, we define an edge by (tensor, edge_name)
    parents: dict[int, list[Tensor]] = defaultdict(list)
    missing_edges: dict[int, set[str]] = {}
    id_to_tensor: dict[int, Tensor] = {}

    # Another complication here is that we usually consider isomporphic tensors to be equal.
    # But here we actually want to distinguish them on edge names, since two copy tensors of
    # the same shape, but with different edge names, can have different edge dimensions.

    # However, we still need the input `shapes` to use a variable rather than an id, since
    # the variable might have changed since the user created it and no longer have the same
    # Compute parents
    new_shapes = {}

    def dfs(t: Tensor):
        print("visiting", t, t in shapes)
        if t in shapes:
            # This is about updating the identity of the variable
            new_shapes[id(t)] = shapes[t]
        missing_edges[id(t)] = set(t.edges)
        id_to_tensor[id(t)] = t
        for t1 in t.tensors:
            parents[id(t1)].append(t)
            # We use missing_edges to check "visisted"
            if id(t1) not in missing_edges:
                dfs(t1)

    dfs(root)

    for v_id, edges in new_shapes.items():
        v = id_to_tensor[v_id]
        rename = dict(zip(v.original_edges, v.edges))
        new_shapes[v_id] = {rename[k]: v for k, v in edges.items()}

    if extra_dims is not None:
        new_shapes.setdefault(id(root), {}).update(extra_dims)

    computed_edges: dict[Tensor, dict[str, int]] = defaultdict(dict)
    tasks: deque[Tensor] = deque()
    for t_id, shape in new_shapes.items():
        # The user might have given us some unnecessary shapes, so we only consider the ones we need.
        if t_id in id_to_tensor:
            computed_edges[t_id] = shape
            tasks.append(id_to_tensor[t_id])
            tasks += parents[t_id]
    while tasks:
        p = tasks.popleft()
        # The parent computes as many edges as it can, based on the information about its
        # children as well as the information about its own edges, already computed.
        computed = p.update_edge_dims({id(t): computed_edges[id(t)] for t in [p] + p.tensors})
        # print(f"{p} returned {computed} from update")
        for t1, e1, size1 in computed:
            if (size := computed_edges[id(t1)].get(e1)) is not None and size != size1:
                raise ValueError(f"Size mismatch: {size} != {size1}")
            # Most of the computed stuff has probably already been computed before, so we only
            # create new tasks for the stuff that hasn't been computed yet.
            if e1 not in computed_edges[id(t1)]:
                computed_edges[id(t1)][e1] = size1
                for t2 in [t1] + parents[id(t1)]:
                    if id(t2) not in map(id, tasks):
                        tasks.append(t2)

    print(f"{computed_edges=}")
    print(f"{missing_edges=}")

    for t_id, edges in missing_edges.items():
        if missing := edges - computed_edges[t_id].keys():
            raise ValueError(f"Missing edge dimensions for {id_to_tensor[t_id]}: {missing}")

    return computed_edges


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

    def simplify(self, args: dict[str, Any] = {}):
        """Apply various simplification rules."""
        return self

    def __add__(self, other):
        return Sum([self, other])

    def __sub__(self, other):
        return Sum([self, other], [1, -1])

    def __matmul__(self, other):
        return Product([self, other])

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        """Contract self and other, but use a 3d-identity to keep the shared edges free."""
        if isinstance(other, int) or isinstance(other, float):
            return Sum([self], [other])
        # These are the edges we multiply over
        shared_edges = set(self.edges) & set(other.edges)
        # Element-wise (Hadamard) product is easy to implement using Copy tensors
        (t0, t1), (rename0, rename1) = make_distinct(
            self, other, preserve_free=False, used_names=shared_edges
        )
        joins = []
        for e in shared_edges:
            joins.append(Copy([e, rename0[e], rename1[e]]))
        return Product([t0, t1] + joins)

    def rename(self, kwargs: dict[str, str]):
        raise NotImplementedError

    def __hash__(self):
        # If we can make a perfect hash, we can save work from sub-computations
        raise NotImplementedError

    def __eq__(self, other):
        # Note: Hash does not care about edge names, so this check won't either
        return hash(self) == hash(other)

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor | Callable],
        *,
        dims: dict[str, int] | None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        raise NotImplementedError

    def update_edge_dims(self, shapes: dict["Tensor", dict[str, int]]) -> Iterable[tuple["Tensor", str, int]]:
        """Return the dimensions of the output edges of the tensor *and its children*.

        Parameters:
        - shapes: The edge_dims already computed for this tensor and its children.

        The reason we need to also output the dimensions of the children is to support propagation
        of dimensions in things like products, where you get a dimension from one child, and use it
        to compute the dimension of another child. You don't have to tell the other child about this
        new information. It will automatically be given to it.
        """
        raise NotImplementedError

    def _check_extras(
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
            extras["edge_dims"] = compute_edge_dims(self, shapes, extra_dims=dims)
        return extras


class Variable(Tensor):
    def __init__(self, name, edges: Iterable[str], surface_edges=None):
        self.name = name
        # The original edges are saved so evaluation can happen with the original
        # edge names given to the variable, without caring about what renaming we
        # might have done in the meantime.
        self.original_edges = list(edges)
        self.edges = self.original_edges if surface_edges is None else surface_edges

    def grad(self, x: "Variable", new_names: Optional[list[str]] = None):
        new_names = ensure_edges_unused(self, x, new_names)
        # We compare using name here, rather than id (x is self), since we might have
        # changed due to renaming.
        if x.name == self.name:
            # Note: This is the product of n identity matrices, which is exactly the identity tensor.
            return Product(Copy([e, new]) for e, new in zip(self.edges, new_names))
        return Zero(self.edges + new_names)

    def __repr__(self):
        return f"Variable({self.name}, {self.original_edges}, {self.edges})"

    def __hash__(self):
        return hash(("Variable", self.name, len(self.edges)))

    def simplify(self, args: dict[str, Any] = {}):
        return self

    def rename(self, kwargs: dict[str, str]):
        v = Variable(
            self.name,
            self.original_edges,
            [kwargs.get(e, e) for e in self.edges],
        )
        return v

    def update_edge_dims(self, shapes: dict[int, dict[str, int]]) -> Iterable[tuple["Tensor", str, int]]:
        # We don't need to update anything from the variables. They are handled automatically.
        return []

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor | Callable],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        extras = self._check_extras(values, dims, extras)
        if self not in values:
            raise ValueError(f"Missing value for {self}")
        # For now we just assume positional consistency
        if values[self].names != tuple(self.original_edges):
            raise ValueError(f"Edge names {values[self].names} don't match original names of {self}")
        if values[self].shape != tuple(extras["edge_dims"][id(self)][e] for e in self.edges):
            raise ValueError(f"Shape of {values[self]} doesn't match expected shape.")
        return values[self].rename(*self.edges)


class Constant(Tensor, ABC):
    def __init__(self, edges: Iterable[str]):
        super().__init__()
        self.edges = list(edges)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.edges})"

    def __hash__(self):
        return hash((type(self).__name__, len(self.edges)))

    def copy(self):
        return type(self)(self.edges)

    def rename(self, kwargs: dict[str, str]):
        return type(self)([kwargs.get(e, e) for e in self.edges])

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = ensure_edges_unused(self, x, new_names)
        return Zero(self.edges + new_names)

    def update_edge_dims(self, shapes: dict[int, dict[str, int]]) -> Iterable[tuple["Tensor", str, int]]:
        # By default constant tensors can't compute any edge dimensions.
        return []

    def _check_extras(self, values, dims=None, extras=None):
        extras = super()._check_extras(values, dims, extras)
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

    def update_edge_dims(self, shapes: dict[int, dict[str, int]]) -> Iterable[tuple["Tensor", str, int]]:
        if not self.edges:
            return []
        dims = list(shapes[id(self)].values())
        if not dims:
            return []
        if len(set(dims)) != 1:
            raise ValueError(f"Edge dimensions must be the same. Got {dims}.")
        return [(self, e, dims[0]) for e in self.edges]

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor | Callable],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        if not self.edges:
            return torch.tensor(1.0)
        extras = self._check_extras(values, dims, extras)
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
        values: dict["Tensor", torch.tensor | Callable],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        if not self.edges:
            return torch.tensor(0.0)
        extras = self._check_extras(values, dims, extras)
        values[self] = torch.zeros([extras["edge_dims"][id(self)][e] for e in self.edges]).rename(*self.edges)
        return values[self]


def Ones(edges: list[str]) -> Tensor:
    """Matrix such that O_{i,j,k} = 1 for all i, j, k"""
    return Product([Copy([e]) for e in edges])


class Function(Tensor):
    """A function that takes one or more tensors as input and produces a new tensor as output.
    Examples:

    # Matrix multiplication:
    Say x and y are matrices with shapes (b, i, j) and (b, j, k).
    We might define f = Function("matmul", [], (x, "j"), (y, "j"))
    This function takes in the j-edge from each matrix, and doesn't produce any new edges.
    The shared edge b is super imposed...
    All in all we end up with a tensor of shape (b, i, k) as we would expect.

    # Unfold
    Say we have a tensor x with shape (b, c, w, h) and we want to unfold it to do convolution.
    We might define f = Function("unfold", ["patch", "i"], (x, "w", "h"))
    This function takes in the w and h edges from x, and produces a number of patches, each with dimension i
    The new tensor will have shape (b, c, patches, i)
    One will typically permute to (b, patches, c, i) and flatten the last two dimensions to (b, patches, j)
    Then one can apply a linear function to get (b, patches, k) before finally reshaping to (b, c, patches, l)
    and unfolding to (b, c, w, h) again.

    # Cross entropy
    This is an example of a function that takes two inputs.
    f = Function("ce", ["ce"], (x, "logits"), (y, "probs")])
    as before, x and y will typically have some shared edges, like (b, logits) and (b, prob).
    So the output will have shape (b, ce).

    # Multi-query Dot Product Attention
    Suppose we have tensors: q with shape (b, n_q, d), k with shape (b, seq, d), and v with shape (b, seq, d_out)
    first we form (b, n_q, seq) then we apply softmax to get (b, n_q, prob) then multiply by v to get (b, n_q, d_out)
    We can define a function f = Function("attention", ["d_out"], (q, "d"), (k, "seq", "d"), (v, "seq", "d_out")).

    Of course, many of these functions could more easily be written as a combination of simpler functions.
    """

    def __init__(
        self,
        name: str,
        edges_out: Iterable[str],
        *inputs: Iterable[tuple[Tensor | str]],
    ):
        self.name = name
        self.edges_out = list(edges_out)
        self.inputs = list(inputs)
        self.tensors = [t for t, *_ in self.inputs]
        # The actual edges of the function is the "output" edges,
        # plus any edges from the inputs that are not "input edges"
        shared_edges = set()
        for t, *es in self.inputs:
            for e in set(t.edges) - set(es):
                if e in shared_edges:
                    # TODO: While it is true that we can avoid shared edges by renaming, and just use a copy tensor,
                    # it's also very inefficient, essentially computing, a loss say, on the whole outer product over
                    # the batch dimension, only to then pick the diagonal at the end. So we probably need to support
                    # broadcasting directly in Function.
                    raise ValueError(
                        f"Edge {e} is already used by another input tensor. Please rename. If you need broadcasting, use a copy tensor."
                    )
                shared_edges.add(e)
        self.edges = list(set(edges_out) | shared_edges)

    #################
    # The following methods need to be implemented by subclasses

    def rename(self, kwargs: dict[str, str]):
        if self.__class__ is not Function:
            raise NotImplementedError(f"Please define rename(...) in {self.__class__.__name__}")
        # We only rename on the surface
        new_edges_out = [kwargs.get(e, e) for e in self.edges]
        return Function(self.name, new_edges_out, *self.inputs)

    def simplify(self, args: dict[str, Any] = {}):
        if self.__class__ is not Function:
            raise NotImplementedError(f"Please define simplify(...) in {self.__class__.__name__}")
        new_inputs = [(t.simplify(args=args), *es) for t, *es in self.inputs]
        return Function(self.name, self.edges, *new_inputs)

    def inner_grad(self, i, new_edges):
        # By default we just return a simple renamed function.
        # But subclasses can override this to do something more clever.
        if self.__class__ is not Function:
            raise NotImplementedError(f"Please define inner_grad(...) in {self.__class__.__name__}")
        return Function(f"D_{i}" + self.name, self.edges + new_edges, *self.inputs)

    def update_edge_dims(self, shapes: dict[int, dict[str, int]]) -> Iterable[tuple["Tensor", str, int]]:
        """Return the dimensions of the output edges of the function, as well as any children you can determine."""
        raise NotImplementedError("Please subclass Function to evaluate it")

    def __call__(self, *values: list[torch.tensor]) -> torch.tensor:
        """Evaluate the function on the given concrete tensors."""
        raise NotImplementedError("Please subclass Function to evaluate it")

    #################
    # The following methods should not be overridden by subclasses

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        # We sum over each input function, just like the normal chain rule:
        # d/dx f(g₁(x), …, gₖ(x)) = Σᵢ₌₁ᵏ (d/dx gᵢ(x)) Dᵢf(g₁(x), …, gₖ(x))

        # D_i adds a new output edge to the function, which is contracted with
        # the normal output edge of the tensor. So we need to make sure this doesn't
        # clash with an existing output edge of f.
        new_names = ensure_edges_unused(self, x, new_names)
        parts = []
        for i, (t, *input_edges) in enumerate(self.inputs):
            # Connection edges have the same shape as the to the function. This is different
            # from new_names, which has the shape of the variable we are taking the derivative with respect to.
            # What's the point with the broadcasted edges? Do we make new edges for them, and then just copy-join them
            # with the existing set?
            connection_edges, rename = unused_edge_names(
                input_edges, set(self.edges) | set(new_names) | set(t.edges)
            )
            parts.append(
                Product([self.inner_grad(i, connection_edges), Derivative(t.rename(rename), x, new_names)])
            )
        return Sum(parts)

    def __hash__(self):
        return hash(
            ("Function", self.name, len(self.edges_out))
            + tuple(map(hash, self.tensors))
            + tuple(map(len, self.inputs))
        )

    def __repr__(self):
        return f"Function({self.name}, {self.edges_out} {self.inputs})"

    # def compute_edge_dims(
    #     self, shapes: dict["Variable", dict[str, int]], edge_dims: dict[str, int]
    # ) -> dict[str, int]:
    #     edge_dims = join_dicts(t.compute_edge_dims(shapes) for t in self.tensors)
    #     subclass_edge_dims = self.edge_dims(edge_dims)
    #     return join_dicts([subclass_edge_dims, edge_dims])

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        extras = self._check_extras(values, dims, extras)
        if self in values:
            return values[self]
        inner_values = [t.evaluate(values, extras=extras) for t in self.tensors]
        out = self(*inner_values)
        # TODO: Use the "original_names" trick from Variable to avoid relying on the order of the edges here.
        out = out.rename(*self.edges)
        values[self] = out
        return out


class Derivative(Tensor):
    def __init__(self, tensor: Tensor, x: Variable, new_names: Optional[list[str]] = None):
        self.tensor = tensor
        self.x = x
        self.new_names = ensure_edges_unused(tensor, x, new_names)
        self.edges = tensor.edges + self.new_names

    def simplify(self, args: dict[str, Any] = {}):
        # args["grad_steps"] allows us to control how far we propagate the derivative.
        # If grad_steps is 0, we pass the simplify through the derivative.
        if args.setdefault("grad_steps", float("inf")) > 0:
            args["grad_steps"] -= 1
            # Have to call simplify twice to avoid an infinite loop when stacking multiple derivatives.
            return self.tensor.simplify(args).grad(self.x, self.new_names).simplify(args)
        return Derivative(self.tensor.simplify(args), self.x, self.new_names)

    def grad(self, x: Variable, new_names: list[str] | None) -> Tensor:
        # Recall grad is about pushing a derivative through a function,
        # so we apply it on the inside of the derivative, not the outside.
        return Derivative(Derivative(self.tensor, x, new_names), self.x, self.new_names)

    def rename(self, kwargs: dict[str, str]):
        return Derivative(
            self.tensor.rename(kwargs),
            self.x,
            [kwargs.get(e, e) for e in self.new_names],
        )

    def __repr__(self):
        return f"Derivative({self.tensor}, {self.x}, {self.new_names})"

    def __hash__(self):
        return hash(("Derivative", hash(self.tensor), hash(self.x), len(self.new_names)))

    def update_edge_dims(self, shapes: dict["Tensor", dict[str, int]]) -> Iterable[tuple["Tensor", str, int]]:
        raise ValueError("Derivative tensors cannot be evaluated directly. Please use simplify() first.")

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor | Callable],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        # We could use numerical differentiation here...  But it would potentially require quite a lot of
        # evaluations, since we need to evaluate the tensor in all directions.
        raise ValueError("Derivative tensors cannot be evaluated directly. Please use simplify() first.")


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
        # We need to rename everything (not just on the surface), since one of the applications of renaming
        # is to combine nested products, and then we need to avoid clashes between "internal" edges in each product.
        return Product([t.rename(kwargs) for t in self.tensors])

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = ensure_edges_unused(self, x, new_names)
        return Sum(
            [
                Product(self.tensors[:i] + [Derivative(t, x, new_names)] + self.tensors[i + 1 :])
                for i, t in enumerate(self.tensors)
            ]
        )

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
        for _ in range(3):
            tensor_hashes = [
                # We always sort the neighbors to make the hash order independent
                hash((h,) + tuple(sorted(hash(t2) for t2 in ns)))
                for h, ns in zip(tensor_hashes, neighbors)
            ]
        return hash(("Product",) + tuple(tensor_hashes))

    def compute_edge_dims(
        self, shapes: dict["Variable", dict[str, int]], edge_dims: dict[str, int]
    ) -> dict[str, int]:
        # TODO: This is not so simple, since we could have a big graph of Copy tensors, whose dimensions
        # all depend on a single Variable. So we need to explore this graph in the right order to
        # allow them to figure out their dimensions.
        # Even this method won't fix Zeros and Ones, which are not Square.
        edge_dims = join_dicts(t.compute_edge_dims(shapes) for t in self.tensors)
        return edge_dims

    def update_edge_dims(self, shapes: dict[int, dict[str, int]]) -> Iterable[tuple[Tensor, str, int]]:
        # We always assume edges have been "de-duplicated" as much as they need to be.
        # So we don't care about what tensor they "come from".
        edge_dims = {}
        for shape in shapes.values():
            edge_dims.update(shape)
        res = []
        for t in [self] + self.tensors:
            for e in t.edges:
                if e in edge_dims:
                    res.append((t, e, edge_dims[e]))
        return res

    def evaluate(
        self,
        values: dict["Variable", torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        extras = self._check_extras(values, dims, extras)
        if self in values:
            return values[self]
        # Keep track of how many contractions we made
        extras["contractions"] = extras.get("contractions", 0) + len(self.contractions)
        # Pytorch only supports single letter einsum names
        all_edges = {e for t in self.tensors for e in t.edges}
        simple_names = {e: chr(ord("a") + i) for i, e in enumerate(all_edges)}
        einsum_string = (
            ", ".join(" ".join(simple_names[e] for e in t.edges) for t in self.tensors)
            + " -> "
            + " ".join(simple_names[e] for e in self.edges)
        )
        res = torch.einsum(
            einsum_string,
            # torch.einsum doesn't support named tensors, so we have to drop them with rename(None)
            *[t.evaluate(values, extras=extras).rename(None) for t in self.tensors],
        )
        # Reinstate the edge names
        values[self] = res.rename(*self.edges)
        return values[self]

    def simplify(self, args: dict[str, Any] = {}):
        tensors = [t.simplify(args=args) for t in self.tensors]
        # If any tensor in a product is 0, so is the whole product
        if any(isinstance(t, Zero) for t in tensors):
            return Zero(self.edges)

        # Decide whether to push products through sums. This can be useful to simplify networks for
        # presentation, but it can also blow up the number of terms. So we make it optional.
        if args.get("distributed_products", False):
            raise NotImplementedError("Not implemented yet")

        # Combine nested products. Note that not combining products may be useful to keep the
        # "natural" contraction order. This can speed up evaluation, since sub-tensors can be reused.
        if args.get("associative_products", True):
            children = []
            for i, t in enumerate(tensors):
                if not isinstance(t, Product):
                    children.append(t)
                    continue
                # We need to rename the hidden edges to avoid clashes
                ts, _renames = make_distinct(
                    *t.tensors,
                    preserve_free=True,
                    used_names={e for t in children for e in t.edges},
                    suffix=f"_{i}",
                )
                children += ts
        else:
            children = tensors

        # Remove identity matrices, connected to something.
        # Note this also covers the higher rank identity tensors, which are just products of these.
        while True:
            cnt_edges = Counter(e for t in self.tensors for e in t.edges)
            for I2 in children:
                if isinstance(I2, Copy) and len(I2.edges) == 2:
                    e0, e1 = I2.edges
                    # Find the edge that is connected to something else.
                    # If both edges are, it doesn't matter which one we rename.
                    if cnt_edges[e0] == 2:
                        rename = {e0: e1}
                    elif cnt_edges[e1] == 2:
                        rename = {e1: e0}
                    # If the Identity matrix is not connected to anything, we don't remove it.
                    else:
                        continue
                    # Rename the edge and remove the identity matrix.
                    # Note that this might create a new identity matrix to remove.
                    children = [t.rename(rename) for t in children if t is not I2]
                    break
            else:
                # If we didn't find any identity matrices to remove, we are done.
                break

        # TODO: Combine connected Copy tensors into a single Copy tensor.
        # This also includes removing "self loops" on Copy's

        # Remove empty Identities and Ones
        # children = [t for t in children if not (isinstance(t, (Copy, Ones)) and t.rank == 0)]
        children = [t for t in children if not (isinstance(t, Copy) and t.rank == 0)]

        # Base cases
        if not children:
            # The only issue here is that we're throwing away edge names.
            return Copy([])
        if len(children) == 1:
            return children[0]
        return Product(children)


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
            self.tensors.append(t @ Copy(missing) if missing else t)
        self.edges = list(edges)
        self.weights = [1] * len(tensors) if weights is None else weights
        assert len(tensors) == len(self.weights)

    def rename(self, kwargs: dict[str, str]):
        return Sum([t.rename(kwargs) for t in self.tensors], self.weights)

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = ensure_edges_unused(self, x, new_names)
        return Sum([Derivative(t, x, new_names) for t in self.tensors], self.weights)

    def simplify(self, args: dict[str, Any] = {}):
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
        ws_tensors = [(w, t) for t, w in ws_tensors.items() if w != 0 and not isinstance(t, Zero)]
        # Base case. Here we can't just return Zero([]), since that would change the signature of the tensor.
        if not ws_tensors:
            return Zero(self.edges)
        weights, tensors = zip(*ws_tensors)
        # If there is just one tensor with weight 1, we don't need LinearComb
        if weights == (1,):
            return tensors[0]
        return Sum(tensors, weights)

    def __repr__(self):
        return f"Sum({self.tensors}, {self.weights})"

    def __hash__(self):
        return hash(("Sum",) + tuple(sorted(map(hash, zip(self.tensors, self.weights)))))

    def compute_edge_dims(self, shapes: dict["Variable", dict[str, int]]) -> dict[str, int]:
        return join_dicts(t.compute_edge_dims(shapes) for t in self.tensors)

    def update_edge_dims(self, shapes: dict[int, dict[str, int]]) -> Iterable[tuple["Tensor", str, int]]:
        # We always assume edges have been "de-duplicated" as much as they need to be.
        # So we don't care about what tensor they "come from".
        edge_dims = {e: size for shape in shapes.values() for e, size in shape.items()}
        res = []
        for t in [self] + self.tensors:
            for e in t.edges:
                if e in edge_dims:
                    res.append((t, e, edge_dims[e]))
        return res

    def evaluate(
        self,
        values: dict["Tensor", torch.tensor | Callable],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        extras = self._check_extras(values, dims, extras)
        if self in values:
            return values[self]
        values[self] = sum(
            w * t.evaluate(values, extras=extras).align_to(*self.edges)
            for w, t in zip(self.weights, self.tensors)
        )
        return values[self]
