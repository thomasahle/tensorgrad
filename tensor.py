from collections import Counter, defaultdict
from typing import Iterable, Optional

# TODO:
# - Add eval: Use torch named tensors, or maybe just a wrapper around numpy arrays?
# - Support for specific functions and their simplification rules (cross entropy, etc)
# - Code generation? (e.g. Triton)
# - Prettier printing


def all_edges(tensor):
    edges = set(tensor.edges)
    for t in getattr(tensor, "tensors", []):
        edges |= all_edges(t)
    return edges


def free_edge_names(edges, used_names):
    used_names = set(used_names)
    new_edges = []
    rename = {}
    for e in edges:
        orig = e
        while e in used_names:
            e += "'"
        new_edges.append(e)
        used_names.add(e)
        if e != orig:
            rename[orig] = e
    return new_edges, rename


def ensure_edges_unused(
    tensor: "Tensor", x: "Variable", new_names: Optional[list[str]] = None
) -> list[str]:
    if new_names is not None:
        assert len(x.edges) == len(new_names)
        used = set(new_names) & set(tensor.edges)
        if used:
            raise ValueError(f"{used} are already present in tensor")
        return new_names
    all_tensor_edges = all_edges(tensor)
    new_edges, _rename = free_edge_names(x.edges, all_tensor_edges)
    return new_edges


class Tensor:
    edges = []

    @property
    def rank(self):
        return len(self.edges)

    def grad(self, x: "Variable", new_names: Optional[list[str]]) -> "Tensor":
        """Take the derivative with respect to the variable x."""
        raise NotImplementedError

    def simplify(self, full=False):
        """Apply various simplification rules."""
        return self

    def copy(self):
        raise NotImplementedError

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
        # Element-wise (Hadamard) product is easy to implement using Copy tensors
        t0, t1 = self, other
        joins = []
        for e in set(t0.edges) & set(t1.edges):
            e0, e1 = e + "_0", e + "_1"
            t0 = t0.rename(**{e: e0})
            t1 = t1.rename(**{e: e1})
            joins.append(Copy([e, e0, e1]))
        return Product([t0, t1] + joins)

    def rename(self, **kwargs: dict[str, str]):
        raise NotImplementedError

    def __hash__(self):
        # If we can make a perfect hash, we can save work from sub-computations
        raise NotImplementedError

    def __eq__(self, other):
        return hash(self) == hash(other)


class Variable(Tensor):
    def __init__(self, name, edges: Iterable[str], surface_edges=None):
        self.name = name
        # The original edges are saved so evaluation can happen with the original
        # edge names given to the variable, without caring about what renaming we
        # might have done in the meantime.
        self.original_edges = list(edges)
        self.edges = self.original_edges if surface_edges is None else surface_edges

    def copy(self):
        return Variable(self.name, self.original_edges, self.edges)

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

    def __eq__(self, value: Tensor) -> bool:
        if not isinstance(value, Variable):
            return False
        return self.name == value.name and self.edges == value.edges

    def __hash__(self):
        return hash(("Variable", self.name) + tuple(self.edges))

    def simplify(self, full=False):
        return self

    def rename(self, **kwargs: dict[str, str]):
        v = Variable(
            self.name,
            self.original_edges,
            [kwargs.get(e, e) for e in self.edges],
        )
        return v


class Constant(Tensor):
    def __init__(self, edges: Iterable[str]):
        super().__init__()
        self.edges = list(edges)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.edges})"

    def __hash__(self):
        # TODO: Sort the edges here? Or should the order matter? Should the names even matter?
        return hash((type(self).__name__,) + tuple(sorted(self.edges)))

    def copy(self):
        return type(self)(self.edges)

    def rename(self, **kwargs: dict[str, str]):
        return type(self)([kwargs.get(e, e) for e in self.edges])

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = ensure_edges_unused(self, x, new_names)
        return Zero(self.edges + new_names)


class Copy(Constant):
    # The "Copy" tensor is defined by C_{i,j,k} = 1 if i == j == k, else 0
    # Or alternatively as Δₙ = ∑ₓ (eₓ)⊗ⁿ, where are e_i are basis vectors
    # For rank 2, this is the identity matrix. For higher rank, the identity tensor is
    # the product of n identity matrices (rank 2 Copy's).
    pass


class Zero(Constant):
    # Matrix such that Z_{i,j,k} = 0 for all i, j, k
    pass


class Ones(Constant):
    # Matrix such that O_{i,j,k} = 1 for all i, j, k
    def simplify(self, full=False):
        if self.rank <= 1:
            return Copy(self.edges)
        return self


class Function(Tensor):
    def __init__(
        self,
        name,
        tensors: Iterable[Tensor],
        edges_in: Iterable[str],
        edges_out: Iterable[str],
    ):
        self.name = name
        # TODO: We are currently assuming just one input edge per input tensor.
        self.tensors = list(tensors)
        self.edges_in = list(edges_in)
        for t, e in zip(self.tensors, self.edges_in):
            assert e in t.edges, f"Edge {e} not in tensor {t}"
        self.edges = list(edges_out)

    def copy(self):
        return Function(self.name, [t.copy() for t in self.tensors], self.edges_in, self.edges)

    def rename(self, **kwargs: dict[str, str]):
        # We only rename on the surface
        return Function(
            self.name,
            self.tensors,
            self.edges_in,
            [kwargs.get(e, e) for e in self.edges],
        )

    def simplify(self, full=False):
        return Function(
            self.name,
            [t.simplify(full) for t in self.tensors],
            self.edges_in,
            self.edges,
        )

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = ensure_edges_unused(self, x, new_names)
        # We sum over each input function, just like the normal chain rule:
        # d/dx f(g₁(x), …, gₖ(x)) = Σᵢ₌₁ᵏ (d/dx gᵢ(x)) Dᵢf(g₁(x), …, gₖ(x))
        # D_i adds a new output edge to the function, which is contracted with
        # the normal output edge of the tensor. So we need to make sure this doesn't
        # clash with an existing output edge of f.
        new_edge = "i"
        while new_edge in self.edges:
            new_edge += "'"
        return Sum(
            Product(
                [
                    Function(
                        self.name + f"_d{i}",
                        self.tensors,
                        self.edges_in,
                        self.edges + [new_edge],
                    ),
                    t.rename(**{e: new_edge}).grad(x, new_names),
                ]
            )
            for i, (t, e) in enumerate(zip(self.tensors, self.edges_in))
        )

    def __hash__(self):
        return hash(
            ("Function", self.name)
            + tuple(map(hash, self.tensors))
            + tuple(self.edges_in)
            + tuple(self.edges)
        )

    def __repr__(self):
        return f"Function({self.name}, {self.tensors}, {self.edges_in}, {self.edges})"


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

    def copy(self):
        return Product([t.copy() for t in self.tensors])

    def rename(self, **kwargs: dict[str, str]):
        # We need to rename everything (not just on the surface), since one of the applications of renaming
        # is to combine nested products, and then we need to avoid clashes between "internal" edges in each product.
        return Product([t.rename(**kwargs) for t in self.tensors])

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = ensure_edges_unused(self, x, new_names)
        return Sum(
            [
                Product(self.tensors[:i] + [t.grad(x, new_names)] + self.tensors[i + 1 :])
                for i, t in enumerate(self.tensors)
            ]
        )

    def __repr__(self):
        return f"Product({self.tensors})"

    def __hash__(self):
        return hash(("Product",) + tuple(sorted(map(hash, self.tensors))))

    def simplify(self, full=False):
        tensors = [t.simplify(full=full) for t in self.tensors]
        # If any tensor in a product is 0, so is the whole product
        if any(isinstance(t, Zero) for t in tensors):
            return Zero(self.edges)
        # Combine nested contractions:
        children = []
        for i, t in enumerate(tensors):
            # TODO: Consider not combining products, since we might want to keep the "natural" contraction order.
            # Maybe this should be the meaning of full=True/False?
            # It could be that simplify can take more complicated arguments, to also control whether we should
            # use the distributive law to push products to the bottom etc.
            if not isinstance(t, Product):
                children.append(t)
                continue
            for t0 in t.tensors:
                # We need to rename the hidden edges to avoid clashes
                # TODO: This kind of renaming is a hack: e_i could have already been used.
                # We need a more general solution.
                children.append(t0.rename(**{e: f"{e}_{i}" for e in t.contractions}))
        # Remove identity matrices, connected to something.
        # Note this also covers the higher rank identity tensors, which are just products of these.
        while True:
            cnt_edges = Counter(e for t in self.tensors for e in t.edges)
            I2 = next(
                (
                    t
                    for t in children
                    if isinstance(t, Copy)
                    and len(t.edges) == 2
                    and (cnt_edges[t.edges[0]] == 2 or cnt_edges[t.edges[1]] == 2)
                ),
                None,
            )
            if I2 is None:
                break
            # Note that it's possible that I2 is connected in both ends.
            # in that case, we must only rename one side.
            e0, e1 = I2.edges
            rename = {e0: e1} if any(e0 in t.edges for t in children) else {e1: e0}
            children = [t.rename(**rename) for t in children if t is not I2]
        # Remove empty Identities and Ones
        children = [t for t in children if not (isinstance(t, (Copy, Ones)) and t.rank == 0)]
        # Base cases
        if not children:
            raise NotImplementedError("Not sure what to do here yet. But it also shouldn't happen")
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

    def copy(self):
        return Sum([t.copy() for t in self.tensors], self.weights)

    def rename(self, **kwargs: dict[str, str]):
        return Sum([t.rename(**kwargs) for t in self.tensors], self.weights)

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = ensure_edges_unused(self, x, new_names)
        return Sum([t.grad(x, new_names) for t in self.tensors], self.weights)

    def simplify(self, full=False):
        # Identify tensors with multiplicity and combine them
        ws_tensors = defaultdict(int)
        for w, t in zip(self.weights, self.tensors):
            t = t.simplify(full=full)
            if isinstance(t, Sum):
                for w1, t1 in zip(t.weights, t.tensors):
                    ws_tensors[t1] += w * w1
            else:
                ws_tensors[t] += w
        # Remove zero tensors or zero weights
        # TODO: It seems this might change the set of edges. Is that a problem?
        # Or are we OK with some operations removing edges?
        # Like, should (x-x) have the same shape as x? or should it have no edges?
        ws_tensors = [(w, t) for t, w in ws_tensors.items() if w != 0 and not isinstance(t, Zero)]
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
