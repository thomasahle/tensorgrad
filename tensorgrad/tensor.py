from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
import math
from typing import Any, Generator, Iterable, Optional
from abc import ABC
from fractions import Fraction
from numbers import Number
import networkx as nx
from sympy import Symbol
import torch


class Tensor(ABC):
    @property
    def edges(self) -> set[str]:
        """Returns an _ordered_ set of edge names"""
        return self.shape.keys()

    @property
    def shape(self) -> dict[str, Symbol]:
        if not hasattr(self, "_shape"):
            raise NotImplementedError
        return self._shape

    @property
    def order(self):
        """The number of edges the tensor has."""
        return len(self.shape)

    def grad(self, x: "Variable", new_names: Optional[dict[str, str]]) -> "Tensor":
        """
        Take the derivative of this tensor with respect to the variable x.

        Args:
            x: The variable to take the derivative with respect to.
            new_names: Optional list of names to use for the new edges created by the derivative.
                If not provided, new names will be generated based on the edges of x.

        Note:
            Pushes the derivative one step through the tensor.
            If you want to push it all the way through, use .simplify().

        Returns:
            The tensor representing the derivative.
        """
        new_names = self._check_grad(x, new_names)
        raise NotImplementedError

    def rename(self, **kwargs: dict[str, str]) -> "Tensor":
        """
        Rename free edges of this tensor.

        Args:
            kwargs: A dictionary mapping old edge names to new edge names.
                Only free edges can be renamed. Inner edges may get renamed
                if necessary to avoid clashes with the new free edge names.

        Returns:
            A new tensor with the edges renamed according to kwargs.
        """
        kwargs = self._check_rename(kwargs)
        raise NotImplementedError

    def simplify(self, args: dict[str, Any] = None) -> "Tensor":
        """
        Apply simplification rules to this tensor.

        This may rename inner edges but should never change the free edges.

        Args:
            args: Optional dictionary of arguments controlling the simplification.

        Returns:
            A simplified version of this tensor.
        """
        return self

    def full_simplify(self) -> "Tensor":
        """Applies multiple simplification rules until the expression no longer changes"""
        expr = self
        while (new := expr.simplify()) != expr:
            expr = new
        expr = expr.simplify({"expand": True})
        while (new := expr.simplify()) != expr:
            expr = new
        return expr

    def __hash__(self) -> int:
        return hash(self.weisfeiler_lehman)

    @cached_property
    def weisfeiler_lehman(self) -> str:
        """Hexadecimal string for WL-hash of the input graph."""
        G, _ = self.edge_structural_graph(match_edges=False)
        return nx.algorithms.weisfeiler_lehman_graph_hash(G, node_attr="name")

    def __eq__(self, other) -> bool:
        return self.is_isomorphic(other)

    def depends_on(self, x: "Variable") -> bool:
        """Check if this tensor depends on the variable x."""
        raise NotImplementedError

    @cached_property
    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, str]]:
        """Create a graph representation of the tensor, which can be used for isomorphism testing.

        Returns:
            A tuple with the following values:
            - A NetworkX directed graph.
                - Each node in the top tree is labeled with the producing tensor.
                - Use name=hashable to label vertices
            - "edges", a dict of edge_name -> node id
            - Node 0 should be the root
        """
        raise NotImplementedError

    def edge_structural_graph(
        self, match_edges=True, edge_names: None | dict[str, str] = None
    ) -> nx.MultiDiGraph:
        """Like structural_graph(), but adds dummy nodes for the outer edges.

        Args:
            if match_edges is True, the edge names are used to match the edges of the two tensors.
            if edge_names is given, it is used to rename the edges of the tensor.
        """
        if edge_names is None:
            edge_names = {}

        G, edges = self.structural_graph()

        for e in edges.keys():
            if e not in edge_names:
                edge_names[e] = ("Outer Edge", e) if match_edges else ""

        for e, node in edges.items():
            n = G.number_of_nodes()
            G.add_node(n, name=edge_names[e])
            G.add_edge(node, n)
        return G, list(edges.keys())

    def graph_to_string(self):
        """Returns an ASCII tree-like representation of the structural graph."""
        G, _ = self.edge_structural_graph(match_edges=True)
        return "\n".join(nx.generate_network_text(G, with_labels="name", sources=[0]))

    # Overloaded ops

    def __add__(self, other) -> "Tensor":
        w = 1
        if isinstance(other, Number):
            other, w = Product([]), other
        return Sum([self, other], [1, w])

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __rsub__(self, other) -> "Tensor":
        # Handle `other - self`
        return -self + other

    def __sub__(self, other) -> "Tensor":
        # Handle `self - other`
        w = 1
        if isinstance(other, Number):
            other, w = Product([]), other
        return Sum([self, other], [1, -w])

    def __neg__(self) -> "Tensor":
        return Sum([self], [-1])

    def __matmul__(self, other) -> "Tensor":
        if isinstance(other, Number):
            other = Sum([Product([])], [other])
        return Product([self, other])

    def __rmatmul__(self, other):
        if isinstance(other, Number):
            return self * other
        return self @ other

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __mul__(self, other) -> "Tensor":
        """Contract self and other, but use a 3d-identity to keep the shared edges free."""
        if isinstance(other, Number):
            if other == 0:
                return Zero(**self.shape)
            if other == 1:
                return self
            return Sum([self], [other])
        # Element-wise (Hadamard) product is easy to implement using Copy tensors
        # These are the edges we multiply over
        shared_edges = self.edges & other.edges
        (t0, t1), (rename0, rename1) = make_distinct(self, other, used_names=shared_edges)
        return Product([t0, t1] + [Copy(self.shape[e], e, rename0[e], rename1[e]) for e in shared_edges])

    def __rtruediv__(self, other) -> "Tensor":
        # Handle other / self
        from tensorgrad.functions import pow  # Avoid circular import

        return other * pow(self, -1)

    def __truediv__(self, other) -> "Tensor":
        # Handle self / other
        from tensorgrad.functions import pow  # Avoid circular import

        if isinstance(other, int):
            return Sum([self], [Fraction(1, other)])
        if isinstance(other, Number):
            return Sum([self], [1 / other])
        return self * pow(other, -1)

    def __pow__(self, other):
        from tensorgrad.functions import pow  # Avoid circular import

        if not isinstance(other, int):
            raise ValueError("Only integer powers are supported.")
        return pow(self, other)

    def is_isomorphic(
        self, other: "Tensor", match_edges=False, edge_names: None | dict[str, str] = None
    ) -> bool:
        if self.weisfeiler_lehman != other.weisfeiler_lehman:
            return False
        G1, _ = self.edge_structural_graph(match_edges=match_edges, edge_names=edge_names)
        G2, _ = other.edge_structural_graph(match_edges=match_edges, edge_names=edge_names)
        return nx.is_isomorphic(G1, G2, node_match=lambda n1, n2: n1.get("name") == n2.get("name"))

    def isomorphisms(self, other: "Tensor") -> Generator[dict[str, str], None, None]:
        """Given self and other are isomorphic, this method returns a dictionary that renames self into other."""
        # We need the edges1 and edges2 lists to keep track of the order of edges added to the graph
        G1, edges1 = self.edge_structural_graph(match_edges=False)
        G2, edges2 = other.edge_structural_graph(match_edges=False)
        for matching in nx.algorithms.isomorphism.MultiDiGraphMatcher(
            G1, G2, node_match=lambda n1, n2: n1.get("name") == n2.get("name")
        ).isomorphisms_iter():
            # Matching is a dict {i: j} where i is the node in G1 and j is the node in G2
            # We are only interested in then `len(self.edges)` last nodes, which correspond to the outer edges
            start_i = G1.number_of_nodes() - len(self.edges)
            start_j = G2.number_of_nodes() - len(self.edges)
            yield {
                edges1[i - start_i]: edges2[j - start_j]
                for i, j in matching.items()
                if i >= start_i and j >= start_j
            }

    @cached_property
    def symmetries(self) -> set[frozenset[str]]:
        """Return the orbits of the automorphism group of the tensor."""
        # The orbits are the sets of edges that ever get mapped to each other.
        G = nx.Graph([(i, j) for mapping in self.isomorphisms(self) for i, j in mapping.items()])
        symmetries = set(map(frozenset, nx.connected_components(G)))
        if hasattr(self, "_symmetries"):
            assert symmetries == self._symmetries, f"{symmetries=} {self._symmetries=}"
        return symmetries

    def evaluate(
        self,
        values: dict["Variable", torch.Tensor],
        dims: dict[Symbol, int] | None = None,
    ) -> torch.Tensor:
        """
        Evaluate this tensor given values for the variable tensors.

        Args:
            values: A dict mapping variable tensors to their values.
            dims: Optional dict specifying the dimensions of free edges.
        """
        if dims is None:
            dims = {}

        # Check dimension consistency for all variables pinned in `values`
        for v, t in values.items():
            if not isinstance(v, Variable):
                continue
            old_to_new = {o: e for e, o in v.orig.items()}

            # For each name 'o' in the actual PyTorch tensor 't',
            # we see if 'o' is a known edge in `v.orig`.
            for o, ts in zip(t.names, t.shape):
                # If 'o' is not an original edge, skip.
                # This handles e.g. 'samples' dimension which is external.
                if o not in old_to_new:
                    continue

                # Otherwise check consistency
                vs = v.shape[old_to_new[o]]  # symbolic dimension
                if vs not in dims:
                    dims[vs] = ts
                elif dims[vs] != ts:
                    raise ValueError(f"Conflicting size for dim {o}")

        # See if we have already evaluated a tensor isomorphic to this one
        if self in values:
            # Find the isomorphic representative that we matched
            other, tensor = next((v, t) for v, t in values.items() if v.is_isomorphic(self))
            mapping = next(other.isomorphisms(self), None)
            # The fix is just this "..." before *self.edges
            res = tensor.rename(**mapping).align_to(..., *self.edges)

            if __debug__:
                expected = self._inner_evaluate(values, dims)
                # We guarantee that inner_evaluate returns the edges in the same order as self.edges,
                # and res has had the order forced on it by align_to.
                assert expected.names == res.names, f"{expected.names=} {res.names=} {self.edges=}"
                torch.testing.assert_close(res.rename(None), expected.rename(None))
            return res

        res = self._inner_evaluate(values, dims)
        # We guarantee that inner_evaluate returns the edges in the same order as self.edges
        assert res.names == tuple(self.edges), f"Expected {self.edges=} but got {res.names=}"
        values[self] = res
        return res

    def _inner_evaluate(self, values: dict["Tensor", torch.Tensor], dims: dict[Symbol, int]) -> torch.Tensor:
        """
        The inner implementation of tensor evaluation.

        Subclasses should override this to define the actual evaluation logic.

        Args:
            values: A dictionary mapping variable tensors to their values.
            dims: A dictionary mapping edge names to their dimensions.

        Returns:
            The result of evaluating this tensor.
        """
        raise NotImplementedError

    def _check_rename(self, kwargs: dict[str, str]):
        """Check that the renaming is valid, and return the renaming dictionary."""
        if len({kwargs.get(e, e) for e in self.edges}) != len(self.edges):
            raise ValueError(f"Renamed an edge to an existing edge name. {self.edges=} {kwargs=}")
        # Restrict renaming to free edges
        return {new: old for new, old in kwargs.items() if new in self.edges}

    def _check_grad(self, x: "Variable", new_names: Optional[dict[str, str]] = None) -> dict[str, str]:
        """
        When taking a derivative with respect to a variable, we need some edge names for the derivative.
        If new_names is not given, we generate names based on the edge names of x.
        However, we need to make sure they don't clash with any names already present in the tensor.
        If new_names is already given, we just check to make sure they are not already present in the tensor.
        """
        if new_names is not None:
            if set(x.orig.values()) != set(new_names.keys()):
                raise ValueError(f"The {new_names.keys()=} must match {x.orig.values()=}")
            # Check that the new names don't clash with self.edges
            if used := self.edges & new_names.values():
                raise ValueError(f"{used} are already present in {self.edges=}")
            return new_names
        # Make new names that are _based_ on x.orig and _avoid_ self.edges
        # This is important, since different instantiations of x might be using different names.
        return unused_edge_names(x.orig.values(), self.edges, suffix="_")
        # Why not base the new_names on x.edges? They are the publicly known ones after all...
        # return unused_edge_names(x.edges, self.edges, suffix="_")

    @staticmethod
    def _check_simplify(args: dict[str, Any] | None = None):
        if args is None:
            args = {}
        # args["grad_steps"] allows us to control how far we propagate the derivative.
        args.setdefault("grad_steps", float("inf"))
        args.setdefault("associative_products", True)
        args.setdefault("associative_sums", True)
        args.setdefault("sum_combine_terms", True)
        args.setdefault("combine_products", True)
        args.setdefault("factor_components", True)
        args.setdefault("expand_functions", True)
        args.setdefault("expand", False)
        return args

    @staticmethod
    def _check_edges(edges: Iterable[str]) -> list[str]:
        if edges is None:
            return None
        if not isinstance(edges, Iterable):
            raise ValueError("Edges must be an iterable of strings")
        edges = list(edges)
        if not all(isinstance(e, str) for e in edges):
            raise ValueError("Edges must be an iterable of strings")
        if len(edges) == 1:
            edges = edges[0].split(",")
            return [e.strip() for e in edges]
        return edges

    @staticmethod
    def _check_shape(shape0: Iterable[Symbol], shape1: dict[str, Symbol]) -> dict[str, Symbol]:
        shape0 = tuple(shape0)
        if not isinstance(shape0, tuple) or not all(isinstance(s, Symbol) for s in shape0):
            raise ValueError("Shape0 must be a tuple of sympy symbols")
        if not isinstance(shape1, dict) or not all(isinstance(s, Symbol) for s in shape1.values()):
            raise ValueError("Shape1 must be a dict of sympy symbols")
        shape0 = {s.name: s for s in shape0}
        if double_keys := shape0.keys() & shape1.keys():
            raise ValueError(f"Duplicate edge names: {double_keys}")
        return shape0 | shape1

    @staticmethod
    def _check_symmetries(shape: dict[str, Symbol], symmetries: str | set[frozenset[str]]):
        if symmetries is None:
            return {frozenset({e}) for e in shape.keys()}
        if isinstance(symmetries, str):
            symmetries = {frozenset(word.split()) for word in symmetries.split(",")}
        # Check symmetries are sets of strings
        assert all(isinstance(e, str) for group in symmetries for e in group), f"{symmetries=}"
        if len({e for group in symmetries for e in group}) != len(shape):
            raise ValueError("Symmetry groups must contain all edges")
        # Check symmetries are compatible with shape
        for group in symmetries:
            if any(e not in shape for e in group):
                raise ValueError(f"Symmetry group {group} contains edges not in shape")
            if len({shape[e] for e in group}) != 1:
                raise ValueError(f"Symmetry group {group} must all have same dim size")
        return symmetries


################################################################################
# Variable
################################################################################


class Variable(Tensor):
    def __init__(
        self,
        name,
        *shape0: Symbol,
        _symmetries: None | str | set[frozenset[str]] = None,
        _orig: None | dict[str, str] = None,
        **shape1: Symbol,
    ):
        """
        A tensor holding a variable.

        Args:
            name: The name of this variable.
            shape: Dict from edge name to size (Sympy symbol). Can also be a string or list of str.
            orig: Dict from new name to original name. If not provided, the names in `edges` are used directly.
            symmetries: Sets of edges (original names) that should be considered equivalent.
        """
        self.name = name
        self._shape = self._check_shape(shape0, shape1)

        self._symmetries = self._check_symmetries(self._shape, _symmetries)

        # The original edges are saved so evaluation can happen with the original
        # edge names given to the variable, without caring about what renaming we
        # might have done in the meantime.
        self.orig = _orig if _orig is not None else {e: e for e in self.edges}
        assert self.orig.keys() == self.edges, f"Missing original name for {self.edges=} {self.orig=}"

    def with_symmetries(self, symmetries: str | set[frozenset[str]]) -> "Variable":
        return Variable(self.name, **self.shape, _symmetries=symmetries, _orig=self.orig)

    def grad(self, x: "Variable", new_names: Optional[dict[str, str]] = None):
        new_names = self._check_grad(x, new_names)
        if x == self:
            orig_shape = {self.orig[e]: s for e, s in self.shape.items()}
            other_shape = {x.orig[e]: s for e, s in x.shape.items()}
            assert orig_shape == other_shape
            # TODO: If X has symmetries, the derivative can actually be more complex than this.
            # See See 2.8.2 Symmetric in the Cookbook: https://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf
            # Note that new_names are based on _the original names_ of x, not the current names.
            return Product(Copy(s, e, new_names[self.orig[e]]) for e, s in self.shape.items())
            # iso_rename = next(self.isomorphisms(x))
            # return Product(Copy(s, iso_rename[e], new_names[e]) for e, s in self.shape.items())
        # Note: We don't need to tell Zero the symmetries, since it's automatically
        # symmetric in all dimensions that have compatible sizes.
        return Zero(**(self.shape | {new_names[e]: s for e, s in x.shape.items()}))

    def __repr__(self):
        args = [f"\"{self.name}\", {', '.join(self.edges)}"]
        symmetries = ""
        if self._symmetries != {frozenset({e}) for e in self.edges}:
            groups = ", ".join(sorted(" ".join(sorted(group)) for group in self._symmetries))
            symmetries = f'.with_symmetries("{groups}")'
        if any(k != v for k, v in self.orig.items()):
            args.append(f"orig={self.orig}")
        return f"Variable({', '.join(args)}){symmetries}"

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        name = f"{self.name}({', '.join(sorted(self.orig.values()))})"
        G.add_node(0, name=("Variable", name), tensor=self)
        edges = {}
        # Symmetries are more fine-grained than shapes, since two dims can have
        # the same size, but not be symmetric. E.g. an assymetric square matrix.
        # Example of variable with 2 sizes, 3 symmetri orbits and 4 edges:
        # +-size 1-sym 1-e 1
        # |  +-----sym 2-e 2
        # +-size 2-sym 3-e 3
        #           +----e 4
        for size in set(self.shape.values()):
            # Symbols are identified by name and assumptions. We might want to have edges
            # with the same name but different assumptions, so add the id to the node name.
            G.add_node(size_node := G.number_of_nodes(), name=(f"size={size.name}", id(size)))
            G.add_edge(0, size_node)
            # Find each orbit with the given size (see note above about fine-grained-ness)
            for orbit in self._symmetries:
                e, *_ = orbit
                if self.shape[e] != size:
                    continue
                orbit_name = " ".join(sorted(self.orig[e] for e in orbit))
                G.add_node(orbit_node := G.number_of_nodes(), name=("Orbit Node", orbit_name))
                G.add_edge(size_node, orbit_node)
                # All the free edges point to an orbit node
                for e in orbit:
                    edges[e] = orbit_node
        assert edges.keys() == self.edges
        return G, edges

    def simplify(self, args: dict[str, Any] = None):
        return self

    def rename(self, **kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)  # Checks only free edges are in kwargs
        return Variable(
            self.name,
            **{kwargs.get(e, e): s for e, s in self.shape.items()},
            _symmetries={frozenset({kwargs.get(e, e) for e in group}) for group in self._symmetries},
            _orig={kwargs.get(e, e): o for e, o in self.orig.items()},
        )

    def depends_on(self, x: "Variable") -> bool:
        return x == self

    def _inner_evaluate(self, values: dict["Tensor", torch.Tensor], dims: dict[Symbol, int]) -> torch.Tensor:
        tensor = values.get(self)
        if tensor is None:
            raise ValueError(f"Missing value for {self}, got {values}")
        # The given named tensor will use original names. We rename the new names
        assert set(tensor.names) == set(self.orig.values()), f"{tensor.names=} != {set(self.orig.values())}"
        old_to_new = {o: e for e, o in self.orig.items()}
        return tensor.rename(*[old_to_new[e] for e in tensor.names])


################################################################################
# Constants
################################################################################


class Constant(Tensor, ABC):
    def __init__(self, *shape0: Symbol, _symmetries: set[frozenset[str]] = None, **shape1: Symbol):
        """
        A constant tensor with the given edges.

        Args:
            edges: The names of the edges of this constant tensor.
            link: An optional variable that this tensor is associated with, used to compute edge dimensions.
        """
        self._shape = self._check_shape(shape0, shape1)
        self._symmetries = self._check_symmetries(self._shape, _symmetries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.shape})"

    def with_symmetries(self, symmetries: str | set[frozenset[str]]) -> "Variable":
        return type(self)(self.name, **self._shape, _symmetries=symmetries)

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=type(self).__name__, tensor=self)
        edges = {}
        for size in set(self.shape.values()):
            # All of this is more or less equal to Variable
            G.add_node(size_node := G.number_of_nodes(), name=f"size={size.name}({id(size)})")
            G.add_edge(0, size_node)
            # Find each orbit with the given size (see note above about fine-grained-ness)
            for orbit in self._symmetries:
                e, *_ = orbit
                if self.shape[e] != size:
                    continue
                G.add_node(orbit_node := G.number_of_nodes(), name="Orbit Node")
                G.add_edge(size_node, orbit_node)
                for e in orbit:
                    edges[e] = orbit_node
        return G, edges

    def rename(self, **kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        c = type(self)(**{kwargs.get(e, e): s for e, s in self.shape.items()})
        assert c.edges == {kwargs.get(e, e) for e in self.edges}
        return c

    def grad(self, x: Variable, new_names: Optional[dict[str, str]] = None):
        new_names = self._check_grad(x, new_names)
        return Zero(**(self.shape | {new_names[e]: s for e, s in x.shape.items()}))

    def depends_on(self, x: "Variable") -> bool:
        return False


class Copy(Constant):
    """The "Copy" tensor is defined by C_{i,j,k} = 1 if i == j == k, else 0
    Or alternatively as Δₙ = ∑ₓ (eₓ)⊗ⁿ, where are e_i are basis vectors
    For order 2, this is the identity matrix.
        Note: For higher order, the identity tensor is the product of n identity matrices,
        (order 2 Copy's), not the order-n Copy tensor.
    For order 0, the Copy tensor is defined as |n|, where n is the size/dimension of the tensor.
    """

    def __init__(self, size: Symbol, *edges: str):
        edges = self._check_edges(edges)
        super().__init__(**{e: size for e in edges})
        assert isinstance(size, (Symbol, Number)), "Size must be a sympy symbol"
        self._size = size

    @property
    def size(self):
        return self._size

    def __repr__(self):
        if not self.edges:
            return f"Copy({self.size})"
        return f"Copy({self.size}, \"{', '.join(self.edges)}\")"

    def _inner_evaluate(self, values: dict["Tensor", torch.Tensor], dims: dict[Symbol, int]) -> torch.Tensor:
        size = dims[self.size]
        if not self.edges:
            return torch.tensor(size)
        copy = torch.zeros([size] * self.order)
        for idx in range(size):
            copy[(idx,) * len(self.edges)] = 1
        return copy.rename(*self.edges)

    def rename(self, **kwargs: dict[str, str]):
        return Copy(self.size, *[kwargs.get(e, e) for e in self.edges])

    @classmethod
    def simplify_outer(cls, tensors: list[Tensor]) -> list[Tensor]:
        """Simplifies a list of tensors assumed to be a product."""
        while True:
            tensors, done = cls._simplify_step(tensors)
            if done:
                break
        return tensors

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=type(self).__name__, tensor=self)
        G.add_node(size_node := G.number_of_nodes(), name=f"size={self.size.name}({id(self.size)})")
        G.add_edge(0, size_node)
        return G, {e: size_node for e in self.edges}

    ################################################################################
    # There are many simplify rules for Copy. We split them into separate methods for clarity.

    @classmethod
    def _simplify_step(cls, tensors: list[Tensor]) -> tuple[list[Tensor], bool]:
        """Performs one step of simplification. Returns a new list if changed, or the original if not."""
        for e, ts in group_edges(tensors).items():
            if len(ts) == 1:
                continue
            t1, t2 = ts
            assert t1.shape[e] == t2.shape[e], f"{t1.shape[e]=} != {t2.shape[e]}, {e=}"

            # Remove t1 and t2 from tensors.  We use the "is" operator, since equality
            # means isomorphic, so it might remove too many unintended tensors.
            other = [t for t in tensors if t is not t1 and t is not t2]

            for simplification in [cls._merge_copy_tensors, cls._remove_identity_matrix]:
                if (new := simplification(t1, t2, e)) is not None:
                    return other + new, False

        return tensors, True

    @staticmethod
    def _merge_copy_tensors(t1: Tensor, t2: Tensor, e: str) -> Optional[list[Tensor]]:
        if not (isinstance(t1, Copy) and isinstance(t2, Copy)):
            return None

        # We can't merge order 0 copy tensors, since we now give them a value equal to their size.
        # In principle we could create a new Copy(size1 * size2, []) tensor, but then we start having
        # arbitrary expressions as sizes, which I'm not sure we want yet.
        # Also, merging an order 0 with and order > 0 was never going to work, unless the size of the
        # order 0 Copy was 1, which it probably never is.
        # In either case, this method should only be called if the tensors share an edge, which they
        # can't if one of them has order 0.
        assert t1.order != 0 and t2.order != 0, "Can't merge order 0 Copy tensors"

        # Since the tensors are connected, we can assume they have the same size
        assert t1.size == t2.size, "Contracted Copy tensors must have same size"
        size = t1.size

        # We don't just remove e, but remove all shared edges
        # The amazing thing is that even in the case where all edges disappear, we
        # still get to keep information on the "size" of the Copy tensor.
        return [Copy(size, *(t1.edges ^ t2.edges))]

    @staticmethod
    def _remove_identity_matrix(t1: Tensor, t2: Tensor, e: str) -> bool:
        # If both are Copy's, we use the previous method
        if isinstance(t1, Copy) and isinstance(t2, Copy):
            return None

        # Make t1 the identity matrix
        if isinstance(t2, Copy) and t2.order == 2:
            t1, t2 = t2, t1

        if isinstance(t1, Copy) and t1.order == 2:
            # Find the edge of t1 that's not e
            other_edge = next(iter(set(t1.edges) - {e}))
            # Don't create self loops. We never connect a tensor to itself.
            # Unless it's another Copy, in which case we already handled it above.
            if other_edge not in t2.edges:
                return [t2.rename(**{e: other_edge})]
        return None


class Zero(Constant):
    """Matrix such that Z_{i,j,k} = 0 for all i, j, k"""

    def _inner_evaluate(self, values: dict["Tensor", torch.Tensor], dims: dict[Symbol, int]) -> torch.Tensor:
        return torch.zeros([dims[s] for s in self.shape.values()]).rename(*self.edges)


def Ones(*shape0: Symbol, **shape1: Symbol) -> Tensor:
    """Matrix such that O_{i,j,k} = 1 for all i, j, k"""
    # Implemented in practice by a simple outer product of Copy's
    shape = Tensor._check_shape(shape0, shape1)
    return Product([Copy(s, e) for e, s in shape.items()])


################################################################################
# Function
################################################################################


@dataclass
class FunctionSignature(ABC):
    """
    An abstract base class representing a function that takes one or more tensors as input
    and produces a new tensor as output.

    This class is designed to capture the symbolic structure of a tensor operation by
    listing its output edges (`edges`) and the input edges (`inputs`) each of its input
    tensors must provide. Subclasses are required to implement the `eval` and `derivative`
    methods, while `simplify` has a default no-op implementation that may be overridden.

    Parameters
    ----------
    name : str
        A descriptive name for the function (e.g., "matmul", "unfold", "ce", "attention").
    edges : Set[str]
        The output edges of this function. These are string identifiers that symbolically
        represent the dimensions of the output tensor.
    inputs : Tuple[Set[str], ...]
        A tuple where each element is a set of edges required from one input tensor.
        For example, if the first input tensor must share dimension edges {"b", "i"} and
        the second input tensor must share dimension edges {"b", "j"}, then
        `inputs = ({"b", "i"}, {"b", "j"})`.

    Examples
    --------
    1) **Matrix multiplication**:
       Suppose `x` has shape (b, i, j) and `y` has shape (b, j, k). We might define:

       .. code-block:: python

          f = MatmulSignature(
              name="matmul",
              edges={"i", "k"},
              inputs=({"i", "j"}, {"j", "k"})
          )

       Here, the function consumes edges `i` and `j` from `x` and edges `j` and `k` from `y`.
       The edge `b` is shared between the two inputs and broadcasted. The output tensor
       eventually has shape (b, i, k).

    2) **Unfold**:
       Suppose we have a tensor `x` with shape (b, c, w, h) and we want to produce a series
       of patches. We might define:

       .. code-block:: python

          f = UnfoldSignature(
              name="unfold",
              edges={"patch", "c_out", "w_out", "h_out"},
              inputs=({"c_in", "w_in", "h_in"},)
          )

       This function consumes edges ``c, w, h`` from `x` and introduces a new edge
       named ``patch`` of size (w_in - w_out + 1)*(h_in - h_out + 1).

    3) **Cross-entropy**:
       This is an example of a function that takes two inputs, for example logits and
       probabilities. Suppose:
       - `x` has shape (b, logits)
       - `y` has shape (b, probs)

       We might define:

       .. code-block:: python

          f = CrossEntropySignature(
              name="ce",
              edges={"ce"},
              inputs=({"logits"}, {"probs"})
          )

       Both inputs share the batch edge ``b``. The new edge ``ce`` could represent the
       cross-entropy output dimension (often scalar per example, so shape (b, ce)).

    4) **Multi-query Dot Product Attention**:
       Suppose we have tensors:
       - `q` with shape (b, n_q, d)
       - `k` with shape (b, seq, d)
       - `v` with shape (b, seq, d_out)

       Conceptually, to compute multi-query attention:

       - We form (b, n_q, seq) via a dot product of `q` and `k`.
       - We apply a softmax to get (b, n_q, prob).
       - We multiply by `v` to get (b, n_q, d_out).

       A possible signature might be:

       .. code-block:: python

          f = AttentionSignature(
              name="attention",
              edges={"n_q", "d_out"},
              inputs=(
                  {"n_q", "d"},
                  {"seq", "d"},
                  {"seq", "d_out"},
              )
          )

       The edge ``b`` is shared among all inputs (broadcast). The final output
       has shape (b, n_q, d_out).
    """

    name: str
    # The output edges of the function. Note no Symbols here, since we don't know the sizes yet.
    edges: set[str]
    # Defines the number of input tensors the function takes, and what edges it needs from each
    inputs: tuple[set[str], ...]

    # TODO: Functions should support symmetric outputs, just like Constant and Variable.
    # Actually we could define symmetries for the inputs too, since the order doesn't always matter.

    def eval(self, *input_tensors: torch.Tensor) -> torch.Tensor:
        """
        Evaluates this function given a list of input torch tensors,
        corresponding to the edges specified in `inputs`.
        """
        raise NotImplementedError

    def derivative(self, i: int, new_edges: dict[str, str] = None) -> "FunctionSignature":
        """
        Returns a new FunctionSignature that represents the derivative
        with respect to the i-th input, and where edges are created
        according to `new_edges` a dict {old_name : new_name}.
        """
        if new_edges is None:
            new_edges = {}
        if self.inputs[i] != new_edges.keys():
            raise ValueError(f"Expected {self.inputs[i]} but got {new_edges.keys()}")
        if self.edges & new_edges.values():
            raise ValueError(f"New edges {new_edges.values()} clash with {self.edges=}")
        return FunctionSignature(f"D_{i}{self.name}", self.edges | new_edges.values(), self.inputs)

    def simplify(self, t: "Function", args: dict[str, Any]) -> Tensor:
        """
        Simplifies the Function tensor that uses this signature.
        Default implementation returns `t` unchanged.
        Subclasses can override this method for custom simplifications.
        """
        return t

    def __repr__(self):
        return f"FunctionSignature('{self.name}', {set(self.edges)}, {[set(s) for s in self.inputs]})"


def function(
    name,
    output_shape: dict[str, Symbol],
    *inputs: tuple[Tensor | str, ...],
    orig_out: dict[str, str] = None,
) -> "Function":
    return Function(
        FunctionSignature(
            name,
            output_shape.keys(),
            [set(es) for (t, *es) in inputs],
        ),
        [t for (t, *es) in inputs],
        output_shape,
        orig_out=orig_out,
    )


class Function(Tensor):
    def __init__(
        self,
        signature: FunctionSignature,
        inputs: Iterable[Tensor],
        shape_out: dict[str, Symbol],
        orig_out: dict[str, str] = None,
    ):
        """
        A function tensor that takes one or more input tensors and produces an output tensor.

        Args:
            signature: The FunctionSignature defining this function, or a string giving the function name.
            inputs: The input tensors and their input edge names. [(t0, e00, e01, ...), (t1, e10, ...), ...]
            shape_out: The names of the output edges of this function.
            orig_out: If provided, a dict mapping each output edge to its signature edge name. (new -> old)
                      If not provided, we assume the output edges are the same as in the signature.
        """
        self.shape_out = self._check_shape([], shape_out)
        self.inputs = list(inputs)
        self.signature = signature

        # We need to keep track of the original edges of the function, since we might rename them.
        self.edges_out = self.shape_out.keys()
        # self.orig_out = {e: e for e in self.edges_out} if not orig_out else orig_out
        self.orig_out = {} if orig_out is None else dict(orig_out)
        self.orig_out |= {e: e for e in self.signature.edges if e not in self.orig_out.values()}
        # assert self.edges_out == self.orig_out.keys()
        assert self.orig_out.keys() - self.edges_out == set(), f"{self.orig_out=}, {self.edges_out=}"

        # Validation
        if (orig_mapped := {self.orig_out.get(e, e) for e in self.shape_out.keys()}) != signature.edges:
            raise ValueError(
                "The shape_out keys must match signature.edges, once mapped back by orig_out."
                + f" Got {orig_mapped=} != {signature.edges}"
            )
        if len(self.inputs) != len(signature.inputs):
            raise ValueError(f"Expected {len(signature.inputs)} inputs, got {len(self.inputs)}")
        for t, es in zip(self.inputs, signature.inputs):
            if es - t.edges:
                raise ValueError(f"Input edges {es} not subset of {t.edges=}")

        # Build final shape = union of output edges + broadcast edges from each input
        # Note that this approach pushes all the broadcasted edges to the end of the shape.
        # But we don't care about the order of the edges, so it's fine.
        self._shape = dict(self.shape_out)
        # We allow “broadcasting” for edges that are not in fn_sig.inputs[i].
        # So if an input has edges that are not used, they’re broadcasted.
        for t, es in zip(self.inputs, self.signature.inputs):
            # any edges in t.edges - es are "broadcast"
            broadcast_edges = t.edges - es
            if broadcast_edges & self._shape.keys():
                raise ValueError("Broadcast edges must not be shared between inputs")
            for b in broadcast_edges:
                self._shape[b] = t.shape[b]

    def rename(self, **kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        renamed_inputs = []
        for t, es in zip(self.inputs, self.signature.inputs):
            # Only rename external edges of input tensors.
            rename = {e: v for e, v in kwargs.items() if e not in es}
            renamed_inputs.append(t.rename(**rename))
        res = Function(
            self.signature,
            renamed_inputs,
            {kwargs.get(e, e): s for e, s in self.shape_out.items()},
            orig_out={kwargs.get(e, e): o for e, o in self.orig_out.items()},
        )
        assert res.edges == {kwargs.get(e, e) for e in self.edges}
        return res

    def simplify(self, args: dict[str, Any] = None):
        args = self._check_simplify(args)
        new_inputs = [t.simplify(args=args) for t in self.inputs]

        # Broadcasting can be pulled out of the function.
        pulled_out = []
        new_inputs2 = []
        for t, es in zip(new_inputs, self.signature.inputs):
            if isinstance(t, Product):
                new_prod = []
                for u in t.tensors:
                    if (
                        isinstance(u, Copy)
                        and u.order == 1
                        and list(u.edges)[0] in t.edges
                        and list(u.edges)[0] not in es
                    ):
                        pulled_out.append(u)
                    else:
                        new_prod.append(u)
                new_inputs2.append(Product(new_prod))
            else:
                new_inputs2.append(t)
        new_inputs = new_inputs2

        res = Function(self.signature, new_inputs, self.shape_out, self.orig_out)
        res = self.signature.simplify(res, args)

        if pulled_out:
            res = Product([res] + pulled_out)

        assert res.shape == self.shape, "Free edges should be preserved"
        return res

    def grad(self, x: Variable, new_names: Optional[dict[str, str]] = None):
        # First find the new names for the gradient edges. These will be guaranteed
        # to avoid any outer edges of the Function tensor as well as the edges of x.
        new_names = self._check_grad(x, new_names)
        new_edges = set(new_names.values())

        # We sum over each input function, just like the normal chain rule:
        # d/dx f(g₁(x), …, gₖ(x)) = Σᵢ₌₁ᵏ (d/dx gᵢ(x)) Dᵢf(g₁(x), …, gₖ(x))

        # D_i adds a new output edge to the function, which is contracted with
        # the normal output edge of the tensor. So we need to make sure this doesn't
        # clash with an existing output edge of f.
        parts = []
        for i, (t, input_edges) in enumerate(zip(self.inputs, self.signature.inputs)):
            # Take the derivative of the outer function
            # We need "connection" edges for each edge in input_edges. Mostly we could just use the same name
            # but they need to avoid clashing with "new_names" and the output edges of the tensor.
            connection_names = unused_edge_names(input_edges, self.edges | new_edges, suffix="_")
            connection_shape = {connection_names[e]: t.shape[e] for e in input_edges}
            # Just like simple calculus, we need the derivative of the outside times the derivative of the inside
            outside = Function(
                self.signature.derivative(i, connection_names),
                self.inputs,
                self.shape_out | connection_shape,
                self.orig_out,
            )
            # The derivative will have the same output edges as the function, but we may have been renamed
            outside = outside.rename(**{o: e for e, o in self.orig_out.items()})
            # Actually, now that we give orig_out to the derivative, we don't need to rename it here.
            assert (
                outside.edges == self.edges | connection_names.values()
            ), f"{outside.edges=}, {self.edges=}, {connection_names=}"

            # The the derivative of the inner function
            # We rename the (former) input edges to the connection edges, but keep the remaining edges
            # (which are part of self.edges) untouched.
            inner = Derivative(t.rename(**connection_names), x, new_names)

            # The two parts are then multiplied together on the connection names,
            # while broadcasted on their remaining shared edges.
            from .functions import dot  # Import here to avoid circular import

            part = dot(outside, inner, connection_names.values())
            parts.append(part)
        res = Sum(parts)
        assert res.edges == self.edges | new_edges, f"{res.edges} != {self.edges} | {new_edges}"
        return res

    def structural_graph(self) -> tuple[nx.MultiDiGraph, int, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=(type(self).__name__, self.signature.name), tensor=self)
        edges = {}
        # (1) We add a node for the "function" tensor itself
        # TODO: Why is this needed?
        G.add_node(1, name=("f", self.signature.name), tensor=self)
        G.add_edge(0, 1)
        # (2) We add a node for each output edge
        # TODO: We should group out-edges by the size-type, similarly to Variable/Constant
        # And of course when we add symmetries to outputs, we need to add that too.
        for i, (e, o) in enumerate(self.orig_out.items()):
            G.add_node(i + 2, name=("Original Edge Out", o))
            G.add_edge(1, i + 2)
            edges[e] = i + 2
        # (3) And we add nodes for all the input tensors
        for i, (t, input_edges) in enumerate(zip(self.inputs, self.signature.inputs)):
            # Compute graph from input tensor, and ensure it uses distinct node numbers
            # Tag the inputs with {i} since order matters for function inputs.
            G, t_edges = add_structural_graph(G, t, root_edge_label=f"{i}")
            # Connect tensor to function. We don't need to label here, since
            # the input tensor already has labeled its own edges as much as it needs to.
            for e in input_edges:
                G.add_edge(t_edges[e], 1)
            # Finally register free edges
            for e in t.edges:
                if e not in input_edges:
                    edges[e] = t_edges[e]
        return G, edges

    def __repr__(self):
        args = []
        args.append(repr(self.signature))
        args.append(", ".join(map(repr, self.inputs)))
        args.append(f"{self.shape_out}")
        if any(e != o for e, o in self.orig_out.items()):
            args.append(f"orig_out={self.orig_out}")
        return f"Function({', '.join(args)})"

    def _inner_evaluate(self, values: dict["Tensor", torch.Tensor], dims: dict[Symbol, int]) -> torch.Tensor:
        xvals = [t.evaluate(values, dims) for t in self.inputs]
        res = self.signature.eval(*xvals)
        # This is weird, I rename back from new names to old
        # but then I assume the names match self. That's backwards, no?
        # Should I use inverse orig out here?
        rename = {o: e for e, o in self.orig_out.items()}
        # res = res.rename(*(self.orig_out.get(e, e) for e in res.names))
        res = res.rename(*(rename.get(e, e) for e in res.names))
        res = res.align_to(*self.edges)  # self.signature might use a different order
        return res

    def depends_on(self, x: "Variable") -> bool:
        return any(t.depends_on(x) for t in self.inputs)


class Derivative(Tensor):
    def __init__(self, tensor: Tensor, x: Variable, new_names: Optional[dict[str]] = None):
        """
        A tensor representing the derivative of another tensor.

        Args:
            tensor: The tensor to take the derivative of.
            x: The variable to take the derivative with respect to.
            new_names: The names to use for the new edges created by the derivative. If not provided, they will be generated.
        """
        self.tensor = tensor
        self.x = x
        # _check_grad makes sure the new_names are not already present in self.edges.
        # But we haven't set self.edges yet, so we call it on tensor instead of self.
        self.new_names = tensor._check_grad(x, new_names)
        self._shape = tensor.shape | {self.new_names[o]: x.shape[e] for e, o in x.orig.items()}

    def simplify(self, args: dict[str, Any] = None):
        args = self._check_simplify(args)
        if not self.tensor.depends_on(self.x):
            return Zero(**self.shape)
        inner = self.tensor.simplify(args)
        if args["grad_steps"] == 0:
            # If grad_steps is 0, we pass the simplify through the derivative.
            res = Derivative(inner, self.x, self.new_names)
        else:
            args["grad_steps"] -= 1
            # Have to call simplify twice to avoid an infinite loop when stacking multiple derivatives.
            res = inner.grad(self.x, self.new_names).simplify(args)
        assert res.shape == self.shape, f"Shape changed from {self.shape} to {res.shape}"
        return res

    def grad(self, x: Variable, new_names: dict[str, str] | None = None) -> Tensor:
        new_names = self._check_grad(x, new_names)
        # To avoid an infinite loop, we let the grad pass through us, rather than creating a double derivative.
        res = Derivative(self.tensor.grad(x, new_names), self.x, self.new_names)
        assert res.edges == self.edges | new_names.values()
        return res

    def rename(self, **kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        # The free edges of Derivative are both the free edges of self.tensor and the new_names.
        # This is the only place where we need to rename the "internal edges" of the tensor.
        res = Derivative(
            self.tensor.rename(**kwargs),
            self.x,
            {o: kwargs.get(e, e) for o, e in self.new_names.items()},
        )
        assert res.edges == {kwargs.get(e, e) for e in self.edges}
        return res

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name="Derivative", tensor=self)
        edges = {}
        # We add a node for the "wrt" tensor
        G, x_edges = add_structural_graph(G, self.x, root_edge_label="self.x")
        # This might be controversial, but we'll have new_edges point to the respective edges in self.x
        x_old_to_new = {oe: e for e, oe in self.x.orig.items()}
        edges |= {e: x_edges[x_old_to_new[oe]] for oe, e in self.new_names.items()}
        # Then we add the differentiated tensor
        G, t_edges = add_structural_graph(G, self.tensor, root_edge_label="self.tensor")
        edges |= t_edges
        assert edges.keys() == self.edges
        return G, edges

    def __repr__(self):
        return f"Derivative({self.tensor}, {self.x}, {self.new_names})"

    def _inner_evaluate(self, values: dict["Tensor", torch.Tensor], dims: dict[Symbol, int]) -> torch.Tensor:
        # We could use numerical differentiation here...  But it would potentially require quite a lot of
        # evaluations, since we need to evaluate the tensor in all directions.
        raise ValueError("Derivative tensors cannot be evaluated directly. Please use simplify() first.")

    def depends_on(self, x: "Variable") -> bool:
        return self.tensor.depends_on(x)


################################################################################
# Product
################################################################################


class Product(Tensor):
    def __init__(self, tensors: Iterable[Tensor]):
        """
        A product of multiple tensors.

        Args:
            tensors: The tensors to multiply together.
        """
        self.tensors = list(tensors)
        self._shape = {}
        for edge, ts in group_edges(self.tensors).items():
            if len(ts) == 1:
                self._shape[edge] = ts[0].shape[edge]
            elif len(ts) == 2:
                s1, s2 = [t.shape[edge] for t in ts]
                if s1 != s2:
                    raise ValueError(f"Edge {edge} has different sizes, {s1} != {s2}")
            else:
                raise ValueError(
                    f"edge {edge} had multiplicity {len(ts)}. Use an identity tensor to combine multiple edges."
                )

    def rename(self, **kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        # Rename the inner edges (contractions) to avoid the new names introduced
        # by the renaming of the free edges.
        new_names = {kwargs.get(e, e) for e in self.edges}
        contractions = {e for t in self.tensors for e in t.edges} - self.edges
        # Note the `unused_edge_names` function avoids introducing new clashes
        # between the edges being renamed.
        rename = unused_edge_names(contractions, new_names)
        # It's safe to add the kwargs to rename, since self._check_rename restricts kwargs to only
        # contain keys that are in self.edges.
        rename |= kwargs
        res = Product([t.rename(**rename) for t in self.tensors])
        assert res.edges == new_names
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
            rename = unused_edge_names(inner_edges, used_edges)
            for t in p.tensors:
                res.append(t.rename(**rename))
            used_edges.update(rename.values())  # Later renames should not clash with this one
        return Product(res)

    def grad(self, x: Variable, new_names: Optional[dict[str, str]] = None):
        # This checks that there is no overlap between the new names and the free edges.
        new_names = self._check_grad(x, new_names)

        # Since we are adding new edges to an internal tensor in the product, we need to make sure
        # none of the other tensors in the product have edges that clash with these new edges.
        inner_names = {e for t in self.tensors for e in t.edges if e not in self.edges}
        new_edges = set(new_names.values()) | self.edges
        rename = unused_edge_names(inner_names, new_edges)
        new_prod = Product([t.rename(**rename) for t in self.tensors])
        assert new_prod == self and new_prod.shape == self.shape, "Renaming should not change the product"
        # Note: We don't have to update they keys of new_names, sinc ethey refer to variable orig_names, which never change

        # The classic product rule of Calculus: d/dx (f * g) = f' * g + f * g'
        res = Sum(
            [
                Product(new_prod.tensors[:i] + [Derivative(t, x, new_names)] + new_prod.tensors[i + 1 :])
                for i, t in enumerate(new_prod.tensors)
            ]
        )
        assert res.edges == self.edges | new_edges, f"{res.edges} != {self.edges} | {new_edges}"
        return res

    def __repr__(self):
        if len(self.tensors) <= 1:
            return f"Product({self.tensors})"
        else:
            inner = "    " + ",\n    ".join(map(repr, self.tensors)) + ","
            return f"Product([\n{inner}\n])"

    def _inner_evaluate(self, values: dict["Tensor", torch.Tensor], dims: dict[Symbol, int]) -> torch.Tensor:
        # TODO: Keep track of how many contractions we made
        # extras["contractions"] = extras.get("contractions", 0) + len(self.contractions)
        if not self.tensors:
            return torch.tensor(1.0)
        # We use "operator" einsum interface, which doesn't require single letter names.
        # e.g.  einsum('i,i', b, b)  =  einsum(b, [0], b, [0])
        # and   einsum('ij,jk->ik', b, c)  =  einsum(b, [0, 1], c, [1, 2], [0, 2])
        edge_numbers = {e: i for i, e in enumerate({e for t in self.tensors for e in t.edges})}
        # TODO: We can make this more efficient by removing Copy tensors.
        parts = []
        for t in self.tensors:
            torch_tensor = t.evaluate(values, dims)
            parts.append(torch_tensor.rename(None))
            parts.append([edge_numbers[e] for e in torch_tensor.names])
        parts.append([edge_numbers[e] for e in self.edges])
        out = torch.einsum(*parts).rename(*self.edges)
        assert out.names == tuple(self.edges)
        return out

    def simplify(self, args: dict[str, Any] = None):
        args = self._check_simplify(args)

        tensors = [t.simplify(args=args) for t in self.tensors]

        # If any tensor in a product is 0, so is the whole product
        if any(isinstance(t, Zero) for t in tensors):
            return Zero(**self.shape)

        # Decide whether to push products through sums. This can be useful to simplify networks for
        # presentation, but it can also blow up the number of terms. So we make it optional.
        if args.get("distributed_products", False):
            raise NotImplementedError("Not implemented yet")

        # Combine nested products. Note that not combining products may be useful to keep the
        # "natural" contraction order. This can speed up evaluation, since sub-tensors can be reused.
        if args["associative_products"]:
            sub_products = [t if isinstance(t, Product) else Product([t]) for t in tensors]
            tensors = Product.merge(sub_products).tensors

        # We can do a "small" kind of distributed products, which is handling children that are single sums
        # Also, if a child is a sum with a single element, we can pull the weight up.
        # In general, we can pull out the least common multiple of the weights of the children.
        if single_sums := [t for t in tensors if isinstance(t, Sum) and len(t.tensors) == 1]:
            tensors = [t if t not in single_sums else t.tensors[0] for t in tensors]
            res_weight = math.prod(t.weights[0] for t in single_sums)
        else:
            res_weight = 1

        def verify_edges(tensors, msg=""):
            cnt = Counter(e for t in tensors for e in t.edges)
            if cnt:
                assert cnt.most_common()[0][1] <= 2, msg

        # Simplify Copy Tensors
        verify_edges(tensors)
        tensors = Copy.simplify_outer(tensors)
        verify_edges(tensors)

        # Combine / Cancel Product Functions
        if args["combine_products"]:
            from tensorgrad.functions import PowerFunction

            verify_edges(tensors)
            before = tensors
            tensors = PowerFunction.simplify_outer(tensors, args)
            verify_edges(tensors, f"{before} -> {tensors}")

        # Base cases
        if len(tensors) == 1:
            res = tensors[0]
        elif args["expand"]:
            terms = [[]]
            weights = [1]
            for t in tensors:
                if isinstance(t, Sum):
                    # Create cartesian product
                    terms = [term + [t0] for term in terms for t0 in t.tensors]
                    weights = [w * w0 for w in weights for w0 in t.weights]
                else:
                    for term in terms:
                        term.append(t)
            # Recurse with expand=False to avoid infinite descent
            res = Sum([Product(ts) for ts in terms], weights).simplify(args={"expand": False})
        else:
            res = Product(tensors)

        if res_weight != 1:
            res = res_weight * res

        assert res.shape == self.shape, f"Edges changed from {self.edges} to {res.edges}"
        return res

    def components(self) -> list["Product"]:
        """Find all disjoint components, that is, subgraphs that are not connected by an edge.

        Returns:
            - A list of disjoint components, each represented by a Product tensor.
        """
        # Create graph of tensor connections
        G = nx.Graph()
        G.add_nodes_from(range(len(self.tensors)))
        for i, t1 in enumerate(self.tensors):
            for j, t2 in enumerate(self.tensors[i + 1 :], i + 1):
                if set(t1.edges) & set(t2.edges):
                    G.add_edge(i, j)

        # Find connected components
        component_sets = list(nx.connected_components(G))
        components = [Product([self.tensors[i] for i in comp]) for comp in component_sets]
        assert Product(components).edges == self.edges
        return components

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=self.__class__.__name__, tensor=self)
        edges = {}
        inner_edges = defaultdict(list)
        for t in self.tensors:
            G, t_edges = add_structural_graph(G, t)
            for e, node in t_edges.items():
                inner_edges[e].append(node)
        for e, nodes in inner_edges.items():
            if len(nodes) == 2:
                # Note that inner edges are bidirectional, since tensor products
                # are associative.
                n1, n2 = nodes
                G.add_edge(n1, n2)
                G.add_edge(n2, n1)
            else:
                # If the edge is only present once, it's an outer edge
                assert e in self.edges, f"{e} not in {self.edges}"
                (n1,) = nodes
                edges[e] = n1
        assert edges.keys() == self.edges
        return G, edges

    def depends_on(self, x: "Variable") -> bool:
        return any(t.depends_on(x) for t in self.tensors)


################################################################################
# Sum
################################################################################


class Sum(Tensor):
    def __init__(self, tensors: Iterable[Tensor], weights: None | Iterable[int] = None):
        """
        A weighted sum of multiple tensors.

        Args:
            tensors: The tensors to add together.
            weights: The weights of each tensor in the sum. If not provided, all weights are 1.
        """
        tensors = list(tensors)

        # Broadcasting means we always upgrade to the super set of edges
        self._shape = {}
        for e, ts in group_edges(tensors).items():
            s = ts[0].shape[e]
            if not all(t.shape[e] == s for t in ts):
                raise ValueError(f"Edge {e} had different sizes in tensors: {ts}")
            self._shape[e] = s

        # Any tensors that lacks an edge will be broadcasted to have that edge.
        all_edges = self._shape.keys()
        self.tensors = []
        for t in tensors:
            missing = {e: self._shape[e] for e in all_edges - t.edges}
            # Note: don't broadcast if the tensor is already full, since that would create new
            # Ones([]) objects after simplification is supposed to have completed.
            self.tensors.append(t @ Ones(**missing) if missing else t)

        self.weights = [1] * len(tensors) if weights is None else list(weights)
        assert len(tensors) == len(self.weights)

    def rename(self, **kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        res = Sum([t.rename(**kwargs) for t in self.tensors], self.weights)
        assert set(res.edges) == {kwargs.get(e, e) for e in self.edges}
        return res

    def grad(self, x: Variable, new_names: Optional[dict[str, str]] = None):
        new_names = self._check_grad(x, new_names)
        return Sum([Derivative(t, x, new_names) for t in self.tensors], self.weights)

    def simplify(self, args: dict[str, Any] = None):
        args = self._check_simplify(args)
        terms = [t.simplify(args=args) for t in self.tensors]

        term_counter = Counter()
        for w, t in zip(self.weights, terms):
            if args["associative_sums"] and isinstance(t, Sum):
                for w1, t1 in zip(t.weights, t.tensors):
                    term_counter[MatchEdgesKey(t1)] += w * w1
            else:
                term_counter[MatchEdgesKey(t)] += w

        if args["sum_combine_terms"]:
            # Identify tensors with multiplicity and combine them. We use tensor.canon_with_edges to identify tensors.
            # It is important that isomorphic tensors with different outer edge labels don't get matched. For example
            # (o-i o-<jk) is isomorphic with (o-j o-<ik), but they shouldn't be combined. This example comes up in the
            # Hessian of softmax.
            ws_tensors = [
                (w, key.value)
                for key, w in term_counter.items()
                if w != 0 and not isinstance(key.value, Zero)
            ]
        else:
            ws_tensors = [(w, key.value) for key, w in term_counter.items()]

        # Remove zero tensors or zero weights.
        # Note: This won't change the shape of the tensor, since all summands have been broadcasted.
        ws_tensors = [(w, t) for w, t in ws_tensors if w != 0 and not isinstance(t, Zero)]
        # Base case. Here we can't just return Zero([]), since that would change the signature of the tensor.
        if not ws_tensors:
            return Zero(**self.shape)
        weights, tensors = zip(*ws_tensors)
        # If there is just one tensor with weight 1, we don't need LinearComb
        if weights == (1,):
            return tensors[0]
        res = Sum(tensors, weights)
        assert res.shape == self.shape, f"Edges changed from {self.edges} to {res.edges}"
        return res

    def __repr__(self):
        return f"Sum({self.tensors}, {self.weights})"

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=self.__class__.__name__, tensor=self)
        edges = {}
        # Create special "Plus nodes" to collect common edges
        for i, e in enumerate(self.edges):
            # Note: No need to add the shape here. It's already in the sub-tensors
            G.add_node(i + 1, name="Plus Node")
            edges[e] = i + 1
        for t, w in zip(self.tensors, self.weights):
            # The weights are a little akward. There a lot of options for how to handle them.
            # E.g. the idea of just using a weighted Copy([]) somehow. But this works.
            G, t_edges = add_structural_graph(G, t, root_edge_label=f"Weight {w}")
            # Connect tensor edges to the plus nodes
            for e, node in t_edges.items():
                G.add_edge(node, edges[e])
        assert edges.keys() == self.edges
        return G, edges

    def _inner_evaluate(self, values: dict["Tensor", torch.Tensor], dims: dict[Symbol, int]) -> torch.Tensor:
        values = [t.evaluate(values, dims).align_to(*self.edges) for t in self.tensors]
        res = sum(float(w) * v for w, v in zip(self.weights, values))
        assert set(res.names) == self.edges, f"Expected {self.edges}, got {res.names}"
        return res

    def depends_on(self, x: "Variable") -> bool:
        return any(t.depends_on(x) for t in self.tensors)


################################################################################
# Some useful functions
################################################################################


class MatchEdgesKey:
    """Normally Tensors use isomorphism as their test for equality, but they don't include
    the edge names in the comparison. This class is used to compare tensors based on their
    edge names. It is used in the Sum tensor to combine tensors with the same edge names."""

    def __init__(self, value, **edge_names: str):
        self.value = value
        self.hash = hash(value)
        self.edge_names = edge_names

    def __eq__(self, other):
        if isinstance(other, MatchEdgesKey):
            return self.value.is_isomorphic(other.value, edge_names=self.edge_names, match_edges=True)
        return False

    def __hash__(self):
        return self.hash


def group_edges(tensors: Iterable[Tensor]) -> dict[str, list[Tensor]]:
    """Group tensors by their edge names."""
    groups = defaultdict(list)
    for t in tensors:
        for e in t.edges:
            groups[e].append(t)
    return groups


def add_structural_graph(G, tensor, root_edge_label=None):
    """Computes the structural graph of tensor, and unions it with G."""
    # We assert that Gx has nodes named [0, Gx.number_of_nodes())
    assert set(G.nodes()) == set(range(G.number_of_nodes())), f"{G.nodes()}"
    Gx, x_edges = tensor.structural_graph()
    offset = G.number_of_nodes()
    assert set(Gx.nodes()) == set(range(Gx.number_of_nodes())), f"{Gx.nodes()}"
    # We also assume that 0 is the root
    Gx = nx.relabel_nodes(Gx, {i: i + offset for i in Gx.nodes()})
    x_root = offset
    x_edges = {e: (n + offset) for e, n in x_edges.items()}
    G = nx.union(G, Gx)
    # Make sure to connect the root of Gx to the root of G, possibly with an "edge label"
    if root_edge_label is not None:
        e = G.number_of_nodes()
        G.add_node(e, name=root_edge_label)
        G.add_edge(e, x_root)
        G.add_edge(0, e)
    else:
        G.add_edge(0, x_root)
    assert x_edges.keys() == tensor.edges, f"{x_edges.keys()} != {tensor.edges}"
    return G, x_edges


def unused_edge_names(edges: Iterable[str], used_names: Iterable[str], suffix: str = "") -> dict[str, str]:
    """Given a list of edges, return a dict[str, str] that renames the original names into
    new fresh names, not in either edges nor used_names.
    Also append suffix to the new names.
    """
    used_names = set(used_names)
    rename = {}
    for e in edges:
        candidate = e + suffix
        while candidate in used_names:
            candidate += "_"
        rename[e] = candidate
        used_names.add(candidate)
    return rename


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

    res, renames = [], []
    for i, t in enumerate(tensors):
        edges = t.edges - unique_names
        rename = unused_edge_names(edges, used_names, suffix=f"_{i}")
        # Make sure all t.edges are present in rename
        rename.update({e: e for e in unique_names})
        used_names.update(rename.values())
        res.append(t.rename(**rename))
        renames.append(rename)
    return res, renames
