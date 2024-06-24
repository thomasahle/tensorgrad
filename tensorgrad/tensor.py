from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
import math
from typing import Any, Callable, Iterable, Optional
from abc import ABC
from fractions import Fraction
from numbers import Number
import networkx as nx

import torch


# TODO:
# - Code generation (e.g. Triton, Pytorch)
# - Prettier printing
# - Introduce a function object, that's not a tensor, but which creates a tensor when called. This makes it easier to define
#   operators, such as taylor expansion, which feeds the function some specific inputs.
# - Support broadcasting in functions, since simple two input functions like Cross Entropy is currently basically not possible.
#   Can't you just user Copy tensors?
# - Taking the derivative with respect to multiple variables at the same time (full backprop)
#   Maybe we can still take derivatives individually, and just use the isomorphic hashing to avoid recomputions?
# - Nested derivatives should create symmetries between the edges they create.
#   This would probably require expanding the Derivative class to take multiple wrts?
# More simplification rules:
# - Optional "function expand" that converts e.g. "softmax" into it's components
# Smaller things:
# - We don't need weights in Sum. We can just use a Consant([]) tensor with a weight and include it in a product.
#   It's nice for pretty printing though, and for simplification rules...
# - Stuff from https://en.wikipedia.org/wiki/Penrose_graphical_notation
#   - Symmetrization/Antisymmetrization
#   - Matrix inverses
# Done:
# X Support taking the Expectation, at least for Gaussian tensors. Can be done via Gaussian integration by parts.
# X Optional "expand" setting that expands the expression to a sum of products
# X Support for specific functions and their simplification rules (pow(-1) cancelation, etc)
# X Use a real algorithm to check isomorphism, like nauty.
#   Part of this is creating a completely new symmetry structure.
# X Support symmetric Variables


class Tensor(ABC):
    edges = []

    @property
    def rank(self):
        """The number of edges the tensor has."""
        return len(self.edges)

    def grad(self, x: "Variable", new_names: Optional[list[str]]) -> "Tensor":
        """
        Take the derivative of this tensor with respect to the variable x.

        Args:
            x: The variable to take the derivative with respect to.
            new_names: Optional list of names to use for the new edges created by the derivative.
                If not provided, new names will be generated based on the edges of x.

        Note:
            Pushes the derivative one step through the tensor.
            If you want to push it all the way through, use simplify.

        Returns:
            The tensor representing the derivative.
        """
        new_names = self._check_grad(x, new_names)
        raise NotImplementedError

    def rename(self, kwargs: dict[str, str]) -> "Tensor":
        """
        Rename the free edges of this tensor.

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
        """Hexadecimal string corresponding to hash of the input graph."""
        return nx.algorithms.weisfeiler_lehman_graph_hash(
            self._edge_structural_graph(match_edges=False),
            node_attr="name",
        )

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

    def _edge_structural_graph(self, match_edges=True) -> nx.MultiDiGraph:
        """Like structural_graph, but adds dummy nodes for the outer edges."""
        G, edges = self.structural_graph()
        for e, node in edges.items():
            n = G.number_of_nodes()
            G.add_node(n, name="Outer Edge" + (f" {e}" if match_edges else ""))
            G.add_edge(n, node)
        return G

    def edge_equivalences(self) -> list[tuple[tuple["Tensor", str], tuple["Tensor", str]]]:
        """
        Return a list of equivalent edges in this tensor.

        Each entry in the returned list is a tuple ((t1, e1), (t2, e2)),
        indicating that edge e1 of tensor t1 is equivalent to edge e2 of tensor t2.
        """
        return []

    def __add__(self, other) -> "Tensor":
        return Sum([self, other])

    def __sub__(self, other) -> "Tensor":
        return Sum([self, other], [1, -1])

    def __neg__(self) -> "Tensor":
        return Sum([self], [-1])

    def __matmul__(self, other) -> "Tensor":
        return Product([self, other])

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __mul__(self, other) -> "Tensor":
        """Contract self and other, but use a 3d-identity to keep the shared edges free."""
        if isinstance(other, Number):
            return Sum([self], [other])
        # Element-wise (Hadamard) product is easy to implement using Copy tensors
        # These are the edges we multiply over
        shared_edges = set(self.edges) & set(other.edges)
        (t0, t1), (rename0, rename1) = make_distinct(self, other, used_names=shared_edges)
        return Product([t0, t1] + [Copy([e, rename0[e], rename1[e]]) for e in shared_edges])

    def __truediv__(self, other) -> "Tensor":
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

    def is_isomorphic(self, other, match_edges=False) -> bool:
        G1 = self._edge_structural_graph(match_edges=match_edges)
        G2 = other._edge_structural_graph(match_edges=match_edges)
        return nx.is_isomorphic(G1, G2, node_match=lambda n1, n2: n1.get("name") == n2.get("name"))

    def isomorphisms(self, other):
        """Given self and other are isomorphic, this method returns a dictionary that renames self into other."""
        G1 = self._edge_structural_graph(match_edges=False)
        G2 = other._edge_structural_graph(match_edges=False)
        for matching in nx.algorithms.isomorphism.MultiDiGraphMatcher(
            G1, G2, node_match=lambda n1, n2: n1.get("name") == n2.get("name")
        ).isomorphisms_iter():
            # Matching is a dict {i: j} where i is the node in G1 and j is the node in G2
            # We are only interested in then `len(self.edges)` last nodes, which correspond to the outer edges
            start_i = G1.number_of_nodes() - len(self.edges)
            start_j = G2.number_of_nodes() - len(self.edges)
            yield {
                self.edges[i - start_i]: other.edges[j - start_j] for i, j in matching.items() if i >= start_i
            }

    @cached_property
    def symmetries(self) -> set[frozenset[str]]:
        G = self._edge_structural_graph(match_edges=False)
        iso = nx.isomorphism.GraphMatcher(G, G, node_match=lambda n1, n2: n1.get("name") == n2.get("name"))

        symmetries = defaultdict(set)
        start_i = G.number_of_nodes() - len(self.edges)
        for auto in iso.isomorphisms_iter():
            for i, j in auto.items():
                if i >= start_i:
                    symmetries[i - start_i].add(self.edges[j - start_i])

        return set(map(frozenset, symmetries.values()))

    def evaluate(
        self,
        values: dict["Variable", torch.tensor],
        *,
        dims: dict[str, int] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> torch.tensor:
        """
        Evaluate this tensor given values for the variable tensors.

        Args:
            values: A dictionary mapping variable tensors to their values.
            dims: An optional dictionary specifying the dimensions of free edges.
            extras: An dictionary containing extra data used during evaluation.

        Returns:
            The result of evaluating this tensor.
        """
        if extras is None:
            extras = {}
        if "edge_dims" not in extras:
            extras["edge_dims"] = {}
        if "cached_values" not in extras:
            extras["cached_values"] = {}
        if id(self) not in extras["edge_dims"]:
            shapes = {v: dict(zip(t.names, t.shape)) for v, t in values.items() if isinstance(v, Variable)}
            extras["edge_dims"] |= self._compute_edge_dims(shapes, extra_dims=dims)
        edge_dims = extras["edge_dims"].get(id(self), {})
        if not edge_dims and self.edges:
            print("Warning: Unable to compute edge dimensions for", self)

        # Variables don't need the value cache, since we can just look up the value directly.
        if isinstance(self, Variable):
            if self not in values:
                raise ValueError(f"Missing value for {self}, got {values}")
            # Since renaming values never change the order, we can just rename back here.
            return values[self].rename(*self.edges)

        cached_values = extras["cached_values"]
        key = (self, tuple(sorted(edge_dims.items())))
        if key in cached_values:
            # Take over the isomorphic representative
            other, value = next((t, v) for (t, ed), v in cached_values.items() if (t, ed) == key)
            rename = next(other.isomorphisms(self), None)
            if rename is None:
                raise ValueError("Graphs must be isomorphic (with edges)")
            res = value.rename(*(rename[e] for e in other.edges)).align_to(*self.edges)
            # Enable this to debug the isomorphic cache
            expected = self.inner_evaluate(edge_dims, values, extras)
            assert expected.names == res.names, f"{expected.names=} {res.names=}"
            assert torch.allclose(res.rename(None), expected.rename(None))
            return res

        res = self.inner_evaluate(edge_dims, values, extras)
        assert res.names == tuple(self.edges), f"Expected {self.edges=} but got {res.names=}"
        cached_values[key] = res
        return res

    def _compute_edge_dims(
        self,
        shapes: dict["Variable", dict[str, int]],
        extra_dims: dict[str, int] | None = None,
    ):
        G = nx.Graph()
        var_ids = defaultdict(list)

        # Make keys using ids, since we want to distinguish between isomorphic tensors
        for (t1, e1), (t2, e2) in self.edge_equivalences():
            G.add_edge((id(t1), e1), (id(t2), e2))

            # Make sure we have the updated variable ids. The same variable can have multiple ids,
            # as it has been renamed differently throughout the graph.
            # Note: Since variables set (v, ("Original", o)) = (v, e) we are guaranteed that all
            # variables show up when calling edge_equivalences.
            for t in [t1, t2]:
                if isinstance(t, Variable):
                    var_ids[t].append(id(t))

        res = defaultdict(dict)
        for v, edges in shapes.items():
            for v_id in var_ids[v]:
                for o, d in edges.items():
                    res[v_id][("Original", o)] = d

        if extra_dims:
            res[id(self)].update(extra_dims)

        # The graph is really a graph of "edges between edges". Hence the
        # connected components of the graph all have the same dimension.
        for component in nx.connected_components(G):
            sizes = [res[t_id].get(e) for t_id, e in component if e in res[t_id]]
            if sizes and len(set(sizes)) > 1:
                raise ValueError(f"Size mismatch in component: {sizes}")
            if sizes:
                size = sizes[0]
                for t_id, e in component:
                    res[t_id][e] = size

        return res

    def inner_evaluate(self, edge_dims: dict[str, int], values: dict, extras: dict) -> torch.tensor:
        """
        The inner implementation of tensor evaluation.

        Subclasses should override this to define the actual evaluation logic.

        Args:
            edge_dims: A dictionary mapping edge names to their dimensions.
            values: A dictionary mapping variable tensors to their values.
            extras: An dictionary containing extra data used during evaluation.

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

    def _check_grad(self, x: "Variable", new_names: Optional[list[str]] = None) -> list[str]:
        """When taking a derivative with respect to a variable, we need some edge names for the derivative.
        If new_names is not given, we generate names based on the edge names of x.
        However, we need to make sure they don't clash with any names already present in the tensor.
        If new_names is already given, we just check to make sure they are not already present in the tensor.
        """
        # It shouldn't be necessary for the edges to be unused in the entire tensor, just in the free edges.
        if new_names is not None:
            assert len(x.edges) == len(new_names), f"{x.edges=} != {new_names=}"
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
        args.setdefault("associative_products", True)
        args.setdefault("associative_sums", True)
        args.setdefault("sum_combine_terms", True)
        args.setdefault("combine_products", True)
        args.setdefault("expand", False)
        return args

    def _check_edges(self, edges: str | Iterable[str]) -> list[str]:
        if edges is None:
            return None
        if isinstance(edges, str):
            return [e.strip() for e in edges.split(",")]
        return list(edges)


################################################################################
# Variable
################################################################################


class Variable(Tensor):
    def __init__(
        self,
        name,
        edges: str | Iterable[str],
        orig: None | str | Iterable[str] = None,
        symmetries: None | str | set[frozenset[str]] = None,
    ):
        """
        A tensor holding a variable.

        Args:
            name: The name of this variable.
            edges: The names of the edges of this variable. Can be a comma-delimited string or iterable of strings.
            orig: The actual edge names to use. If not provided, the names in `edges` are used directly.
            symmetries: Sets of edges (original names) that should be considered equivalent.
        """
        edges = self._check_edges(edges)
        self.name = name

        # The original edges are saved so evaluation can happen with the original
        # edge names given to the variable, without caring about what renaming we
        # might have done in the meantime.
        self.edges = list(edges)
        self.original_edges = self._check_edges(orig) if orig is not None else edges

        if symmetries is None:
            symmetries = {frozenset({o}) for o in self.original_edges}
        if isinstance(symmetries, str):
            symmetries = {frozenset(word.split()) for word in symmetries.split(",")}
        assert all(
            e in self.original_edges for group in symmetries for e in group
        ), f"{symmetries} not in {self.original_edges}"
        # Note that self.symmetries is the Tensor property, which concerns current edge names,
        # not original edge names.
        self.o_symmetries = symmetries

    def grad(self, x: "Variable", new_names: Optional[list[str]] = None):
        new_names = self._check_grad(x, new_names)
        if x == self:
            assert x.original_edges == self.original_edges
            return Product(Copy([e, new], link=self) for e, new in zip(self.edges, new_names))
        return Zero(self.edges + new_names)

    def __repr__(self):
        args = [f"\"{self.name}\", \"{', '.join(self.edges)}\""]
        if self.edges != self.original_edges:
            args.append(f"orig=\"{', '.join(self.original_edges)}\"")
        if self.o_symmetries != {frozenset({o}) for o in self.original_edges}:
            groups = ", ".join(sorted(" ".join(sorted(group)) for group in self.o_symmetries))
            args.append(f'symmetries="{groups}"')
        return f"Variable({', '.join(args)})"

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=("Variable", self.name), tensor=self)
        edges = {}
        name_map = dict(zip(self.original_edges, self.edges))
        for i, group in enumerate(self.o_symmetries):
            G.add_node(i + 1, name=("Original Edge Name", " ".join(sorted(group))))
            G.add_edge(i + 1, 0)
            for o in group:
                edges[name_map[o]] = i + 1
        return G, edges

    def simplify(self, args: dict[str, Any] = None):
        return self

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)  # Checks only free edges are in kwargs
        return Variable(
            self.name,
            edges=[kwargs.get(e, e) for e in self.edges],
            orig=self.original_edges,
            symmetries=self.o_symmetries,
        )

    def edge_equivalences(self):
        # Since the user gives edge dimensions in terms of variables, it's important to keep track
        # of renamed edge names.
        for o, e in zip(self.original_edges, self.edges):
            yield (self, ("Original", o)), (self, e)

        # Next make a complete graph for each symmetry class
        name_map = dict(zip(self.original_edges, self.edges))
        for group in self.o_symmetries:
            for o1, o2 in combinations(group, 2):
                # Sorting makes it easier to write unit tests
                e1, e2 = sorted([name_map[o1], name_map[o2]])
                yield (self, e1), (self, e2)

    def depends_on(self, x: "Variable") -> bool:
        return x == self


################################################################################
# Constants
################################################################################


class Constant(Tensor, ABC):
    def __init__(self, edges: Iterable[str], link: Optional[Variable] = None, tag: Optional[int] = None):
        """
        A constant tensor with the given edges.

        Args:
            edges: The names of the edges of this constant tensor.
            link: An optional variable that this tensor is associated with, used to compute edge dimensions.
            tag: An optional tag for this tensor.
        """
        edges = self._check_edges(edges)
        self.edges = list(edges)
        self.link = link
        self.tag = tag

    def __repr__(self) -> str:
        extra = ""
        if self.link is not None:
            extra = f", link={self.link}"
        if self.tag is not None:
            extra += f", tag={self.tag}"
        return f"{type(self).__name__}({self.edges}{extra})"

    def structural_graph(self) -> tuple[nx.MultiDiGraph, int, dict[str, int]]:
        # This method assumes the constant is completely symmetric in the edges.
        # That's true for all of Copy, Zero and Ones
        # Note: it might not be true once edge dimensions are introduced, so we can't
        # necessarily use this same graph to replace edge_equivalences()...
        G = nx.MultiDiGraph()
        name = type(self).__name__
        if self.tag is not None:
            name += f" ({self.tag})"
        G.add_node(0, name=name, tensor=self)
        # TODO: Should we also include link here?
        if self.link is not None:
            add_structural_graph(G, self.link, root_edge_label="self.link")
        return G, {e: 0 for e in self.edges}

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        link = self.link.rename(kwargs) if self.link is not None else None
        c = type(self)([kwargs.get(e, e) for e in self.edges], link=link)
        assert set(c.edges) == {kwargs.get(e, e) for e in self.edges}
        return c

    def grad(self, x: Variable, new_names: Optional[list[str]] = None):
        new_names = self._check_grad(x, new_names)
        return Zero(self.edges + new_names)

    def edge_equivalences(self) -> Iterable[tuple[tuple["Tensor", str], tuple["Tensor", str]]]:
        if self.link is not None:
            yield from self.link.edge_equivalences()
            for e in self.link.edges:
                if e in self.edges:
                    yield (self, e), (self.link, e)

    def _check_evaluate(self, values, dims=None, extras=None):
        extras = super()._check_evaluate(values, dims, extras)
        for e in self.edges:
            if e not in extras["edge_dims"][id(self)]:
                raise ValueError(f"Missing edge dimension for {e}.")
        return extras

    def depends_on(self, x: "Variable") -> bool:
        return False


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

    def inner_evaluate(self, edge_dims: dict[str, int], values: dict, extras: dict) -> torch.tensor:
        if not self.edges:
            return torch.tensor(1.0)
        shape = [edge_dims[e] for e in self.edges if e in edge_dims]
        assert len(shape) == len(self.edges) and len(set(shape)) == 1, shape
        copy = torch.zeros(shape)
        for idx in range(shape[0]):
            copy[(idx,) * len(self.edges)] = 1
        return copy.rename(*self.edges)

    @classmethod
    def simplify_outer(cls, tensors: list[Tensor]) -> list[Tensor]:
        """Simplifies a list of tensors assumed to be a product."""
        while True:
            tensors, done = cls._simplify_step(tensors)
            if done:
                break
        # Remove empty Copy's (they are just the constant 1)
        tensors = [t for t in tensors if not (isinstance(t, Copy) and t.rank == 0)]
        return tensors

    @classmethod
    def _simplify_step(cls, tensors: list[Tensor]) -> tuple[list[Tensor], bool]:
        """Performs one step of simplification. Returns a new list if changed, or the original if not."""
        for e in [edge for t in tensors for edge in t.edges]:
            connected = [t for t in tensors if e in t.edges]
            if len(connected) != 2:
                continue
            t1, t2 = connected

            # Remove t1 and t2 from tensors.  We use the "is" operator, since we've overloaded equality to mean isomorphic.
            other = [t for t in tensors if t is not t1 and t is not t2]

            for simplification in [cls._merge_copy_tensors, cls._remove_identity_matrix]:
                if (new := simplification(t1, t2, e)) is not None:
                    tensors = other + new
                    return tensors, False

        return tensors, True

    @staticmethod
    def _merge_copy_tensors(t1: Tensor, t2: Tensor, e: str) -> Optional[list[Tensor]]:
        if not (isinstance(t1, Copy) and isinstance(t2, Copy)):
            return None

        # Don't create a singleton Copy tensor
        if set(t1.edges) == set(t2.edges):
            # If the two copy's share all edges with each other, we reduce to a single
            # shared edge, which is how we represent a constant with value = edge dim.
            if t1.rank != 1:
                # TODO: If there is some linked edge, we should try to let that be the one we keep
                return [
                    Copy([e], link=t1.link),
                    Copy([e], link=t2.link),
                ]
            return None

        # We don't just remove e, but remove all shared edges
        return [Copy(list(set(t1.edges) ^ set(t2.edges)), link=t1.link or t2.link)]

    @staticmethod
    def _remove_identity_matrix(t1: Tensor, t2: Tensor, e: str) -> bool:
        # Make t1 the identity matrix
        if isinstance(t2, Copy) and t2.rank == 2:
            t1, t2 = t2, t1

        if isinstance(t1, Copy) and t1.rank == 2:
            other_edge = next(iter(set(t1.edges) - {e}))
            # Don't create self loops. We never connect a tensor to itself.
            if other_edge not in t2.edges:
                return [t2.rename({e: other_edge})]


class Zero(Constant):
    """Matrix such that Z_{i,j,k} = 0 for all i, j, k"""

    def inner_evaluate(self, edge_dims: dict[str, int], values: dict, extras: dict) -> torch.tensor:
        if not self.edges:
            return torch.tensor(0.0)
        return torch.zeros([edge_dims[e] for e in self.edges]).rename(*self.edges)


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
    simplify: Callable[["Function", dict[str, Any]], Tensor] = None


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
        """
        A function tensor that takes one or more input tensors and produces an output tensor.

        Args:
            fn_info: The FunctionInfo defining this function, or a string giving the function name.
            edges_out: The names of the output edges of this function.
            *inputs: The input tensors and their input edge names. [(t0, e00, e01, ...), (t1, e10, ...), ...]
            orig_edges_out: The original output edge names. If not provided, `edges_out` is used.
            orig_edges_in: The original input edge names for each input. If not provided, the actual input edge names are used.
            orig_edges_ts: The original edges of each complete input tensor. If not provided, the actual tensor edges are used.
        """
        edges_out = self._check_edges(edges_out)
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

        res = Function(
            self.fn_info,
            self.edges_out,
            *new_inputs,
            orig_edges_out=self.orig_edges_out,
            orig_edges_in=self.orig_edges_in,
            orig_edges_ts=orig_edges_ts,
        )
        if pulled_out:
            return Product([res] + pulled_out)

        if self.fn_info.simplify is not None:
            res = self.fn_info.simplify(res, args)

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

            # Take the derivative of the function and the inner function (chain rule)
            canonical_d_edges, rename1 = unused_edge_names(input_edges, self.orig_edges, suffix="_")
            part_a = self.fn_info.derivative(i, canonical_d_edges, *[t for t, *_ in self.inputs])
            # Rename to avoid clashing with upcoming new_edges
            avoid = set(new_names) | set(self.edges) | set(self.orig_edges)
            connection_edges, rename2 = unused_edge_names(canonical_d_edges, avoid)
            part_a = part_a.rename(rename2)
            # We need to rename the output to the current names used by the function, which may have changed since
            # the fn_info.derivative was created.
            part_a = part_a.rename(dict(zip(self.orig_edges, self.edges)))
            assert set(part_a.edges) == set(self.edges + connection_edges)

            part_b = Derivative(t.rename(rename1).rename(rename2), x, new_names)

            # The "external edges" are going to be joined with copy tensors, resulting in joint broadcasting.
            # So we need to rename them in both the function and the inner. Like we do in __mul__.
            avoid |= set(connection_edges)
            broadcasted_edges = [e for e in t.edges if e not in input_edges]
            assert set(broadcasted_edges) == set(t.rename(rename1).rename(rename2).edges) - set(
                connection_edges
            )
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

    def structural_graph(self) -> tuple[nx.MultiDiGraph, int, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=(type(self).__name__, self.fn_info.name), tensor=self)
        edges = {}
        # We add a node for the "function" tensor itself
        G.add_node(1, name=("f", self.fn_info.name), tensor=self)
        G.add_edge(1, 0)
        for i, (e, o) in enumerate(zip(self.edges_out, self.orig_edges_out)):
            G.add_node(i + 2, name=("Original Edge Out", o))
            G.add_edge(i + 2, 1)
            edges[e] = i + 2
        # And we add nodes for all the input tensors
        for (t, *input_edges), oes in zip(self.inputs, self.orig_edges_in):
            # Compute graph from input tensor, and ensure it uses distinct node numbers
            G, t_edges = add_structural_graph(G, t)
            # Connect tensor to function. This is another place we need to label the edges
            # with original names, since argument order matters.
            for e, o in zip(input_edges, oes):
                # Create labeled edge
                n = G.number_of_nodes()
                G.add_node(n, name=("Function Input Edge", o))
                G.add_edge(t_edges[e], n)
                # Connect to function
                G.add_edge(n, 1)
            # Finally register free edges
            # TODO: Do we need self.original_edges_ts here somehow? I don't see it..
            # I think they are just needed for eval, and only because we somehow assume
            # functions get confused if edges are renamed. Which I don't actually think
            # they do, when it's not the input dimensions.
            for e in t.edges:
                if e not in input_edges:
                    edges[e] = t_edges[e]
        return G, edges

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

    def inner_evaluate(self, edge_dims: dict[str, int], values, extras) -> torch.tensor:
        inner_values = []
        for (t, *input_edges), oes in zip(self.inputs, self.orig_edges_ts):
            inner_value = t.evaluate(values, extras=extras)
            assert inner_value.names == tuple(t.edges), f"Expected {t.edges}, got {inner_value.names}"
            # After evaluation we need to rename the output edges to the original names of the function,
            # as they are expected by fn_info.eval.
            # TODO: Are they really? Shouldnt it just be the input edges that need to be consistent?
            # Could I do rename(input_edges: orig_edges_in)?
            inner_values.append(inner_value.rename(*oes))
        out = self.fn_info.eval(*inner_values)
        # After evaluation we need to rename the output edges back to their current values.
        assert out.names == tuple(self.orig_edges)
        out = out.rename(*self.edges)
        assert out.names == tuple(self.edges)
        return out

    def depends_on(self, x: "Variable") -> bool:
        return any(t.depends_on(x) for t, *_ in self.inputs)


class Derivative(Tensor):
    def __init__(self, tensor: Tensor, x: Variable, new_names: Optional[list[str]] = None):
        """
        A tensor representing the derivative of another tensor.

        Args:
            tensor: The tensor to take the derivative of.
            x: The variable to take the derivative with respect to.
            new_names: The names to use for the new edges created by the derivative. If not provided, they will be generated.
        """
        new_names = self._check_edges(new_names)
        self.tensor = tensor
        self.x = x
        # _check_grad makes sure the new_names are not already present in self.edges.
        # But we haven't set self.edges yet, so we call it on tensor instead of self.
        self.new_names = tensor._check_grad(x, new_names)
        self.edges = tensor.edges + self.new_names

    def simplify(self, args: dict[str, Any] = None):
        args = self._check_simplify(args)
        if not self.tensor.depends_on(self.x):
            return Zero(self.edges)
        inner = self.tensor.simplify(args)
        if args["grad_steps"] == 0:
            # If grad_steps is 0, we pass the simplify through the derivative.
            res = Derivative(inner, self.x, self.new_names)
        else:
            args["grad_steps"] -= 1
            # Have to call simplify twice to avoid an infinite loop when stacking multiple derivatives.
            res = inner.grad(self.x, self.new_names).simplify(args)
        assert set(res.edges) == set(self.edges), f"Edges changed from {self.edges} to {res.edges}"
        return res

    def grad(self, x: Variable, new_names: list[str] | None) -> Tensor:
        new_names = self._check_grad(x, new_names)
        # To avoid an infinite loop, we let the grad pass through us, rather than creating a double derivative.
        res = Derivative(self.tensor.grad(x, new_names), self.x, self.new_names)
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

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name="Derivative", tensor=self)
        edges = {}
        # We add a node for the "wrt" tensor
        G, x_edges = add_structural_graph(G, self.x, root_edge_label="self.x")
        # This might be controversial, but we'll have new_edges point to the respective edges in self.x
        edges |= {e: x_edges[xe] for e, xe in zip(self.new_names, self.x.edges)}
        # Then we add the differentiated tensor
        G, t_edges = add_structural_graph(G, self.tensor, root_edge_label="self.tensor")
        edges |= t_edges
        return G, edges

    def __repr__(self):
        return f"Derivative({self.tensor}, {self.x}, {self.new_names})"

    def inner_evaluate(self, edge_dims: dict[str, int], values: dict, extras: dict) -> torch.tensor:
        # We could use numerical differentiation here...  But it would potentially require quite a lot of
        # evaluations, since we need to evaluate the tensor in all directions.
        raise ValueError("Derivative tensors cannot be evaluated directly. Please use simplify() first.")

    def edge_equivalences(self):
        # Not really needed since we don't expect to be evaluated
        yield from self.tensor.edge_equivalences()

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

    def inner_evaluate(self, edge_dims: dict[str, int], values: dict, extras: dict) -> torch.tensor:
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
        out = torch.einsum(*parts).rename(*self.edges)
        assert out.names == tuple(self.edges)
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

        # Simplify Copy Tensors
        tensors = Copy.simplify_outer(tensors)

        # Combine / Cancel Product Functions
        if args["combine_products"]:
            from tensorgrad.functions import PowFunctionInfo

            tensors = PowFunctionInfo.simplify_outer(tensors)

        # Base cases
        if not tensors:
            # The only issue here is that we're throwing away edge names.
            res = Copy([])
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
            res = Sum([Product(ts) for ts in terms], weights)
        else:
            res = Product(tensors)

        if res_weight != 1:
            res = res_weight * res

        assert set(res.edges) == set(self.edges), f"Edges changed from {self.edges} to {res.edges}"
        return res

    def components(self, return_colors=False):
        """Find all disjoint components, that is, subgraphs that are not connected by an edge.

        Returns:
            - A list of disjoint components, each represented by a Product tensor.
            - If return_colors is True, also returns color mapping and grouping.
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
        assert {e for c in components for e in c.edges} == set(self.edges)

        if return_colors:
            colors = {i: c for c, comp in enumerate(component_sets) for i in comp}
            return components, colors, component_sets

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
        return G, edges

    def depends_on(self, x: "Variable") -> bool:
        return any(t.depends_on(x) for t in self.tensors)


################################################################################
# Sum
################################################################################


class Sum(Tensor):
    def __init__(self, tensors: Iterable[Tensor], weights: list[int] = None):
        """
        A weighted sum of multiple tensors.

        Args:
            tensors: The tensors to add together.
            weights: The weights of each tensor in the sum. If not provided, all weights are 1.
        """
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
        term_counter = Counter()

        for w, t in zip(self.weights, (t.simplify(args=args) for t in self.tensors)):
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
            ws_tensors = [(w, t.value) for t, w in term_counter.items()]

        # Remove zero tensors or zero weights.
        # Note: This won't change the shape of the tensor, since all summands have been broadcasted.
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

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=self.__class__.__name__, tensor=self)
        edges = {}
        # Create special "Plus nodes" to collect common edges
        for i, e in enumerate(self.edges):
            G.add_node(i + 1, name="Plus Node")
            edges[e] = i + 1
        for t, w in zip(self.tensors, self.weights):
            # The weights are a little akward. There a lot of options for how to handle them.
            # E.g. the idea of just using a weighted Copy([]) somehow. But this works.
            G, t_edges = add_structural_graph(G, t, root_edge_label=f"Weight {w}")
            # Connect tensor edges to the plus nodes
            for e, node in t_edges.items():
                G.add_edge(node, edges[e])
        return G, edges

    def edge_equivalences(self):
        for t in self.tensors:
            yield from t.edge_equivalences()
            for e in t.edges:
                yield (t, e), (self, e)

    def inner_evaluate(self, edge_dims: dict[str, int], values: dict, extras: dict) -> torch.tensor:
        res = sum(
            w * t.evaluate(values, extras=extras).align_to(*self.edges)
            for w, t in zip(self.weights, self.tensors)
        )
        assert res.names == tuple(self.edges), f"Expected {self.edges}, got {res.names}"
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

    def __init__(self, value):
        self.value = value
        self.hash = hash(value)

    def __eq__(self, other):
        if isinstance(other, MatchEdgesKey):
            return self.value.is_isomorphic(other.value, match_edges=True)
        return False

    def __hash__(self):
        return self.hash


def add_structural_graph(G, tensor, root_edge_label=None):
    """Computes the structural graph of tensor, and unions it with G."""
    # We assert that Gx has nodes named [0, Gx.number_of_nodes())
    assert set(G.nodes()) == set(range(G.number_of_nodes())), f"{G.nodes()}"
    Gx, x_edges = tensor.structural_graph()
    assert set(Gx.nodes()) == set(range(Gx.number_of_nodes())), f"{Gx.nodes()}"
    # We also assume that 0 is the root
    Gx = nx.relabel_nodes(Gx, {i: i + G.number_of_nodes() for i in range(Gx.number_of_nodes())})
    x_root = G.number_of_nodes()
    x_edges = {e: i + G.number_of_nodes() for e, i in x_edges.items()}
    G = nx.union(G, Gx)
    # Make sure to connect the root of Gx to the root of G, possibly with an "edge label"
    if root_edge_label is not None:
        e = G.number_of_nodes()
        G.add_node(e, name=root_edge_label)
        G.add_edge(x_root, e)
        G.add_edge(e, 0)
    else:
        G.add_edge(x_root, 0)
    assert set(x_edges.keys()) == set(tensor.edges), f"{x_edges.keys()} != {tensor.edges}"
    return G, x_edges


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
