from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from string import ascii_letters
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Generator,
    Iterable,
    KeysView,
    Optional,
    Sequence,
    Union,
    cast,
    final,
)

if TYPE_CHECKING:  # type-only: tensor.py imports tensorgrad.structure lazily
    from tensorgrad.structure import Structure
from abc import ABC, ABCMeta
from fractions import Fraction
from numbers import Number

import networkx as nx
from sympy import Symbol

# Plain numeric scalars accepted by Tensor's arithmetic operators and as Sum
# weights. `numbers.Number` covers numeric types that only register with the
# ABC at runtime (e.g. sympy/numpy scalars); int/float/Fraction are listed
# explicitly because static checkers don't see virtual (ABC-registered)
# subclassing, so `int` is not statically a `Number`.
Scalar = Union[int, float, Fraction, Number]

# Runtime twin of `Scalar` for isinstance checks. Semantically identical to
# `isinstance(x, Number)` (int/float/Fraction all register with the Number
# ABC), but listing the concrete types lets the type checker narrow the
# non-scalar branch down to Tensor.
_SCALAR_TYPES = (int, float, Fraction, Number)

# Lazily imported tensorgrad.structure module (declarative structural
# identity: the graph and fingerprint folds). Loaded on first use to keep
# `import tensorgrad` light; always available (stdlib + networkx only).
_canon_mod: Optional[ModuleType] = None


def _get_canon() -> ModuleType:
    global _canon_mod
    if _canon_mod is None:
        import tensorgrad.structure as canon

        _canon_mod = canon
    return _canon_mod


# Lazily imported operation modules (tensor.py defines the data types; the
# operations over the closed node-type set live in their own modules).
# tensorgrad.simplify holds the simplification rule catalog + engine and
# tensorgrad.grad the differentiation rules; both import this module, so the
# back edges here must be lazy.
_simplify_mod: Optional[ModuleType] = None
_grad_mod: Optional[ModuleType] = None


def _get_simplify() -> ModuleType:
    global _simplify_mod
    if _simplify_mod is None:
        import tensorgrad.simplify as simplify

        _simplify_mod = simplify
    return _simplify_mod


def _get_grad() -> ModuleType:
    global _grad_mod
    if _grad_mod is None:
        import tensorgrad.grad as grad

        _grad_mod = grad
    return _grad_mod


class TensorMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        """Calling Tensor(...) is a shortcut to wrapping the argument in a tensor."""

        # Since we're hacking the metaclass, we need to check if we're being called on a subclass
        if cls is not Tensor:
            return super().__call__(*args, **kwargs)

        if len(args) != 1 or kwargs:
            raise TypeError("Tensor(...) expects exactly one argument, e.g. Tensor(1).")
        val = args[0]

        if isinstance(val, Number):
            if val == 0:
                return Zero()
            elif val == 1:
                return Product([])
            else:
                return Sum([Product([])], [val])

        if isinstance(val, Symbol):
            return Delta(val)

        if isinstance(val, Tensor):
            return val

        raise ValueError(f"Invalid argument {val}")


class Tensor(metaclass=TensorMeta):
    # Set by concrete subclasses in __init__ (declared here so the type is
    # known; `shape` still uses hasattr to detect abstract tensors).
    _shape: dict[str, Symbol]
    _symmetries: set[frozenset[str]]

    @property
    def edges(self) -> KeysView[str]:
        """Returns an _ordered_ set of edge names"""
        return self.shape.keys()

    @property
    def shape(self) -> dict[str, Symbol]:
        if not hasattr(self, "_shape"):
            raise NotImplementedError
        return self._shape.copy()

    @property
    def order(self) -> int:
        """The number of edges the tensor has. Same as ndim in torch/numpy"""
        return len(self.shape)

    @final
    def grad(self, x: "Variable", new_names: Optional[dict[str, str]] = None) -> "Tensor":
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
        result = _get_grad().grad_step(self, x, new_names)

        # Check that the shape is preserved
        assert result.shape == self.shape | {n: x.shape[o] for o, n in new_names.items()}

        return result

    def _check_grad(self, x: "Variable", new_names: Optional[dict[str, str]]) -> dict[str, str]:
        # When taking a derivative with respect to a variable, we need some edge names for the derivative.
        # If new_names is not given, we generate names based on the edge names of x.
        # However, we need to make sure they don't clash with any names already present in the tensor.
        # If new_names is already given, we just check to make sure they are not already present in the tensor.
        if new_names is not None:
            if x.edges != new_names.keys():
                raise ValueError(f"The {new_names.keys()=} must match {x.edges=}")
            # Check that the new names don't clash with self.edges
            if used := self.edges & new_names.values():
                raise ValueError(f"{used} are already present in {self.edges=}")
            return new_names

        # Make new names that are _based_ on x.edges and _avoid_ self.edges
        return _unused_edge_names(x.edges, self.edges)

    def _grad(self, x: "Variable", new_names: dict[str, str]) -> "Tensor":
        # Fallback protocol for types outside the core dispatch table in
        # tensorgrad/grad.py (extras like Expectation override this).
        raise NotImplementedError(f"Derivative not implemented for {type(self)}")

    @final
    def rename(self, **kwargs: str) -> "Tensor":
        """
        Rename free edges of this tensor.

        Args:
            kwargs: A dictionary mapping old edge names to new edge names.
                Only free edges can be renamed. Inner edges may get renamed
                if necessary to avoid clashes with the new free edge names.

        Returns:
            A new tensor with the edges renamed according to kwargs.

        Note:
            Renaming a composite (Sum/Product/Function) is LAZY: it wraps the
            tensor in a Rename node in O(1) instead of rebuilding the tree
            (which is exponential when subexpressions are reused — residual
            networks!). The wrapper is isomorphism-transparent; simplify
            merges chains into one wrapper (push_rename_down in
            tensorgrad/simplify.py) and the compiler lowers it as pure edge
            relabeling. Code that pattern-matches tree structure must
            therefore either simplify first or peel wrappers via
            :func:`peel_rename`.
        """

        # Check that the renaming is valid, and return the renaming dictionary.
        if len({kwargs.get(e, e) for e in self.edges}) != len(self.edges):
            raise ValueError(f"Renamed an edge to an existing edge name. {self.edges=} {kwargs=}")

        # Restrict renaming to free edges
        kwargs = {new: old for new, old in kwargs.items() if new in self.edges}

        if all(k == v for k, v in kwargs.items()):
            return self
        if isinstance(self, (Sum, Product, Function)):
            result = Rename(self, kwargs)
        else:
            # Leaves (Variable/Delta/Constant) rename cheaply and directly;
            # Rename merges mappings; Derivative/Expectation re-wrap in O(1).
            result = self._rename(**kwargs)

        # Check that the shape is preserved
        assert result.shape == {kwargs.get(e, e): s for e, s in self.shape.items()}

        return result

    def _rename(self, **kwargs: str) -> "Tensor":
        # Override this method to implement renaming
        raise NotImplementedError

    @final
    def simplify(self, args: Optional[dict[str, Any]] = None) -> "Tensor":
        """
        Apply simplification rules to this tensor.

        This may rename inner edges but should never change the free edges.

        Args:
            args: Optional dictionary of arguments controlling the simplification.
                The knobs (grad_steps, expand, memoize, ...) are documented at
                the engine in tensorgrad/simplify.py.

        Returns:
            A simplified version of this tensor.
        """
        return _get_simplify().simplify(self, args)

    def _simplify(self, args: dict[str, Any]) -> "Tensor":
        # Fallback protocol for types outside the core dispatch table in
        # tensorgrad/simplify.py (extras like Expectation override this).
        return self

    def full_simplify(self, expand: bool = True) -> "Tensor":
        """Applies multiple simplification rules until the expression no longer changes"""
        expr = self
        while (new := expr.simplify()) != expr:
            expr = new
        if expand:
            expr = expr.simplify({"expand": True})
            while (new := expr.simplify({"expand": True})) != expr:
                expr = new
        return expr

    @final
    def substitute(self, x: "Variable", y: "Tensor") -> "Tensor":
        """Substitute a variable with a tensor.

        Sharing-preserving and memoized: each DAG node is rewritten at most
        once (keyed by object identity), and unchanged subtrees are returned
        as-is. The stock recursion rebuilt the whole tree per call, which is
        exponential on expressions with shared subexpressions (residual
        networks)."""
        # Ideally we'd like y to be able to have extra edges that are not in x
        # but which will be broadcasted as new edges of the tensor. This will
        # allow us to implement numerical expectation, and other situations where
        # we originally didn't have broadcasting, but we want to add it.
        # Maybe we could even remove broadcasted edges of x as well.
        # The tricky thing is to ensure it doesn't clash with the existing edges.
        return _substitute_memo(self, x, y, {})

    def _substitute(self, x: "Variable", y: "Tensor", memo: dict) -> "Tensor":
        """Override this method to implement substitution.
        Recurse into children via _substitute_memo(child, x, y, memo)."""
        raise NotImplementedError

    def __hash__(self) -> int:
        # Isomorphism-invariant hash (a == b implies equal hashes), computed
        # compositionally and memoized on the object by structure.canon_info,
        # so shared subexpressions are hashed once.
        return _get_canon().structural_hash(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Tensor):
            return self.is_isomorphic(other)
        return False

    @final
    def depends_on(self, x: "Variable") -> bool:
        """Check if this tensor depends on the variable x.

        Memoized per (self, x) object pair: the raw recursion visits every
        root-to-leaf PATH, which is exponential on shared-subtree DAGs
        (residual networks: x = x + f(x) doubles the path count per layer),
        and Derivative._simplify queries it per Derivative node. Same disease
        and cure as the memoized Tensor.substitute.
        """
        # (cast: pyright resolves instance `__dict__` through the metaclass as
        # a read-only mappingproxy; instances have a plain mutable dict.)
        cache = cast(dict[str, Any], self.__dict__).setdefault("_depends_on_cache", {})
        if (hit := cache.get(id(x))) is None:
            cache[id(x)] = hit = (self._depends_on(x), x)  # x ref pins the id
        return hit[0]

    def _depends_on(self, x: "Variable") -> bool:
        raise NotImplementedError

    def structure(self) -> "Structure":
        """Declarative structural identity of this node (the single source of
        truth for isomorphism and hashing; see tensorgrad/structure.py).

        Returns a Structure(label, children, child_roles, junctions) from
        which both the networkx structural graph and the canonical
        fingerprints are derived by generic folds."""
        raise NotImplementedError

    def edge_structural_graph(
        self, match_edges: bool = True, edge_names: Optional[dict[str, Any]] = None
    ) -> tuple[nx.MultiDiGraph, list[str]]:
        """
        Build a structural graph of the tensor with dummy nodes for outer edges.

        Args:
            match_edges: If True the names are used to match edges.
            edge_names: An optional mapping of edge names.

        Returns:
            A tuple (G, edge_list) where G is the graph and edge_list is the list of edge names.
        """
        from tensorgrad.structure import build_graph

        if edge_names is None:
            edge_names = {}

        G, edges = build_graph(self)

        for e in edges.keys():
            if e not in edge_names:
                edge_names[e] = ("Outer Edge", e) if match_edges else ""

        for e, node in edges.items():
            n = G.number_of_nodes()
            G.add_node(n, name=edge_names[e])
            G.add_edge(node, n)
        return G, list(edges.keys())

    def graph_to_string(self) -> str:
        """
        Returns an ASCII tree-like representation of the structural graph.

        Returns:
            A string showing the graph structure.
        """
        G, _ = self.edge_structural_graph(match_edges=True)
        # (networkx accepts an attribute name for with_labels; the stub says bool)
        return "\n".join(
            nx.generate_network_text(G, with_labels="name", sources=[0])  # pyright: ignore[reportArgumentType]
        )

    # Overloaded operators.
    # All binary operators accept a plain numeric scalar (int, float,
    # Fraction, or any runtime numbers.Number) or another Tensor, and return a
    # Tensor. A scalar is treated as a (weighted) order-0 tensor, so e.g.
    # `2 * x == x * 2`, `1 - x`, `x / 3` and `1 @ x` (scalar product) all work.

    def __add__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        w: Any = 1  # Any: numbers.Number supports no static arithmetic
        if isinstance(other, _SCALAR_TYPES):
            other, w = Product([]), other
        return Sum([self, other], [1, w])

    def __radd__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        return self + other

    def __rsub__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        # Handle `other - self`
        return -self + other

    def __sub__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        # Handle `self - other`
        w: Any = 1  # Any: numbers.Number supports no static arithmetic
        if isinstance(other, _SCALAR_TYPES):
            other, w = Product([]), other
        return Sum([self, other], [1, -w])

    def __neg__(self) -> "Tensor":
        return Sum([self], [-1])

    def __matmul__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        if isinstance(other, _SCALAR_TYPES):
            other = Sum([Product([])], [other])
        return Product([self, other])

    def __rmatmul__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        # `scalar @ tensor` is scalar multiplication (same as `scalar * tensor`).
        if isinstance(other, _SCALAR_TYPES):
            return self * other
        return self @ other

    def __rmul__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        return self * other

    def __mul__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        """
        Contract self with other using tensor product and contraction rules.
        """
        if isinstance(other, _SCALAR_TYPES):
            if other == 0:
                return Zero(_symmetries=None, **self.shape)
            if other == 1:
                return self
            return Sum([self], [other])

        from tensorgrad.functions import prod  # Avoid circular import

        return prod(self, other)

    def __rtruediv__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        # Handle other / self
        from tensorgrad.functions import pow  # Avoid circular import

        return other * pow(self, -1)

    def __truediv__(self, other: Union[Scalar, "Tensor"]) -> "Tensor":
        # Handle self / other
        from tensorgrad.functions import pow  # Avoid circular import

        if isinstance(other, int):
            return Sum([self], [Fraction(1, other)])
        if isinstance(other, _SCALAR_TYPES):
            w: Any = other  # Any: numbers.Number supports no static arithmetic
            return Sum([self], [1 / w])
        return self * pow(other, -1)

    def __pow__(self, other: Union[int, Fraction]) -> "Tensor":
        from tensorgrad.functions import pow  # Avoid circular import

        if not isinstance(other, (int, Fraction)):
            raise ValueError("Only integer and fractional powers are supported.")
        return pow(self, other)

    def is_isomorphic(
        self, other: "Tensor", match_edges: bool = False, edge_names: None | dict[str, Any] = None
    ) -> bool:
        """
        Test whether this tensor is isomorphic to another tensor.

        Args:
            other: The tensor to compare with.
            match_edges: Whether to require a matching of edge names.
            edge_names: Optional mapping to use for edge renaming.

        Returns:
            True if the tensors are isomorphic; False otherwise.
        """
        canon = _get_canon()
        a, b = canon.canon_info(self), canon.canon_info(other)
        # Sound reject: any isomorphism (with or without edge matching)
        # implies equal invariant hashes.
        if a.coarse_fp != b.coarse_fp:
            return False
        if a.refined_fp == b.refined_fp:
            if not match_edges and not edge_names:
                # Sound accept: equal fingerprints imply isomorphic.
                return True

            # Name-sensitive accept: by (I1) equal fingerprints yield a
            # color-preserving isomorphism, and by (I2) any
            # color-preserving edge permutation is an automorphism, so a
            # label-preserving isomorphism exists iff the (color, label)
            # multisets agree. (Labels mirror edge_structural_graph.)
            def label(e: str):
                default = ("Outer Edge", e) if match_edges else ""
                return default if edge_names is None else edge_names.get(e, default)

            if Counter((a.refined_colors[e], label(e)) for e in self.edges) == Counter(
                (b.refined_colors[e], label(e)) for e in other.edges
            ):
                return True
        # Ambiguous (hash-equal but fingerprint/label-distinct): only now
        # pay for the exact nx isomorphism test.
        G1, _ = self.edge_structural_graph(match_edges=match_edges, edge_names=edge_names)
        G2, _ = other.edge_structural_graph(match_edges=match_edges, edge_names=edge_names)
        return nx.is_isomorphic(G1, G2, node_match=lambda n1, n2: n1.get("name") == n2.get("name"))
        # return nx.vf2pp_is_isomorphic(G1, G2, node_label="name")

    def isomorphisms(self, other: "Tensor") -> Generator[dict[str, str], None, None]:
        """
        Yield all isomorphisms (edge renamings) between self and other.

        Args:
            other: The other tensor to compare.

        Yields:
            Dictionaries mapping edge names in self to edge names in other.
        """
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

    @staticmethod
    def _check_edges(edges: Iterable[str]) -> list[str]:
        """
        Validate and standardize an iterable of edge names.

        Args:
            edges: An iterable of edge names.

        Returns:
            A list of edge names.

        Raises:
            ValueError: If any edge is not a string.
        """
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
        """
        Check and merge a shape given as an iterable and a dict.

        Args:
            shape0: An iterable of Sympy symbols.
            shape1: A dictionary mapping edge names to Sympy symbols.

        Returns:
            A dictionary merging the two representations.

        Raises:
            ValueError: If inputs are not of the expected types.
        """
        shape0 = tuple(shape0)
        if not isinstance(shape0, tuple) or not all(isinstance(s, Symbol) for s in shape0):
            raise ValueError("Shape0 must be a tuple of sympy symbols")
        if not isinstance(shape1, dict) or not all(isinstance(s, Symbol) for s in shape1.values()):
            raise ValueError("Shape1 must be a dict of sympy symbols")
        # Check for duplicate positional dimension names
        names = [s.name for s in shape0]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate positional dimension names: {duplicates}")
        shape0_dict = {s.name: s for s in shape0}
        if double_keys := shape0_dict.keys() & shape1.keys():
            raise ValueError(f"Duplicate edge names: {double_keys}")
        return shape0_dict | shape1

    @staticmethod
    def _check_symmetries(
        shape: dict[str, Symbol], symmetries: Union[str, set[frozenset[str]], None]
    ) -> set[frozenset[str]]:
        """
        Validate the symmetries for a tensor.

        Args:
            shape: A dictionary mapping edge names to sizes.
            symmetries: Either a string (with groups separated by commas) or a set of frozensets.

        Returns:
            A set of frozensets representing the symmetry groups.

        Raises:
            ValueError: If the symmetries are inconsistent with the shape.
        """
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
        name: str,
        *shape0: Symbol,
        _symmetries: None | str | set[frozenset[str]] = None,
        _constraints: None | Iterable[tuple["Tensor", "Tensor"]] = None,
        **shape1: Symbol,
    ):
        """
        A tensor holding a variable.

        Args:
            name: The name of the variable.
            shape0: Positional Sympy symbols for dimensions.
            _symmetries: Optional symmetries (as a string or set of frozensets).
            _constraints: Optional value constraints as (lhs, rhs) Tensor equation
                pairs. Use :meth:`with_constraint` instead of passing this directly.
            shape1: Keyword arguments for dimensions.
        """
        self.name = name
        self._shape = self._check_shape(shape0, shape1)
        self._symmetries = self._check_symmetries(self._shape, _symmetries)
        self._constraints = self._check_constraints(_constraints)
        # Deterministic, hash-stable identity keys for the equations, used by
        # structure() (the structural-identity label) and repr (tensors aren't
        # orderable).
        self._constraint_keys = tuple(sorted(self._equation_key(l, r) for l, r in self._constraints))

    @staticmethod
    def _equation_key(lhs: "Tensor", rhs: "Tensor") -> tuple:
        canon = _get_canon()
        return (
            canon.structural_hash(lhs),
            tuple(sorted(lhs.edges)),
            canon.structural_hash(rhs),
            tuple(sorted(rhs.edges)),
        )

    def _check_constraints(
        self, constraints: Optional[Iterable[tuple["Tensor", "Tensor"]]]
    ) -> tuple[tuple["Tensor", "Tensor"], ...]:
        if constraints is None:
            return ()
        # Dedup by the edge-name-aware key, NOT by tensor equality (which is
        # isomorphism and would collapse distinct equations that differ only
        # in free-edge names — e.g. the i-sum and j-sum of a symmetric matrix).
        constraints = tuple({self._equation_key(l, r): (l, r) for l, r in constraints}.values())
        for lhs, rhs in constraints:
            if not (isinstance(lhs, Tensor) and isinstance(rhs, Tensor)):
                raise ValueError("Constraints must be (lhs, rhs) Tensor equation pairs")
            if lhs.shape != rhs.shape:
                raise ValueError(f"Constraint sides must have equal shape: {lhs.shape} != {rhs.shape}")
            if not any(v.name == self.name and v.shape == self.shape for v in self._find_variables(lhs)):
                raise ValueError(f"Constraint lhs does not mention variable {self.name!r}")
        # The equation SET must be invariant under the variable's symmetry
        # orbits: an automorphism may permute an orbit's edges, so for every
        # orbit transposition the swapped equation must itself be declared.
        # (Generalizes the old "constrained edges must cover the orbit" check
        # to arbitrary equations.)
        bare = None
        for orbit in self._symmetries:
            orbit = sorted(orbit)
            for a, b in zip(orbit, orbit[1:]):
                swap = {a: b, b: a}
                if bare is None:
                    bare = Variable(self.name, **self.shape, _symmetries=self._symmetries, _constraints=None)
                swapped_anchor = bare.rename(**swap)
                for lhs, rhs in constraints:
                    sl = lhs.substitute(bare, swapped_anchor)
                    sr = rhs.substitute(bare, swapped_anchor)
                    sl = sl.rename(**{k: v for k, v in swap.items() if k in sl.edges})
                    sr = sr.rename(**{k: v for k, v in swap.items() if k in sr.edges})
                    if not any(
                        sl.is_isomorphic(l2, match_edges=True) and sr.is_isomorphic(r2, match_edges=True)
                        for l2, r2 in constraints
                    ):
                        raise ValueError(
                            f"Constraint equations on {self.name!r} are not closed under the "
                            f"symmetry orbit {orbit}: the {a!r}<->{b!r} swap of an equation is "
                            "not itself declared (declare it for all edges of the orbit, or none)"
                        )
        return constraints

    @staticmethod
    def _find_variables(t: "Tensor", cap: int = 64) -> list["Variable"]:
        """Bounded walk collecting Variable leaves (cap guards huge subtrees:
        beyond it we may only MISS constraint matches, never miscompute)."""
        seen: set[int] = set()
        out: list[Variable] = []
        stack = [t]
        while stack and len(seen) < cap:
            u = stack.pop()
            if id(u) in seen:
                continue
            seen.add(id(u))
            if isinstance(u, Variable):
                out.append(u)
                continue
            stack.extend(getattr(u, "factors", ()) or ())
            stack.extend(getattr(u, "terms", ()) or ())
            stack.extend(getattr(u, "inputs", ()) or ())
            if (inner := getattr(u, "tensor", None)) is not None:
                stack.append(inner)
        return out

    @staticmethod
    def _strip_constraints(t: "Tensor") -> "Tensor":
        """Replace every constrained Variable in t with its bare twin.
        Equations are stored anchored on bare variables (storing them on the
        constrained variable would make its identity self-referential), so
        matching compares bare against bare."""
        for v in Variable._find_variables(t):
            if v._constraints:
                bare = Variable(v.name, **v.shape, _symmetries=v._symmetries, _constraints=None)
                t = t.substitute(v, bare)
        return t

    def with_symmetries(self, symmetries: str | set[frozenset[str]]) -> "Variable":
        # Constraint equations are anchored on this variable's bare (constraint-
        # free) twin, whose identity includes the symmetries — re-anchor them.
        constraints = self._constraints
        if constraints:
            old_bare = Variable(self.name, **self.shape, _symmetries=self._symmetries, _constraints=None)
            new_bare = Variable(self.name, **self.shape, _symmetries=symmetries, _constraints=None)
            constraints = tuple(
                (l.substitute(old_bare, new_bare), r.substitute(old_bare, new_bare)) for l, r in constraints
            )
        return Variable(self.name, **self.shape, _symmetries=symmetries, _constraints=constraints)

    def with_eq_constraint(self, lhs: "Tensor", rhs: Union["Tensor", int, float]) -> "Variable":
        """Declare a value constraint on this variable, as an EQUATION written
        in tensorgrad itself: ``with_eq_constraint(lhs, rhs)`` states that the
        expression ``lhs`` (which must mention this variable) always EQUALS the
        expression ``rhs``. During ``simplify``, any subnetwork isomorphic to
        ``lhs`` is replaced by ``rhs``.

        Examples::

            y = Variable("y", b, v)
            y = y.with_eq_constraint(F.sum(y, ["v"]), Ones(b))  # rows sum to 1
            W = Variable("W", d=d, i=i)
            W = W.with_eq_constraint(W @ W.rename(i="j"), Delta(i, "i", "j"))  # orthonormal
            u = Variable("u", i)
            u = u.with_eq_constraint(F.sum(u * u, ["i"]), 1)    # unit norm

        A plain number ``rhs`` is promoted to ``number * Ones(...)`` over the
        lhs's free edges.

        The rhs must be strictly smaller (node count) than the lhs: equations
        are applied as left-to-right rewrite rules during simplify, so
        shrinking is what guarantees termination.

        Matching currently recognizes equations whose (simplified) lhs is a
        contraction of two factors — which covers sums against ones-vectors,
        Hadamard powers (``x*x`` canonicalizes to ``pow``), and two-instance
        products like orthogonality. If the variable has symmetries, the
        declared equation set must be closed under the symmetry orbits
        (checked at declaration; e.g. sum-to-one over edge ``i`` of an
        ``i j``-symmetric matrix requires the ``j`` equation too).

        NOTE: Constraints describe the VALUES the variable will hold; they do
        not change differentiation. ``grad`` still treats the variable as a
        free, unconstrained parameter (derivatives are taken in the full
        ambient space).
        """
        lhs = self._strip_constraints(lhs).simplify()
        if isinstance(rhs, (int, float)):
            ones = Product([Delta(s, e) for e, s in lhs.shape.items()])
            rhs = ones if rhs == 1 else rhs * ones
        rhs = self._strip_constraints(rhs).simplify()
        if (nr := self._node_count(rhs)) >= (nl := self._node_count(lhs)):
            raise ValueError(
                f"Constraint rhs must be strictly smaller than lhs ({nr} vs {nl} nodes): "
                "equations rewrite lhs -> rhs during simplify, and shrinking is what "
                "guarantees termination"
            )
        return Variable(
            self.name,
            **self.shape,
            _symmetries=self._symmetries,
            _constraints=self._constraints + ((lhs, rhs),),
        )

    @staticmethod
    def _node_count(t: "Tensor") -> int:
        seen: set[int] = set()
        stack = [t]
        while stack:
            u = stack.pop()
            if id(u) in seen:
                continue
            seen.add(id(u))
            stack.extend(getattr(u, "factors", ()) or ())
            stack.extend(getattr(u, "terms", ()) or ())
            stack.extend(getattr(u, "inputs", ()) or ())
            if (inner := getattr(u, "tensor", None)) is not None:
                stack.append(inner)
        return len(seen)

    def __repr__(self) -> str:
        args = [f'"{self.name}", {", ".join(self.edges)}']
        symmetries = ""
        if self._symmetries != {frozenset({e}) for e in self.edges}:
            groups = ", ".join(sorted(" ".join(sorted(group)) for group in self._symmetries))
            symmetries = f'.with_symmetries("{groups}")'
        constraints = (
            f".with_eq_constraint(<{len(self._constraints)} equations>)" if self._constraints else ""
        )
        return f"Variable({', '.join(args)}){symmetries}{constraints}"

    def structure(self) -> "Structure":
        from tensorgrad.structure import Structure, size_key

        # Constraints are part of the label (like the variable name itself),
        # so a constrained variable is never isomorphic to an unconstrained
        # one, and rename/substitute (which go through isomorphism) keep
        # track of them for free.
        label = ("Variable", self.name)
        if self._constraints:
            label += (self._constraint_keys,)
        # One junction per symmetry orbit: its edges are the same wire.  The
        # orbit port is named after the (sorted) edge names, so variables are
        # only isomorphic to variables with the SAME edge names — the names
        # are load-bearing identity for Variables (unlike every other node).
        junctions = []
        for orbit in self._symmetries:
            e0 = next(iter(orbit))
            port = ("port", ("orbit", size_key(self._shape[e0]), " ".join(sorted(orbit))))
            junctions.append(frozenset({port} | {("free", e) for e in orbit}))
        return Structure(label, junctions=frozenset(junctions))

    def _rename(self, **kwargs: str) -> Tensor:
        if not kwargs:
            return self
        return Rename(self, kwargs)

    def _substitute(self, x: "Variable", y: "Tensor", memo: dict) -> "Tensor":
        # Name pre-check: Variables with different names are never isomorphic
        # (the name is part of the structural graph), so the expensive
        # isomorphism __eq__ only runs on same-named variables.
        if self is x or (self.name == x.name and x == self):
            return y
        return self

    def _depends_on(self, x: "Variable") -> bool:
        return x == self


################################################################################
# Constants
################################################################################


class Rename(Tensor):
    """
    A tensor that renames the edges of an inner tensor.

    This is used to change free edge names (and sometimes inner edge names)
    without modifying the underlying contraction.
    """

    def __init__(self, inner: Tensor, rename: dict[str, str]):
        """
        Initialize a Rename tensor.

        Args:
            inner: The tensor to be renamed.
            rename: A dictionary mapping old edge names to new names.
        """
        self.name = "Rename"
        self.tensor = inner
        self.mapping = {k: v for k, v in rename.items() if k != v}
        self._shape = {rename.get(e, e): s for e, s in inner.shape.items()}

    def __repr__(self) -> str:
        argstring = ", ".join(f'{k}="{v}"' for k, v in self.mapping.items())
        return f"{self.tensor}.rename({argstring})"

    def _depends_on(self, x: "Variable") -> bool:
        return self.tensor.depends_on(x)

    @classmethod
    def merge_renames(cls, *renames: dict[str, str]) -> dict[str, str]:
        """
        Merge several renaming dictionaries into one.

        Args:
            *renames: Arbitrary number of renaming dictionaries.

        Returns:
            A single dictionary representing the merged renaming.
        """
        merged: dict[str, str] = {}
        for rename in renames:
            used = merged.keys() | merged.values()
            # Apply the rename to the existing chains
            for o, n in merged.items():
                merged[o] = rename.get(n, n)
            # Add new renames
            for o, n in rename.items():
                if o not in used:
                    merged[o] = n
        return merged

    def _rename(self, **kwargs: str) -> Tensor:
        return Rename(self.tensor, self.merge_renames(self.mapping, kwargs))

    def _substitute(self, x: "Variable", y: "Tensor", memo: dict) -> "Tensor":
        inner = _substitute_memo(self.tensor, x, y, memo)
        return self if inner is self.tensor else Rename(inner, self.mapping)

    def structure(self) -> "Structure":
        from tensorgrad.structure import Structure

        # Rename is the only tensor that doesn't exist structurally
        # (transparent=True): the folds splice the inner tensor through and
        # just relocate its free edges.  This allows
        # Variable(i, j=i).with_symmetries("i j") to be isomorphic to
        # Rename(Variable(j, i=j).with_symmetries("i j"), {"i": "j", "j": "i"})
        junctions = frozenset(
            frozenset({("child", 0, o), ("free", self.mapping.get(o, o))}) for o in self.tensor.edges
        )
        return Structure(("Rename",), (self.tensor,), ("inner",), junctions, transparent=True)


class Constant(Tensor, ABC):
    """
    An abstract tensor that represents a constant (non-variable) tensor.
    """

    def __init__(
        self, *shape0: Symbol, _symmetries: None | str | set[frozenset[str]] = None, **shape1: Symbol
    ):
        """
        A constant tensor with the given edges.

        Args:
            shape0: Positional dimensions (as Sympy symbols).
            _symmetries: Optional symmetry groups.
            shape1: Keyword dimensions.
        """
        self.name = "NotImplemented"
        self._shape = self._check_shape(shape0, shape1)
        self._symmetries = self._check_symmetries(self._shape, _symmetries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.shape})"

    def with_symmetries(self, symmetries: str | set[frozenset[str]]) -> "Constant":
        return type(self)(_symmetries=symmetries, **self._shape)

    def structure(self) -> "Structure":
        from tensorgrad.structure import Structure, size_key

        # Like Variable, but orbits are anonymous (no edge names in the
        # port): orbits of equal size are interchangeable as blocks.
        junctions = []
        for orbit in self._symmetries:
            e0 = next(iter(orbit))
            port = ("port", ("orbit", size_key(self._shape[e0])))
            junctions.append(frozenset({port} | {("free", e) for e in orbit}))
        return Structure(("Constant", type(self).__name__), junctions=frozenset(junctions))

    def _substitute(self, x: "Variable", y: "Tensor", memo: dict) -> "Tensor":
        return self

    def _rename(self, **kwargs: str) -> Tensor:
        return type(self)(_symmetries=None, **{kwargs.get(e, e): s for e, s in self.shape.items()})

    def _grad(self, x: Variable, new_names: dict[str, str]) -> Tensor:
        # Thin delegate to the zero-gradient rule (tensorgrad.grad
        # dispatches Delta/Zero there directly; this protocol fallback covers
        # the Constant subclasses outside the core table: Convolution,
        # Reshape, Affine).
        return _get_grad().grad_constant(self, x, new_names)

    def _depends_on(self, x: "Variable") -> bool:
        return False


class Delta(Constant):
    """The "Delta" tensor is defined by C_{i,j,k} = 1 if i == j == k, else 0
    Or alternatively as Δₙ = ∑ₓ (eₓ)⊗ⁿ, where are e_i are basis vectors
    For order 2, this is the identity matrix.
        Note: For higher order, the identity tensor is the product of n identity matrices,
        (order 2 Delta's), not the order-n Delta tensor.
    For order 0, the Delta tensor is defined as |n|, where n is the size/dimension of the tensor.
    """

    def __init__(self, size: Symbol, *edges: str):
        """
        Initialize a Delta tensor.

        Args:
            size: The size (dimension) of the tensor.
            edges: The names of the tensor edges.
        """
        edge_list = self._check_edges(edges)
        super().__init__(_symmetries=None, **{e: size for e in edge_list})
        assert isinstance(size, (Symbol, Number)), "Size must be a sympy symbol"
        self._size = size

    @property
    def size(self) -> Symbol:
        return self._size

    def __repr__(self) -> str:
        if not self.edges:
            return f"Delta({self.size})"
        return f'Delta({self.size}, "{", ".join(self.edges)}")'

    def _rename(self, **kwargs: str) -> Tensor:
        return Delta(self.size, *[kwargs.get(e, e) for e in self.edges])

    def structure(self) -> "Structure":
        from tensorgrad.structure import Structure, size_key

        # ALL edges are the same wire: one junction with one port.  The order
        # of the Delta is implied by the junction's free-edge count.
        sk = size_key(self._size)
        junction = frozenset({("port", ("Delta", sk))} | {("free", e) for e in self.edges})
        return Structure(("Delta", sk), junctions=frozenset({junction}))

    # The Delta pair rules (merging copy tensors, removing identity matrices,
    # applying constraint equations) live in tensorgrad/simplify.py.


class Zero(Constant):
    """Matrix such that Z_{i,j,k} = 0 for all i, j, k"""


def Ones(*shape0: Symbol, **shape1: Symbol) -> Tensor:
    """Matrix such that O_{i,j,k} = 1 for all i, j, k"""
    # Implemented in practice by a simple outer product of Delta's
    shape = Tensor._check_shape(shape0, shape1)
    return Product([Delta(s, e) for e, s in shape.items()])


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
    inputs : tuple[Set[str], ...]
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
    # (Any set-like collection of edge names: set, frozenset or dict keys.)
    edges: AbstractSet[str]
    # Defines the number of input tensors the function takes, and what edges it needs from each
    inputs: Sequence[AbstractSet[str]]

    # TODO: Functions should support symmetric outputs, just like Constant and Variable.
    # Actually we could define symmetries for the inputs too, since the order doesn't always matter.

    def derivative(self, i: int, new_edges: Optional[dict[str, str]] = None) -> "FunctionSignature":
        """
        Returns a new FunctionSignature that represents the derivative
        with respect to the i-th input, and where edges are created
        according to `new_edges` a dict {old_name : new_name}.
        """
        if new_edges is None:
            new_edges = {}
        if self.inputs[i] != new_edges.keys():
            raise ValueError(f"Expected {self.inputs[i]} but got {new_edges.keys()}")
        if self.edges & set(new_edges.values()):
            raise ValueError(f"New edges {new_edges.values()} clash with {self.edges=}")
        return FunctionSignature(f"D_{i}{self.name}", self.edges | set(new_edges.values()), self.inputs)

    def simplify(self, t: "Function", args: dict[str, Any], /) -> Tensor:
        """
        Simplifies the Function tensor that uses this signature.
        Default implementation returns `t` unchanged.
        Subclasses can override this method for custom simplifications.

        (The parameters are positional-only: subclasses name the Function
        parameter differently, e.g. `f` or `func`.)
        """
        return t

    def param_key(self) -> tuple:
        """Structural identity of the signature's parameters beyond `name`.

        Enters the Function's structural label (hashing/isomorphism).
        Subclasses whose behavior depends on parameters NOT reflected in
        their `name` should override this (pow/scale already bake `k`/`alpha`
        into the name).  Must return a (nested) tuple of str/int/bool."""
        return ()

    def __repr__(self) -> str:
        return f"FunctionSignature('{self.name}', {set(self.edges)}, {[set(s) for s in self.inputs]})"


def function(
    name: str,
    output_shape: dict[str, Symbol],
    *inputs: tuple[Tensor | str, ...],
) -> "Function":
    """
    Create a Function tensor from a name, output shape and inputs.

    Args:
        name: The function name.
        output_shape: A dictionary mapping output edge names to sizes.
        inputs: A sequence of tuples where each tuple starts with a tensor (or string)
                followed by the required edge names.

    Returns:
        A Function tensor.
    """
    # Each input tuple is (tensor, edge_name, ...); the heterogeneous tuple
    # type can't express that, hence the casts.
    return Function(
        FunctionSignature(
            name,
            output_shape.keys(),
            [cast(set[str], set(es)) for (t, *es) in inputs],
        ),
        [cast(Tensor, t) for (t, *es) in inputs],
        output_shape,
    )


class Function(Tensor):
    """
    A tensor that represents a function applied to one or more input tensors.

    Attributes:
        signature: The FunctionSignature for this function.
        inputs: A list of input tensors.
        shape_out: A dictionary mapping output edge names to sizes.
    """

    def __init__(
        self,
        signature: FunctionSignature,
        inputs: Iterable[Tensor],
        shape_out: dict[str, Symbol],
    ):
        """
        A function tensor that takes one or more input tensors and produces an output tensor.

        Args:
            signature: The FunctionSignature defining this function, or a string giving the function name.
            inputs: The input tensors and their input edge names. [(t0, e00, e01, ...), (t1, e10, ...), ...]
            shape_out: The names of the output edges of this function.
        """
        self.shape_out = self._check_shape([], shape_out)
        self.inputs = list(inputs)
        self.signature = signature

        # Validation
        if self.shape_out.keys() != signature.edges:
            raise ValueError(f"Signature doesn't match shape: {self.shape_out.keys()=} != {signature.edges}")
        if len(self.inputs) != len(signature.inputs):
            raise ValueError(f"Expected {len(signature.inputs)} inputs, got {len(self.inputs)}")
        for t, es in zip(self.inputs, signature.inputs):
            if es - t.edges:
                raise ValueError(f"Input edges {es} not subset of {t.edges=}")

        # Build final shape = union of output edges + broadcast edges from each input
        # Note that this approach pushes all the broadcasted edges to the end of the shape.
        # But we don't care about the order of the edges, so it's fine.
        self._shape = dict(self.shape_out)
        # We allow "broadcasting" for edges that are not in fn_sig.inputs[i].
        # So if an input has edges that are not used, they're broadcasted.
        for t, es in zip(self.inputs, self.signature.inputs):
            # any edges in t.edges - es are "broadcast"
            broadcast_edges = t.edges - es
            if broadcast_edges & self._shape.keys():
                raise ValueError("Broadcast edges must not be shared between inputs")
            for b in broadcast_edges:
                self._shape[b] = t.shape[b]

    def _rename(self, **kwargs: str) -> Tensor:
        # Names to avoid when generating fresh middle names below.
        used = self.edges | set(kwargs.values())
        for t in self.inputs:
            used |= t.edges
        renamed_inputs = []
        final_rename = {}  # applied on top of the rebuilt Function at the end
        for t, es in zip(self.inputs, self.signature.inputs):
            # Only rename external (broadcast) edges of input tensors.
            rename = {e: v for e, v in kwargs.items() if e in t.edges and e not in es}
            # A rename target may collide with an edge of t that we are not renaming,
            # typically an edge consumed by the function (e.g. renaming broadcast edge
            # 's' -> 't' while the function consumes t's edge 't'). Route such renames
            # through fresh middle names and fix them up with a final Rename,
            # mirroring the middle-name routing in Rename._grad.
            remaining = t.edges - rename.keys()
            colliding = {e: v for e, v in rename.items() if v in remaining}
            if colliding:
                middle = _unused_edge_names(colliding.values(), used)
                used |= set(middle.values())
                for e, v in colliding.items():
                    rename[e] = middle[v]
                    final_rename[middle[v]] = v
            renamed_inputs.append(t.rename(**rename))
        output_rename = {k: v for k, v in kwargs.items() if k in self.shape_out}
        final_rename |= output_rename
        res = Function(
            self.signature,
            renamed_inputs,
            self.shape_out,
        )
        if any(v != k for k, v in final_rename.items()):
            res = Rename(res, final_rename)
        return res

    def structure(self) -> "Structure":
        from tensorgrad.structure import Structure

        # Inputs are ordered (slotted roles 0..n-1).  A consumed edge wires
        # its input to an anonymous per-input port (which consumed edges an
        # input provides is distinguished by the input's own structure, not
        # by the edge names); a non-consumed input edge broadcasts through as
        # a free edge; output edges are ports named after the output edge —
        # output NAMES are part of the function's identity.
        junctions = set()
        for k, (t, es) in enumerate(zip(self.inputs, self.signature.inputs)):
            for e in t.edges:
                if e in es:
                    junctions.add(frozenset({("child", k, e), ("port", ("in", k))}))
                else:
                    junctions.add(frozenset({("child", k, e), ("free", e)}))
        for e in self.shape_out:
            junctions.add(frozenset({("port", ("out", e)), ("free", e)}))
        return Structure(
            ("Function", self.signature.name, self.signature.param_key()),
            tuple(self.inputs),
            tuple(range(len(self.inputs))),
            frozenset(junctions),
        )

    def __repr__(self) -> str:
        args = []
        args.append(repr(self.signature))
        args.append(f"inputs=[{', '.join(map(repr, self.inputs))}]")
        args.append(f"shape_out={self.shape_out}")
        return f"Function({', '.join(args)})"

    def _depends_on(self, x: "Variable") -> bool:
        return any(t.depends_on(x) for t in self.inputs)

    def _substitute(self, x: "Variable", y: "Tensor", memo: dict) -> "Tensor":
        inputs = [_substitute_memo(t, x, y, memo) for t in self.inputs]
        if all(a is b for a, b in zip(inputs, self.inputs)):
            return self
        return Function(self.signature, inputs, self.shape_out)


class Derivative(Tensor):
    """
    A tensor representing the derivative of another tensor with respect to a variable.
    """

    def __init__(self, tensor: Tensor, wrt: Variable, new_names: Optional[dict[str, str]] = None):
        """
        A tensor representing the derivative of another tensor.

        Args:
            tensor: The tensor to differentiate.
            x: The variable with respect to which to differentiate.
            new_names: A mapping for renaming edges (if not provided, it will be generated).
        """
        self.tensor = tensor
        self.x = wrt
        # If no edges were given, pick some names based on x, but avoid clashes with tensor.
        self.new_names = tensor._check_grad(wrt, new_names)
        self._shape = tensor.shape | {self.new_names[e]: wrt.shape[e] for e in wrt.edges}

    def _rename(self, **kwargs: str) -> Tensor:
        # The free edges of Derivative are both the free edges of self.tensor and the new_names.
        # This is the only place where we need to rename the "internal edges" of the tensor.
        return Derivative(
            self.tensor.rename(**kwargs),
            self.x,
            {o: kwargs.get(e, e) for o, e in self.new_names.items()},
        )

    def structure(self) -> "Structure":
        from tensorgrad.structure import Structure

        # The wrt-variable is a slotted child (its full identity counts); each
        # new_names edge is a free edge on the same wire as the corresponding
        # edge of x (so symmetric x-edges make the new edges interchangeable,
        # and the CHOICE of new names is transparent).
        junctions = {frozenset({("child", 1, e), ("free", e)}) for e in self.tensor.edges}
        junctions |= {frozenset({("child", 0, o), ("free", n)}) for o, n in self.new_names.items()}
        return Structure(("Derivative",), (self.x, self.tensor), ("wrt", "tensor"), frozenset(junctions))

    def __repr__(self) -> str:
        return f"Derivative({self.tensor}, {self.x}, {self.new_names})"

    def _depends_on(self, x: "Variable") -> bool:
        return self.tensor.depends_on(x)


################################################################################
# Product
################################################################################


class Product(Tensor):
    """
    A tensor representing the product (contraction) of several tensors.
    """

    def __init__(self, factors: Iterable[Tensor]):
        """
        Initialize a Product tensor.

        Args:
            factors: An iterable of tensors to be multiplied/contracted.
        """
        self.factors = list(factors)
        self._shape = {}
        for edge, ts in _group_edges(self.factors).items():
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

    def _rename(self, **kwargs: str) -> Tensor:
        # Rename the inner edges (contractions) to avoid the new names introduced
        # by the renaming of the free edges.
        new_names = {kwargs.get(e, e) for e in self.edges}
        contractions = {e for t in self.factors for e in t.edges} - self.edges
        # Note the `unused_edge_names` function avoids introducing new clashes
        # between the edges being renamed.
        rename = _unused_edge_names(contractions, new_names)
        # It's safe to add the kwargs to rename, since self._check_rename restricts kwargs to only
        # contain keys that are in self.edges.
        rename |= kwargs
        return Product([t.rename(**rename) for t in self.factors])

    @staticmethod
    def merge(products: list["Product"]) -> "Product":
        """
        Merge several Product tensors into one, renaming inner edges so they are distinct.

        Args:
            products: A list of Product tensors.

        Returns:
            A new Product tensor merging the inputs.
        """
        used_edges = {e for p in products for e in p.edges}
        res = []
        for p in products:
            # Maybe this could also be expressed in terms of avoid_internal_edges
            inner_edges = {e for t in p.factors for e in t.edges if e not in p.edges}
            rename = _unused_edge_names(inner_edges, used_edges)
            for t in p.factors:
                res.append(t.rename(**rename))
            used_edges.update(rename.values())  # Later renames should not clash with this one
        return Product(res)

    def _substitute(self, x: "Variable", y: "Tensor", memo: dict) -> "Tensor":
        factors = [_substitute_memo(t, x, y, memo) for t in self.factors]
        if all(a is b for a, b in zip(factors, self.factors)):
            return self
        return Product(factors)

    def __repr__(self) -> str:
        if len(self.factors) <= 1:
            return f"Product({self.factors})"
        else:
            inner = "    " + ",\n    ".join(map(repr, self.factors)) + ","
            return f"Product([\n{inner}\n])"

    def components(self) -> list["Product"]:
        """
        Decompose the product into disjoint components (subgraphs).

        Returns:
            A list of Product tensors representing disjoint components.
        """
        # Create graph of tensor connections
        G = nx.Graph()
        G.add_nodes_from(range(len(self.factors)))
        for i, t1 in enumerate(self.factors):
            for j, t2 in enumerate(self.factors[i + 1 :], i + 1):
                if set(t1.edges) & set(t2.edges):
                    G.add_edge(i, j)

        # Find connected components
        component_sets = list(nx.connected_components(G))
        components = [Product([self.factors[i] for i in comp]) for comp in component_sets]
        assert Product(components).edges == self.edges
        return components

    def structure(self) -> "Structure":
        from tensorgrad.structure import Structure

        # Factors are an interchangeable multiset (all the same role).  An
        # edge name shared by exactly two factors is an inner wire (the name
        # itself is discarded); a single-occurrence edge is free.
        owners: dict[str, list[int]] = defaultdict(list)
        for i, f in enumerate(self.factors):
            for e in f.edges:
                owners[e].append(i)
        junctions = set()
        for e, os in owners.items():
            if len(os) == 2:
                junctions.add(frozenset({("child", os[0], e), ("child", os[1], e)}))
            else:
                junctions.add(frozenset({("child", os[0], e), ("free", e)}))
        return Structure(
            ("Product",), tuple(self.factors), ("factor",) * len(self.factors), frozenset(junctions)
        )

    def _depends_on(self, x: "Variable") -> bool:
        return any(t.depends_on(x) for t in self.factors)

    def _to_einsum_eq(self) -> tuple[list[Tensor], str]:
        """
        Convert the product to an einsum equation.

        The main utility is that Delta tensors are mostly removed in favor of hyper edges, which are more efficient to compute.
        However, some Deltas are still needed, so we return the list of tensors that should be
        used with the einsum equation.

        Returns:
            A tuple (factors, equation) where factors is a list of tensors and
            equation is the corresponding einsum string.

        Raises:
            ValueError: If there are too many unique edges.
        """
        unique_edges = {e for t in self.factors for e in t.edges}
        if (n := len(unique_edges)) > 52:
            raise ValueError(f"Too many unique edges ({n} > 52) to convert to einsum equation")
        name_to_idx = defaultdict(lambda: ascii_letters[len(name_to_idx)])

        # The main challenge is to convert Delta tensors to hyper edges
        factors = []
        for ft in self.factors:
            if isinstance(ft, Delta) and ft.order >= 1:
                output_edges = ft.edges & self.edges
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
                else:
                    factors.append(ft)
            else:
                factors.append(ft)

        equation = ",".join("".join(name_to_idx[e] for e in ft.edges) for ft in factors)
        equation += "->" + "".join(name_to_idx[e] for e in self.edges)
        return factors, equation


################################################################################
# Sum
################################################################################


class Sum(Tensor):
    """
    A weighted sum of several tensors.
    """

    def __init__(self, terms: Iterable[Tensor], weights: Optional[Iterable[Scalar]] = None):
        """
        A weighted sum of multiple tensors.

        Args:
            terms: The tensors to add together.
            weights: The weights of each tensor in the sum. If not provided, all weights are 1.
        """
        terms = list(terms)

        # Broadcasting means we always upgrade to the super set of edges
        self._shape = {}
        for e, ts in _group_edges(terms).items():
            s = ts[0].shape[e]
            if not all(t.shape[e] == s for t in ts):
                raise ValueError(f"Edge {e} had different sizes in tensors: {ts}")
            self._shape[e] = s

        # Any tensors that lacks an edge will be broadcasted to have that edge.
        # (Iterate in shape order, not set order, for deterministic results.)
        self.terms = []
        for t in terms:
            missing = {e: s for e, s in self._shape.items() if e not in t.edges}
            # Note: don't broadcast if the tensor is already full, since that would create new
            # Ones([]) objects after simplification is supposed to have completed.
            self.terms.append(t @ Ones(**missing) if missing else t)

        # Typed list[Any] (not list[Scalar]): numbers.Number is a virtual ABC
        # with no static arithmetic, so precise typing would force a cast at
        # every `w1 * w2` site.
        self.weights: list[Any] = [1] * len(terms) if weights is None else list(weights)
        assert len(terms) == len(self.weights)

    def _rename(self, **kwargs: str) -> Tensor:
        return Sum([t.rename(**kwargs) for t in self.terms], self.weights)

    def _substitute(self, x: "Variable", y: "Tensor", memo: dict) -> "Tensor":
        terms = [_substitute_memo(t, x, y, memo) for t in self.terms]
        if all(a is b for a, b in zip(terms, self.terms)):
            return self
        return Sum(terms, self.weights)

    def __repr__(self) -> str:
        return f"Sum({self.terms}, {self.weights})"

    def structure(self) -> "Structure":
        from tensorgrad.structure import Structure

        # Terms with equal weights are interchangeable (roles carry str(w),
        # matching numerically-equal weights of different types the same way
        # the old nx labels f"Weight {w}" did).  Per free edge, one hyper-
        # junction over all terms: __init__ broadcasts, so every term has
        # every edge.
        n = len(self.terms)
        junctions = frozenset(
            frozenset({("child", i, e) for i in range(n)} | {("free", e)}) for e in self.edges
        )
        return Structure(
            ("Sum",), tuple(self.terms), tuple(("term", str(w)) for w in self.weights), junctions
        )

    def _depends_on(self, x: "Variable") -> bool:
        return any(t.depends_on(x) for t in self.terms)


################################################################################
# Some useful functions
################################################################################


def peel_rename(t: Tensor) -> Tensor:
    """Peel a (lazy) top-level Rename off `t`: merge a chain of Rename
    wrappers into one mapping and materialize it ONE level into the inner
    tensor via its ``_rename``.

    THE CONTRACT for structural pattern-matching: matching is defined on
    SIMPLIFIED expressions, and because ``Tensor.rename`` is always lazy on
    composites, even simplified trees carry Rename wrappers — simplify
    merges chains into a single wrapper but deliberately KEEPS it above a
    composite, since the inner tensor is typically shared under several
    renamings and pushing each into a private copy of the tree would destroy
    that sharing (see push_rename_down in tensorgrad/simplify.py). Matchers
    that dispatch on node type must therefore either simplify first or call
    this helper at every level they descend.

    The result's root is the inner tensor's own node type — except the two
    irreducible forms where edge names are part of the node's identity, which
    eager renaming produced too and matchers already treat as opaque:
    ``Rename(Variable)`` and ``Rename(Function)`` (renamed function OUTPUT
    edges). If the result is still a Rename, it is one of those.
    """
    if not isinstance(t, Rename):
        return t
    inner, mapping = t.tensor, t.mapping
    while isinstance(inner, Rename):
        mapping = Rename.merge_renames(inner.mapping, mapping)
        inner = inner.tensor
    # _rename implementations assume the mapping is restricted to free edges
    # (Tensor.rename guarantees this; merged/hand-built mappings may not be).
    mapping = {k: v for k, v in mapping.items() if k in inner.edges and k != v}
    return inner._rename(**mapping) if mapping else inner


def _substitute_memo(t: Tensor, x: "Variable", y: Tensor, memo: dict) -> Tensor:
    """Memoized (by object identity) driver for Tensor.substitute.

    The memo keys are ids of descendants of the substitution root, which stay
    alive for the duration of the call, so ids cannot be recycled."""
    res = memo.get(id(t))
    if res is None:
        res = t._substitute(x, y, memo)
        memo[id(t)] = res
    return res


def _group_edges(tensors: Iterable[Tensor]) -> dict[str, list[Tensor]]:
    """Group tensors by their edge names."""
    groups = defaultdict(list)
    for t in tensors:
        for e in t.edges:
            groups[e].append(t)
    return groups


def _unused_edge_names(edges: Iterable[str], used_names: Iterable[str], suffix: str = "") -> dict[str, str]:
    """
    Generate a mapping from each edge in 'edges' to a new name that is not in used_names.

    Args:
        edges: An iterable of original edge names.
        used_names: A collection of names that must be avoided.
        suffix: A suffix to append to new names.

    Returns:
        A dictionary mapping original edge names to fresh names.
    """
    used_names = set(used_names)
    rename = {}
    # Sort so collision-resolution (the "_" suffixes) doesn't depend on the
    # iteration order of the (frequently set-typed, hence PYTHONHASHSEED-
    # dependent) `edges` argument.
    for e in sorted(edges):
        candidate = e + suffix
        while candidate in used_names:
            candidate += "_"
        rename[e] = candidate
        used_names.add(candidate)
    return rename
