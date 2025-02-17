from tensorgrad.tensor import (
    Derivative,
    Product,
    Rename,
    Sum,
    Tensor,
    Variable,
    _add_structural_graph,
    _unused_edge_names,
    Delta,
    Zero,
    Function,
)
import tensorgrad.functions as F
from tensorgrad.functions import _PowerFunction
import networkx as nx


class Expectation(Tensor):
    def __init__(
        self,
        tensor: Tensor,
        wrt: Variable,
        mu: None | Tensor = None,
        covar: None | Tensor = None,
        covar_names: None | dict[str, str] = None,
    ):
        """
        Take the Expectation of a tensor with respect to a variable, assumed to have a multi-normal
        distribution with the given expectation and covariance.

        Note:
            The covariance tensor should be the expectation, covar = E[x x^T - mu mu^T].
            The edges of covar should include the edges of x, as well as a disjoint copy
            of the same edges. The mapping from the original edges to the new edges should be
            given in covar_names.

        Args:
            tensor (Tensor): The input tensor.
            wrt (Variable): The variable with respect to which the expectation is computed.
            mu (Tensor): The mean tensor. Defaults to the zero tensor.
            covar (Tensor): The covariance tensor. Defaults to the identity tensor.
            covar_names (dict[str, str]): Map from original (wrt) name to covar name.
            Note: It is not a map from wrt.original name, like in the Derivative class.
        """
        self.tensor = tensor
        self._shape = tensor.shape
        self.wrt = wrt

        if mu is None:
            mu = Zero(**wrt.shape)
        assert mu.shape == wrt.shape, f"{mu.shape=} != {wrt.shape=}"
        self.mu = mu

        if covar is None:
            if covar_names is None:
                covar_names = _unused_edge_names(wrt.edges, wrt.edges)
            covar = Product([Delta(wrt.shape[e], e, e2) for e, e2 in covar_names.items()])
        elif covar_names is None:
            raise ValueError("If covar is not given, covar_names must be given.")
        if covar.shape != wrt.shape | {covar_names[k]: s for k, s in wrt.shape.items()}:
            co_size = {covar_names[k]: s for k, s in wrt.shape.items()}
            raise ValueError(f"{covar.shape=} != {wrt.shape=} | {co_size=}")
        assert covar.order % 2 == 0, f"{covar.order=}"

        assert covar_names.keys() == wrt.edges, f"{covar_names.keys()=} != {wrt.edges=}"
        for k, v in covar_names.items():
            assert any((k in s and v in s) for s in covar.symmetries), f"{k}, {v} should be symmetric"
        self.covar_names = covar_names
        self.covar = covar

    def _simplify(self, args: dict[str, str]):
        # We prefer products to not be factored using the pow function when taking expectations
        inner = self.tensor.simplify(args=args | {"factor_components": False})

        # Constants
        if not inner.depends_on(self.wrt):
            return inner

        if isinstance(inner, Sum):
            return Sum(
                [Expectation(t, self.wrt, self.mu, self.covar, self.covar_names) for t in inner.tensors],
                inner.weights,
            )

        if isinstance(inner, Rename):
            return Rename(
                Expectation(inner.tensor, self.wrt, self.mu, self.covar, self.covar_names),
                inner.mapping,
            )

        if isinstance(inner, Variable):
            assert inner == self.wrt, "A variable can only depend on wrt if they are the same"
            iso_rename = next(self.wrt.isomorphisms(inner))
            return self.mu.rename(**iso_rename)

        if isinstance(inner, Product):
            return self._simplify_product(inner, args)

        if isinstance(inner, Function):
            return self._simplify_function(inner, args)

        # If nothing was found that we know how to simplify, we just return the original
        return Expectation(inner, self.wrt, self.mu, self.covar, self.covar_names)

    def _simplify_product(self, prod: Product, args: dict[str, str]):
        # Right now we only support expectations of products where wrt is directly in the product.
        assert isinstance(prod, Product), f"{prod=}"
        constants = []
        dependents = []
        for t in prod.tensors:
            if not t.depends_on(self.wrt):
                constants.append(t)
            else:
                dependents.append(t)
        assert dependents, "Should have at least one dependent variable in the product"
        if len(dependents) == 1:
            x, = dependents
            return Product(constants) @ Expectation(x, self.wrt, self.mu, self.covar, self.covar_names)

        if self.wrt in dependents:
            # 1) Look for an instance of wrt in the product
            x = next(x for x in dependents if x == self.wrt)

            # Rename the mu and covar to match the actual edges of x
            # E.g. if x is actually the transpose of wrt
            iso_rename = next(self.wrt.isomorphisms(x))
            mu = self.mu.rename(**iso_rename)

            # 2) Form x * rest by removing x from the product
            # Note subs.remove will only remove _the first_ occurrence of x, not all of them.
            subs = dependents[:]
            subs.remove(x)
            rest = Product(subs)

            # 3) Expand: x * rest = (x - mu + mu) * rest = mu * rest + (x - mu) * rest
            res = mu @ Expectation(rest, self.wrt, self.mu, self.covar, self.covar_names)
            assert res.edges == Product(dependents).edges

            # Before we can rename covar with iso_rename, we have to make sure it there's
            # no clash with the covar_names.
            out_rename = _unused_edge_names(self.covar_names.values(), x.edges | rest.edges)
            covar = self.covar.rename(**(out_rename | iso_rename))
            expected = x.shape | {out_rename[self.covar_names[k]]: s for k, s in self.wrt.shape.items()}
            assert covar.shape == expected, f"{covar.shape=} != {expected=}"
            new_edges = {k: out_rename[v] for k, v in self.covar_names.items()}

            # We use the covar_names as the new_names for the derivative. Note that these will eventually
            # be consumed by the multiplication with @ covar.
            res = res + covar @ Expectation(
                # We have to take the derivative wrt x, not wrt. Or maybe it works with wrt too?
                # I guess derivatives don't really care about renamings of the variable, as long as the
                # new edges are consistent
                Derivative(rest, self.wrt, new_edges),
                self.wrt,
                self.mu,
                self.covar,
                self.covar_names,
            )
            if constants:
                res = Product(constants) @ res
            return res

        # Unable to simplify
        return Expectation(prod, self.wrt, self.mu, self.covar, self.covar_names)

    def _simplify_function(self, fn: Function, args: dict[str, str]):
        # If the function is a power function with a positive exponent, we can simplify it
        if isinstance(fn.signature, _PowerFunction) and fn.signature.k >= 0:
            (inner,) = fn.inputs
            prod = F.prod(*[inner] * fn.signature.k)
            assert isinstance(prod, Product), f"{prod=}"
            res = self._simplify_product(prod, args)
            return res
        return fn

    def _grad(self, x: Variable, new_names: dict[str, str]) -> Tensor:
        if x == self.wrt:
            raise ValueError("Cannot take the gradient wrt the variable we're taking the expectation over")
        return Expectation(Derivative(self.tensor, x, new_names), self.wrt, self.covar, self.covar_names)

    def __repr__(self):
        return f"E[{self.tensor}]"

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=type(self).__name__, tensor=self)
        G, t_edges = _add_structural_graph(G, self.tensor, root_edge_label="self.tensor")
        G, _ = _add_structural_graph(G, self.wrt, root_edge_label="self.wrt")
        G, _ = _add_structural_graph(G, self.mu, root_edge_label="self.mu")
        # TODO: We should add the covar_names here
        G, _ = _add_structural_graph(G, self.covar, root_edge_label="self.covar")
        return G, t_edges

    def _rename(self, **kwargs: dict[str, str]):
        # The variables, wrt, mu, covar shouldn't influence our free edge names
        return Expectation(self.tensor.rename(**kwargs), self.wrt, self.mu, self.covar, self.covar_names)

    def depends_on(self, x: "Variable") -> bool:
        return self.tensor.depends_on(x)
