from sympy import Symbol
import torch
from tensorgrad.tensor import (
    Constant,
    Derivative,
    Product,
    Sum,
    Tensor,
    Variable,
    add_structural_graph,
    unused_edge_names,
    Copy,
    Zero,
)
import tensorgrad.functions as F
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
                covar_names = unused_edge_names(wrt.edges, wrt.edges)
            covar = Product([Copy(wrt.shape[e], e, e2) for e, e2 in covar_names.items()])
        elif covar_names is None:
            raise ValueError("If covar is not given, covar_names must be given.")
        if covar.shape != wrt.shape | {covar_names[k]: s for k, s in wrt.shape.items()}:
            raise ValueError(f"{covar.shape=} != {wrt.shape=} | {covar_names=}")
        assert covar.order % 2 == 0, f"{covar.order=}"

        assert covar_names.keys() == wrt.edges, f"{covar_names.keys()=} != {wrt.edges=}"
        self.covar_names = covar_names
        self.covar = covar

        # Compute the mapping between the two sets of edge names in covar:
        # self.covar_out_edges = [e for e in covar.edges if e not in mu.edges]

    def evaluate(
        self,
        values: dict["Variable", torch.Tensor],
        dims: dict[Symbol, int] | None = None,
    ) -> torch.Tensor:
        # TODO: Could do a numerical integration here
        # A neat approach would be if could substitute self.wrt with a tensor with an extra edge.
        # This edge would be "number of samples" and we'd evantually take the mean over it.
        # So we'd add a method tensor.substitute(variable_old, variable_new) which would require
        # that variable_new's edges are a superset of variable_old's edges.
        # It would be nice to have one more application of such a method to help inform the design.
        raise NotImplementedError

    def simplify(self, args=None):
        # We don't currently support Product functions directly, so we prefer to expand them.
        args = self._check_simplify(args) | {"combine_products": False}
        inner = self.tensor.simplify(args=args)

        # Just a small optimization
        if not inner.depends_on(self.wrt):
            return inner

        if args["grad_steps"] == 0:
            # We just steal the grad_steps name for now
            res = Expectation(inner, self.wrt, self.mu, self.covar, self.covar_names)
        else:
            args["grad_steps"] -= 1

        if isinstance(inner, Sum):
            return Sum(
                [Expectation(t, self.wrt, self.mu, self.covar, self.covar_names) for t in inner.tensors],
                inner.weights,
            )

        if isinstance(inner, Constant):
            return inner.simplify(args=args)

        if isinstance(inner, Variable):
            assert inner == self.wrt, "A variable can only depend on wrt if they are the same"
            iso_rename = next(self.wrt.isomorphisms(inner))
            return self.mu.rename(**iso_rename)

        if isinstance(inner, Product):
            prod = inner
            # Right now we only support expectations of products where wrt is directly in the product.
            if self.wrt in prod.tensors:
                # 1) Look for an instance of wrt in the product
                x = next(x for x in prod.tensors if x == self.wrt)

                # Rename the mu and covar to match the actual edges of x
                # E.g. if x is actually the transpose of wrt
                iso_rename = next(self.wrt.isomorphisms(x))
                mu = self.mu.rename(**iso_rename)

                # 2) Form x * rest by removing x from the product
                # Note subs.remove will only remove _the first_ occurrence of x, not all of them.
                subs = prod.tensors[:]
                subs.remove(x)
                rest = Product(subs)

                # 3) Expand: x * rest = (x - mu + mu) * rest = mu * rest + (x - mu) * rest
                res = mu @ Expectation(rest, self.wrt, self.mu, self.covar, self.covar_names)
                assert res.edges == self.edges, f"{res.edges=} != {self.edges=}"

                # Before we can rename covar with iso_rename, we have to make sure it there's
                # no clash with the covar_names.
                out_rename = unused_edge_names(self.covar_names.values(), x.edges | rest.edges)
                covar = self.covar.rename(**(out_rename | iso_rename))
                expected = x.shape | {out_rename[self.covar_names[k]]: s for k, s in self.wrt.shape.items()}
                assert covar.shape == expected, f"{covar.shape=} != {expected=}"

                # With derivatives, we have to use the original names of the variables
                # new_edges = {x.orig[iso_rename[k]]: out_rename[v] for k, v in self.covar_names.items()}
                new_edges = {self.wrt.orig[k]: out_rename[v] for k, v in self.covar_names.items()}

                # We use the covar_names as the new_names for the derivative. Note that these will eventually
                # be consumed by the multiplication with @ covar.
                res += covar @ Expectation(
                    # We have to take the derivative wrt x, not wrt. Or maybe it works with wrt too?
                    # I guess derivatives don't really care about renamings of the variable, as long as the
                    # new edges are consistent?
                    # Derivative(rest, x, new_edges),
                    Derivative(rest, self.wrt, new_edges),
                    self.wrt,
                    self.mu,
                    self.covar,
                    self.covar_names,
                )
                assert res.edges == self.edges, f"{res.edges=} != {self.edges=}"
                return res.simplify(args=args)

        # If nothing was found that we know how to simplify, we just return the original
        return Expectation(inner, self.wrt, self.mu, self.covar, self.covar_names)

    def grad(self, x: Variable, new_names: dict[str, str] | None = None) -> Tensor:
        new_names = self._check_grad(x, new_names)
        res = Expectation(Derivative(self.tensor, x, new_names), self.wrt, self.covar, self.covar_names)
        assert res.shape == self.shape | {new_names[k]: s for k, s in x.shape.items()}
        return res

    def __repr__(self):
        return f"E[{self.tensor}]"

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=type(self).__name__, tensor=self)
        G, t_edges = add_structural_graph(G, self.tensor, root_edge_label="self.tensor")
        G, _ = add_structural_graph(G, self.wrt, root_edge_label="self.wrt")
        G, _ = add_structural_graph(G, self.mu, root_edge_label="self.mu")
        # TODO: We should add the covar_names here
        G, _ = add_structural_graph(G, self.covar, root_edge_label="self.covar")
        return G, t_edges

    def rename(self, **kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        # The variables, wrt, mu, covar shouldn't influence our free edge names
        res = Expectation(self.tensor.rename(**kwargs), self.wrt, self.mu, self.covar, self.covar_names)
        assert res.edges == {kwargs.get(e, e) for e in self.edges}
        return res

    def depends_on(self, x: "Variable") -> bool:
        return self.tensor.depends_on(x)
