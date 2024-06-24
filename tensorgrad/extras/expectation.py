from tensorgrad.tensor import (
    Constant,
    Derivative,
    Product,
    Sum,
    Tensor,
    Variable,
    add_structural_graph,
    unused_edge_names,
    Function,
    Copy,
    Zero,
)
import networkx as nx


class Expectation(Tensor):
    def __init__(self, tensor: Tensor, wrt: Variable, mu: Tensor = None, covar: Tensor = None):
        """
        Take the Expectation of a tensor with respect to a variable, assumed to have a multi-normal
        distribution with the given expectation and covariance.

        Note:
            The covariance tensor should be the expectation, covar = E[x x^T - mu mu^T].
            The edges of covar should include the edges of x, as well as a disjoint copy
            of the same edges. The second copy can have any names, but is assumed to be
            in the same order as the original set. (Similar to Derivative(tensor, x))

        Args:
            tensor (Tensor): The input tensor.
            wrt (Variable): The variable with respect to which the expectation is computed.
            mu (Tensor): The mean tensor. Defaults to the zero tensor.
            covar (Tensor): The covariance tensor. Defaults to the identity tensor.

        Raises:
            ValueError: If `mu` does not have the same edges as `wrt` in the same order.
            ValueError: If `covar` does not have twice the edges of `wrt`.
            ValueError: If the first half of `covar` edges does not match `mu`.
        """
        self.tensor = tensor
        self.edges = tensor.edges
        self.wrt = wrt
        if mu is None:
            mu = Zero(wrt.edges)
        if covar is None:
            new_names, _ = unused_edge_names(wrt.edges, wrt.edges)
            covar = Product([Copy([e, e2]) for e, e2 in zip(wrt.edges, new_names)])
        self.mu = mu
        self.covar = covar
        if mu.edges != wrt.edges:
            # Maybe we don't really need the same order. It just seems natural.
            raise ValueError("mu must have the same edges as wrt, in the same order")
        if len(covar.edges) != 2 * len(wrt.edges):
            raise ValueError("covar must have twice the edges of wrt")
        if len(set(covar.edges) - set(mu.edges)) != len(mu.edges):
            raise ValueError("first half of covar edges must match mu")
        # Compute the mapping between the two sets of edge names in covar:
        self.covar_out_edges = [e for e in covar.edges if e not in mu.edges]

    def evaluate(self, values):
        # Could do a numerical integration here
        raise NotImplementedError

    def simplify(self, args=None):
        args = self._check_simplify(args) | {"combine_products": False}
        inner = self.tensor.simplify(args=args)

        if not inner.depends_on(self.wrt):
            return inner

        if args["grad_steps"] == 0:
            # We just steal the grad_steps name for now
            res = Expectation(inner, self.wrt, self.mu, self.covar)
        else:
            args["grad_steps"] -= 1

        if isinstance(inner, Sum):
            return Sum([Expectation(t, self.wrt, self.mu, self.covar) for t in inner.tensors], inner.weights)
        if isinstance(inner, Constant):
            return inner.simplify(args=args)
        if isinstance(inner, Variable):
            assert inner == self.wrt, "A variable can only depend on wrt if they are the same"
            iso_rename = next(self.wrt.isomorphisms(inner))
            return self.mu.rename(iso_rename)
        if isinstance(inner, Product):
            prod = inner
            if self.wrt in prod.tensors:
                # 1) Look for an instance of wrt in the product
                x = next(x for x in prod.tensors if x == self.wrt)
                # Rename the mu and covar to match the actual edges of x
                # E.g. if x is actually the transpose of wrt
                iso_rename = next(self.wrt.isomorphisms(x))
                mu = self.mu.rename(iso_rename)
                # 2) Form x * rest by removing x from the product
                subs = prod.tensors[:]
                subs.remove(x)
                rest = Product(subs)
                # 2) Expand: (x - mu + mu) rest = mu * rest + (x - mu) * rest
                res = mu @ Expectation(rest, self.wrt, self.mu, self.covar)
                assert set(res.edges) == set(self.edges), (res.edges, self.edges)

                # Before we can rename covar with iso_rename, we have to make sure it there's
                # no clash with the out names
                new_out_names, out_rename = unused_edge_names(
                    self.covar_out_edges, set(self.wrt.edges) | set(x.edges) | set(rest.edges)
                )
                covar = self.covar.rename(out_rename).rename(iso_rename)

                res += Expectation(Derivative(rest, x, new_out_names), self.wrt, self.mu, self.covar) @ covar
                assert set(res.edges) == set(self.edges)
                return res.simplify(args=args)
        return Expectation(inner, self.wrt, self.mu, self.covar)

    def grad(self, x: Variable, new_names: list[str] | None) -> Tensor:
        new_names = self._check_grad(x, new_names)
        return Expectation(Derivative(self.tensor, x, new_names), self.wrt)

    def __repr__(self):
        return f"E[{self.tensor}]"

    def structural_graph(self) -> tuple[nx.MultiDiGraph, dict[str, int]]:
        G = nx.MultiDiGraph()
        G.add_node(0, name=type(self).__name__, tensor=self)
        G, t_edges = add_structural_graph(G, self.tensor, root_edge_label="self.tensor")
        G, _ = add_structural_graph(G, self.wrt, root_edge_label="self.wrt")
        G, _ = add_structural_graph(G, self.mu, root_edge_label="self.mu")
        G, _ = add_structural_graph(G, self.covar, root_edge_label="self.covar")
        return G, t_edges

    def _compute_canonical(self):
        base = hash(("Expectation", self.tensor, self.wrt, self.mu, self.covar))
        return base, [hash((base, e)) for e in self.tensor.canonical_edge_names]

    def rename(self, kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        # The variables, wrt, mu, covar shouldn't influence our free edge names
        res = Expectation(
            self.tensor.rename(kwargs),
            self.wrt,
            self.mu,
            self.covar,
        )
        assert set(res.edges) == {kwargs.get(e, e) for e in self.edges}
        return res

    def depends_on(self, x: "Variable") -> bool:
        return self.tensor.depends_on(x)
