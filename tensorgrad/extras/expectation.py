from tensorgrad.tensor import Constant, Derivative, Product, Sum, Tensor, Variable, unused_edge_names


class Expectation(Tensor):
    def __init__(self, tensor: Tensor, wrt: Variable, mu: Tensor, covar: Tensor):
        self.tensor = tensor
        self.edges = tensor.edges
        self.wrt = wrt
        self.mu = mu
        self.covar = covar
        if set(mu.edges) != set(wrt.edges):
            raise ValueError("mu must have the same edges as wrt")
        if len(covar.edges) != 2 * len(wrt.edges):
            raise ValueError("covar must have twice the edges of wrt")
        if set(covar.edges[: len(mu.edges)]) != set(mu.edges):
            raise ValueError("first half of covar edges must match mu")

    def evaluate(self, values):
        # Could do a numerical integration here
        raise NotImplementedError

    def simplify(self, args=None):
        inner = self.tensor.simplify(args=args)
        if isinstance(inner, Sum):
            return Sum([Expectation(t, self.wrt, self.mu, self.covar) for t in inner.tensors], inner.weights)
        if isinstance(inner, Constant):
            return inner.simplify(args=args)
        if isinstance(inner, Variable):
            return self.mu if inner == self.wrt else inner
        if isinstance(inner, Product):
            prod = inner
            if not prod.depends_on(self.wrt):
                return prod
            if self.wrt in prod.tensors:
                # 1) Look for an instance of wrt in the product
                x = next(x for x in prod.tensors if x == self.wrt)
                # Rename the mu and covar to match the actual edges of x
                rename = self.wrt.get_isomorphism(x)
                mu = self.mu.rename(rename)
                covar = self.covar.rename(rename)
                # 2) Form x * rest by removing x from the product
                subs = prod.tensors[:]
                subs.remove(x)
                rest = Product(subs)
                # 2) Expand: (x - mu + mu) rest = mu * rest + (x - mu) * rest
                res = mu @ Expectation(rest, self.wrt, self.mu, self.covar)
                assert set(res.edges) == set(self.edges), (res.edges, self.edges)
                if not rest.depends_on(self.wrt):
                    return res.simplify(args=args)
                # 3) Use Gaussian Integration of Parts on (x - mu) * rest
                covar_out_edges = covar.edges[len(mu.edges) :]
                new_names, rename = unused_edge_names(
                    covar_out_edges, set(rest.edges) | set(x.edges), suffix="_"
                )
                print(f"{covar_out_edges=}, {new_names=}, {rename=}")
                res += Expectation(
                    Derivative(rest, x, new_names), self.wrt, self.mu, self.covar
                ) @ covar.rename(rename)
                assert set(res.edges) == set(self.edges)
                return res.simplify(args=args)
            pass
        return Expectation(inner, self.wrt, self.mu, self.covar)

    def grad(self, x: Variable, new_names: list[str] | None) -> Tensor:
        new_names = self._check_grad(x, new_names)
        return Expectation(Derivative(self.tensor, x, new_names), self.wrt)

    def __repr__(self):
        return f"E[{self.tensor}]"

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
