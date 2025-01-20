from sympy import Symbol
import torch
from tensorgrad.tensor import (
    Constant,
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

        # Compute the mapping between the two sets of edge names in covar:
        # self.covar_out_edges = [e for e in covar.edges if e not in mu.edges]

    def evaluate(
        self,
        values: dict["Variable", torch.Tensor],
        dims: dict[Symbol, int] | None = None,
    ) -> torch.Tensor:
        """
        Numerically approximate E[tensor] via Monte Carlo sampling from N(mu, covar),
        storing a *batch* dimension ("samples") in wrt and taking the mean over that dimension.

        Steps:
        - Evaluate mu, covar => align/flatten => sample in R^D
        - Reshape to (samples, *wrt.shape)
        - Insert that into values[self.wrt]
        - Evaluate self.tensor(...) => shape (samples, *rest)
        - .mean over 'samples' => final result
        """

        # If we already computed this Expectation, return cached value:
        if self in values:
            return values[self]

        # If user pinned a specific wrt => just evaluate the tensor directly
        if self.wrt in values:
            out = self.tensor.evaluate(values, dims)
            values[self] = out
            return out

        if dims is None:
            dims = {}

        # 1) Evaluate mu and covar numerically
        mu_torch = self.mu.evaluate(values, dims)  # named edges = same as self.mu.edges
        covar_torch = self.covar.evaluate(values, dims)

        # 2) Align mu => shape(*self.wrt.edges)
        #    Example: if wrt.edges = (i, j), we call mu_torch.align_to("i","j").
        #    That ensures the dimension order matches wrt's order.
        mu_aligned = mu_torch.align_to(*self.wrt.edges)
        # Flatten to (D,)
        D = mu_aligned.numel()
        mu_flat = mu_aligned.rename(None).reshape(D)

        # 3) Align covar => shape(*self.wrt.edges, *mapped_wrt_edges)
        #    For example: (i, j, i2, j2).  Let's define the order explicitly:
        #    first the "row" edges (i, j), then the "column" edges (i2, j2).
        col_edges = [self.covar_names[e] for e in self.wrt.edges]
        covar_aligned = covar_torch.align_to(*self.wrt.edges, *col_edges)
        # Flatten to (D, D)
        covar_reshaped = covar_aligned.rename(None).reshape(D, D)

        # 4) Number of samples
        num_samples = dims.get("samples", 1000)
        if "seed" in dims:
            torch.manual_seed(dims["seed"])

        # 5) Cholesky
        L = torch.linalg.cholesky(covar_reshaped)

        # 6) Sample => shape (samples, D)
        z = torch.randn(num_samples, D, device=mu_flat.device, dtype=mu_flat.dtype)
        x_flat = mu_flat + (z @ L.T)  # shape (samples, D)

        # 7) Reshape => (samples, *wrt.shape)  => name the leading dim "samples"
        wrt_shape = mu_aligned.shape  # e.g. (2,3)
        x_batched = x_flat.reshape(num_samples, *wrt_shape)
        x_batched_named = x_batched.refine_names("samples", *self.wrt.edges)

        # 8) Insert into values => call self.tensor.evaluate
        temp_values = dict(values)
        temp_values[self.wrt] = x_batched_named  # Now wrt has a leading 'samples' dimension
        out_batched = self.tensor.evaluate(temp_values, dims)  # => shape (samples, ...)

        # 9) Take mean over the "samples" dimension
        out_batched_unnamed = out_batched.rename(None)  # drop named dims to index them
        out_mean = out_batched_unnamed.mean(dim=0)  # remove 'samples' from the front

        # 10) Rename the result to the free edges => i.e. self.tensor.edges
        #     but "out_batched" might have shape (samples, E...) => after mean => shape(E...).
        #     So we can do:
        out_named = out_mean.refine_names(*self.tensor.edges)

        # Cache
        values[self] = out_named
        return out_named

    def simplify(self, args=None):
        # We don't currently support pow() functions directly, so we prefer to expand them.
        args = self._check_simplify(args) | {"combine_products": False}
        inner = self.tensor.simplify(args=args)

        # Constants
        if not inner.depends_on(self.wrt):
            return inner

        if isinstance(inner, Sum):
            return Sum(
                [
                    Expectation(t, self.wrt, self.mu, self.covar, self.covar_names).simplify(args)
                    for t in inner.tensors
                ],
                inner.weights,
            )

        if isinstance(inner, Rename):
            return Rename(
                Expectation(inner.tensor, self.wrt, self.mu, self.covar, self.covar_names).simplify(args),
                inner.mapping,
            )

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
                out_rename = _unused_edge_names(self.covar_names.values(), x.edges | rest.edges)
                covar = self.covar.rename(**(out_rename | iso_rename))
                expected = x.shape | {out_rename[self.covar_names[k]]: s for k, s in self.wrt.shape.items()}
                assert covar.shape == expected, f"{covar.shape=} != {expected=}"
                new_edges = {k: out_rename[v] for k, v in self.covar_names.items()}

                # We use the covar_names as the new_names for the derivative. Note that these will eventually
                # be consumed by the multiplication with @ covar.
                res += covar @ Expectation(
                    # We have to take the derivative wrt x, not wrt. Or maybe it works with wrt too?
                    # I guess derivatives don't really care about renamings of the variable, as long as the
                    # new edges are consistent
                    # Derivative(rest, x, new_edges),
                    Derivative(rest, self.wrt, new_edges),
                    self.wrt,
                    self.mu,
                    self.covar,
                    self.covar_names,
                )
                assert res.edges == self.edges, f"{res.edges=} != {self.edges=}"
                return res.simplify(args=args)

            # Look for a power function with exponent >= 1 and pull out a factor
            if (
                fn := next(
                    (
                        t
                        for t in prod.tensors
                        if isinstance(t, Function)
                        and isinstance(t.signature, _PowerFunction)
                        and t.signature.k >= 1
                    ),
                    None,
                )
            ) is not None:
                assert isinstance(fn, Function)
                subs = prod.tensors[:]
                subs.remove(fn)
                (inner,) = fn.inputs
                subs.append(inner * fn.weight)  # We pull the weight out as well
                if fn.signature.k > 1:
                    subs.append(pow(inner, fn.signature.k - 1))
                return Expectation(Product(subs), self.wrt, self.mu, self.covar, self.covar_names).simplify(
                    args=args
                )

            # Otherwise we look for constant factors to pull out
            elif args.get("extract_constants_from_expectation") and any(
                not t.depends_on(self.wrt) for t in prod.tensors
            ):
                # Separate into constant and wrt-dependent factors
                constant_terms, wrt_terms = [], []
                for t in prod.tensors:
                    if t.depends_on(self.wrt):
                        wrt_terms.append(t)
                    else:
                        constant_terms.append(t)
                assert len(wrt_terms) > 0
                assert len(constant_terms) > 0

                # Pull out the constant terms.
                # Note we need to avoid introducing a Product with a single element,
                # so we don't get an infinite loop in the simplify method.
                constant_prod = Product(constant_terms).simplify(args=args)
                wrt_prod = Product(wrt_terms).simplify(args=args)

                # Compute E[wrt-dependent part] and multiply by constants
                return (
                    constant_prod @ Expectation(wrt_prod, self.wrt, self.mu, self.covar, self.covar_names)
                ).simplify(args=args)

            # Finally check if any factors contain sums that we can expand
            if False and (
                sum_idx := next(
                    (i for i, t in enumerate(prod.tensors) if isinstance(t, Sum) and t.depends_on(self.wrt)),
                    None,
                )
                is not None
            ):
                sum_term = prod.tensors[sum_idx]
                assert isinstance(sum_term, Sum)
                other_terms = prod.tensors[:sum_idx] + prod.tensors[sum_idx + 1 :]

                # Distribute the product over the sum
                return Sum(
                    [
                        Expectation(
                            Product([t] + other_terms), self.wrt, self.mu, self.covar, self.covar_names
                        )
                        for t in sum_term.tensors
                    ],
                    sum_term.weights,
                ).simplify(args=args)

        # If nothing was found that we know how to simplify, we just return the original
        return Expectation(inner, self.wrt, self.mu, self.covar, self.covar_names)

    def grad(self, x: Variable, new_names: dict[str, str] | None = None) -> Tensor:
        new_names = self._check_grad(x, new_names)
        # TODO: There's some issue here if x == self.wrt
        res = Expectation(Derivative(self.tensor, x, new_names), self.wrt, self.covar, self.covar_names)
        assert res.shape == self.shape | {new_names[k]: s for k, s in x.shape.items()}
        return res

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

    def rename(self, **kwargs: dict[str, str]):
        kwargs = self._check_rename(kwargs)
        # The variables, wrt, mu, covar shouldn't influence our free edge names
        res = Expectation(self.tensor.rename(**kwargs), self.wrt, self.mu, self.covar, self.covar_names)
        assert res.edges == {kwargs.get(e, e) for e in self.edges}
        return res

    def depends_on(self, x: "Variable") -> bool:
        return self.tensor.depends_on(x)
