import itertools
from sympy import Symbol, symbols
import torch
from typing import Iterable, Tuple, Dict
import random
import string
from tensorgrad import Copy, Ones, Tensor, Zero, Variable, Product, Sum
import networkx as nx


def rand_values(variables: Iterable[Variable], shape: Dict[Symbol, int] = {}):
    values = {}
    for v in variables:
        if v.order == 0:
            values[v] = torch.randn([])
        else:
            edges, sizes = zip(*v.shape.items())
            values[v] = torch.randn([shape[s] for s in sizes], names=edges)
    return values


def assert_close(a, b, rtol=1e-4, atol=1e-5):
    assert set(a.names) == set(b.names)
    a = a.align_to(*b.names)
    torch.testing.assert_close(a.rename(None), b.rename(None), rtol=rtol, atol=atol)


def broadcast_tensors(left_torch, right_torch):
    all_dims = list(set(left_torch.names) | set(right_torch.names))
    left_aligned = left_torch.align_to(*all_dims)
    right_aligned = right_torch.align_to(*all_dims)
    return left_aligned, right_aligned


def generate_copy(dim: int, edges: Iterable[str]):
    copy = torch.zeros((dim,) * len(edges))
    for i in range(dim):
        copy[(i,) * len(edges)] = 1
    return copy.rename(*edges)


def generate_random_tensor_expression(
    max_size: int,
) -> Tuple[Tensor, torch.Tensor, Dict[Variable, torch.Tensor]]:
    def generate_recursive(size: int, variables: Dict[Variable, torch.Tensor]) -> Tuple[Tensor, torch.Tensor]:
        if size == 1:
            # Base case: single variable or constant with different edge configurations
            # if random.random() < 0.5 and variables:
            if random.random() < 1:
                var, tensor = random.choice(list(variables.items()))
                return var, tensor
            else:
                tensor_class, torch_func = random.choice(
                    [(Zero, torch.zeros), (Ones, torch.ones), (Copy, generate_copy)]
                )
                edges = random.choice([["a"], ["a", "b"], ["a", "b", "c"]])
                if tensor_class == Copy:
                    dim = random.choice([2, 3])
                    return tensor_class(edges), torch_func(dim, edges)
                else:
                    dims = tuple(random.choice([2, 3]) for _ in range(len(edges)))
                    return tensor_class(edges), torch_func(dims, names=edges)
        else:
            # Recursive case: generate subexpressions and combine them
            left_size = random.randint(1, size // 2 + 1)
            right_size = size - left_size

            for _ in range(10):
                left_tensor, left_torch = generate_recursive(left_size, variables)
                right_tensor, right_torch = generate_recursive(right_size, variables)

                if random.random() < 0.2:
                    left_aligned, right_aligned = broadcast_tensors(left_torch, right_torch)
                    try:
                        return left_tensor + right_tensor, left_aligned + right_aligned
                    except RuntimeError as e:
                        # print(e)
                        continue
                else:
                    contracted = set(left_tensor.edges) & set(right_tensor.edges)
                    rhs = "".join(e for e in left_torch.names + right_torch.names if e not in contracted)
                    eq = f"{''.join(left_torch.names)},{''.join(right_torch.names)}->{rhs}"
                    try:
                        torch_result = torch.einsum(eq, left_torch.rename(None), right_torch.rename(None))
                    except RuntimeError as e:
                        # print(eq, e)
                        continue
                    return left_tensor @ right_tensor, torch_result.rename(*rhs)
            # Give up
            raise ValueError("Failed to generate random tensor expression")

    variables = {}
    ds = {c: random.choice([2, 3]) for c in "abc"}
    for var_name in "xyztuv":
        edges = {"x": ["a"], "y": ["b"], "z": ["c"], "t": ["a", "b"], "u": ["a", "b"], "v": ["a", "b", "c"]}[
            var_name
        ]
        # dims = [random.choice([2, 3]) for _ in range(len(edges))]
        dims = [ds[e] for e in edges]
        variables[Variable(var_name, edges)] = torch.randn(dims, names=edges)

    while True:
        try:
            expr, tensor = generate_recursive(max_size, variables)
            return expr, tensor, variables
        except ValueError:
            continue


def make_random_tree(nodes: int):
    components = [i for i in range(nodes)]

    def find(x):
        if components[x] != x:
            components[x] = find(components[x])
        return components[x]

    def union(x, y):
        components[find(x)] = find(y)

    edges = []
    adj = [[] for _ in range(nodes)]
    while len(edges) < nodes - 1:
        x, y = random.randint(0, nodes - 1), random.randint(0, nodes - 1)
        if len(adj[x]) < 3 and len(adj[y]) < 3 and find(x) != find(y):
            union(x, y)
            edges.append((x, y))
            adj[x].append(y)
            adj[y].append(x)

    # 3n edges, n-1 used, 2(n-1) used for connections, n+1 leaf nodes, 1 free edge left.
    names = string.ascii_uppercase
    vectors = []
    variables = []
    for i in range(nodes):
        ts = [f"{names[min(i,j)]}|{names[max(i,j)]}" for j in adj[i]]
        while len(ts) < 3 and len(vectors) < nodes + 1:
            vi = len(vectors)
            vectors.append(Variable(f"V{vi}", f"v{vi}"))
            ts.append(f"v{vi}")
        if len(ts) != 3:
            ts.append("free")
        variables.append(Variable(names[i], ts))

    return vectors, variables


def atlas_generate_random_tensor_expression():
    atlas = nx.graph_atlas_g()
    for _ in range(100):
        gs = [random.choice(atlas)]
        while random.random() < 0.5:
            gs.append(random.choice(atlas))


def random_tensor_expr(max_depth=4, max_dim=4) -> tuple[Tensor, torch.Tensor, dict[Variable, torch.Tensor]]:
    assert max_dim >= 1
    if max_dim == 1:
        symbols_list = [symbols("a")]
    else:
        symbols_list = symbols(" ".join(string.ascii_letters[:max_dim]))
    sizes = {s: random.randrange(1, 2 * max_dim + 1) for s in symbols_list}
    vars = [
        (
            Variable(f"var_{symbols}", *symbols),
            torch.randn([sizes[s] for s in symbols], names=list(map(str, symbols))),
        )
        for r in range(1, len(symbols_list) + 1)
        for symbols in itertools.combinations(symbols_list, r)
    ]
    copys = [
        (Copy(s0, *map(str, symbols)), generate_copy(sizes[s0], list(map(str, symbols))))
        for s0 in symbols_list
        for r in range(1, len(symbols_list) + 1)
        for symbols in itertools.combinations(symbols_list, r)
    ]

    def inner(depth):
        if depth == 0:
            # return random.choice(random.choice([vars, copys]))
            return random.choice(vars)
        left, left_torch = inner(depth - 1)
        right, right_torch = inner(depth - 1)
        if random.random() < 0.5:
            left_aligned, right_aligned = broadcast_tensors(left_torch, right_torch)
            return left + right, left_aligned + right_aligned
        else:
            contracted = left.edges & right.edges
            rhs = "".join(e for e in left_torch.names + right_torch.names if e not in contracted)
            eq = f"{''.join(left_torch.names)},{''.join(right_torch.names)}->{rhs}"
            torch_result = torch.einsum(eq, left_torch.rename(None), right_torch.rename(None))
            return left @ right, torch_result.rename(*rhs)

    tensor, tensor_torch = inner(depth=max_depth)
    return tensor, tensor_torch, {v: t for v, t in vars}
