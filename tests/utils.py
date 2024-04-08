import torch
from typing import Generator, Tuple, Dict
import itertools
from typing import Tuple, Dict
import torch
import random

from tensorgrad.tensor import Copy, Ones, Tensor, Variable, Zero


def rand_values(variables, **shape):
    return {v: torch.randn([shape[e] for e in v.edges], names=v.edges) for v in variables}


def assert_close(a, b):
    assert set(a.names) == set(b.names)
    a = a.align_to(*b.names)
    torch.testing.assert_close(a.rename(None), b.rename(None))


def generate_random_tensor_expression(
    max_size: int,
) -> Tuple[Tensor, torch.Tensor, Dict[Variable, torch.Tensor]]:
    def generate_copy(dim, edges):
        copy = torch.zeros((dim,) * len(edges))
        for i in range(dim):
            copy[(i,) * len(edges)] = 1
        return copy.rename(*edges)

    def broadcast_tensors(left_torch, right_torch):
        all_dims = list(set(left_torch.names) | set(right_torch.names))
        left_aligned = left_torch.align_to(*all_dims)
        right_aligned = right_torch.align_to(*all_dims)
        return left_aligned, right_aligned

    def generate_recursive(size: int, variables: Dict[Variable, torch.Tensor]) -> Tuple[Tensor, torch.Tensor]:
        if size == 1 or random.random() < 0.3:
            # Base case: single variable or constant with different edge configurations
            if random.random() < 0.5 and variables:
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

            left_tensor, left_torch = generate_recursive(left_size, variables)
            right_tensor, right_torch = generate_recursive(right_size, variables)

            if random.random() < 0.5:
                left_aligned, right_aligned = broadcast_tensors(left_torch, right_torch)
                try:
                    return left_tensor + right_tensor, left_aligned + right_aligned
                except RuntimeError as e:
                    print(e)
                    raise ValueError("Failed to generate random tensor expression")
            else:
                contracted = set(left_tensor.edges) & set(right_tensor.edges)
                rhs = "".join(e for e in left_torch.names + right_torch.names if e not in contracted)
                eq = f"{''.join(left_torch.names)},{''.join(right_torch.names)}->{rhs}"
                try:
                    torch_result = torch.einsum(eq, left_torch.rename(None), right_torch.rename(None))
                except RuntimeError as e:
                    print(eq, e)
                    raise ValueError("Failed to generate random tensor expression")
                return left_tensor @ right_tensor, torch_result.rename(*rhs)

    variables = {}
    for var_name in ["x", "y", "z"]:
        edges = random.choice([["a"], ["a", "b"], ["a", "b", "c"]])
        dims = [random.choice([2, 3]) for _ in range(len(edges))]
        variables[Variable(var_name, edges)] = torch.randn(dims, names=edges)

    while True:
        try:
            expr, tensor = generate_recursive(max_size, variables)
            return expr, tensor, variables
        except ValueError:
            continue
