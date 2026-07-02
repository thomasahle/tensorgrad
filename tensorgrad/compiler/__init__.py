"""An ahead-of-time compiler for tensorgrad expressions.

Turns symbolic tensor algebra (gradients, simplifications, structural sparsity,
all worked out offline) into straight-line PyTorch kernel calls with no
per-step planning, dispatch, or constant-building overhead.

Usage:
    from tensorgrad.compiler import compile_to_callable
    f = compile_to_callable(loss, *grads)
    loss_val, *grad_vals = f({x: x_val, y: y_val, ...}, dims)
"""

from tensorgrad.compiler.runtime import compile_to_callable
from tensorgrad.compiler.affine import (
    Affine,
    affine_basis,
    affine_convolution,
    affine_delta,
    affine_flatten,
    affine_shift,
)

__all__ = [
    "compile_to_callable",
    "Affine",
    "affine_basis",
    "affine_convolution",
    "affine_delta",
    "affine_flatten",
    "affine_shift",
]
