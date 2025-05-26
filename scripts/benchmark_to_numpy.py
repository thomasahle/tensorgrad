#!/usr/bin/env python3
"""Benchmark script to measure performance improvements in the optimized to_numpy module."""

import time
import numpy as np
import torch
import sympy
from tensorgrad import Tensor, Variable, Delta, Sum, Product
from tensorgrad import functions as F
from tensorgrad.extras.to_numpy import compile_to_callable as compile_orig
from tensorgrad.extras.to_numpy_optimized import compile_to_callable as compile_opt


class Timer:
    def __init__(self, name):
        self.name = name
        self.times = []
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.times.append(self.end - self.start)
    
    def average(self):
        return sum(self.times) / len(self.times) if self.times else 0
    
    def total(self):
        return sum(self.times)


def benchmark_compilation(name, tensor_fn, warmup=1, iterations=5):
    """Benchmark compilation time."""
    print(f"\n{name}:")
    
    # Build tensor expression
    tensor = tensor_fn()
    
    # Warmup
    for _ in range(warmup):
        _ = compile_orig(tensor)
        _ = compile_opt(tensor)
    
    # Benchmark original
    timer_orig = Timer("Original compilation")
    for _ in range(iterations):
        with timer_orig:
            fn_orig = compile_orig(tensor)
    
    # Benchmark optimized
    timer_opt = Timer("Optimized compilation")
    for _ in range(iterations):
        with timer_opt:
            fn_opt = compile_opt(tensor)
    
    avg_orig = timer_orig.average() * 1000  # Convert to ms
    avg_opt = timer_opt.average() * 1000
    speedup = avg_orig / avg_opt if avg_opt > 0 else float('inf')
    
    print(f"  Original:  {avg_orig:.2f} ms")
    print(f"  Optimized: {avg_opt:.2f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    
    return fn_orig, fn_opt


def benchmark_execution(name, fn_orig, fn_opt, values_fn, shapes=None, warmup=10, iterations=100):
    """Benchmark execution time."""
    print(f"\n{name} - Execution:")
    
    # Generate test data
    values = values_fn()
    
    # Warmup
    for _ in range(warmup):
        _ = fn_orig(values, shapes)
        _ = fn_opt(values, shapes)
    
    # Benchmark original
    timer_orig = Timer("Original execution")
    for _ in range(iterations):
        with timer_orig:
            result_orig = fn_orig(values, shapes)
    
    # Benchmark optimized
    timer_opt = Timer("Optimized execution")
    for _ in range(iterations):
        with timer_opt:
            result_opt = fn_opt(values, shapes)
    
    avg_orig = timer_orig.average() * 1000  # Convert to ms
    avg_opt = timer_opt.average() * 1000
    speedup = avg_orig / avg_opt if avg_opt > 0 else float('inf')
    
    print(f"  Original:  {avg_orig:.3f} ms")
    print(f"  Optimized: {avg_opt:.3f} ms")
    print(f"  Speedup:   {speedup:.2f}x")


def benchmark_matrix_multiply():
    """Benchmark matrix multiplication."""
    d = sympy.Symbol('d')
    
    def tensor_fn():
        X = Variable('X', i=d, j=d)
        Y = Variable('Y', i=d, j=d)
        return X @ Y
    
    fn_orig, fn_opt = benchmark_compilation("Matrix Multiplication - Compilation", tensor_fn)
    
    # Test different sizes
    for size in [10, 50, 100, 200]:
        def values_fn():
            return {
                Variable('X', i=d, j=d): torch.randn(size, size),
                Variable('Y', i=d, j=d): torch.randn(size, size)
            }
        
        benchmark_execution(f"Matrix Multiplication ({size}x{size})", 
                           fn_orig, fn_opt, values_fn)


def benchmark_neural_network():
    """Benchmark a simple neural network."""
    batch = sympy.Symbol('batch')
    d_in = sympy.Symbol('d_in')
    d_hidden = sympy.Symbol('d_hidden')
    d_out = sympy.Symbol('d_out')
    
    def tensor_fn():
        X = Variable('X', i=batch, j=d_in)
        W1 = Variable('W1', i=d_in, j=d_hidden)
        W2 = Variable('W2', i=d_hidden, j=d_out)
        b1 = Variable('b1', d_hidden)
        b2 = Variable('b2', d_out)
        
        # Two-layer network
        # X @ W1 has edges (i=batch, j=d_hidden)
        bias1_expanded = Product([Delta(batch, 'i'), b1.rename(d_hidden='j')])
        hidden = F.relu(X @ W1 + bias1_expanded)
        # hidden @ W2 has edges (i=batch, j=d_out)
        bias2_expanded = Product([Delta(batch, 'i'), b2.rename(d_out='j')])
        output = hidden @ W2 + bias2_expanded
        return output
    
    fn_orig, fn_opt = benchmark_compilation("Neural Network - Compilation", tensor_fn)
    
    # Test different batch sizes
    for batch_size in [32, 128, 512]:
        def values_fn():
            return {
                Variable('X', i=batch, j=d_in): torch.randn(batch_size, 128),
                Variable('W1', i=d_in, j=d_hidden): torch.randn(128, 256),
                Variable('W2', i=d_hidden, j=d_out): torch.randn(256, 10),
                Variable('b1', d_hidden): torch.randn(256),
                Variable('b2', d_out): torch.randn(10)
            }
        
        benchmark_execution(f"Neural Network (batch={batch_size})", 
                           fn_orig, fn_opt, values_fn)


def benchmark_convolution():
    """Benchmark convolution operation."""
    w_in = sympy.Symbol('w_in')
    k_size = sympy.Symbol('k_size')
    w_out = sympy.Symbol('w_out')
    
    def tensor_fn():
        return F.Convolution(input=w_in, kernel=k_size, output=w_out)
    
    fn_orig, fn_opt = benchmark_compilation("Convolution - Compilation", tensor_fn)
    
    # Test different sizes
    test_configs = [
        (28, 3, 26),  # Small
        (100, 5, 96), # Medium
        (200, 7, 194) # Large
    ]
    
    for in_size, kernel, out_size in test_configs:
        shapes = {w_in: in_size, k_size: kernel, w_out: out_size}
        
        benchmark_execution(f"Convolution ({in_size}x{kernel}→{out_size})",
                           fn_orig, fn_opt, lambda: {}, shapes)


def benchmark_complex_expression():
    """Benchmark a complex expression with multiple operations."""
    d = sympy.Symbol('d')
    
    def tensor_fn():
        A = Variable('A', i=d, j=d)
        B = Variable('B', i=d, j=d)
        C = Variable('C', i=d, j=d)
        D = Variable('D', i=d, j=d)
        
        # Complex expression: (A @ B + C) @ D + A - 2*B
        temp = A @ B + C
        result = temp @ D + A - 2*B
        return result
    
    fn_orig, fn_opt = benchmark_compilation("Complex Expression - Compilation", tensor_fn)
    
    # Test different sizes
    for size in [20, 50, 100]:
        def values_fn():
            return {
                Variable('A', i=d, j=d): torch.randn(size, size),
                Variable('B', i=d, j=d): torch.randn(size, size),
                Variable('C', i=d, j=d): torch.randn(size, size),
                Variable('D', i=d, j=d): torch.randn(size, size)
            }
        
        benchmark_execution(f"Complex Expression ({size}x{size})",
                           fn_orig, fn_opt, values_fn)


def benchmark_sparse_operations():
    """Benchmark sparse tensor operations."""
    d = sympy.Symbol('d')
    
    def tensor_fn():
        # Sum of delta tensors
        I = Delta(d, 'i', 'j')
        X = Variable('X', i=d, j=d)
        return I + X + 2*I - X
    
    fn_orig, fn_opt = benchmark_compilation("Sparse Operations - Compilation", tensor_fn)
    
    # Test different sizes
    for size in [50, 100, 200]:
        def values_fn():
            return {
                Variable('X', i=d, j=d): torch.randn(size, size)
            }
        
        shapes = {d: size}
        benchmark_execution(f"Sparse Operations ({size}x{size})",
                           fn_orig, fn_opt, values_fn, shapes)


def benchmark_permutations():
    """Benchmark operations requiring many permutations."""
    a, b, c, d = sympy.symbols('a b c d')
    
    def tensor_fn():
        # Create tensors that will require transpositions
        X = Variable('X', i=a, j=b, k=c)
        Y = Variable('Y', i=b, j=c, k=d)
        Z = Variable('Z', i=c, j=d, k=a)
        
        # Operations that require permutations
        # X has edges (i, j, k), Y has edges (i, j, k) 
        # Contract over j and k to get shape (a, d)
        temp1 = Product([X, Y.rename(i='j', j='k', k='l')])  # Result edges: i, l
        # Contract temp1 (i, l) with Z (i, j, k) over i and l (renamed)
        temp2 = Product([temp1, Z.rename(i='k', j='l')])  # Result edges: k
        return temp2
    
    fn_orig, fn_opt = benchmark_compilation("Permutation Heavy - Compilation", tensor_fn)
    
    # Test
    def values_fn():
        return {
            Variable('X', i=a, j=b, k=c): torch.randn(10, 20, 30),
            Variable('Y', i=b, j=c, k=d): torch.randn(20, 30, 40),
            Variable('Z', i=c, j=d, k=a): torch.randn(30, 40, 10)
        }
    
    benchmark_execution("Permutation Heavy Operations",
                       fn_orig, fn_opt, values_fn)


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("Benchmarking optimized to_numpy module")
    print("=" * 60)
    
    benchmark_matrix_multiply()
    print("\n" + "-" * 60)
    
    benchmark_neural_network()
    print("\n" + "-" * 60)
    
    benchmark_convolution()
    print("\n" + "-" * 60)
    
    benchmark_complex_expression()
    print("\n" + "-" * 60)
    
    benchmark_sparse_operations()
    print("\n" + "-" * 60)
    
    benchmark_permutations()
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()