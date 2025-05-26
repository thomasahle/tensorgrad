#!/usr/bin/env python3
"""Simple benchmark to compare original vs optimized to_numpy."""

import time
import numpy as np
import torch
import sympy
from tensorgrad import Variable, Delta, Product, Sum
from tensorgrad import functions as F
from tensorgrad.extras.to_numpy import compile_to_callable as compile_orig
from tensorgrad.extras.to_numpy_optimized import compile_to_callable as compile_opt


def time_function(fn, inputs, n_runs=100):
    """Time a function over multiple runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = fn(inputs)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times[10:])  # Skip first few for warmup


def benchmark_simple():
    """Simple benchmarks."""
    print("=== SIMPLE BENCHMARKS ===\n")
    
    # 1. Matrix multiplication
    print("1. Matrix Multiplication")
    d = sympy.Symbol('d')
    X = Variable('X', i=d, j=d)
    Y = Variable('Y', i=d, j=d)
    Z = X @ Y
    
    # Compilation time
    start = time.perf_counter()
    fn_orig = compile_orig(Z)
    compile_time_orig = time.perf_counter() - start
    
    start = time.perf_counter()
    fn_opt = compile_opt(Z)
    compile_time_opt = time.perf_counter() - start
    
    print(f"  Compilation - Original: {compile_time_orig*1000:.2f}ms, Optimized: {compile_time_opt*1000:.2f}ms")
    print(f"  Speedup: {compile_time_orig/compile_time_opt:.2f}x")
    
    # Execution time
    for size in [50, 100, 200]:
        values = {
            X: torch.randn(size, size),
            Y: torch.randn(size, size)
        }
        
        time_orig = time_function(fn_orig, values)
        time_opt = time_function(fn_opt, values)
        
        print(f"  Execution ({size}x{size}) - Original: {time_orig*1000:.3f}ms, Optimized: {time_opt*1000:.3f}ms")
        print(f"  Speedup: {time_orig/time_opt:.2f}x")
    
    # 2. Complex sum expression
    print("\n2. Complex Sum Expression")
    A = Variable('A', i=d, j=d)
    B = Variable('B', i=d, j=d)
    C = Variable('C', i=d, j=d)
    expr = A + 2*B - 3*C + A@B - B@C
    
    fn_orig = compile_orig(expr)
    fn_opt = compile_opt(expr)
    
    values = {
        A: torch.randn(100, 100),
        B: torch.randn(100, 100),
        C: torch.randn(100, 100)
    }
    
    time_orig = time_function(fn_orig, values)
    time_opt = time_function(fn_opt, values)
    
    print(f"  Execution - Original: {time_orig*1000:.3f}ms, Optimized: {time_opt*1000:.3f}ms")
    print(f"  Speedup: {time_orig/time_opt:.2f}x")
    
    # 3. Convolution
    print("\n3. Convolution")
    w_in = sympy.Symbol('w_in')
    k_size = sympy.Symbol('k_size')
    w_out = sympy.Symbol('w_out')
    conv = F.Convolution(input=w_in, kernel=k_size, output=w_out)
    
    fn_orig = compile_orig(conv)
    fn_opt = compile_opt(conv)
    
    shapes = {w_in: 100, k_size: 5, w_out: 96}
    
    # Need to pass shapes differently
    def run_conv_orig():
        return fn_orig({}, shapes)
    
    def run_conv_opt():
        return fn_opt({}, shapes)
    
    times_orig = []
    times_opt = []
    for _ in range(50):
        start = time.perf_counter()
        _ = run_conv_orig()
        times_orig.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        _ = run_conv_opt()
        times_opt.append(time.perf_counter() - start)
    
    time_orig = np.mean(times_orig[10:]) * 1000
    time_opt = np.mean(times_opt[10:]) * 1000
    
    print(f"  Execution - Original: {time_orig:.3f}ms, Optimized: {time_opt:.3f}ms")
    print(f"  Speedup: {time_orig/time_opt:.2f}x")
    
    # 4. Many variable access
    print("\n4. Many Variables (tests arg passing efficiency)")
    vars = [Variable(f'v{i}', d) for i in range(20)]
    expr = Sum(vars)
    
    fn_orig = compile_orig(expr)
    fn_opt = compile_opt(expr)
    
    values = {v: torch.randn(100) for v in vars}
    
    # Run many times to test argument passing overhead
    time_orig = time_function(fn_orig, values, n_runs=1000)
    time_opt = time_function(fn_opt, values, n_runs=1000)
    
    print(f"  Execution (1000 runs) - Original: {time_orig*1000:.3f}ms, Optimized: {time_opt*1000:.3f}ms")
    print(f"  Speedup: {time_orig/time_opt:.2f}x")


if __name__ == "__main__":
    benchmark_simple()