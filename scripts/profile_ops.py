"""
Script to profile common tensor operations in tensorgrad.

This script runs a series of standard tensor operations and profiles their performance.
It shows which operations are most expensive and helps identify optimization targets.

Usage:
    python scripts/profile_ops.py [output_file]
"""
import cProfile
import pstats
from pstats import SortKey
import os
import sys
from sympy import symbols
from tensorgrad import Variable
import tensorgrad.functions as F

def profile_tensor_operations():
    """Profile a set of common tensor operations."""
    # Define symbols for dimensions
    i, j, k, l = symbols("i j k l")
    n, m, p, q = symbols("n m p q")
    
    # Create test variables
    X = Variable("X", i, j)
    W = Variable("W", j, k)
    B = Variable("B", i, k)
    Y = Variable("Y", i, k)
    
    # Operations to profile
    operations = []
    
    # 1. Matrix multiplication and gradient
    matmul = X @ W
    operations.append(("Matrix Multiplication", lambda: X @ W))
    operations.append(("MatMul Gradient", lambda: matmul.grad(W)))
    
    # 2. Addition, subtraction, and their gradients
    addition = X @ W + B
    operations.append(("Addition", lambda: X @ W + B))
    operations.append(("Addition Gradient", lambda: addition.grad(B)))
    
    # 3. L2 Loss and its gradient
    l2_loss = F.frobenius2(X @ W - Y)
    operations.append(("L2 Loss", lambda: F.frobenius2(X @ W - Y)))
    operations.append(("L2 Loss Gradient", lambda: l2_loss.grad(W)))
    
    # 4. Simplification of complex expressions
    complex_expr = l2_loss.grad(W)
    operations.append(("Simplify", lambda: complex_expr.simplify()))
    operations.append(("Full Simplify", lambda: complex_expr.full_simplify()))
    
    # 5. Function composition and gradients
    softmax_expr = F.softmax(X @ W, dim='k')
    operations.append(("Softmax", lambda: F.softmax(X @ W, dim='k')))
    operations.append(("Softmax Gradient", lambda: softmax_expr.grad(W)))
    
    # 6. Hessian computation (second derivative)
    hessian = l2_loss.grad(W).grad(W)
    operations.append(("Hessian", lambda: l2_loss.grad(W).grad(W)))
    operations.append(("Hessian Simplify", lambda: hessian.simplify()))
    
    # 7. Tensor isomorphism tests
    operations.append(("Isomorphism Test", lambda: X @ W == X.rename(i="i_") @ W.rename(j="j_")))
    
    # Run the operations and collect timing data
    results = {}
    for name, op in operations:
        # Clear any cached computations
        op()  # Warm-up run
        
        # Time the operation
        results[name] = op()
    
    return results

if __name__ == "__main__":
    # Set up the output file path
    output_file = "profiling/tensor_ops.prof"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Run the profiling
    print("Profiling tensor operations...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    results = profile_tensor_operations()
    
    profiler.disable()
    profiler.dump_stats(output_file)
    
    # Print a summary
    print(f"\nProfile data saved to {output_file}")
    print("\nOperation summary:")
    for name in results:
        print(f"- {name}")
    
    print("\nTo analyze the profile, run:")
    print(f"python scripts/analyze_profile.py {output_file}")