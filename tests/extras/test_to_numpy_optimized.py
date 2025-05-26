#!/usr/bin/env python3
"""Test script to verify the optimized to_numpy module works correctly."""

import numpy as np
import torch
import sympy
from tensorgrad import Tensor, Variable, Delta, Sum, Product
from tensorgrad import functions as F
from tensorgrad.extras.to_numpy import compile_to_callable as compile_orig
from tensorgrad.extras.to_numpy_optimized import compile_to_callable as compile_opt


def test_basic_operations():
    """Test basic tensor operations."""
    print("Testing basic operations...")
    
    # Test 1: Simple matrix multiplication
    d = sympy.Symbol('d')
    X = Variable('X', i=d, j=d)
    Y = Variable('Y', i=d, j=d)
    Z = X @ Y
    
    # Create test data
    values = {
        X: torch.randn(3, 3),
        Y: torch.randn(3, 3)
    }
    
    # Compile with both versions
    fn_orig = compile_orig(Z)
    fn_opt = compile_opt(Z)
    
    # Execute and compare
    result_orig = fn_orig(values)
    result_opt = fn_opt(values)
    
    assert torch.allclose(result_orig.rename(None), result_opt.rename(None), rtol=1e-5), "Matrix multiplication results differ"
    print("✓ Matrix multiplication test passed")
    
    # Test 2: Sum of tensors
    A = Variable('A', i=d, j=d)
    B = Variable('B', i=d, j=d)
    C = Variable('C', i=d, j=d)
    S = A + 2*B - C
    
    values = {
        A: torch.randn(4, 4),
        B: torch.randn(4, 4),
        C: torch.randn(4, 4)
    }
    
    fn_orig = compile_orig(S)
    fn_opt = compile_opt(S)
    
    result_orig = fn_orig(values)
    result_opt = fn_opt(values)
    
    assert torch.allclose(result_orig.rename(None), result_opt.rename(None), rtol=1e-5), "Sum results differ"
    print("✓ Sum test passed")
    
    # Test 3: Delta tensor
    I = Delta(d, 'i', 'j')
    
    fn_orig = compile_orig(I)
    fn_opt = compile_opt(I)
    
    result_orig = fn_orig({}, {d: 5})
    result_opt = fn_opt({}, {d: 5})
    
    assert torch.allclose(result_orig.rename(None), result_opt.rename(None)), "Delta results differ"
    print("✓ Delta test passed")


def test_functions():
    """Test function operations."""
    print("\nTesting functions...")
    
    d = sympy.Symbol('d')
    X = Variable('X', i=d, j=d)
    
    # Test ReLU
    R = F.relu(X)
    values = {X: torch.randn(3, 3)}
    
    fn_orig = compile_orig(R)
    fn_opt = compile_opt(R)
    
    result_orig = fn_orig(values)
    result_opt = fn_opt(values)
    
    assert torch.allclose(result_orig.rename(None), result_opt.rename(None)), "ReLU results differ"
    print("✓ ReLU test passed")
    
    # Test exp
    E = F.exp(X)
    fn_orig = compile_orig(E)
    fn_opt = compile_opt(E)
    
    result_orig = fn_orig(values)
    result_opt = fn_opt(values)
    
    assert torch.allclose(result_orig.rename(None), result_opt.rename(None), rtol=1e-5), "Exp results differ"
    print("✓ Exp test passed")
    
    # Test power
    P = X ** 2
    fn_orig = compile_orig(P)
    fn_opt = compile_opt(P)
    
    result_orig = fn_orig(values)
    result_opt = fn_opt(values)
    
    assert torch.allclose(result_orig.rename(None), result_opt.rename(None), rtol=1e-5), "Power results differ"
    print("✓ Power test passed")


def test_convolution():
    """Test convolution operation."""
    print("\nTesting convolution...")
    
    w_in = sympy.Symbol('w_in')
    k_size = sympy.Symbol('k_size')
    w_out = sympy.Symbol('w_out')
    
    Conv = F.Convolution(input=w_in, kernel=k_size, output=w_out)
    
    shapes = {w_in: 10, k_size: 3, w_out: 8}
    
    fn_orig = compile_orig(Conv)
    fn_opt = compile_opt(Conv)
    
    result_orig = fn_orig({}, shapes)
    result_opt = fn_opt({}, shapes)
    
    # Convert sparse to dense for comparison if needed
    if hasattr(result_orig, 'todense'):
        result_orig = torch.from_numpy(result_orig.todense())
    if hasattr(result_opt, 'todense'):
        result_opt = torch.from_numpy(result_opt.todense())
    
    assert torch.allclose(result_orig.rename(None), result_opt.rename(None)), "Convolution results differ"
    print("✓ Convolution test passed")


def test_complex_expression():
    """Test a more complex expression."""
    print("\nTesting complex expression...")
    
    batch = sympy.Symbol('batch')
    d = sympy.Symbol('d')
    
    X = Variable('X', i=batch, j=d)
    W1 = Variable('W1', j=d, k=d)
    W2 = Variable('W2', j=d, k=d)
    b1 = Variable('b1', j=d)
    b2 = Variable('b2', j=d)
    
    # Two-layer neural network with ReLU
    # X @ W1 results in shape (batch, d) with edges i, k
    # Broadcast bias: need to expand b1 from shape (d,) to (batch, d)
    bias1 = Product([Delta(batch, 'i'), b1.rename(j='k')])
    hidden = F.relu(X @ W1 + bias1)
    
    # hidden @ W2 also results in shape (batch, d) with edges i, k  
    bias2 = Product([Delta(batch, 'i'), b2.rename(j='k')])
    output = hidden @ W2 + bias2
    
    values = {
        X: torch.randn(32, 64),
        W1: torch.randn(64, 64),
        W2: torch.randn(64, 64),
        b1: torch.randn(64),
        b2: torch.randn(64)
    }
    
    fn_orig = compile_orig(output)
    fn_opt = compile_opt(output)
    
    result_orig = fn_orig(values)
    result_opt = fn_opt(values)
    
    
    torch.testing.assert_close(result_orig.rename(None), result_opt.rename(None), rtol=1e-3, atol=1e-3)
    print("✓ Complex expression test passed")


def test_multiple_outputs():
    """Test multiple output tensors."""
    print("\nTesting multiple outputs...")
    
    d = sympy.Symbol('d')
    X = Variable('X', i=d, j=d)
    Y = Variable('Y', i=d, j=d)
    
    A = X + Y
    B = X - Y
    C = X @ Y
    
    values = {
        X: torch.randn(3, 3),
        Y: torch.randn(3, 3)
    }
    
    fn_orig = compile_orig(A, B, C)
    fn_opt = compile_opt(A, B, C)
    
    results_orig = fn_orig(values)
    results_opt = fn_opt(values)
    
    assert len(results_orig) == len(results_opt) == 3
    for i, (orig, opt) in enumerate(zip(results_orig, results_opt)):
        assert torch.allclose(orig.rename(None), opt.rename(None), rtol=1e-5), f"Output {i} differs"
    
    print("✓ Multiple outputs test passed")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    # Empty product
    empty = Product([])
    fn_orig = compile_orig(empty)
    fn_opt = compile_opt(empty)
    
    result_orig = fn_orig({})
    result_opt = fn_opt({})
    
    assert torch.allclose(result_orig.rename(None), result_opt.rename(None)), "Empty product results differ"
    print("✓ Empty product test passed")
    
    # Single factor product
    d = sympy.Symbol('d')
    X = Variable('X', d)
    single = Product([X])
    
    values = {X: torch.randn(5)}
    
    fn_orig = compile_orig(single)
    fn_opt = compile_opt(single)
    
    result_orig = fn_orig(values)
    result_opt = fn_opt(values)
    
    assert torch.allclose(result_orig.rename(None), result_opt.rename(None)), "Single factor product results differ"
    print("✓ Single factor product test passed")


def main():
    """Run all tests."""
    print("Running correctness tests for optimized to_numpy module...\n")
    
    test_basic_operations()
    test_functions()
    test_convolution()
    test_complex_expression()
    test_multiple_outputs()
    test_edge_cases()
    
    print("\n✅ All tests passed! The optimized version produces identical results.")


if __name__ == "__main__":
    main()
