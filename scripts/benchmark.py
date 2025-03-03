"""
A simple benchmark script to test the performance of the cached structural graph.
"""
import time
from sympy import symbols
from tensorgrad import Variable
import tensorgrad.functions as F

def run_benchmark(iterations=10000):
    # Create test variables
    i, j, k = symbols("i j k")
    s = symbols("s")  # Same size for symmetric dimensions
    X = Variable("X", i, j)
    W = Variable("W", j, k)
    Y = Variable("Y", i, k)
    Z = Variable("Z", i=s, j=s)  # For symmetry testing
    
    # Perform operations that use structural graph
    start_time = time.time()
    
    for _ in range(iterations):
        # Hash computation
        hash(X)
        hash(W)
        
        # Isomorphism checking
        X.is_isomorphic(X.rename(i="i_new", j="j_new"))
        
        # Symmetries computation
        Z_sym = Z.with_symmetries("i j")
        Z_sym.symmetries
        
        # Complex expression with multiple isomorphism tests
        expr = F.frobenius2(X @ W - Y)
        grad = expr.grad(W)
        grad.full_simplify()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    return {
        "total_time": elapsed,
        "avg_time_per_iteration": elapsed / iterations,
    }

if __name__ == "__main__":
    print("Running benchmark...")
    results = run_benchmark(iterations=5)
    print(f"Total time: {results['total_time']:.4f} seconds")
    print(f"Average time per iteration: {results['avg_time_per_iteration']:.4f} seconds")
