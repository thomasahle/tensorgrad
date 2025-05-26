import sympy
from tensorgrad import Variable
from tensorgrad.extras.to_numpy_optimized import compile_to_callable

# Create a simple test
d = sympy.Symbol('d')
X = Variable('X', i=d, j=d)
Y = Variable('Y', i=d, j=d)
Z = X @ Y

print(f"Symbol d: {d}, id: {id(d)}")
print(f"X.shape: {X.shape}")
print(f"X.shape values: {list(X.shape.values())}")
print(f"X.shape value ids: {[id(v) for v in X.shape.values()]}")

# Try to compile
fn = compile_to_callable(Z, verbose=True)