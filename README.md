# tensorgrad
Tensorgrad is a library for visualizing symbolic differentiation through tensor networks.

Tensor network diagrams, also known as Penrose graphical notation or tensor diagram notation, provide a powerful visual representation for tensors and their contractions. Introduced by Roger Penrose in 1971, these diagrams have become widely used in various fields such as quantum physics, multilinear algebra, and machine learning. They allow complex tensor equations to be represented in a way that makes the structure of the contraction clear, without cluttering the notation with explicit indices.

## Examples

### Derivative of L2 Loss

```python
from tensorgrad import Variable, Derivative
import tensorgrad.functions as F
# ||Ax - y||_2^2
x = Variable("x", ["x"])
y = Variable("y", ["y"])
A = Variable("A", ["x", "y"])
Axmy = A @ x - y
frob = F.frobenius2(Axmy)
grad = Derivative(frob, x)
```


# Basic Elements 

# Matrix Calculus

# Transformers

# Convolution Neural Netowrks

# Tensor Sketch