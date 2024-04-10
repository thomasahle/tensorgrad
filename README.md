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

This will output the tensor diagram:

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/l2_grad_w_single_step.png" width="50%">

Together with the pytorch code for numerically computing the gradient with respect to W:
```python
import torch
WX = torch.einsum('xy,bx -> by', W, X)
subtraction = WX - Y
X_subtraction = torch.einsum('bx,by -> xy', X, subtraction)
final_result = 2 * X_subtraction
```

The neat thing about tensorgrad is that it will also give you the step by step instructions to see how the rules of derivatives are computed on tensors like this:

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/l2_grad_w.png" width="66%">




# Basic Elements 

# Matrix Calculus

# Transformers

# Convolution Neural Netowrks

# Tensor Sketch
