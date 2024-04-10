# tensorgrad
Tensorgrad is a library for visualizing symbolic differentiation through tensor networks.

Tensor network diagrams, also known as [Penrose graphical notation](https://en.wikipedia.org/wiki/Penrose_graphical_notation) or [tensor diagram notation](https://tensornetwork.org/), provide a powerful visual representation for tensors and their contractions. Introduced by Roger Penrose in 1971, these diagrams have become widely used in various fields such as quantum physics, multilinear algebra, and machine learning. They allow complex tensor equations to be represented in a way that makes the structure of the contraction clear, without cluttering the notation with explicit indices.

## Examples

To run the examples for yourself, you can use [the main.py file](https://github.com/thomasahle/tensorgrad/blob/main/main.py) or [this colab notebook](https://colab.research.google.com/drive/10Lk39tTgRd-cCo5gNNe3KvdDcVP2F5aB?usp=sharing).

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

For a beautiful introduction to tensor networks, see [Jordan Taylor's blog post](https://www.lesswrong.com/posts/BQKKQiBmc63fwjDrj/graphical-tensor-notation-for-interpretability)
in which he gives analogies like this one:
<img src="https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/BQKKQiBmc63fwjDrj/sn4nuvu7eocdljp3pc6g" width="100%">

# Matrix Calculus

In Penrose's book, The Road to Reality: A Complete Guide to the Laws of the Universe, he introduces a notation for taking derivatives on tensor networks. In this library we try to follow Penrose's notation, expanding it as needed to handle a full "chain rule" on tensor functions.
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Penrose_covariant_derivate.svg/2880px-Penrose_covariant_derivate.svg.png" width="100%">

Another source of inspiration was Yaroslav Bulatov's [derivation of the hessian of neural networks](https://community.wolfram.com/groups/-/m/t/2437093):


# Transformers

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/attention.png" width="100%">

# Convolution Neural Netowrks

# Tensor Sketch

# See also

- [Tool for creating tensor diagrams from einsum](https://thomasahle.com/blog/einsum_to_dot.html?q=abc,cde,efg,ghi,ija-%3Ebdfhj&layout=circo) by Thomas Ahle
