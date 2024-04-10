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

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/hessian_yaroslaw.png">

# Transformers

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/attention.png">

# Convolution Neural Netowrks

# Tensor Sketch

Taken from [this Twitter thread](https://twitter.com/thomasahle/status/1674572437953089536):
I wish I had know about Tensor Graphs back when i worked on Tensor-sketching.
Let me correct this now and explain dimensionality reduction for tensors using Tensor Networks:

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/ts_simple.png" width="66%">

The second version is the "original" Tensor Sketch by 
Rasmus Pagh and Ninh Pham. (https://rasmuspagh.net/papers/tensorsketch.pdf) Each fiber is reduced by a JL sketch, and the result is element-wise multiplied.
Note the output of each JL is larger than in the "simple" sketch to give the same output size.

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/ts_pp.png" width="66%">

Next we have the "recursive" sketch by myself and coauthors in https://thomasahle.com/#paper-tensorsketch-joint.
In the paper we sometimes describe this as a tree, but it doesn't really matter. We just had already created the tree-graphic when we realized.

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/ts_tree.png" width="66%">

The main issue with the AKKRVWZ-sketch was that we used order-3 tensors internally, which require more space/time than simple random matrices in the PP-sketch.
We can mitigate this issue by replacing each order-3 tensor with a simple order-2 PP-sketch.

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/ts_hybrid.png" width="66%">


Finally we can speed up each matrix multiplication by using FastJL, which is itself basically an outer product of a bunch of tiny matrices. But at this point my picture is starting to get a bit overwhelming.

# See also

- [Tool for creating tensor diagrams from einsum](https://thomasahle.com/blog/einsum_to_dot.html?q=abc,cde,efg,ghi,ija-%3Ebdfhj&layout=circo) by Thomas Ahle
