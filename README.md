<div align="center">
<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/basics.png" width="100%">
<h3>
  
[Book](https://tensorcookbook.com) | [Documentation](https://tensorcookbook.com/docs) | [API Reference](https://tensorcookbook.com/docs/api)

</h3>
</div>

# Tensorgrad
A tensor & deep learning framework. PyTorch meets SymPy.

Tensor diagrams let you manipulate high dimensional tensors are graphs in a way that makes derivatives and complex products easy to understand.
The [Tensor Cookbook (draft)](https://github.com/thomasahle/tensorgrad/blob/main/paper/cookbook.pdf) contains everything you need to know.

## Examples

To run the examples for yourself, see [this colab notebook](https://colab.research.google.com/drive/10Lk39tTgRd-cCo5gNNe3KvdDcVP2F5aB?usp=sharing).

Install tensorgrad with
```bash
pip install tensorg
```

For visualizations we need some latex packages:
```bash
apt-get install texlive-luatex
apt-get install texlive-latex-extra
apt-get install texlive-fonts-extra
apt-get install poppler-utils
```

### Derivative of L2 Loss

```python
from tensorgrad import Variable
import tensorgrad.functions as F
# ||Ax - y||_2^2
b, x, y = sp.symbols("b x y")
X = tg.Variable("X", b, x)
Y = tg.Variable("Y", b, y)
W = tg.Variable("W", x, y)
XWmY = X @ W - Y
l2 = XWmY @ XWmY
grad = l2.grad(W)
display_pdf_image(to_tikz(grad.full_simplify()))
```

This will output the tensor diagram:

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/l2_grad_w_single_step.png" width="50%">

Tensorgrad can also output pytorch code for numerically computing the gradient with respect to W:
```python
>>> to_pytorch(grad)
import torch
WX = torch.einsum('xy,bx -> by', W, X)
subtraction = WX - Y
X_subtraction = torch.einsum('bx,by -> xy', X, subtraction)
final_result = 2 * X_subtraction
```

### Hessian of CE Loss

For a more complicated example, consider the following program for computing the Entropy of Cross Entropy Loss:

```python
from tensorgrad import Variable
import tensorgrad.functions as F

logits = Variable("logits", ["C"])
target = Variable("target", ["C"])

e = F.exp(logits)
softmax = e / F.sum(e)
ce = -F.sum(target * F.log(softmax))

H = ce.grad(logits).grad(logits)

display_pdf_image(to_tikz(H.full_simplify()))
```

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/hess_ce.png" width="50%">

This is tensor diagram notation for `(diag(p) - pp^T) sum(target)`, where `p = softmax(logits)`.

### Expectations

Tensorgrad can also take expectations of arbitrary functions with respect to Gaussian tensors.

As an example, consider the L2 Loss program from before:
```python
X = Variable("X", "b, x")
Y = Variable("Y", "b, y")
W = Variable("W", "x, y")
mu = Variable("mu", "x, y")
C = Variable("C", "x, y, x2, y2")
XWmY = X @ W - Y
l2 = F.sum(XWmY * XWmY)
E = Expectation(l2, W, mu, C)
display_pdf_image(to_tikz(E.full_simplify()))
```

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/expectation.png" width="50%">

Note that the covariance is a rank-4 tensor (illustrated with a star) since we take the expectation with respect to a matrix.
This is different from the normal "matrix shaped" covariance you get if you take expectation with respect to a vector.

### Evaluation

Tensorgrad can evaluate your diagrams using [Pytorch Named Tensors](https://pytorch.org/docs/stable/named_tensor.html).
It uses graph isomorphism detection to eliminated common subexpressions.

### Code Generation

Tensorgrad can convert your diagrams back into pytorch code.
This gives a super optimized way to do gradients and higher order derivatives in neural networks.


### Matrix Calculus

In Penrose's book, The Road to Reality: A Complete Guide to the Laws of the Universe, he introduces a notation for taking derivatives on tensor networks. In this library we try to follow Penrose's notation, expanding it as needed to handle a full "chain rule" on tensor functions.
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Penrose_covariant_derivate.svg/2880px-Penrose_covariant_derivate.svg.png" width="100%">

Another source of inspiration was Yaroslav Bulatov's [derivation of the hessian of neural networks](https://community.wolfram.com/groups/-/m/t/2437093):

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/hessian_yaroslaw.png">

# More stuff

## Transformers

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/attention.png">

## Convolutional Neural Networks 

The main ingredient in CNNs are the linear operations Fold and Unfold. 
Unfold takes an image, with dimensions HxW and outputs P "patches" of size K^2, where K is the kernel size. Fold is the reverse operation. 
Since they are linear operations (they consists only of copying/adding) we can express them as a tensor with shape (H, W, P, K^2).

<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/uCrOg.png" widht="80%">

<a href="https://arxiv.org/abs/1908.04471">Hayashi et al.</a> show that if you define a tensor `(∗)_{i,j,k} = [i=j+k]`, then the "Unfold" operator factors along the spacial dimensions, and you can write a bunch of different convolutional neural networks easily as tensor networks:
<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d3141305235795371446e48715939624650677163546f6b347735416a516c666572.png">

With tensorgrad you can write the "standard" convolutional neural network like this:
```python
data = Variable("data", ["b", "c", "w", "h"])
unfold = Convolution("w", "j", "w2") @ Convolution("h", "i", "h2")
kernel = Variable("kernel", ["c", "i", "j", "c2"])
expr = data @ unfold @ kernel
```

And then easily find the jacobian symbolically with `expr.grad(kernel)`:
<img src="https://raw.githubusercontent.com/thomasahle/tensorgrad/main/docs/images/conv_jac.png">

## Tensor Sketch

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

## See also

- [Tool for creating tensor diagrams from einsum](https://thomasahle.com/blog/einsum_to_dot.html?q=abc,cde,efg,ghi,ija-%3Ebdfhj&layout=circo) by Thomas Ahle
- [Ideograph: A Language for Expressing and Manipulating Structured Data](https://arxiv.org/pdf/2303.15784.pdf) by Stephen Mell, Osbert Bastani, Steve Zdancewic

