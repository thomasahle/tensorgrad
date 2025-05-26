import pytest
from sympy import symbols
import torch

from tensorgrad.extras._convolution import conv_einsum_dispatch
import tensorgrad.functions as F
from tensorgrad.tensor import Product, Variable, Zero, Delta, Sum, Derivative, Ones

# from tensorgrad.extras.to_pytorch import compile_to_callable
from tensorgrad.extras.to_numpy import compile_to_callable
from tensorgrad.testutils import assert_close, rand_values
from tensorgrad.extras.evaluate import evaluate


@pytest.mark.parametrize("compile", [False, True])
def test_codegen_zero(compile):
    """Test generation of a Zero tensor code."""
    i, j = symbols("i j")
    zero_tensor = Zero(i, j)

    # Compile
    compiled_fn = compile_to_callable(zero_tensor, verbose=False, torch_compile=compile)

    # Evaluate reference
    dims = {i: 2, j: 3}
    ref = evaluate(zero_tensor, {}, dims)
    # Evaluate compiled
    out = compiled_fn({}, dims)
    assert_close(ref, out)


@pytest.mark.parametrize("compile", [False, True])
def test_codegen_copy(compile):
    """Test generation of a Copy (identity) tensor code."""
    i = symbols("i")
    copy_tensor = Delta(i, "i, j")

    # Compile
    compiled_fn = compile_to_callable(copy_tensor, verbose=False, torch_compile=compile)

    # Evaluate reference
    dims = {i: 3}
    ref = evaluate(copy_tensor, {}, dims)
    # Evaluate compiled
    out = compiled_fn({}, dims)
    assert_close(ref, out)


@pytest.mark.parametrize("compile", [False, True])
def test_codegen_sum(compile):
    """Test generation of a Sum tensor code."""
    i, j = symbols("i j")
    x = Variable("x", i, j)
    y = Variable("y", i, j)

    expr = Sum([x, y], weights=[2, -1])

    # Compile
    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=compile)

    # Evaluate reference
    dims = {i: 4, j: 5}
    vals = rand_values([x, y], dims)
    ref = evaluate(expr, vals, dims)
    # Evaluate compiled
    out = compiled_fn(vals, dims)
    assert_close(ref, out)


@pytest.mark.parametrize("compile", [False, True])
def test_codegen_product(compile):
    """Test generation of a Product tensor code."""
    i, j, k = symbols("i j k")
    # a has shape (i, j), b has shape (j, k)
    a = Variable("a", i, j)
    b = Variable("b", j, k)
    expr = a @ b

    # Compile
    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=compile)

    # Evaluate reference
    dims = {i: 2, j: 3, k: 4}
    vals = rand_values([a, b], dims)
    ref = evaluate(expr, vals, dims)
    # Evaluate compiled
    out = compiled_fn(vals, dims)
    assert_close(ref, out)


def test_codegen_simple_function():
    """Test generation of a simple function like ReLU."""
    i = symbols("i")
    x = Variable("x", i)
    expr = F.relu(x)  # edges = [i]

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 5}
    vals = rand_values([x], dims)
    ref = evaluate(expr, vals, dims)
    out = compiled_fn(vals, dims)
    assert_close(ref, out)


def test_codegen_argmax():
    """Test generation of the argmax function."""
    i, j = symbols("i j")
    x = Variable("x", i, j)
    # Argmax over j => shape is (i,)
    expr = F.argmax(x, dim="j")

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 4, j: 3}
    vals = rand_values([x], dims)
    ref = evaluate(expr, vals, dims)
    out = compiled_fn(vals, dims)
    assert (ref == out).all()


def test_codegen_power():
    """Test generation of a power function (x^2)."""
    i = symbols("i")
    x = Variable("x", i)
    expr = F.pow(x, 2)

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 6}
    vals = rand_values([x], dims)
    ref = evaluate(expr, vals, dims)
    out = compiled_fn(vals, dims)
    assert_close(ref, out)


def test_codegen_derivative_placeholder():
    """
    Test generation of a Derivative object.
    The current code generation stub returns a zero tensor,
    but we verify it compiles and has the correct shape.
    """
    i = symbols("i")
    x = Variable("x", i)
    expr = Derivative(x, x).simplify()  # partial derivative w.r.t. x

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 5}
    vals = rand_values([x], dims)
    ref = evaluate(expr, vals, dims)  # We expect a zero of shape (i, i_) in the current stub
    out = compiled_fn(vals, dims)
    assert_close(ref, out)


def test_codegen_full_expression():
    """
    Test a more involved expression combining Sum, Product, ReLU, etc.
    """
    i, j, k = symbols("i j k")
    x = Variable("x", i, j)
    w = Variable("w", j, k)
    b = Variable("b", i, k)

    expr = F.relu(x @ w + b)  # shape (i, k)
    grad = Derivative(expr, x).simplify()  # shape (i, k, i', j)

    # Call the code generator with two tensors
    compiled_fn = compile_to_callable(expr, grad, verbose=True)

    dims = {i: 2, j: 3, k: 4}
    vals = rand_values([x, w, b], dims)
    ref = evaluate(expr, vals, dims)
    grad_ref = evaluate(grad, vals, dims)
    out, grad_out = compiled_fn(vals, dims)
    assert out.names == tuple(expr.edges)
    assert_close(ref, out)
    assert_close(grad_ref, grad_out)


def test_codegen_ones():
    """Test generation of an expression that includes a constant ones tensor."""
    i, j = symbols("i j")
    # We'll create a small expression: x + 2 * (Ones(i,j))
    x = Variable("x", i, j)
    ones_expr = Ones(i, j)  # This is not an explicit class in the posted code, but let's assume
    # we can test something similar or your own internal "Ones" approach
    # If not, use e.g. Zero() or Copy() for demonstration.
    expr = x + 2 * ones_expr

    compiled_fn = compile_to_callable(expr, verbose=False, torch_compile=False)

    dims = {i: 3, j: 2}
    vals = rand_values([x], dims)
    ref = evaluate(expr, vals, dims)
    out = compiled_fn(vals, dims)
    assert_close(ref, out)


def test_from_random():
    a, b = symbols("a b")
    var_a = Variable("var_a", a)
    var_b = Variable("var_b", b)
    var_a_b = Variable("var_a_b", a, b)
    expr = Sum(
        [
            Product(
                [
                    Delta(b, "b, b_1"),
                    var_a_b.rename(b="b_1"),
                ]
            ),
            Product(
                [
                    var_a,
                    Delta(b, "b"),
                ]
            ),
        ],
    )
    compiled_fn = compile_to_callable(expr, verbose=True)
    dims = {a: 3, b: 2}
    vals = rand_values([var_a, var_b, var_a_b], dims)
    ref = evaluate(expr, vals, dims)
    out = compiled_fn(vals, dims)
    assert_close(ref, out)


def test_convolution():
    batch, c_in, c_out, h_in, h_out, w_in, w_out, ks = symbols("batch c_in c_out h_in h_out w_in w_out ks")
    data = Variable("data", batch, c_in, h_in, w_in)
    h_conv = F.Convolution(h_in, h_out, hk=ks)
    w_conv = F.Convolution(w_in, w_out, wk=ks)
    kernel = Variable("kernel", c_in, c_out, hk=ks, wk=ks)
    expr = Product([data, kernel, h_conv, w_conv])
    compiled_fn = compile_to_callable(expr, verbose=True)
    dims = {batch: 2, c_in: 3, c_out: 4, h_in: 5, h_out: 3, w_in: 5, w_out: 3, ks: 3}
    vals = rand_values([data, kernel], dims)
    ref = evaluate(expr, vals, dims)
    out = compiled_fn(vals, dims)
    assert_close(ref, out)


@pytest.mark.parametrize(
    "X,Z",
    [
        (2, 1),  # Minimal corner case (X=2 => Y=1, Z=1)
        (3, 2),  # Smaller case
        (6, 3),  # Mid-size case
        (10, 4),  # Example from your code
    ],
)
@pytest.mark.parametrize("B", [1, 2, 5])  # various batch sizes
@pytest.mark.parametrize(
    "einsum_str",
    [
        "xb,xzy->bzy",
        "by,xzy->bxz",
        "bz,xzy->bxy",
        "bxy,xzy->bz",
        "xz,xzy->y",
        "bzy,xzy->bx",
        "byz,xzy->bx",
        "byz,xzy->bzx",
        "bxyz,xzy->bxyz",
    ],
)
def test_all_einsums(einsum_str, X, Z, B):
    """
    Tests the conv_einsum_dispatch function by comparing to a reference
    torch.einsum call that uses a naive (X,Y,Z) Toeplitz conv kernel.
    We loop over various shapes (X,Y,Z), batch sizes B, and einsum patterns.
    """
    Y = X - Z + 1  # X = Y + Z - 1
    lhs, rhs = einsum_str.split("->")
    A_dims, B_dims = lhs.split(",")
    assert B_dims == "xzy"

    # Build the naive Toeplitz conv kernel: conv[x,z,y] = 1 if x == y+z
    conv_kernel = torch.zeros(X, Z, Y)
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if x == y + z:
                    conv_kernel[x, z, y] = 1

    # Map 'x'->X, 'y'->Y, 'z'->Z, 'b'->B to build A's shape
    dim_map = {"x": X, "y": Y, "z": Z, "b": B}
    A_shape = tuple(dim_map[d] for d in A_dims)
    A = torch.randn(*A_shape)

    # Reference via standard torch.einsum
    ref = torch.einsum(einsum_str, A, conv_kernel)

    # Call your dispatcher: conv_einsum_dispatch must be in scope
    out = conv_einsum_dispatch(einsum_str, A, (X, Z, Y))

    # Compare
    assert torch.allclose(ref, out, atol=1e-6, rtol=1e-5), (
        f"Mismatch in pattern={einsum_str}, A.shape={A_shape}, "
        f"X={X},Y={Y},Z={Z},B={B}\n"
        f"ref.shape={ref.shape}, out.shape={out.shape}"
    )


@pytest.mark.parametrize("model", ["linear-1", "linear-2", "conv-1"])
def test_mnist(model):
    n_epochs = 10
    batch_size = 32
    lr = 5e-2

    # Create model
    batch, c0, w0, h0, out, kernel_size = symbols("batch c0 w0 h0 out ks")
    data = Variable("data", batch, c0, w0, h0)
    targets = Variable("targets", batch, out)
    shapes = {
        batch: batch_size,
        c0: 1,
        w0: 18,
        h0: 18,
        kernel_size: 3,
        out: 10,
    }

    # Just one batch of data is fine
    mnist = torch.randn(batch_size, shapes[c0], shapes[w0], shapes[h0])
    mnist_targets = torch.randint(0, shapes[out], (batch_size,))

    # Repeat the same data so the model can overfit
    training_data = [(mnist, mnist_targets)] * 100

    layers = []

    def conv_layer(channels: int):
        # Declare heigth and weidth convolutions
        i = len(layers)
        c_in, c_out, h_in, h_out, w_in, w_out = symbols(f"c{i} c{i+1} h{i} h{i+1}, w{i} w{i+1}")
        h_conv = F.Convolution(h_in, h_out, hk=kernel_size)
        w_conv = F.Convolution(w_in, w_out, wk=kernel_size)
        kernel = Variable(f"kernel_{i}", c_in, c_out, hk=kernel_size, wk=kernel_size)
        # Save the layer and shapes of the inner dimensions
        layers.append(kernel)
        shapes[c_out] = channels
        shapes[h_out] = shapes[h_in] - shapes[kernel_size] + 1
        shapes[w_out] = shapes[w_in] - shapes[kernel_size] + 1
        # Apply the convolution
        return kernel @ h_conv @ w_conv

    # Build the model
    x = data

    if model == "conv-2":
        x = F.relu(x @ conv_layer(channels=2)).simplify()
        x = F.relu(x @ conv_layer(channels=3)).simplify()
        c2, h2, w2, c3 = symbols("c2 h2 w2 c3")
        shapes[c3] = shapes[c2] * shapes[w2] * shapes[h2]
        layers.append(linear := Variable("lin", c2, h2, w2, out))
        logits = x @ linear

    elif model == "conv-1":
        x = F.relu(x @ conv_layer(channels=2)).simplify()
        c1, h1, w1, c2 = symbols("c1 h1 w1 c2")
        shapes[c2] = shapes[c1] * shapes[w1] * shapes[h1]
        layers.append(linear := Variable("lin", c1, h1, w1, out))
        logits = x @ linear

    elif model == "linear-2":
        shapes[mid := symbols("mid")] = 40  # Arbitrary number of hidden units
        layers.append(linear1 := Variable("lin1", c0, h0, w0, mid))
        layers.append(linear2 := Variable("lin2", mid, out))
        x = F.relu(x @ linear1)
        logits = x @ linear2

    elif model == "linear-1":
        layers.append(linear := Variable("lin", c0, w0, h0, out))
        logits = x @ linear

    layers.append(bias := Variable("bias", out))
    logits += bias

    # y = F.cross_entropy(logits, targets, dim='out')
    y = F.mean((logits - targets) ** 2, dim="out")
    y = F.mean(y, dim="batch")
    y = y.full_simplify()
    prediction = F.argmax(logits, dim="out")

    print("Computing and simplifying gradients")
    grad_tensors = [y.grad(param).full_simplify() for param in layers]

    backprop = compile_to_callable(prediction, y, *grad_tensors, verbose=True, torch_compile=False)

    # Train
    print("Training...")
    parameters = rand_values(layers, shapes)
    parameters = {s: t / sum(t.shape) ** 0.5 for s, t in parameters.items()}
    for _ in range(n_epochs):
        total_loss = 0
        corr = 0
        batches = 0
        for t_data, t_target in training_data:
            shapes[batch] = t_data.shape[0]
            input_params = {t: p.clone() for t, p in parameters.items()}
            input_params[data] = t_data.rename("batch", "c0", "w0", "h0")
            input_params[targets] = torch.eye(10)[t_target].rename("batch", "out")

            # Forward and backward pass
            pred_out, y_out, *grad_outputs = backprop(input_params, shapes)

            # Grad update
            for layer, grad in zip(layers, grad_outputs):
                g = grad.align_to(*parameters[layer].names)
                parameters[layer] -= lr * g

            # Forward pass
            total_loss += y_out / shapes[batch]
            corr += (pred_out == t_target).sum() / shapes[batch]
            batches += 1

        # We just hope it learned a little bit
        assert corr / batches > 0.15, f"Accuracy too low: {corr / batches}"

