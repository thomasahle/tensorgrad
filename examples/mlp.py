from tensorgrad import Variable
import tensorgrad.functions as F
from tensorgrad.extras.to_pytorch import compile_to_callable as compile_to_pytorch
from tensorgrad.extras.to_numpy import compile_to_callable as compile_to_numpy
from tensorgrad.extras.evaluate import evaluate
from tensorgrad.testutils import init_tensor

import torch
from sympy import symbols
from functools import partial
import argparse
import time

import torch.nn.functional as torch_F
from sklearn.datasets import load_wine

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train MLP with tensorgrad")
parser.add_argument(
    "--backend",
    type=str,
    default="pytorch",
    choices=["pytorch", "pytorch-compile", "numpy", "eval"],
    help="Backend to use for compilation (default: pytorch)",
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output during compilation."
)
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
args = parser.parse_args()


def evaluate_callable(*tensors, verbose=False):
    """Wrapper to make evaluate work like compiled backends"""
    def forward_with_values(values, dims):
        results = []
        for tensor in tensors:
            output = evaluate(tensor, values, dims)
            # If this output has the same shape as an input variable but different edge order,
            # align it to match the variable (common for gradients after simplification)
            for var in values.keys():
                if isinstance(var, Variable) and set(tensor.edges) == set(var.edges) and list(tensor.edges) != list(var.edges):
                    output = output.align_to(*var.edges)
                    break
            results.append(output)
        if len(results) == 1:
            return results[0]
        return tuple(results)
    return forward_with_values


if args.backend == "numpy":
    compiler = partial(compile_to_numpy, verbose=args.verbose)
elif args.backend == "pytorch":
    compiler = partial( compile_to_pytorch, torch_compile=False, verbose=args.verbose)
elif args.backend == "pytorch-compile":
    compiler = partial( compile_to_pytorch, torch_compile=True, verbose=args.verbose)
elif args.backend == "eval":
    compiler = partial(evaluate_callable, verbose=args.verbose)


print("Defining MLP architecture and loss symbolically...")

# Step 1: Define network architecture symbolically
batch, in_dim, hidden, out_dim = symbols("batch in_dim hidden out_dim")

# Input and target
x = Variable("x", batch, in_dim)
y = Variable("y", batch, out_dim)

# Parameters
W1 = Variable("W1", in_dim, hidden)
b1 = Variable("b1", batch, hidden)
W2 = Variable("W2", hidden, out_dim)
b2 = Variable("b2", batch, out_dim)

# Step 2: Define forward pass
h = F.relu(x @ W1 + b1)  # Hidden layer with ReLU
logits = h @ W2 + b2  # Output layer

# Step 3: Define loss
loss = F.mean(F.cross_entropy(logits, y, dim="out_dim"))  # Cross-entropy loss

# Step 4: Define accuracy symbolically using tensorgrad
# Get predicted and target class indices
pred_classes = F.argmax(logits, dim="out_dim")
target_classes = F.argmax(y, dim="out_dim")
accuracy = F.mean(F.equal(pred_classes, target_classes))

print("Simplifying...")
# Step 5: Compute all gradients symbolically using a loop
params_list = [W1, b1, W2, b2]
grads = [loss.grad(param).full_simplify() for param in params_list]

print("Compiling to backend...")
# Step 6: Compile to optimized backend
compute_loss_and_grads = compiler(loss, *grads)
compute_accuracy = compiler(accuracy)

print("Generating data and training the MLP...")
# Step 7: Load a simple multi-class dataset (Wine, 13 features, 3 classes)
torch.manual_seed(42)
wine = load_wine()
X = torch.tensor(wine.data, dtype=torch.float)
y_labels = torch.tensor(wine.target, dtype=torch.long)

# Eval backend has limitations with very large batch sizes due to Delta tensor materialization
# Reduce dataset size for eval backend
if args.backend == "eval":
    max_samples = 24  # Limit to 24 samples for eval backend (same as before optimization)
    X = X[:max_samples]
    y_labels = y_labels[:max_samples]
    print(f"Note: Eval backend using reduced dataset ({max_samples} samples) to avoid memory issues with large Delta tensors")

# Normalize features for easier optimization
X = (X - X.mean(dim=0)) / X.std(dim=0)
y_onehot = torch_F.one_hot(y_labels, num_classes=3).float()

x_train = X.refine_names("batch", "in_dim")
y_train = y_onehot.refine_names("batch", "out_dim")
x_test, y_test = x_train, y_train

dims = {
    batch: x_train.shape[0],
    in_dim: x_train.shape[1],
    hidden: 32,
    out_dim: y_train.shape[1],
}

# Initialize parameters using unified init_tensor function
params = {
    var: init_tensor(var, dims, method="he" if var.name.startswith("W") else "zeros")
    for var in params_list
}

# Training hyperparameters
learning_rate = 0.01  # Cross-entropy loss is more stable than Frobenius
num_epochs = args.epochs

# Step 8: Training loop (simplified without gradient clipping and NaN check)
start_time = time.time()
for epoch in range(num_epochs):
    # Forward and backward pass (all computed symbolically in tensorgrad!)
    loss_val, *grad_vals = compute_loss_and_grads(
        {x: x_train, y: y_train, **params}, dims
    )

    # SGD parameter updates - simplified without NaN check
    for param, grad in zip(params_list, grad_vals):
        params[param] -= learning_rate * grad

    # Compute test accuracy periodically
    if epoch < 5 or epoch % 20 == 0 or epoch == num_epochs - 1:
        test_acc = compute_accuracy({x: x_test, y: y_test, **params}, dims)
        elapsed = time.time() - start_time
        eps = (epoch + 1) / elapsed

        print(
            f"Epoch {epoch:3d} | Loss: {loss_val.item():8.4f} | "
            f"Test Acc: {test_acc.item()*100:5.1f}% | "
            f"Epochs/s: {eps:5.2f}"
        )
