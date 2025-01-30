from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sympy import symbols
import tqdm
import torch
import argparse

from tensorgrad import Variable
from tensorgrad.testutils import rand_values
from tensorgrad.extras.to_pytorch import compile_to_callable as compile_torch
from tensorgrad.extras.to_numpy import compile_to_callable as compile_numpy
import tensorgrad.functions as F


def main(args):
    n_epochs = 10
    batch_size = 32
    lr = 1e-2

    # Load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_args = dict(root="./data", download=True, transform=transform)
    train_dataset = datasets.MNIST(train=True, **mnist_args)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataset = datasets.MNIST(train=False, **mnist_args)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    batch, c0, w0, h0, out, kernel_size = symbols("batch c0 w0 h0 out ks")
    data = Variable("data", batch, c0, w0, h0)
    targets = Variable("targets", batch, out)
    shapes = {
        batch: batch_size,
        c0: 1,
        w0: 28,
        h0: 28,
        kernel_size: 3,
        out: 10,
    }

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

    # Build the mode
    x = data

    if args.model == "conv-2":
        x = F.relu(x @ conv_layer(channels=2)).simplify()
        x = F.relu(x @ conv_layer(channels=3)).simplify()
        c2, h2, w2 = symbols("c2 h2 w2")
        layers.append(linear := Variable("lin", c2, h2, w2, out))
        logits = x @ linear

    elif args.model == "conv-1":
        x = F.relu(x @ conv_layer(channels=2)).simplify()
        c1, h1, w1 = symbols("c1 h1 w1")
        layers.append(linear := Variable("lin", c1, h1, w1, out))
        logits = x @ linear

    elif args.model == "linear-2":
        shapes[mid := symbols("mid")] = 40
        layers.append(linear1 := Variable("lin1", c0, h0, w0, mid))
        layers.append(linear2 := Variable("lin2", mid, out))
        x = F.relu(x @ linear1)
        logits = x @ linear2

    elif args.model == "linear-1":
        layers.append(linear := Variable("lin", c0, w0, h0, out))
        logits = x @ linear

    # y = F.cross_entropy(logits, targets, dim='out')
    logits = logits.full_simplify(expand=False)
    y = F.mean((logits - targets) ** 2, dim="out")
    y = F.mean(y, dim="batch")
    # y = y.full_simplify()
    prediction = F.argmax(logits, dim="out")

    print("Computing and simplifying gradients")
    grad_tensors = [y.grad(param).full_simplify(expand=False) for param in layers]

    compile_func = compile_numpy if args.backend == "numpy" else compile_torch
    backprop = compile_func(prediction, y, *grad_tensors, verbose=True, torch_compile=True)

    # Train
    print("Training...")
    parameters = rand_values(layers, shapes)
    parameters = {s: t / sum(t.shape) ** 0.5 for s, t in parameters.items()}
    for _ in range(n_epochs):
        total_loss = 0
        corr = 0
        batches = 0
        for t_data, t_target in tqdm.tqdm(train_loader):
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

        print(f"Loss: {total_loss/batches}")
        print(f"Acc: {corr/batches}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, default="linear-1",
        choices=["conv-1", "conv-2", "linear-1", "linear-2"]
    )
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "numpy"])
    args = parser.parse_args()
    main(args)
