from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sympy import symbols
import time
import tqdm
import torch

from tensorgrad import Variable
from tensorgrad.testutils import rand_values
from tensorgrad.serializers.to_pytorch import compile_to_callable
import tensorgrad.functions as F


def main():
    n_epochs = 10
    batch_size = 32
    lr = 1e-2

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    args = dict(root='./data', download=True, transform=transform)
    train_dataset = datasets.MNIST(train=True, **args)
    test_dataset = datasets.MNIST(train=False, **args)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    batch, c0, w0, h0, out, kernel_size = symbols(f"batch c0 w0 h0 out ks")
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
    def conv_layer(i, in_channels, out_channels):
        c_in, c_out = symbols(f"c{i} c{i+1}")

        h_in, h_out, w_in, w_out = symbols(f"h{i} h{i+1}, w{i} w{i+1}")
        h_conv = F.Convolution(h_in, h_out, hk=kernel_size)
        w_conv = F.Convolution(w_in, w_out, wk=kernel_size)

        kernel = Variable(f"kernel_{i}", c_in, c_out, hk=kernel_size, wk=kernel_size)
        layers.append(kernel)

        shapes[c_in] = in_channels
        shapes[c_out] = out_channels
        shapes[h_out] = shapes[h_in] - shapes[kernel_size] + 1
        shapes[w_out] = shapes[w_in] - shapes[kernel_size] + 1

        return kernel @ h_conv @ w_conv


    x = data

    if False:
        x = F.relu(x @ conv_layer(0, 1, 2)).simplify()
        x = F.relu(x @ conv_layer(1, 2, 3)).simplify()
        c2, h2, w2, c3 = symbols("c2 h2 w2 c3")
        shapes[c3] = 3 * 24**2  # c2*w2*h2
        layers.append(linear := Variable("lin", c2, h2, w2, out))
        logits = x @ linear

    elif False:
        x = F.relu(x @ conv_layer(0, 1, 2)).simplify()
        c1, h1, w1, c2 = symbols("c1 h1 w1 c2")
        shapes[c2] = 2 * 26**2  # c1*w1*h1
        layers.append(linear := Variable("lin", c1, h1, w1, out))
        logits = x @ linear

    elif False:
        shapes[mid := symbols("mid")] = 40
        layers.append(linear1 := Variable("lin1", c0, h0, w0, mid))
        layers.append(linear2 := Variable("lin2", mid, out))
        x = F.relu(x @ linear1)
        logits = x @ linear2

    else:
        layers.append(linear := Variable("lin", c0, w0, h0, out))
        logits = x @ linear

    #y = F.cross_entropy(logits, targets, dim='out')
    y = F.mean((logits - targets)**2, dim='out')
    y = F.mean(y, dim='batch')
    y = y.full_simplify()
    prediction = F.argmax(logits, dim='out')

    # parameters = rand_values(kernels + [linear, data, targets], shapes)
    # clb = compile_to_callable(y, verbose=False, torch_compile=False)
    # clb_fast = compile_to_callable(y, verbose=False, torch_compile=True)

    # print("Time normal callback")
    # print(clb(parameters, shapes))
    # start = time.time()
    # for _ in range(10):
    #     _ = clb(parameters, shapes)
    # print(time.time() - start)

    # print("Time compiled callback")
    # print(clb_fast(parameters, shapes))
    # start = time.time()
    # for _ in range(10):
    #     _ = clb_fast(parameters, shapes)
    # print(time.time() - start)

    # print("Time evaluate")
    # print(y.evaluate(parameters, shapes))
    # start = time.time()
    # for _ in range(10):
    #     _ = y.evaluate(parameters, shapes)
    # print(time.time() - start)

    # return

    print("Computing and simplifying gradients")
    grad_tensors = [y.grad(param).full_simplify() for param in layers]

    backprop = compile_to_callable(prediction, y, *grad_tensors, verbose=True, torch_compile=False)

    # Train
    print("Training...")
    parameters = rand_values(layers, shapes)
    parameters = {s: t/sum(t.shape)**.5 for s, t in parameters.items()}
    for epoch in range(n_epochs):
        total_loss = 0
        corr = 0
        batches = 0
        for t_data, t_target in tqdm.tqdm(train_loader):
            shapes[batch] = t_data.shape[0]
            input_params = {t: p.clone() for t, p in parameters.items()}
            input_params[data] = t_data.rename("batch", "c0", "w0", "h0")
            #print(f"{data.edges=}")
            input_params[targets] = torch.eye(10)[t_target].rename("batch", "out")
            #print(f"{targets.edges=}")
            #for ip, par in input_params.items():
            #    print(ip.shape, par.names)

            outputs = backprop(input_params, shapes)
            #outputs = {gr: gr.evaluate(input_params, shapes) for gr in grad_tensors}
            #outputs[y] = y.evaluate(input_params, shapes)
            #outputs[prediction] = prediction.evaluate(input_params, shapes)

            # Grad update
            for layer, grad_layer in zip(layers, grad_tensors):
                g = outputs[grad_layer].align_to(*parameters[layer].names)
                parameters[layer] -= lr * g

            # Forward pass
            total_loss += outputs[y] / shapes[batch]
            corr += (outputs[prediction] == t_target).sum() / shapes[batch]
            batches += 1

        print(f"Loss: {total_loss/batches}")
        print(f"Acc: {corr/batches}")

if __name__ == "__main__":
    main()

