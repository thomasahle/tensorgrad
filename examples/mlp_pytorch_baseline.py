"""Naive PyTorch baseline for the Wine dataset to compare against tensorgrad."""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_wine


def load_wine_tensor():
    wine = load_wine()
    X = torch.tensor(wine.data, dtype=torch.float)
    y = torch.tensor(wine.target, dtype=torch.long)

    # Standardize features for smoother optimization
    X = (X - X.mean(dim=0)) / X.std(dim=0)
    return X, y


class FullBatchMLP(nn.Module):
    """Match tensorgrad architecture: full-batch bias parameters per example."""

    def __init__(self, batch_size: int, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(batch_size, hidden))
        self.W2 = nn.Parameter(torch.empty(hidden, out_dim))
        self.b2 = nn.Parameter(torch.zeros(batch_size, out_dim))

        nn.init.kaiming_normal_(self.W1, nonlinearity="relu")
        nn.init.zeros_(self.b1)
        nn.init.kaiming_normal_(self.W2, nonlinearity="relu")
        nn.init.zeros_(self.b2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = F.relu(X @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="Naive PyTorch MLP on Wine (full-batch to match tensorgrad example)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate (default: 1e-2)")
    parser.add_argument("--hidden", type=int, default=32, help="Hidden width (default: 32)")
    args = parser.parse_args()

    torch.manual_seed(42)

    X, y = load_wine_tensor()
    in_dim = X.shape[1]
    out_dim = int(y.max().item() + 1)

    model = FullBatchMLP(batch_size=X.shape[0], in_dim=in_dim, hidden=args.hidden, out_dim=out_dim)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        elapsed = time.time() - start
        eps = (epoch + 1) / elapsed

        model.eval()
        acc = accuracy(model, X, y)
        print(
            f"Epoch {epoch:3d} | Loss: {loss.item():8.4f} | Acc: {acc*100:5.1f}% | Epochs/s: {eps:5.2f}"
        )


if __name__ == "__main__":
    main()
