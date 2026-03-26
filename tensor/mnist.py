import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

train_data = datasets.MNIST("data", train=True, download=True)
test_data = datasets.MNIST("data", train=False)

X_train = train_data.data.float().reshape(-1, 784) / 255.0
y_train = train_data.targets
X_test = test_data.data.float().reshape(-1, 784) / 255.0
y_test = test_data.targets

np.save("data/X_train.npy", X_train.numpy())
np.save("data/y_train.npy", y_train.float().numpy())
np.save("data/X_test.npy", X_test.numpy())
np.save("data/y_test.npy", y_test.float().numpy())

W1 = (torch.randn(784, 128) * (2 / 784) ** 0.5).requires_grad_(True)
b1 = torch.zeros(1, 128, requires_grad=True)
W2 = (torch.randn(128, 10) * (1 / 128) ** 0.5).requires_grad_(True)
b2 = torch.zeros(1, 10, requires_grad=True)

vW1 = torch.zeros_like(W1)
vb1 = torch.zeros_like(b1)
vW2 = torch.zeros_like(W2)
vb2 = torch.zeros_like(b2)

lr = 0.1
batch_size = 128
momentum = 0.9
epochs = 10


def forward(X):
    return torch.relu(X @ W1 + b1) @ W2 + b2


for epoch in range(epochs):
    perm = torch.randperm(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]

    total_loss = 0.0
    n_batches = 0

    for i in range(0, X_train.shape[0], batch_size):
        loss = F.cross_entropy(forward(X_train[i : i + batch_size]), y_train[i : i + batch_size])
        loss.backward()

        with torch.no_grad():
            for p, v in zip([W1, b1, W2, b2], [vW1, vb1, vW2, vb2]):
                if p.grad is not None:
                    v.mul_(momentum).add_(p.grad)
                    p.sub_(lr * v)
                    p.grad.zero_()

        total_loss += loss.item()
        n_batches += 1

    with torch.no_grad():
        acc = (forward(X_test).argmax(dim=1) == y_test).float().mean().item()
        print(
            f"Epoch {epoch + 1:2d}  Loss: {total_loss / n_batches:.4f}  Accuracy: {100 * acc:.1f}%"
        )
