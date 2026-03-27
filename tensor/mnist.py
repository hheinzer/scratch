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

Wc1 = (torch.randn(16, 1, 3, 3) * (2 / 9) ** 0.5).requires_grad_(True)
bc1 = torch.zeros(16, requires_grad=True)
Wc2 = (torch.randn(32, 16, 3, 3) * (2 / 144) ** 0.5).requires_grad_(True)
bc2 = torch.zeros(32, requires_grad=True)
W = (torch.randn(1568, 10) * (1 / 1568) ** 0.5).requires_grad_(True)
b = torch.zeros(1, 10, requires_grad=True)

params = [Wc1, bc1, Wc2, bc2, W, b]
vels = [torch.zeros_like(p) for p in params]

lr = 0.1
batch_size = 128
momentum = 0.9
epochs = 10


def forward(X):
    x = torch.relu(F.conv2d(X.reshape(-1, 1, 28, 28), Wc1, bc1, stride=2, padding=1))
    x = torch.relu(F.conv2d(x, Wc2, bc2, stride=2, padding=1))
    return x.flatten(1) @ W + b


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
            for p, v in zip(params, vels):
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
