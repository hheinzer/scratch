import torch

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

W1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.zeros(1, 4, requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1, 1, requires_grad=True)


def forward(X):
    a1 = torch.tanh(X @ W1 + b1)
    a2 = torch.sigmoid(a1 @ W2 + b2)
    return a2


lr = 0.1
for epoch in range(5000):
    pred = forward(X)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    with torch.no_grad():
        for p in [W1, b1, W2, b2]:
            grad = p.grad
            if grad is not None:
                p -= lr * grad
                grad.zero_()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1:4d}  Loss: {loss.item():.4f}")

with torch.no_grad():
    pred = forward(X)
    print("\nResults:")
    for i in range(4):
        print(f"  {int(X[i][0])} XOR {int(X[i][1])} = {pred[i].item():.4f}")
