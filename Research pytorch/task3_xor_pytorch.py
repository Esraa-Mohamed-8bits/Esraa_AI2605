"""
IEEE AI Team - Level 2, Task 3: Build Your First Neural Network using PyTorch
-------------------------------------------------------------------------------
Solves the 4-input XOR problem (4D XOR): output = 1 if an ODD number of the
4 binary inputs are 1, else 0 (i.e. parity / XOR across a,b,c,d).

This matches the truth table given in the assignment (16 rows, a,b,c,d,Out).

NOTE: This script requires PyTorch (`pip install torch`). It could not be
executed in this sandbox because it has no network access to install torch,
but the code follows a standard, tested PyTorch training pattern.
"""

import itertools
import torch
import torch.nn as nn


def build_dataset():
    """All 16 combinations of 4 binary inputs; label = parity (XOR) of the 4 bits."""
    X, y = [], []
    for bits in itertools.product([0, 1], repeat=4):
        X.append(bits)
        y.append(sum(bits) % 2)  # 1 if odd number of 1s, else 0
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y


class XORNet(nn.Module):
    """Simple feedforward network: 4 -> 8 -> 8 -> 1, with ReLU hidden
    activations and a sigmoid output (binary classification)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)

    X, y = build_dataset()

    model = XORNet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    epochs = 3000
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            preds = (outputs.detach() > 0.5).float()
            acc = (preds == y).float().mean().item()
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Accuracy: {acc*100:.2f}%")

    # Final evaluation on all 16 input combinations
    print("\nFinal predictions vs expected output:")
    print(f"{'a':>2} {'b':>2} {'c':>2} {'d':>2} | {'Predicted':>9} | {'Expected':>8}")
    with torch.no_grad():
        final_outputs = model(X)
        final_preds = (final_outputs > 0.5).float()
        for i in range(len(X)):
            a, b, c, d = X[i].int().tolist()
            pred = int(final_preds[i].item())
            expected = int(y[i].item())
            print(f"{a:>2} {b:>2} {c:>2} {d:>2} | {pred:>9} | {expected:>8}")

        final_acc = (final_preds == y).float().mean().item()
        print(f"\nOverall accuracy on all 16 XOR combinations: {final_acc*100:.2f}%")


if __name__ == "__main__":
    main()
