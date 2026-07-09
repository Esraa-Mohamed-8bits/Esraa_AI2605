"""
IEEE AI Team - Level 2, Task 1: Fact behind Softmax
----------------------------------------------------
Shows the effect of the temperature parameter T on the softmax function.

softmax(z_i) = exp(z_i / T) / sum_j( exp(z_j / T) )

- Low T  (e.g. 0.001) -> softmax becomes very "peaky" / close to argmax (one-hot).
- T = 1                -> standard softmax.
- High T (e.g. 500)    -> softmax becomes very "flat" / close to a uniform distribution.
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(z, T=1.0):
    """
    Numerically stable softmax with temperature.

    Parameters
    ----------
    z : np.ndarray
        Vector of logits.
    T : float
        Temperature. T -> 0 sharpens the distribution, T -> inf flattens it.
    """
    z = np.asarray(z, dtype=np.float64)
    scaled = z / T
    # subtract max for numerical stability (does not change the result)
    scaled = scaled - np.max(scaled)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled)


def main():
    # Any vector of logits works here.
    z = np.array([2.0, 1.0, 0.1, 3.5, -1.0])
    labels = [f"z{i}" for i in range(len(z))]

    temperatures = [0.001, 0.01, 0.1, 0.3, 0.5, 0.9, 1, 1.5, 2, 10, 100, 500]

    n = len(temperatures)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    for ax, T in zip(axes, temperatures):
        probs = softmax(z, T=T)
        ax.bar(labels, probs, color="tab:orange")
        ax.set_title(f"T = {T}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("probability")

    # hide any unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Effect of Temperature on Softmax  (logits z = {z.tolist()})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("softmax_temperature_effect.png", dpi=150)
    print("Saved plot to softmax_temperature_effect.png")

    # print numeric results too
    for T in temperatures:
        probs = softmax(z, T=T)
        print(f"T={T:>7}: {np.round(probs, 4)}")


if __name__ == "__main__":
    main()
