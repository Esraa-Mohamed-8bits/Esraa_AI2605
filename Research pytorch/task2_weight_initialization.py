"""
IEEE AI Team - Level 2, Task 2: Smart Initialization
------------------------------------------------------
Implements and compares four weight-initialization strategies for a
weight matrix W of shape (fan_in, fan_out):

    1. Zero Initialization
    2. Random Initialization      (small Uniform / Normal)
    3. Xavier (Glorot) Initialization
    4. He (Kaiming) Initialization

See task2_writeup.md for the conceptual explanation (Xavier vs He:
when to use each, advantages/disadvantages).
"""

import numpy as np
import matplotlib.pyplot as plt


def zero_init(fan_in, fan_out):
    """All weights = 0. Causes the symmetry problem: every neuron in a
    layer gets an identical gradient and stays identical forever."""
    return np.zeros((fan_in, fan_out))


def random_init(fan_in, fan_out, scale=0.01, distribution="uniform"):
    """Small random weights. Breaks symmetry but the fixed scale (e.g. 0.01)
    ignores layer size, so variance can shrink/explode as depth grows."""
    if distribution == "uniform":
        return np.random.uniform(-scale, scale, size=(fan_in, fan_out))
    return np.random.normal(0, scale, size=(fan_in, fan_out))


def xavier_init(fan_in, fan_out, distribution="uniform"):
    """Xavier/Glorot initialization.
    Keeps the variance of activations (and gradients) roughly constant
    across layers when using symmetric activations like tanh or sigmoid.

    Uniform:  W ~ U(-limit, limit),  limit = sqrt(6 / (fan_in + fan_out))
    Normal:   W ~ N(0, std^2),       std   = sqrt(2 / (fan_in + fan_out))
    """
    if distribution == "uniform":
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_in, fan_out))
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, std, size=(fan_in, fan_out))


def he_init(fan_in, fan_out, distribution="normal"):
    """He/Kaiming initialization.
    Designed for ReLU-family activations, which zero out roughly half
    of their inputs, so it uses fan_in only and a larger variance to
    compensate for that lost signal.

    Normal:   W ~ N(0, std^2),      std   = sqrt(2 / fan_in)
    Uniform:  W ~ U(-limit, limit), limit = sqrt(6 / fan_in)
    """
    if distribution == "normal":
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, size=(fan_in, fan_out))
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


def main():
    np.random.seed(42)
    fan_in, fan_out = 256, 256  # e.g. a hidden layer of a deep MLP

    inits = {
        "Zero": zero_init(fan_in, fan_out),
        "Random (U[-0.01,0.01])": random_init(fan_in, fan_out, scale=0.01),
        "Xavier (uniform)": xavier_init(fan_in, fan_out, "uniform"),
        "He (normal)": he_init(fan_in, fan_out, "normal"),
    }

    fig, axes = plt.subplots(1, len(inits), figsize=(4 * len(inits), 3.5))
    for ax, (name, W) in zip(axes, inits.items()):
        ax.hist(W.flatten(), bins=50, color="tab:blue")
        ax.set_title(f"{name}\nvar={W.var():.6f}")
        ax.set_xlabel("weight value")
    fig.suptitle(f"Weight distributions for fan_in={fan_in}, fan_out={fan_out}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("weight_initialization_comparison.png", dpi=150)
    print("Saved plot to weight_initialization_comparison.png")

    for name, W in inits.items():
        print(f"{name:30s} mean={W.mean(): .6f}  var={W.var(): .6f}")


if __name__ == "__main__":
    main()
