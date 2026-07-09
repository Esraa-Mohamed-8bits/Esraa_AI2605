# Task 2: Smart Initialization — Xavier & He

## 1. Why initialization matters
As shown in the slides, **zero initialization** causes the *symmetry problem*: every
neuron in a layer starts identical, produces identical outputs, and receives identical
gradients, so it never differentiates. **Small random initialization** breaks symmetry,
but a fixed scale (e.g. `U(-0.01, 0.01)`) ignores the size of the layer. In a deep
network this causes the variance of activations to shrink or grow uncontrollably as
signals pass through many layers — leading to vanishing or exploding gradients.

Xavier and He initialization fix this by scaling the random weights according to the
number of input/output connections of the layer, so that the variance of activations
stays roughly the same from layer to layer.

## 2. Xavier (Glorot) Initialization
Proposed by Glorot & Bengio (2010) for networks using **symmetric, saturating
activations** (tanh, sigmoid).

- Normal: `W ~ N(0, 2 / (fan_in + fan_out))`
- Uniform: `W ~ U(-limit, limit)`, `limit = sqrt(6 / (fan_in + fan_out))`

It balances variance using *both* fan_in and fan_out, keeping the forward-pass signal
variance and the backward-pass gradient variance stable at the same time.

**Advantages**
- Prevents vanishing/exploding signals in networks with tanh/sigmoid activations.
- Theoretically grounded (derived from variance-preservation assumptions).

**Disadvantages**
- Assumes a linear/symmetric activation around 0; the assumption breaks down for ReLU,
  because ReLU zeroes out roughly half the activations, which effectively halves the
  variance Xavier was trying to preserve. Using Xavier with deep ReLU networks tends to
  make activations shrink toward zero in later layers.

## 3. He (Kaiming) Initialization
Proposed by He et al. (2015) specifically for **ReLU and ReLU-family activations**
(ReLU, Leaky ReLU, PReLU).

- Normal: `W ~ N(0, 2 / fan_in)`
- Uniform: `W ~ U(-limit, limit)`, `limit = sqrt(6 / fan_in)`

Because ReLU zeroes out about half its inputs, He initialization only uses `fan_in`
and doubles the variance (factor of 2 instead of Xavier's implicit ~1) to compensate
for that lost signal, keeping activation variance stable through deep ReLU networks.

**Advantages**
- Keeps signal variance stable specifically in deep ReLU networks — the default choice
  for most modern CNNs/MLPs.
- Empirically enables training much deeper ReLU networks than Xavier.

**Disadvantages**
- Less appropriate for tanh/sigmoid networks, since it does not account for the
  activation's saturation behavior the way Xavier does.
- Like Xavier, it's a heuristic based on a simplified variance analysis — it doesn't
  account for things like batch normalization, residual connections, or dropout,
  which also affect signal propagation in practice.

## 4. When to use which
| Activation function            | Recommended init |
|---------------------------------|-------------------|
| Sigmoid, Tanh                   | Xavier/Glorot      |
| ReLU, Leaky ReLU, ELU, GELU     | He/Kaiming         |
| Zero / naive small random       | Avoid in deep nets |

## 5. Summary
| Method   | Formula (normal)                | Breaks symmetry | Depth-safe | Best activation |
|----------|----------------------------------|-----------------|------------|------------------|
| Zero     | `W = 0`                          | No              | No         | —                |
| Random   | `W ~ N(0, 0.01^2)`                | Yes             | No (fixed scale) | —          |
| Xavier   | `W ~ N(0, 2/(fan_in+fan_out))`    | Yes             | Yes        | tanh / sigmoid   |
| He       | `W ~ N(0, 2/fan_in)`              | Yes             | Yes        | ReLU family      |

See `task2_weight_initialization.py` for a NumPy implementation of all four methods
and a plot comparing their resulting weight distributions
(`weight_initialization_comparison.png`).
