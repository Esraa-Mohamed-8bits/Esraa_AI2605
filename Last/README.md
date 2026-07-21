# IEEE AI Team – Level 2: PyTorch Regression Library

A regression library built from scratch on PyTorch tensors and autograd —
no `sklearn.linear_model`, no `torch.nn.Linear`, no `torch.optim.SGD`.

## Files

- `regression_models.py` — the library: `Model` base class + 5 regressors
- `demo.py` — synthetic-data demo that trains every model and plots results

## Design

`Model` (abstract base class) holds everything shared across algorithms:

| Method | Purpose |
|---|---|
| `fit(X, y)` | Manual batch Gradient Descent using `loss.backward()` |
| `predict(X)` | Forward pass with learned parameters |
| `get_weights()` | Returns `(weights, bias)` as numpy arrays |
| `get_mse/get_rmse/get_mae(X, y)` | Error metrics |

Each subclass only overrides `_compute_loss` (to add its regularization term)
and, for `PolynomialRegression`, `_transform_features` (to expand `X` into
powers before the linear forward pass).

```
Model (ABC)
├── LinearRegression        loss = MSE
├── PolynomialRegression    loss = MSE, features expanded to x, x², ..., x^d
├── RidgeRegression         loss = MSE + α · Σw²
├── LassoRegression         loss = MSE + α · Σ|w|
└── ElasticNetRegression    loss = MSE + α·(l1_ratio·Σ|w| + (1-l1_ratio)·Σw²)
```

Training loop (identical for every model, lives in the base class):

```python
y_pred = self._forward(X)          # forward propagation
loss = self._compute_loss(y_pred, y)
loss.backward()                    # autograd
with torch.no_grad():
    self.weights -= lr * self.weights.grad
    self.bias   -= lr * self.bias.grad
    self.weights.grad.zero_()
    self.bias.grad.zero_()
```

## Usage

```python
from regression_models import LinearRegression, RidgeRegression

model = LinearRegression(lr=0.05, epochs=500)
model.fit(X_train, y_train)

preds = model.predict(X_test)
w, b = model.get_weights()
print(model.get_mse(X_test, y_test), model.get_rmse(X_test, y_test), model.get_mae(X_test, y_test))
```

Run the demo:

```bash
pip install torch numpy matplotlib
python demo.py
```

## Notes / things you may be asked about

- **Polynomial features**: this implementation expands each feature into
  `x, x², ..., x^degree` independently (no cross-terms between different
  input columns). That's enough for single/few-feature curve fitting;
  if you need interaction terms across many features, extend
  `_transform_features` with `itertools.combinations_with_replacement`.
- **Normalize inputs before high-degree polynomial fits** — large `x^3`,
  `x^4` values otherwise blow up the gradients. `demo.py` does this.
- **Learning rate & regularization strength (`alpha`)** are the two knobs
  you'll likely need to tune per dataset — start small (`lr=0.01–0.05`)
  and watch `loss_history` for divergence vs. plateauing.
