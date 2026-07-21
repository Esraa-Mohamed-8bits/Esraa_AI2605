"""
IEEE AI Team - Level 2: PyTorch & Neural Networks
Regression library built from scratch on top of PyTorch tensors and autograd.

No sklearn.linear_model, no torch.nn.Linear, no torch.optim.SGD.
All forward passes, losses, and gradient-descent updates are written by hand.

Author: Esraa
"""

from abc import ABC, abstractmethod
import torch


class Model(ABC):
    """
    Abstract base class shared by every regression algorithm in this library.

    Subclasses must implement `_compute_loss`, which lets each algorithm
    plug in its own regularization term while reusing everything else
    (parameter init, forward pass, training loop, prediction, metrics).
    """

    def __init__(self, lr: float = 0.01, epochs: int = 1000, verbose: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

        self.weights = None          # torch tensor, shape (n_features,)
        self.bias = None              # torch tensor, shape (1,)
        self.loss_history = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Hooks that subclasses may override
    # ------------------------------------------------------------------
    def _transform_features(self, X: torch.Tensor) -> torch.Tensor:
        """Identity by default. PolynomialRegression overrides this to
        expand the feature space before training/prediction."""
        return X

    @abstractmethod
    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Every subclass defines its own loss (plain MSE, MSE + L1, MSE + L2, ...)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(data) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(dtype=torch.float32)
        return torch.tensor(data, dtype=torch.float32)

    def _initialize_parameters(self, n_features: int):
        """Manual parameter initialization (zeros) with autograd tracking enabled."""
        self.weights = torch.zeros(n_features, dtype=torch.float32, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        """Linear forward pass: y_hat = X @ w + b."""
        return X @ self.weights + self.bias

    def fit(self, X, y):
        """
        Train the model with manual batch Gradient Descent using PyTorch autograd.

        X: array-like, shape (n_samples, n_features) or (n_samples,)
        y: array-like, shape (n_samples,)
        """
        X_t = self._to_tensor(X)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(1)
        y_t = self._to_tensor(y).view(-1)

        X_t = self._transform_features(X_t)
        self._initialize_parameters(X_t.shape[1])
        self.loss_history = []

        for epoch in range(self.epochs):
            # ---- forward propagation ----
            y_pred = self._forward(X_t)

            # ---- loss computation ----
            loss = self._compute_loss(y_pred, y_t)

            # ---- automatic differentiation ----
            loss.backward()

            # ---- manual gradient descent weight update ----
            with torch.no_grad():
                self.weights -= self.lr * self.weights.grad
                self.bias -= self.lr * self.bias.grad
                self.weights.grad.zero_()
                self.bias.grad.zero_()

            self.loss_history.append(loss.item())

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                print(f"[{self.__class__.__name__}] epoch {epoch:5d}  loss={loss.item():.6f}")

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict outputs for new input samples. Returns a numpy array."""
        self._check_fitted()
        X_t = self._to_tensor(X)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(1)
        X_t = self._transform_features(X_t)
        with torch.no_grad():
            y_pred = self._forward(X_t)
        return y_pred.numpy()

    def get_weights(self):
        """Return learned weights and bias as numpy arrays."""
        self._check_fitted()
        return self.weights.detach().numpy(), self.bias.detach().numpy()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_mse(self, X, y) -> float:
        y_pred = self._to_tensor(self.predict(X))
        y_true = self._to_tensor(y).view(-1)
        return torch.mean((y_pred - y_true) ** 2).item()

    def get_rmse(self, X, y) -> float:
        return self.get_mse(X, y) ** 0.5

    def get_mae(self, X, y) -> float:
        y_pred = self._to_tensor(self.predict(X))
        y_true = self._to_tensor(y).view(-1)
        return torch.mean(torch.abs(y_pred - y_true)).item()

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fit() before this call.")


# ==========================================================================
# Derived classes
# ==========================================================================

class LinearRegression(Model):
    """Ordinary least squares, trained with manual Gradient Descent."""

    def _compute_loss(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)


class PolynomialRegression(Model):
    """
    Polynomial Regression: expands each input feature into powers
    [x, x^2, ..., x^degree] before running ordinary linear-in-parameters
    gradient descent on the expanded feature matrix.
    """

    def __init__(self, degree: int = 2, lr: float = 0.01, epochs: int = 1000, verbose: bool = False):
        super().__init__(lr=lr, epochs=epochs, verbose=verbose)
        self.degree = degree

    def _transform_features(self, X: torch.Tensor) -> torch.Tensor:
        # X: (n_samples, n_features) -> concatenate powers 1..degree per feature
        powers = [X ** d for d in range(1, self.degree + 1)]
        return torch.cat(powers, dim=1)

    def _compute_loss(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)


class RidgeRegression(Model):
    """Linear Regression with L2 regularization (Ridge)."""

    def __init__(self, alpha: float = 0.1, lr: float = 0.01, epochs: int = 1000, verbose: bool = False):
        super().__init__(lr=lr, epochs=epochs, verbose=verbose)
        self.alpha = alpha

    def _compute_loss(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2)
        l2 = self.alpha * torch.sum(self.weights ** 2)
        return mse + l2


class LassoRegression(Model):
    """Linear Regression with L1 regularization (Lasso)."""

    def __init__(self, alpha: float = 0.1, lr: float = 0.01, epochs: int = 1000, verbose: bool = False):
        super().__init__(lr=lr, epochs=epochs, verbose=verbose)
        self.alpha = alpha

    def _compute_loss(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2)
        l1 = self.alpha * torch.sum(torch.abs(self.weights))
        return mse + l1


class ElasticNetRegression(Model):
    """Linear Regression with combined L1 + L2 regularization (Elastic Net)."""

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5,
                 lr: float = 0.01, epochs: int = 1000, verbose: bool = False):
        super().__init__(lr=lr, epochs=epochs, verbose=verbose)
        self.alpha = alpha
        self.l1_ratio = l1_ratio  # 0 -> pure Ridge, 1 -> pure Lasso

    def _compute_loss(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2)
        l1 = torch.sum(torch.abs(self.weights))
        l2 = torch.sum(self.weights ** 2)
        penalty = self.alpha * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)
        return mse + penalty
