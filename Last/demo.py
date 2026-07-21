"""
Demo / smoke test for the regression library.

Generates synthetic data, trains every model, prints metrics,
and plots loss curves + fitted lines for a quick visual sanity check.

Run:  python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from regression_models import (
    LinearRegression,
    PolynomialRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression,
)

np.random.seed(42)


def make_linear_data(n=200, noise=1.0):
    X = np.random.uniform(-5, 5, size=(n, 1))
    y = 3.0 * X[:, 0] - 2.0 + np.random.normal(0, noise, size=n)
    return X, y


def make_nonlinear_data(n=200, noise=2.0):
    X = np.random.uniform(-3, 3, size=(n, 1))
    y = 0.5 * X[:, 0] ** 3 - 2 * X[:, 0] ** 2 + X[:, 0] + np.random.normal(0, noise, size=n)
    return X, y


def train_test_split(X, y, test_ratio=0.2):
    n = len(X)
    idx = np.random.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def report(name, model, X_train, X_test, y_train, y_test):
    w, b = model.get_weights()
    print(f"\n--- {name} ---")
    print(f"weights: {np.round(w, 4)}  bias: {np.round(b, 4)}")
    print(f"train  -> MSE: {model.get_mse(X_train, y_train):.4f}  "
          f"RMSE: {model.get_rmse(X_train, y_train):.4f}  "
          f"MAE: {model.get_mae(X_train, y_train):.4f}")
    print(f"test   -> MSE: {model.get_mse(X_test, y_test):.4f}  "
          f"RMSE: {model.get_rmse(X_test, y_test):.4f}  "
          f"MAE: {model.get_mae(X_test, y_test):.4f}")


def main():
    # ---------- Linear / Ridge / Lasso / Elastic Net on linear data ----------
    X, y = make_linear_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    models = {
        "Linear Regression": LinearRegression(lr=0.05, epochs=500),
        "Ridge Regression":  RidgeRegression(alpha=0.5, lr=0.05, epochs=500),
        "Lasso Regression":  LassoRegression(alpha=0.1, lr=0.05, epochs=500),
        "Elastic Net":       ElasticNetRegression(alpha=0.1, l1_ratio=0.5, lr=0.05, epochs=500),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        report(name, model, X_train, X_test, y_train, y_test)

    # ---------- Polynomial Regression on nonlinear data ----------
    Xp, yp = make_nonlinear_data()
    # normalize inputs so higher powers don't blow up during gradient descent
    Xp_norm = (Xp - Xp.mean()) / Xp.std()
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp_norm, yp)

    poly = PolynomialRegression(degree=3, lr=0.01, epochs=2000)
    poly.fit(Xp_train, yp_train)
    report("Polynomial Regression (deg 3)", poly, Xp_train, Xp_test, yp_train, yp_test)

    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # loss curves
    for name, model in models.items():
        axes[0].plot(model.loss_history, label=name)
    axes[0].plot(poly.loss_history, label="Polynomial (deg 3)")
    axes[0].set_title("Training Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # fitted line for plain linear regression
    lin_model = models["Linear Regression"]
    order = np.argsort(X[:, 0])
    axes[1].scatter(X, y, s=10, alpha=0.4, label="data")
    axes[1].plot(X[order], lin_model.predict(X)[order], color="red", label="Linear fit")

    order_p = np.argsort(Xp[:, 0])
    axes[1].scatter(Xp, yp, s=10, alpha=0.2, label="nonlinear data")
    axes[1].plot(Xp[order_p], poly.predict(Xp_norm)[order_p], color="green", label="Polynomial fit")
    axes[1].set_title("Fitted Curves")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("demo_results.png", dpi=150)
    print("\nSaved plot to demo_results.png")


if __name__ == "__main__":
    main()
