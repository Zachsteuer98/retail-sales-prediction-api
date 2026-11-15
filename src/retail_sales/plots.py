"""Report figures written by the training pipeline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np

FIGURE_DPI = 150
PRIMARY = "#2b5c8f"
HIGHLIGHT = "#e67e22"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_cv_comparison(results: dict[str, dict], selected: str, path: Path) -> None:
    names = sorted(results, key=lambda n: results[n]["cv_rmse"], reverse=True)
    rmse = [results[n]["cv_rmse"] for n in names]
    std = [results[n]["cv_rmse_std"] for n in names]
    colors = [HIGHLIGHT if n == selected else PRIMARY for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, rmse, xerr=std, color=colors, capsize=3)
    ax.set(xlabel="Cross-validated RMSE (lower is better)", title="Model selection on the training split")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    _save(fig, path)


def save_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4, color=PRIMARY)
    bounds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(bounds, bounds, color=HIGHLIGHT, linestyle="--", label="Perfect prediction")
    ax.set(xlabel="Actual sales", ylabel="Predicted sales", title="Test set: predicted vs actual")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, path)


def save_importance(importances: dict[str, float], path: Path) -> None:
    names = list(importances)[::-1]
    values = [importances[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, values, color=PRIMARY)
    ax.set(
        xlabel="Increase in RMSE when the feature is shuffled",
        title="Permutation importance (test set)",
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    _save(fig, path)
