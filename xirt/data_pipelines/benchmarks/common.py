"""Artifact integrity, prediction metrics and plots shared by benchmark pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def file_hash(path: Path) -> str:
    """Return a streaming SHA256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value: Any) -> str:
    """Hash canonical JSON, rejecting nonfinite values."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Publish complete JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    try:
        pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """Compute errors in seconds and R² from aligned finite observations."""
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not len(truth) or truth.shape != prediction.shape:
        raise ValueError("Truth and prediction must be nonempty arrays with equal shapes.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Prediction tables contain nonfinite values.")
    error = prediction - truth
    denominator = float(np.square(truth - truth.mean()).sum())
    return {
        "rmse_seconds": float(np.sqrt(np.square(error).mean())),
        "mae_seconds": float(np.abs(error).mean()),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator else None,
    }


def verify_archive(directory: Path) -> dict[str, Any]:
    """Verify every indexed artifact before treating an archive as complete."""
    manifest = json.loads((directory / "archive.json").read_text())
    base = directory.resolve()
    for name, expected in manifest["files"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base) or file_hash(path) != expected:
            raise ValueError(f"Archive artifact changed: {name}")
    return manifest


def seal_archive(directory: Path, metadata: dict[str, Any]) -> None:
    """Index published files; the manifest intentionally does not hash itself."""
    files = {
        str(path.relative_to(directory)): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "archive.json"
    }
    write_json(directory / "archive.json", {**metadata, "files": files})


def plot_predictions(predictions: pd.DataFrame, scores: pd.DataFrame, output: Path) -> None:
    """Plot held-out predictions, residuals and seed dispersion without ensembling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    test = predictions.loc[predictions["split"] == "test"]
    models = list(test["model"].unique())
    if not models:
        raise ValueError("No held-out predictions to plot.")
    figure, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for axis, model in zip(axes[0], models, strict=True):
        rows = test.loc[test["model"] == model]
        axis.scatter(rows["truth_seconds"] / 60, rows["prediction_seconds"] / 60, s=5, alpha=0.2)
        low = min(rows["truth_seconds"].min(), rows["prediction_seconds"].min()) / 60
        high = max(rows["truth_seconds"].max(), rows["prediction_seconds"].max()) / 60
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set(title=model, xlabel="Observed RT (min)", ylabel="Predicted RT (min)")
    figure.suptitle("Held-out predictions: individual runs, not an ensemble")
    figure.tight_layout()
    figure.savefig(output / "predictions.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    for model in models:
        rows = test.loc[test["model"] == model]
        axis.hist(
            (rows["prediction_seconds"] - rows["truth_seconds"]) / 60,
            bins=50,
            density=True,
            alpha=0.45,
            label=model,
        )
    axis.set(xlabel="Prediction − observed RT (min)", ylabel="Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "residuals.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(max(6, len(models) * 1.3), 4))
    held_out = scores.loc[scores["split"] == "test"]
    grouped = held_out.groupby("model", sort=False)["rmse_seconds"]
    means = grouped.mean().reindex(models)
    deviations = grouped.std().reindex(models).fillna(0)
    axis.bar(models, means, yerr=deviations, capsize=4, color="#426b9b")
    axis.set(ylabel="Test RMSE (s)", title="Mean and seed SD; not a confidence interval")
    axis.tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    figure.savefig(output / "rmse.png", dpi=160)
    plt.close(figure)
