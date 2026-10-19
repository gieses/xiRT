"""Check standalone prediction and fine-tuning archives without running training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xirt.data_pipelines.benchmarks.common import identity, metrics, verify_archive


def verify_predictions(directory: Path) -> int:
    """Recompute every indexed metric from aligned, finite prediction rows."""
    evidence = verify_archive(directory)
    for name in (
        "tables/predictions.csv",
        "tables/metrics.csv",
        "data/manifest.json",
        "data/pairs.csv",
    ):
        if name not in evidence["files"]:
            raise ValueError(f"Missing required prediction artifact: {name}")
    predictions = pd.read_csv(directory / "tables/predictions.csv")
    scores = pd.read_csv(directory / "tables/metrics.csv")
    if scores.duplicated(["model", "seed", "split"]).any():
        raise ValueError("Duplicate metric identities.")
    manifest = json.loads((directory / "data/manifest.json").read_text())
    pairs = pd.read_csv(directory / "data/pairs.csv")
    expected_keys = {
        (model, seed, split)
        for model in ("legacy_tf", "pytorch")
        for seed in evidence["seeds"]
        for split in manifest["splits"]
    }
    actual_keys = set(scores[["model", "seed", "split"]].itertuples(index=False, name=None))
    grouped = predictions.groupby(["model", "seed", "split"], sort=False)
    if actual_keys != expected_keys or set(grouped.groups) != expected_keys:
        raise ValueError("Prediction/metric coverage differs from the archived protocol.")
    for (model, seed, split), rows in grouped:
        if rows["pair_index"].tolist() != manifest["splits"][split]:
            raise ValueError("Prediction order differs from its partition.")
        if not np.allclose(
            rows["truth_seconds"],
            pairs.iloc[manifest["splits"][split]]["rp_seconds"],
            rtol=1e-7,
            atol=1e-3,
        ):
            raise ValueError("Prediction truth differs from archived observations.")
        score = scores.loc[
            (scores["model"] == model) & (scores["seed"] == seed) & (scores["split"] == split)
        ].iloc[0]
        for key, value in metrics(rows["truth_seconds"], rows["prediction_seconds"]).items():
            if not (pd.isna(score[key]) if value is None else np.isclose(score[key], value)):
                raise ValueError(f"Recomputed {key} differs: {model}/{seed}/{split}")
    for seed in evidence["seeds"]:
        for model in ("pytorch", "pretrain"):
            name = f"models/{model}/s{seed}/model.pt"
            if name not in evidence["files"]:
                raise ValueError(f"Missing native checkpoint: {name}")
    return len(scores)


def verify_fine_tuning(directory: Path) -> int:
    """Verify run coverage, common protocol identity and recomputed per-run scores."""
    evidence = verify_archive(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    if identity(protocol) != evidence["protocol_sha256"]:
        raise ValueError("Fine-tuning protocol identity changed.")
    expected = {
        (dataset, strategy, seed)
        for dataset in protocol["datasets"]
        for strategy in protocol["strategies"]
        for seed in protocol["seeds"]
    }
    observed = set()
    scores = []
    for path in directory.glob("runs/*/*/*/record.json"):
        run = verify_archive(path.parent)
        record = json.loads(path.read_text())
        required = {"record.json", "model.pt", "predictions.csv", "history.csv"}
        if not required <= run["files"].keys():
            raise ValueError("Missing required fine-tuning run artifact.")
        key = record["dataset"], record["strategy"], record["seed"]
        dataset, strategy, seed = key
        if (
            key in observed
            or key not in expected
            or path != (directory / "runs" / dataset / strategy / f"s{seed}" / "record.json")
        ):
            raise ValueError("Duplicate or unexpected fine-tuning run.")
        observed.add(key)
        if (
            run["protocol_sha256"] != evidence["protocol_sha256"]
            or record["protocol_sha256"] != evidence["protocol_sha256"]
            or record["status"] != "ok"
            or record["frozen_parameters_unchanged"] is not True
        ):
            raise ValueError("Mixed fine-tuning protocol.")
        rows = pd.read_csv(path.parent / "predictions.csv")
        if not (
            rows["dataset"].eq(dataset).all()
            and rows["model"].eq(strategy).all()
            and rows["seed"].eq(seed).all()
        ):
            raise ValueError("Fine-tuning prediction identity differs from its run.")
        if set(rows["split"]) != set(record["quality"]):
            raise ValueError("Fine-tuning prediction split coverage changed.")
        for split, group in rows.groupby("split"):
            quality = metrics(group["truth_seconds"], group["prediction_seconds"])
            for name, value in quality.items():
                saved = record["quality"][split][name]
                # Historical CSVs round float32 predictions; records keep in-memory metrics.
                if not (
                    saved is None
                    if value is None
                    else saved is not None and np.isclose(saved, value, rtol=1e-6, atol=1e-4)
                ):
                    raise ValueError(f"Fine-tuning {name} differs from predictions.")
            scores.append(
                {
                    "dataset": dataset,
                    "model": strategy,
                    "seed": seed,
                    "split": split,
                    **record["quality"][split],
                    "trainable_parameters": record["trainable_parameters"],
                    "training_seconds": record["training_seconds"],
                }
            )
    if observed != expected or len(observed) != evidence["runs_verified"]:
        raise ValueError("Incomplete fine-tuning run coverage.")
    expected_metrics = pd.DataFrame(scores).astype({"r2": float})
    keys = ["dataset", "model", "seed", "split"]
    actual_metrics = pd.read_csv(directory / "results/metrics.csv")
    pd.testing.assert_frame_equal(
        expected_metrics.sort_values(keys).reset_index(drop=True),
        actual_metrics[expected_metrics.columns].sort_values(keys).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-8,
        atol=1e-8,
    )
    summary = (
        expected_metrics.loc[expected_metrics["split"] == "test"]
        .groupby(["dataset", "model"])
        .agg(
            rmse_mean=("rmse_seconds", "mean"),
            rmse_seed_sd=("rmse_seconds", "std"),
            mae_mean=("mae_seconds", "mean"),
            r2_mean=("r2", "mean"),
            trainable_parameters=("trainable_parameters", "first"),
            training_seconds_mean=("training_seconds", "mean"),
        )
    )
    pd.testing.assert_frame_equal(
        summary.sort_index(),
        pd.read_csv(directory / "results/test-summary.csv")
        .set_index(["dataset", "model"])
        .sort_index(),
        check_dtype=False,
        rtol=1e-8,
        atol=1e-8,
    )
    return len(observed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=Path("benchmark-tf-torch"))
    args = parser.parse_args()
    cohorts = sorted(args.archive.glob("*/results/archive.json"))
    if not cohorts:
        parser.error("No archived TF/PyTorch cohorts found.")
    for path in cohorts:
        print(f"{path.parent}: {verify_predictions(path.parent)} metric records verified")
    fine_tuning = args.archive / "fine-tune-torch"
    if (fine_tuning / "archive.json").exists():
        print(f"{fine_tuning}: {verify_fine_tuning(fine_tuning)} runs verified")


if __name__ == "__main__":
    main()
