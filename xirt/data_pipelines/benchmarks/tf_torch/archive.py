"""Seal matched historical TF/PyTorch predictions, models, metrics and figures."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from xirt.data_pipelines.benchmarks.common import (
    file_hash,
    metrics,
    plot_predictions,
    seal_archive,
    verify_archive,
)
from xirt.data_pipelines.benchmarks.dataset import BenchmarkDataset, recurrent_parameters
from xirt.data_pipelines.benchmarks.verify import verify_predictions


def convert_native(
    source: Path,
    target: Path,
    params: dict,
    data: BenchmarkDataset,
    predictions: pd.DataFrame | None,
) -> None:
    """Convert frozen state tensors to a self-describing native checkpoint without retraining."""
    import torch

    from xirt.xirtnet import RetentionModel

    model = RetentionModel(
        params,
        data.manifest["max_length"],
        len(data.manifest["alphabet"]) + 1,
        data.manifest["siamese"],
    )
    state = torch.load(source, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    if predictions is not None:
        indices = data.splits["test"]
        actual = data.prediction_table(model.predict(data.features(indices)), indices)
        if not np.allclose(
            actual["prediction_seconds"], predictions["prediction_seconds"], rtol=1e-5, atol=1e-3
        ):
            raise ValueError("Converted native checkpoint changed held-out predictions.")
    model.save(target)


def archive_cohort(run: Path, dataset: str, output: Path, replay_root: Path) -> None:
    """Copy verified predictions into a standalone, checksum-indexed cohort archive."""
    provenance = json.loads((run / "provenance.json").read_text())
    config = provenance["config"]
    destination = output / dataset / "results"
    if destination.exists():
        manifest = verify_archive(destination)
        if manifest["protocol_sha256"] != provenance["protocol_sha256"]:
            raise ValueError("Existing prediction archive belongs to another protocol.")
        verify_predictions(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".archive-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "results"
        stage.mkdir()
        source = run / "data" / dataset
        shutil.copytree(source, stage / "data")
        pairs = pd.read_csv(source / "pairs.csv")
        manifest = json.loads((source / "manifest.json").read_text())
        data = BenchmarkDataset(source)
        if file_hash(source / "pairs.csv") != manifest["pairs_sha256"]:
            raise ValueError("Measured pair table changed.")
        frames, scores = [], []
        for model in ("legacy_tf", "pytorch"):
            for seed in config["seeds"]:
                original = run / "final" / dataset / model / f"s{seed}"
                record = json.loads((original / "result.json").read_text())
                if (
                    record["status"] != "ok"
                    or record["protocol_sha256"] != provenance["protocol_sha256"]
                    or record["pairs_sha256"] != manifest["pairs_sha256"]
                    or file_hash(original / "checkpoint.bin") != record["checkpoint_sha256"]
                ):
                    raise ValueError(f"Invalid measured checkpoint: {original}")
                target = stage / "models" / model / f"s{seed}"
                target.mkdir(parents=True)
                shutil.copy2(original / "checkpoint.bin", target / "checkpoint.bin")
                shutil.copy2(original / "result.json", target / "record.json")
                for split, indices in manifest["splits"].items():
                    predictions = pd.read_csv(
                        replay_root / dataset / model / f"s{seed}" / f"{split}.csv"
                    )
                    if predictions["pair_index"].tolist() != indices:
                        raise ValueError("Prediction rows do not match the frozen partition.")
                    truth = pairs.iloc[indices]["rp_seconds"].to_numpy(dtype=np.float64)
                    if not np.allclose(predictions["truth_seconds"], truth, rtol=1e-7, atol=1e-3):
                        raise ValueError("Prediction targets do not match the measured pair table.")
                    quality = metrics(
                        predictions["truth_seconds"], predictions["prediction_seconds"]
                    )
                    if split != "train":
                        if (
                            file_hash(original / f"{split}.csv")
                            != record[f"{split}_predictions_sha256"]
                        ):
                            raise ValueError("Recorded prediction table changed.")
                        for key, value in quality.items():
                            recorded = record[f"{split}_quality"][key]
                            if not (
                                recorded is None
                                if value is None
                                else recorded is not None
                                and np.isclose(value, recorded, rtol=1e-5, atol=1e-4)
                            ):
                                raise ValueError(
                                    f"Replayed {key} differs from the recorded metric."
                                )
                    predictions.insert(0, "split", split)
                    predictions.insert(0, "seed", seed)
                    predictions.insert(0, "model", model)
                    for column in ("peptide_a", "peptide_b", "group_key"):
                        predictions[column] = pairs.iloc[indices][column].to_numpy()
                    frames.append(predictions)
                    scores.append({"model": model, "seed": seed, "split": split, **quality})
                if model == "pytorch":
                    params = recurrent_parameters(manifest, config, record["candidate"])
                    convert_native(
                        original / "checkpoint.bin",
                        target / "model.pt",
                        params,
                        data,
                        pd.read_csv(replay_root / dataset / model / f"s{seed}" / "test.csv"),
                    )
                    pretrain = run / "pretrain" / dataset / f"s{seed}"
                    pretrain_record = json.loads((pretrain / "result.json").read_text())
                    if (
                        file_hash(pretrain / "checkpoint.bin")
                        != pretrain_record["checkpoint_sha256"]
                    ):
                        raise ValueError("Train-only base checkpoint changed.")
                    base = stage / "models" / "pretrain" / f"s{seed}"
                    base.mkdir(parents=True)
                    shutil.copy2(pretrain / "result.json", base / "record.json")
                    convert_native(
                        pretrain / "checkpoint.bin", base / "model.pt", params, data, None
                    )
        predictions, score_table = pd.concat(frames, ignore_index=True), pd.DataFrame(scores)
        tables = stage / "tables"
        tables.mkdir()
        predictions.to_csv(tables / "predictions.csv", index=False)
        score_table.to_csv(tables / "metrics.csv", index=False)
        summary = (
            score_table.loc[score_table["split"] == "test"]
            .groupby("model")
            .agg(
                rmse_mean=("rmse_seconds", "mean"),
                rmse_seed_sd=("rmse_seconds", "std"),
                mae_mean=("mae_seconds", "mean"),
                r2_mean=("r2", "mean"),
            )
        )
        summary.to_csv(tables / "test-summary.csv")
        plot_predictions(predictions, score_table, stage / "plots")
        evidence = stage / "provenance"
        evidence.mkdir()
        for name in ("provenance.json", "config.json", "verification.json", "source.tar.gz"):
            shutil.copy2(run / name, evidence / name)
        text = (
            f"# {dataset}: TensorFlow reference and PyTorch\n\n"
            "Predictions were regenerated from the validation-selected checkpoints. "
            "Train, validation and test rows retain the original grouped split; "
            "only test metrics describe held-out accuracy. Errors are in seconds.\n\n"
            "```text\n" + summary.round(3).to_string() + "\n```\n\n"
            "![Held-out RMSE](plots/rmse.png)\n\n"
            "![Predictions](plots/predictions.png)\n\n"
            "![Residuals](plots/residuals.png)\n\n"
            "Error bars are seed standard deviations, not confidence intervals. "
            "TensorFlow is retired from the maintained package. Its checkpoint format, "
            "source and original environment locks are retained only as archival evidence.\n"
        )
        (stage / "README.md").write_text(text)
        seal_archive(
            stage,
            {
                "schema_version": 1,
                "use_case": dataset,
                "protocol_sha256": provenance["protocol_sha256"],
                "source_snapshot_sha256": provenance["source_snapshot_sha256"],
                "prediction_origin": "regenerated from frozen selected checkpoints",
                "models": ["legacy_tf", "pytorch"],
                "seeds": config["seeds"],
                "test_rows_per_seed": len(manifest["splits"]["test"]),
            },
        )
        stage.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("benchmarks/results/scheduled-v2"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-tf-torch"))
    parser.add_argument("--replay-root", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads((args.run / "provenance.json").read_text())
    for dataset in record["config"]["datasets"]:
        archive_cohort(args.run, dataset, args.output, args.replay_root)
        print(f"Archived {dataset}", flush=True)


if __name__ == "__main__":
    main()
