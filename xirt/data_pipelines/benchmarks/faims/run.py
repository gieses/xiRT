"""Execute the frozen FAIMS protocol on verified local xiFAIMS CSV files.

Run from a source checkout; raw observations and tensor checkpoints remain in
ignored local-data while compact evaluation evidence is written to reports.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from xirt import qc
from xirt.faims import (
    FAIMSPredictor,
    default_faims_params,
    prepare_faims_observations,
    split_precursors,
)

ROOT = Path(__file__).resolve().parents[4]
logger = logging.getLogger(__name__)


def write_json(path: Path, data: Any) -> None:
    """Write strict JSON suitable for notebook inspection and archival."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    """Hash a source, dataset or checkpoint."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_metrics(frame: pd.DataFrame, **labels: Any) -> list[dict[str, Any]]:
    """Evaluate the whole prediction set and prespecified strata."""
    rows = []
    subsets = [("all", "all", frame)]
    for column in ("precursor_charge", "source"):
        subsets.extend((column, str(key), part) for key, part in frame.groupby(column))
    subsets.append(("token_coverage", "known_only", frame.loc[frame["unknown_token_count"] == 0]))
    for stratification, value, subset in subsets:
        if len(subset):
            rows.append(
                {
                    **labels,
                    "stratification": stratification,
                    "stratum": value,
                    **qc.prediction_metrics(subset["faims_cv"], subset["faims_cv-prediction"]),
                    "unknown_precursors": int((subset["unknown_token_count"] > 0).sum()),
                }
            )
    for row in rows:
        row["rmse_v"] = row.pop("rmse")
        row["mae_v"] = row.pop("mae")
    return rows


def evaluation_table(data: pd.DataFrame, predictions: Any, unknown: Any = None) -> pd.DataFrame:
    """Keep stable IDs and evaluation labels, not raw identification metadata."""
    columns = [
        "precursor_id",
        "source",
        "pair_group",
        "precursor_charge",
        "faims_cv",
        "observation_count",
        "distinct_cv_count",
        "cv_spread_v",
    ]
    result = data[columns].copy()
    result["faims_cv-prediction"] = np.asarray(predictions, dtype=float)
    result["unknown_token_count"] = 0 if unknown is None else np.asarray(unknown)
    return result


def make_plots(reports: Path) -> None:
    """Use the public RT/FAIMS QC plots and consistent model/seed colors."""
    protocol = json.loads((reports / "manifest.json").read_text())["protocol"]
    models = list(protocol["models"])
    labels = ["Sequence", "Sequence + charge", "Sequence + charge + m/z"]
    colors = ["#6b7a90", "#287b8e", "#d67b30"]
    seed = protocol["evaluation"]["plot_initialization_seed"]
    metrics = pd.read_csv(reports / "metrics.csv")
    heldout = metrics.loc[(metrics["evaluation"] == "test") & (metrics["stratification"] == "all")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (cohort, group) in zip(axes, heldout.groupby("cohort", sort=False)):
        summary = group.groupby("model", sort=False)["rmse_v"].agg(["mean", "std"]).reindex(models)
        ax.barh(labels, summary["mean"], xerr=summary["std"], color=colors)
        ax.set(title=cohort, xlabel="Held-out RMSE (V)")
    fig.savefig(reports / "heldout_rmse.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, cohort in zip(axes, heldout["cohort"].drop_duplicates()):
        for model, label, color in zip(models, labels, colors, strict=True):
            for initialization in protocol["initialization_seeds"]:
                history = pd.read_csv(reports / f"{cohort}__{model}__{initialization}__history.csv")
                ax.plot(
                    history.index + 1,
                    np.sqrt(history["val_loss"]) * history["target_scale_v"].iloc[0],
                    label=label if initialization == protocol["initialization_seeds"][0] else None,
                    color=color,
                    alpha=0.6,
                )
        ax.set(title=cohort, xlabel="Epoch", ylabel="Validation RMSE (V)")
        ax.legend(fontsize=7)
    fig.savefig(reports / "learning_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
    for column, cohort in enumerate(heldout["cohort"].drop_duplicates()):
        for row, (model, label, color) in enumerate(zip(models, labels, colors, strict=True)):
            data = pd.read_csv(reports / f"{cohort}__{model}__{seed}__test.csv")
            qc.plot_predictions(
                data["faims_cv"],
                data["faims_cv-prediction"],
                ax=axes[row, column],
                name=f"{cohort} · {label}",
                unit="V",
                color=color,
            )
            residuals = qc.prediction_qc(
                data["faims_cv"],
                data["faims_cv-prediction"],
                name=f"{cohort} · {label}",
                unit="V",
            )
            residuals.savefig(reports / f"{cohort}__{model}__qc.png", dpi=150)
            plt.close(residuals)
    fig.savefig(reports / "heldout_predictions.png", dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    """Verify inputs and protocol before any model sees training data."""
    protocol = json.loads(args.protocol.read_text())
    torch.set_num_threads(protocol["training"]["cpu_threads"])
    args.reports.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    source_files = [
        ROOT / "xirt/faims.py",
        ROOT / "xirt/xirtnet.py",
        ROOT / "xirt/config.py",
        ROOT / "xirt/training.py",
        ROOT / "xirt/processing.py",
        ROOT / "xirt/sequences.py",
        ROOT / "xirt/predictor.py",
        ROOT / "xirt/qc.py",
        ROOT / "xirt/explain.py",
        Path(__file__),
    ]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    data, audits = {}, []
    for name, spec in protocol["datasets"].items():
        path = args.data / spec["file"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {path}")
        data[name], audit = prepare_faims_observations(
            pd.read_csv(path), name, maximum_fdr=protocol["target"]["maximum_fdr"]
        )
        audits.append(audit)
    cohorts = {
        name: split_precursors(
            pd.concat([data[source] for source in sources], ignore_index=True),
            seed=protocol["split"]["seed"],
            fractions=tuple(protocol["split"]["fractions"]),
        )
        for name, sources in protocol["training_cohorts"].items()
    }
    manifest = {
        "protocol": protocol,
        "source_sha256": hashes,
        "git_base": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "test_used_for_selection": False,
    }
    manifest_path = args.reports / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if any(value != existing.get(key) for key, value in manifest.items() if key != "git_base"):
            raise ValueError(
                "Existing experiment manifest differs; choose fresh work/reports paths."
            )
    else:
        write_json(manifest_path, manifest)
        with tarfile.open(args.reports / "source.tar.gz", "w:gz") as bundle:
            for path in source_files:
                bundle.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    write_json(args.reports / "data_audit.json", audits)
    for name, cohort in cohorts.items():
        cohort.to_csv(args.work / f"{name}__precursors.csv", index=False)
        cohort[["precursor_id", "pair_group", "source", "split"]].to_csv(
            args.reports / f"{name}__partitions.csv", index=False
        )
        assert cohort.groupby("pair_group")["split"].nunique().max() == 1
        archived = (
            ROOT / "benchmark-faims" / protocol["archive"] / "reports" / f"{name}__partitions.csv"
        )
        pd.testing.assert_frame_equal(
            cohort[["precursor_id", "pair_group", "source", "split"]].reset_index(drop=True),
            pd.read_csv(archived),
        )
    metrics: list[dict[str, Any]] = []
    runs = []
    transfer_audit = []
    for cohort_name, cohort in cohorts.items():
        train = cohort.loc[cohort["split"] == "train"]
        validation = cohort.loc[cohort["split"] == "validation"]
        test = cohort.loc[cohort["split"] == "test"]
        evaluations = {"test": test}
        if cohort_name == protocol["transfer"]["source"]:
            for target in protocol["transfer"]["targets"]:
                external = cohorts[target]
                keep = ~external["pair_group"].isin(set(cohort["pair_group"]))
                evaluations[f"transfer_{target}"] = external.loc[keep]
                transfer_audit.append(
                    {
                        "source": cohort_name,
                        "target": target,
                        "excluded_shared_pair_precursors": int((~keep).sum()),
                        "evaluated_precursors": int(keep.sum()),
                    }
                )
        for model_name in protocol["models"]:
            for seed in protocol["initialization_seeds"]:
                tag = f"{cohort_name}__{model_name}__{seed}"
                checkpoint = args.work / f"{tag}.pt"
                history_path = args.reports / f"{tag}__history.csv"
                status_path = args.reports / f"{tag}__run.json"
                if status_path.exists():
                    status = json.loads(status_path.read_text())
                    if sha256(checkpoint) != status["checkpoint_sha256"]:
                        raise ValueError(f"Checkpoint changed: {checkpoint}")
                    predictor = FAIMSPredictor.load(checkpoint)
                else:
                    params = default_faims_params(
                        seed=seed, features=protocol["models"][model_name]
                    )
                    if (
                        params["numeric_features"]
                        != protocol["feature_sets"][protocol["models"][model_name]]
                    ):
                        raise ValueError("Model numeric features differ from the frozen protocol.")
                    training, architecture = protocol["training"], protocol["architecture"]
                    params["learning"].update(
                        epochs=training["maximum_epochs"],
                        batch_size=training["batch_size"],
                        optimizer=training["optimizer"],
                        learningrate=training["learning_rate"],
                        weight_decay=training["weight_decay"],
                    )
                    params["callbacks"].update(
                        early_stopping_patience=training["early_stopping_patience"],
                        reduce_lr_patience=training["reduce_lr_patience"],
                        reduce_lr_factor=training["reduce_lr_factor"],
                    )
                    params["LSTM"]["units"] = architecture["gru_units"]
                    params["embedding"]["length"] = architecture["embedding_width"]
                    params["dense"]["neurons"] = architecture["dense_units"]
                    params["dense"]["dropout"] = [architecture["dropout"]] * 2
                    params["siamese"]["merge_type"] = architecture["merge"]
                    started = time.monotonic()
                    predictor = FAIMSPredictor(
                        params, maximum_residues=protocol["architecture"]["maximum_residues"]
                    ).fit(train, validation)
                    predictor.save(checkpoint)
                    history = pd.DataFrame(predictor.history)
                    history["target_scale_v"] = predictor.metadata["target_scale"]
                    history.to_csv(history_path, index=False)
                    status = {
                        "cohort": cohort_name,
                        "model": model_name,
                        "seed": seed,
                        "epochs": len(history),
                        "best_epoch": int(history["val_loss"].argmin()) + 1,
                        "seconds": time.monotonic() - started,
                        "checkpoint_sha256": sha256(checkpoint),
                        "checkpoint": os.path.relpath(checkpoint, ROOT),
                        "params": predictor.model.params,
                    }
                    write_json(status_path, status)
                runs.append({k: v for k, v in status.items() if k != "params"})
                for evaluation, subset in evaluations.items():
                    prediction = predictor.predict(subset)
                    result = evaluation_table(
                        subset,
                        prediction["faims_cv-prediction"],
                        prediction["unknown_token_count"],
                    )
                    result.to_csv(args.reports / f"{tag}__{evaluation}.csv", index=False)
                    rows = record_metrics(
                        result,
                        cohort=cohort_name,
                        model=model_name,
                        seed=seed,
                        evaluation=evaluation,
                    )
                    metrics.extend(rows)
                logger.info("%s: epochs=%s best=%s", tag, status["epochs"], status["best_epoch"])
                pd.DataFrame(metrics).to_csv(args.reports / "metrics.csv", index=False)
        pd.DataFrame(metrics).to_csv(args.reports / "metrics.csv", index=False)
    pd.DataFrame(runs).to_csv(args.reports / "runs.csv", index=False)
    write_json(args.reports / "transfer_audit.json", transfer_audit)
    metric_frame = pd.DataFrame(metrics)
    summary = (
        metric_frame.loc[metric_frame["stratification"] == "all"]
        .groupby(["cohort", "model", "evaluation"], sort=False)[
            ["rmse_v", "mae_v", "r2", "pearson_r", "spearman_rho"]
        ]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.reset_index().to_csv(args.reports / "summary.csv", index=False)
    make_plots(args.reports)
    write_json(
        args.reports / "completion.json",
        {
            "recurrent_runs": len(runs),
            "protocol_sha256": sha256(args.protocol),
            "report_files": sorted(path.name for path in args.reports.iterdir() if path.is_file()),
        },
    )
    print(summary.to_string())


def main() -> None:
    """Resolve checkout-local defaults and run the complete experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark-faims/protocol.json")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "local-data/xifaims/b85482ffc246b2dc669c4592f752568fd9b1baed",
    )
    parser.add_argument("--work", type=Path, default=ROOT / "local-data/faims-charge")
    parser.add_argument("--reports", type=Path, default=ROOT / "benchmark-faims/reports")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
