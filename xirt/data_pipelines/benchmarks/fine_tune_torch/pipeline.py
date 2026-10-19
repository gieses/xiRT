"""Compare native recurrent fine-tuning strategies on archived grouped RP cohorts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from xirt.data_pipelines.benchmarks.common import (
    file_hash,
    identity,
    metrics,
    plot_predictions,
    seal_archive,
    verify_archive,
    write_json,
)
from xirt.data_pipelines.benchmarks.dataset import BenchmarkDataset
from xirt.training import seed_everything
from xirt.xirtnet import read_checkpoint, xiRTNET

STRATEGIES = ("recurrent_full", "recurrent_frozen", "recurrent_lokr", "recurrent_lora")


class ExperimentConfig(BaseModel):
    """Fixed, validation-led protocol shared by all four strategies."""

    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=30, gt=0, strict=True)
    batch_size: int = Field(default=64, gt=0, strict=True)
    learning_rate: FiniteFloat = Field(default=0.003, gt=0)
    weight_decay: FiniteFloat = Field(default=0.01, ge=0)
    patience: int = Field(default=8, gt=0, strict=True)
    rank: int = Field(default=4, gt=0, strict=True)
    alpha: FiniteFloat = Field(default=4.0, gt=0)
    adapter_learning_rate_ratio: FiniteFloat = Field(default=0.1, gt=0)
    threads: int = Field(default=2, gt=0, strict=True)


def _source_files(root: Path) -> list[Path]:
    names = [
        "xirt/adapters.py",
        "xirt/config.py",
        "xirt/training.py",
        "xirt/tracking.py",
        "xirt/xirtnet.py",
        "xirt/data_pipelines/benchmarks/common.py",
        "xirt/data_pipelines/benchmarks/dataset.py",
        "xirt/data_pipelines/benchmarks/fine_tune_torch/pipeline.py",
    ]
    return [root / name for name in names]


def run_one(
    archive: Path,
    output: Path,
    dataset: str,
    seed: int,
    strategy: str,
    config: ExperimentConfig,
    protocol: str,
) -> dict[str, Any]:
    """Train on outer training rows and restore the validation-selected checkpoint."""
    destination = output / "runs" / dataset / strategy / f"s{seed}"
    if (destination / "archive.json").exists():
        evidence = verify_archive(destination)
        if evidence["protocol_sha256"] != protocol:
            raise ValueError("Completed fine-tuning run has a different protocol.")
        return json.loads((destination / "record.json").read_text())
    if destination.exists():
        raise ValueError(f"Incomplete output needs inspection before retry: {destination}")
    cohort = archive / dataset / "results"
    data = BenchmarkDataset(cohort / "data")
    base = cohort / "models/pretrain" / f"s{seed}" / "model.pt"
    constructor = read_checkpoint(base)
    params = constructor["params"]
    params["learning"].update(
        optimizer="adamw",
        learningrate=config.learning_rate,
        weight_decay=config.weight_decay,
        device="cpu",
        seed=seed,
        deterministic=True,
    )
    seed_everything(seed)
    torch.set_num_threads(config.threads)
    network = xiRTNET(params, constructor["input_dim"], constructor["alphabet"])
    network.fine_tune(
        base,
        strategy,
        rank=config.rank,
        alpha=config.alpha,
        factor=4,
        adapter_learning_rate_ratio=config.adapter_learning_rate_ratio,
    )
    model = network.model
    if model is None:
        raise RuntimeError("Fine-tuning did not construct a model.")
    network.compile()
    frozen = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    frozen_buffers = {
        name: value.detach().clone()
        for name, value in model.named_buffers()
        if name.startswith(("embedding.", "recurrent.", "encoder_norm."))
        and strategy != "recurrent_full"
    }
    seed_everything(seed)
    started = time.perf_counter()
    train, validation = data.splits["train"], data.splits["validation"]
    history = model.fit(
        data.features(train),
        data.targets[train],
        validation_data=(data.features(validation), data.targets[validation]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
        callbacks={
            "early_stopping": True,
            "early_stopping_patience": config.patience,
            "reduce_lr": True,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 3,
        },
    ).history
    training_seconds = time.perf_counter() - started
    for name, value in model.named_parameters():
        if name in frozen and not torch.equal(value, frozen[name]):
            raise RuntimeError(f"Frozen weight changed: {name}")
    for name, value in model.named_buffers():
        if name in frozen_buffers and not torch.equal(value, frozen_buffers[name]):
            raise RuntimeError(f"Frozen buffer changed: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".job-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        model.save(stage / "model.pt")
        restored = xiRTNET(params, constructor["input_dim"], constructor["alphabet"])
        restored.load_model(stage / "model.pt")
        if restored.model is None:
            raise RuntimeError("Saved checkpoint could not be reconstructed.")
        rows, scores = [], {}
        inference_seconds = None
        for split, indices in data.splits.items():
            before = time.perf_counter()
            values = restored.model.predict(data.features(indices), batch_size=config.batch_size)
            if split == "test":
                inference_seconds = time.perf_counter() - before
            frame = data.prediction_table(values, indices)
            scores[split] = metrics(frame["truth_seconds"], frame["prediction_seconds"])
            frame.insert(0, "split", split)
            frame.insert(0, "seed", seed)
            frame.insert(0, "model", strategy)
            frame.insert(0, "dataset", dataset)
            rows.append(frame)
        pd.concat(rows, ignore_index=True).to_csv(stage / "predictions.csv", index=False)
        pd.DataFrame(history).assign(epoch=np.arange(1, len(history["loss"]) + 1)).to_csv(
            stage / "history.csv", index=False
        )
        record = {
            "status": "ok",
            "dataset": dataset,
            "seed": seed,
            "strategy": strategy,
            "protocol_sha256": protocol,
            "base_checkpoint_sha256": file_hash(base),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "epochs_executed": len(history["loss"]),
            "best_epoch": int(np.argmin(history["val_loss"])) + 1,
            "training_seconds": training_seconds,
            "test_inference_seconds": inference_seconds,
            "quality": scores,
            "frozen_parameters_unchanged": True,
        }
        write_json(stage / "record.json", record)
        seal_archive(stage, {"protocol_sha256": protocol})
        stage.rename(destination)
    print(
        f"{dataset}/{strategy}/s{seed}: test RMSE {scores['test']['rmse_seconds']:.2f}s", flush=True
    )
    return record


def summarize(output: Path, records: list[dict[str, Any]]) -> None:
    """Publish per-seed scores and cohort plots after every requested job succeeds."""
    rows = []
    for record in records:
        for split, quality in record["quality"].items():
            rows.append(
                {
                    "dataset": record["dataset"],
                    "model": record["strategy"],
                    "seed": record["seed"],
                    "split": split,
                    **quality,
                    "trainable_parameters": record["trainable_parameters"],
                    "training_seconds": record["training_seconds"],
                }
            )
    table = pd.DataFrame(rows)
    results = output / "results"
    results.mkdir(exist_ok=True)
    table.to_csv(results / "metrics.csv", index=False)
    summary = (
        table.loc[table["split"] == "test"]
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
    summary.to_csv(results / "test-summary.csv")
    for dataset in table["dataset"].unique():
        frames = [
            pd.read_csv(output / "runs" / dataset / strategy / f"s{seed}" / "predictions.csv")
            for strategy in STRATEGIES
            for seed in sorted(table.loc[table["dataset"] == dataset, "seed"].unique())
        ]
        plot_predictions(
            pd.concat(frames, ignore_index=True),
            table.loc[table["dataset"] == dataset],
            results / "plots" / dataset,
        )
    (results / "README.md").write_text(
        "# Native recurrent fine-tuning\n\n"
        "All strategies start from the same train-only checkpoint for each cohort and seed. "
        "The learning rate and adapter settings are fixed before evaluation. "
        "Early stopping and learning-rate reductions use validation loss; test rows are scored "
        "only after checkpoint selection and reconstruction. This is a separate protocol "
        "from the historical hyperparameter sweep.\n\n"
        "```text\n" + summary.round(3).to_string() + "\n```\n\n"
        "RMSE and MAE are seconds. Seed SD is not a confidence interval. "
        "Repeated cohorts are correlated views of two studies. These are staged within-cohort "
        "training experiments, not external transfer-learning validation.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("benchmark-tf-torch"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-tf-torch/fine-tune-torch"))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(epochs=args.epochs)
    available = sorted(
        path.parent.parent.name for path in args.archive.glob("*/results/archive.json")
    )
    datasets = args.datasets or available
    if not datasets or set(datasets) - set(available) or len(set(datasets)) != len(datasets):
        parser.error("Choose existing, unique archived cohorts.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be nonempty and unique.")
    for dataset in datasets:
        evidence = verify_archive(args.archive / dataset / "results")
        if set(args.seeds) - set(evidence["seeds"]):
            parser.error(f"Requested seeds are not archived for {dataset}.")
    root = Path(__file__).resolve().parents[4]
    source = {str(path.relative_to(root)): file_hash(path) for path in _source_files(root)}
    environment = {
        "python": platform.python_version(),
        "torch_runtime": torch.__version__,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "pandas",
                "pydantic",
                "scikit-learn",
            )
        },
    }
    protocol = {
        "config": config.model_dump(),
        "datasets": datasets,
        "seeds": args.seeds,
        "strategies": STRATEGIES,
        "source_files": source,
        "environment": environment,
        "cohort_archives": {
            dataset: file_hash(args.archive / dataset / "results/archive.json")
            for dataset in datasets
        },
    }
    digest = identity(protocol)
    if args.dry_run:
        print(f"{len(datasets) * len(args.seeds) * len(STRATEGIES)} native fine-tuning jobs")
        print(f"Protocol SHA256: {digest}")
        return
    if (args.output / "archive.json").exists():
        previous = verify_archive(args.output)
        if previous["protocol_sha256"] != digest:
            raise ValueError("Completed experiment is immutable; choose another output directory.")
        print("All requested fine-tuning results are already complete.")
        return
    if (args.output / "protocol.json").exists():
        if json.loads((args.output / "protocol.json").read_text()) != json.loads(
            json.dumps(protocol)
        ):
            raise ValueError(
                "Experiment code/configuration changed; choose another output directory."
            )
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Nonempty experiment output has no protocol.")
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "protocol.json", protocol)
        with tarfile.open(args.output / "source.tar.gz", "w:gz") as bundle:
            for path in _source_files(root):
                bundle.add(path, arcname=str(path.relative_to(root)))
        write_json(args.output / "environment.json", environment)
    records = [
        run_one(args.archive, args.output, dataset, seed, strategy, config, digest)
        for dataset in datasets
        for seed in args.seeds
        for strategy in STRATEGIES
    ]
    summarize(args.output, records)
    seal_archive(
        args.output,
        {
            "schema_version": 1,
            "protocol_sha256": digest,
            "runs_verified": len(records),
            "datasets": datasets,
            "seeds": args.seeds,
        },
    )


if __name__ == "__main__":
    main()
