"""Integrity and recomputation contracts for standalone benchmark archives."""

import json

import numpy as np
import pandas as pd
import pytest

from xirt.data_pipelines.benchmarks.common import (
    file_hash,
    identity,
    metrics,
    seal_archive,
    verify_archive,
    write_json,
)
from xirt.data_pipelines.benchmarks.dataset import BenchmarkDataset
from xirt.data_pipelines.benchmarks.verify import verify_fine_tuning, verify_predictions


def test_metrics_reject_invalid_predictions_and_leave_constant_r2_undefined():
    assert metrics([1, 1], [1, 2])["r2"] is None
    assert metrics([0, 2], [1, 3])["rmse_seconds"] == 1
    for truth, prediction in [([], []), ([1], [1, 2]), ([np.nan], [1]), ([1], [np.inf])]:
        with pytest.raises(ValueError):
            metrics(truth, prediction)


def test_archive_detects_changes_and_never_ignores_nested_indexes(tmp_path):
    write_json(tmp_path / "nested/archive.json", {"nested": True})
    seal_archive(tmp_path, {"schema_version": 1})
    assert "nested/archive.json" in verify_archive(tmp_path)["files"]
    write_json(tmp_path / "nested/archive.json", {"nested": False})
    with pytest.raises(ValueError, match="changed"):
        verify_archive(tmp_path)


def test_archive_rejects_paths_outside_root(tmp_path):
    write_json(tmp_path / "archive.json", {"files": {"../outside": "not-allowed"}})
    with pytest.raises(ValueError, match="changed"):
        verify_archive(tmp_path)


@pytest.fixture
def predictions(tmp_path):
    splits = {"train": [0, 1], "validation": [2], "test": [3]}
    truth = np.array([100.0, 200.0, 300.0, 400.0])
    write_json(tmp_path / "data/manifest.json", {"splits": splits})
    pd.DataFrame({"rp_seconds": truth}).to_csv(tmp_path / "data/pairs.csv", index=False)
    frames, scores = [], []
    for model in ("legacy_tf", "pytorch"):
        for split, indices in splits.items():
            frame = pd.DataFrame(
                {
                    "model": model,
                    "seed": 17,
                    "split": split,
                    "pair_index": indices,
                    "truth_seconds": truth[indices],
                    "prediction_seconds": truth[indices] + 1,
                }
            )
            frames.append(frame)
            scores.append(
                {
                    "model": model,
                    "seed": 17,
                    "split": split,
                    **metrics(truth[indices], truth[indices] + 1),
                }
            )
    (tmp_path / "tables").mkdir()
    pd.concat(frames).to_csv(tmp_path / "tables/predictions.csv", index=False)
    pd.DataFrame(scores).to_csv(tmp_path / "tables/metrics.csv", index=False)
    for model in ("pytorch", "pretrain"):
        path = tmp_path / f"models/{model}/s17/model.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture checkpoint")
    seal_archive(tmp_path, {"seeds": [17]})
    return tmp_path


def test_prediction_archive_recomputes_metrics_and_requires_exports(predictions):
    assert verify_predictions(predictions) == 6
    manifest = json.loads((predictions / "archive.json").read_text())
    del manifest["files"]["models/pretrain/s17/model.pt"]
    write_json(predictions / "archive.json", manifest)
    with pytest.raises(ValueError, match="Missing native"):
        verify_predictions(predictions)


def test_prediction_archive_rejects_self_consistent_but_wrong_scores(predictions):
    path = predictions / "tables/metrics.csv"
    table = pd.read_csv(path)
    table.loc[0, "mae_seconds"] = 42
    table.to_csv(path, index=False)
    seal_archive(predictions, {"seeds": [17]})
    with pytest.raises(ValueError, match="Recomputed"):
        verify_predictions(predictions)


def test_fine_tuning_archive_requires_all_runs(tmp_path):
    protocol = {"datasets": ["cohort"], "strategies": ["recurrent_full"], "seeds": [17]}
    write_json(tmp_path / "protocol.json", protocol)
    seal_archive(tmp_path, {"protocol_sha256": identity(protocol), "runs_verified": 1})
    with pytest.raises(ValueError, match="Incomplete"):
        verify_fine_tuning(tmp_path)


def test_fine_tuning_verification_checks_aggregate_tables(tmp_path):
    protocol = {"datasets": ["cohort"], "strategies": ["recurrent_full"], "seeds": [17]}
    digest = identity(protocol)
    write_json(tmp_path / "protocol.json", protocol)
    run = tmp_path / "runs/cohort/recurrent_full/s17"
    run.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "dataset": "cohort",
            "model": "recurrent_full",
            "seed": 17,
            "split": ["train", "validation", "test"],
            "pair_index": [0, 1, 2],
            "truth_seconds": [100.0, 200.0, 300.0],
            "prediction_seconds": [101.0, 201.0, 301.0],
        }
    )
    frame.to_csv(run / "predictions.csv", index=False)
    quality = {
        split: metrics(group["truth_seconds"], group["prediction_seconds"])
        for split, group in frame.groupby("split")
    }
    record = {
        "dataset": "cohort",
        "strategy": "recurrent_full",
        "seed": 17,
        "status": "ok",
        "protocol_sha256": digest,
        "frozen_parameters_unchanged": True,
        "quality": quality,
        "trainable_parameters": 10,
        "training_seconds": 1.0,
    }
    write_json(run / "record.json", record)
    (run / "model.pt").write_bytes(b"checkpoint fixture")
    (run / "history.csv").write_text("loss,val_loss\n1,1\n")
    seal_archive(run, {"protocol_sha256": digest})
    table = pd.DataFrame(
        [
            {
                "dataset": "cohort",
                "model": "recurrent_full",
                "seed": 17,
                "split": split,
                **values,
                "trainable_parameters": 10,
                "training_seconds": 1.0,
            }
            for split, values in quality.items()
        ]
    )
    results = tmp_path / "results"
    results.mkdir()
    table.to_csv(results / "metrics.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "dataset": "cohort",
                "model": "recurrent_full",
                "rmse_mean": 1.0,
                "rmse_seed_sd": np.nan,
                "mae_mean": 1.0,
                "r2_mean": np.nan,
                "trainable_parameters": 10,
                "training_seconds_mean": 1.0,
            }
        ]
    )
    summary.to_csv(results / "test-summary.csv", index=False)
    seal_archive(tmp_path, {"protocol_sha256": digest, "runs_verified": 1})
    assert verify_fine_tuning(tmp_path) == 1
    summary.loc[0, "rmse_mean"] = 42.0
    summary.to_csv(results / "test-summary.csv", index=False)
    seal_archive(tmp_path, {"protocol_sha256": digest, "runs_verified": 1})
    with pytest.raises(AssertionError):
        verify_fine_tuning(tmp_path)


def test_prediction_export_preserves_double_precision_seconds(tmp_path):
    pairs = pd.DataFrame(
        {
            "rp_seconds": [100.0, 200.0, 300.0, 400.0],
            "group_key": ["a", "b", "c", "d"],
            "tokens_a": ['["A"]'] * 4,
        }
    )
    pairs.to_csv(tmp_path / "pairs.csv", index=False)
    write_json(
        tmp_path / "manifest.json",
        {
            "pairs_sha256": file_hash(tmp_path / "pairs.csv"),
            "alphabet": ["A"],
            "max_length": 2,
            "siamese": False,
            "target_normalization": {"mean": 150.0, "std": 50.0},
            "splits": {"train": [0, 1], "validation": [2], "test": [3]},
        },
    )
    data = BenchmarkDataset(tmp_path)
    frame = data.prediction_table(np.array([[1.23456789]], dtype=np.float32), [3])
    assert frame["prediction_seconds"].dtype == np.float64
    frame.to_csv(tmp_path / "export.csv", index=False)
    actual = pd.read_csv(tmp_path / "export.csv")
    assert actual["prediction_seconds"][0] == pytest.approx(
        frame["prediction_seconds"][0], abs=1e-12
    )


def test_fine_tuning_pipeline_rejects_duplicate_cohorts(tmp_path, monkeypatch):
    from xirt.data_pipelines.benchmarks.fine_tune_torch import pipeline

    directory = tmp_path / "cohort/results"
    directory.mkdir(parents=True)
    seal_archive(directory, {"seeds": [17]})
    monkeypatch.setattr(
        "sys.argv",
        [
            "fine-tune",
            "--archive",
            str(tmp_path),
            "--datasets",
            "cohort",
            "cohort",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as error:
        pipeline.main()
    assert error.value.code == 2
