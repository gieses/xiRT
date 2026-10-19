"""Read-only notebook analysis and partial/corrupt artifact contracts."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from xirt.data_pipelines.benchmarks.common import identity, write_json
from xirt.data_pipelines.benchmarks.tf_torch.analyze_results import (
    app,
    digest,
    load_run,
    predictions,
    read_json,
)


@pytest.fixture
def measured(tmp_path):
    directory = tmp_path / "scheduled-test"
    directory.mkdir()
    config = {
        "schema_version": 1,
        "datasets": ["cohort"],
        "variants": ["legacy_tf", "pytorch"],
        "seeds": [17, 29],
        "tuning_seeds": [17, 29],
        "candidates": [
            {
                "id": "reference",
                "learning_rate": 0.001,
                "dropout": 0.1,
                "weight_decay": 0.01,
                "scheduler": "cosine",
            },
            {
                "id": "constant",
                "learning_rate": 0.001,
                "dropout": 0.0,
                "weight_decay": 0.01,
                "scheduler": "constant",
            },
        ],
    }
    provenance = {
        "config": config,
        "scope": "smoke_subset",
        "protocol_sha256": identity(config),
        "source_snapshot_sha256": "fixture",
    }
    write_json(directory / "provenance.json", provenance)
    data = directory / "data/cohort"
    data.mkdir(parents=True)
    pd.DataFrame({"rp_seconds": [100, 200, 300, 400, 500, 600]}).to_csv(
        data / "pairs.csv", index=False
    )
    manifest = {
        "pair_count": 6,
        "pairs_sha256": digest(data / "pairs.csv"),
        "splits": {"train": [0, 1], "validation": [2, 3], "test": [4, 5]},
        "target_normalization": {"mean": 150, "std": 50},
    }
    write_json(data / "manifest.json", manifest)
    for variant in config["variants"]:
        selection_path = directory / "selection/cohort" / f"{variant}.json"
        write_json(selection_path, {"candidate_id": "reference", "test_used": False})
        for seed in config["seeds"]:
            common = {
                "status": "ok",
                "scope": "smoke_subset",
                "dataset": "cohort",
                "variant": variant,
                "seed": seed,
                "protocol_sha256": identity(config),
                "source_snapshot_sha256": "fixture",
                "data_manifest_sha256": digest(data / "manifest.json"),
                "pairs_sha256": manifest["pairs_sha256"],
                "model": {"device": "cpu", "trainable_parameters": 17, "total_parameters": 17},
                "history": [
                    {
                        "epoch": 1,
                        "train_loss": 0.7,
                        "validation_loss": 0.5,
                        "learning_rate_ratio": 1.0,
                    }
                ],
                "best_epoch": 1,
                "epochs_budget": 2,
                "checkpoint_bytes": 128,
                "training_seconds": 2.0,
                "setup_seconds": 1.0,
                "job_seconds": 4.0,
                "initial_adapter_gradient_l2": None,
                "diagnostics": {"frozen_parameters_unchanged": True, "adapter_update_l2": None},
            }
            if variant != "legacy_tf":
                common["model"]["warmstart_sha256"] = None
            for candidate in config["candidates"]:
                record = {
                    **copy.deepcopy(common),
                    "candidate": candidate,
                    "stage": "tune",
                    "test_evaluated": False,
                    "validation_loss": 0.5 if candidate["id"] == "reference" else 0.8,
                }
                write_json(
                    directory
                    / "tuning/cohort"
                    / variant
                    / candidate["id"]
                    / f"s{seed}/result.json",
                    record,
                )
            output = directory / "final/cohort" / variant / f"s{seed}"
            output.mkdir(parents=True)
            for split, indices in manifest["splits"].items():
                if split == "train":
                    continue
                truth = np.array([100 * (index + 1) for index in indices])
                pd.DataFrame(
                    {
                        "pair_index": indices,
                        "truth_seconds": truth,
                        "prediction_seconds": truth + seed,
                    }
                ).to_csv(output / f"{split}.csv", index=False)
            record = {
                **copy.deepcopy(common),
                "candidate": config["candidates"][0],
                "stage": "final",
                "test_evaluated": True,
                "validation_loss": 0.5,
                "test_quality": {
                    "rmse_seconds": float(seed),
                    "mae_seconds": float(seed),
                    "r2": 0.5,
                },
                "inference_seconds": 0.001,
                "selection_sha256": digest(selection_path),
                **{
                    f"{split}_predictions_sha256": digest(output / f"{split}.csv")
                    for split in ("validation", "test")
                },
            }
            write_json(output / "result.json", record)
    paths = sorted(directory.glob("final/**/result.json"))
    write_json(
        directory / "verification.json",
        {
            "status": "ok",
            "final_runs_verified": len(paths),
            "records": [
                {"path": str(path.relative_to(directory)), "sha256": digest(path)} for path in paths
            ],
        },
    )
    return directory


def test_complete_analysis_is_read_only_and_handles_legacy_metadata(measured):
    before = {path: digest(path) for path in measured.rglob("*") if path.is_file()}
    run = load_run(measured)
    assert len(run["finals"]) == 4
    assert len(run["tuning"]) == 8
    assert (run["coverage"]["final_done"] == 2).all()
    assert (run["coverage"]["tuning_done"] == 4).all()
    assert "hashes still match" in run["verification_state"]
    assert run["finals"]["warmstart_training_s"].eq(0).all()
    assert before == {path: digest(path) for path in measured.rglob("*") if path.is_file()}


def test_partial_run_preserves_missingness_and_failure_events(measured):
    for path in measured.glob("final/**/result.json"):
        path.unlink()
    (measured / "verification.json").unlink()
    write_json(
        measured / "failures/attempt.json",
        {
            "error_type": "TimeoutError",
            "error": "budget exceeded",
            "status": "failed",
        },
    )
    run = load_run(measured)
    assert run["finals"].empty
    assert run["coverage"]["final_done"].eq(0).all()
    assert run["coverage"]["final_expected"].eq(2).all()
    assert "unverified" in run["verification_state"]
    assert run["failures"].iloc[0]["error_type"] == "TimeoutError"


@pytest.mark.parametrize("field", ["protocol_sha256", "source_snapshot_sha256", "scope"])
def test_mixed_experiments_are_rejected(measured, field):
    path = next(measured.glob("final/**/result.json"))
    record = read_json(path)
    record[field] = "changed"
    write_json(path, record)
    with pytest.raises(ValueError, match="Mixed protocol/source/scope"):
        load_run(measured)


def test_stale_verification_is_not_reported_as_verified(measured):
    path = next(measured.glob("final/**/result.json"))
    record = read_json(path)
    record["training_seconds"] = 3
    write_json(path, record)
    assert "STALE" in load_run(measured)["verification_state"]


def test_prediction_partitions_units_and_tamper_detection(measured):
    run = load_run(measured)
    relative = "final/cohort/pytorch/s17/result.json"
    validation = predictions(run, relative, "validation")
    test = predictions(run, relative, "test")
    assert validation["pair_index"].tolist() == [2, 3]
    assert test["pair_index"].tolist() == [4, 5]
    assert test["residual_seconds"].eq(17).all()
    with pytest.raises(ValueError, match="validation or test"):
        predictions(run, relative, "train")
    path = measured / "final/cohort/pytorch/s17/test.csv"
    path.write_text(path.read_text().replace("517", "999"))
    with pytest.raises(ValueError, match="fingerprint changed"):
        predictions(run, relative, "test")


def test_notebook_executes_reactive_tables_plots_and_prediction_metrics(measured):
    outputs, definitions = app.run(defs={"results_root": measured.parent})
    assert len(definitions["comparison"]) == 2
    assert definitions["comparison"]["seeds"].eq(2).all()
    assert definitions["comparison"]["rmse_s"].eq(23).all()
    assert definitions["prediction_metrics"]["rows"] == 2
    assert definitions["prediction_metrics"]["rmse_seconds"] == 17
    assert definitions["prediction_metrics"]["train_mean_baseline_rmse_seconds"] == pytest.approx(
        np.sqrt((350**2 + 450**2) / 2)
    )
    assert sum("<img" in output.text for output in outputs if hasattr(output, "text")) == 3


def test_notebook_handles_no_runs(tmp_path):
    outputs, definitions = app.run(defs={"results_root": tmp_path})
    assert "run" not in definitions
    assert any(
        "No scheduled runs found" in output.text for output in outputs if hasattr(output, "text")
    )
