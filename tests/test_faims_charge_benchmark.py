"""Audit the completed three-model comparison without raw observations or weights."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from xirt.data_pipelines.benchmarks.faims.run import evaluation_table
from xirt.qc import prediction_metrics

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark-faims"
REPORTS = BENCHMARK / "reports"
ARCHIVE = BENCHMARK / "archive/2026-10-20-precursor-baselines"


def test_prediction_csv_preserves_float32_model_values():
    data = pd.DataFrame(
        {
            "precursor_id": ["a", "b", "c"],
            "source": ["example"] * 3,
            "pair_group": ["a", "b", "c"],
            "precursor_charge": [3, 4, 5],
            "faims_cv": [-70, -50, -40],
            "observation_count": [2, 2, 2],
            "distinct_cv_count": [1, 1, 1],
            "cv_spread_v": [0, 0, 0],
        }
    )
    prediction = np.array([-68.793434, -53.146713, -40.312347], dtype=np.float32)
    result = evaluation_table(data, prediction)
    saved = pd.read_csv(io.StringIO(result.to_csv(index=False)))
    np.testing.assert_allclose(
        saved["faims_cv-prediction"], prediction.astype(float), rtol=0, atol=1e-12
    )


def test_original_benchmark_archive_is_unchanged():
    hashes = json.loads((ARCHIVE / "sha256.json").read_text())
    for relative, expected in hashes.items():
        assert hashlib.sha256((ARCHIVE / relative).read_bytes()).hexdigest() == expected


def test_charge_comparison_inventory_features_and_training_controls():
    protocol = json.loads((BENCHMARK / "protocol.json").read_text())
    manifest = json.loads((REPORTS / "manifest.json").read_text())
    completion = json.loads((REPORTS / "completion.json").read_text())
    assert manifest["protocol"] == protocol
    assert protocol["evaluation"]["previous_test_results_seen"]
    assert not manifest["test_used_for_selection"]
    assert completion["recurrent_runs"] == 27
    assert (
        completion["protocol_sha256"]
        == hashlib.sha256((BENCHMARK / "protocol.json").read_bytes()).hexdigest()
    )
    with tarfile.open(REPORTS / "source.tar.gz") as source:
        for relative, expected in manifest["source_sha256"].items():
            stream = source.extractfile(relative)
            assert stream is not None
            assert hashlib.sha256(stream.read()).hexdigest() == expected
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    runs = pd.read_csv(REPORTS / "runs.csv")
    expected_runs = {
        (cohort, model, seed)
        for cohort in protocol["training_cohorts"]
        for model in protocol["models"]
        for seed in protocol["initialization_seeds"]
    }
    assert len(runs) == len(expected_runs)
    assert (
        set(runs[["cohort", "model", "seed"]].itertuples(index=False, name=None)) == expected_runs
    )
    for cohort in protocol["training_cohorts"]:
        partition = pd.read_csv(REPORTS / f"{cohort}__partitions.csv")
        pd.testing.assert_frame_equal(
            partition, pd.read_csv(ARCHIVE / "reports" / f"{cohort}__partitions.csv")
        )
        assert partition["precursor_id"].is_unique
        assert partition.groupby("pair_group")["split"].nunique().eq(1).all()
        test_ids = partition.loc[partition["split"] == "test", "precursor_id"].tolist()
        for run in runs.loc[runs["cohort"] == cohort].itertuples(index=False):
            tag = f"{cohort}__{run.model}__{run.seed}"
            details = json.loads((REPORTS / f"{tag}__run.json").read_text())
            params = details["params"]
            features = protocol["feature_sets"][protocol["models"][run.model]]
            assert params["numeric_features"] == features
            assert params["learning"]["seed"] == run.seed
            assert params["learning"]["epochs"] == protocol["training"]["maximum_epochs"]
            assert params["learning"]["learningrate"] == protocol["training"]["learning_rate"]
            assert params["callbacks"]["early_stopping"]
            assert params["callbacks"]["reduce_lr"]
            assert params["_faims"]["units"] == "V"
            if features:
                assert params["_numeric_scaling"]["training_rows"] == int(
                    partition["split"].eq("train").sum()
                )
            else:
                assert "_numeric_scaling" not in params
            assert details["checkpoint_sha256"] == run.checkpoint_sha256
            assert not Path(details["checkpoint"]).is_absolute()
            history = pd.read_csv(REPORTS / f"{tag}__history.csv")
            assert len(history) == run.epochs <= protocol["training"]["maximum_epochs"]
            assert history["val_loss"].argmin() + 1 == run.best_epoch
            assert np.isfinite(history).all().all()
            predictions = pd.read_csv(REPORTS / f"{tag}__test.csv")
            assert predictions["precursor_id"].tolist() == test_ids


def test_charge_comparison_metrics_transfer_and_correlations():
    metrics = pd.read_csv(REPORTS / "metrics.csv")
    source_pairs = set(pd.read_csv(REPORTS / "pooled_4pm_8pm__partitions.csv")["pair_group"])
    for keys, group in metrics.groupby(["cohort", "model", "seed", "evaluation"], sort=False):
        cohort, model, seed, evaluation = keys
        predictions = pd.read_csv(REPORTS / f"{cohort}__{model}__{seed}__{evaluation}.csv")
        assert predictions["precursor_id"].is_unique
        if evaluation.startswith("transfer_"):
            target = evaluation.removeprefix("transfer_")
            partition = pd.read_csv(REPORTS / f"{target}__partitions.csv")
            eligible = partition.loc[~partition["pair_group"].isin(source_pairs)]
            assert predictions["precursor_id"].tolist() == eligible["precursor_id"].tolist()
        for row in group.itertuples(index=False):
            if row.stratification == "all":
                subset = predictions
            elif row.stratification == "token_coverage":
                subset = predictions.loc[predictions["unknown_token_count"] == 0]
            else:
                subset = predictions.loc[
                    predictions[row.stratification].astype(str) == str(row.stratum)
                ]
            actual = prediction_metrics(subset["faims_cv"], subset["faims_cv-prediction"])
            assert actual["n"] == row.n
            for key, value in actual.items():
                column = {"rmse": "rmse_v", "mae": "mae_v"}.get(key, key)
                if value is None:
                    assert pd.isna(getattr(row, column))
                else:
                    np.testing.assert_allclose(getattr(row, column), value, rtol=1e-6, atol=1e-6)
            assert row.unknown_precursors == subset["unknown_token_count"].gt(0).sum()
