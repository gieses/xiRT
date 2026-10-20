"""Validate checked-in scientific evidence without raw data or local weights."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xirt.faims import cv_metrics

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "benchmark-faims/archive/2026-10-20-precursor-baselines"
REPORTS = ARCHIVE / "reports"


def test_completed_benchmark_follows_frozen_run_inventory_and_group_partitions():
    protocol = json.loads((ARCHIVE / "protocol.json").read_text())
    manifest = json.loads((REPORTS / "manifest.json").read_text())
    completion = json.loads((REPORTS / "completion.json").read_text())
    assert manifest["protocol"] == protocol
    assert completion["recurrent_runs"] == 18
    assert not manifest["test_used_for_selection"]
    runs = pd.read_csv(REPORTS / "runs.csv")
    expected = {
        (cohort, model, seed)
        for cohort in protocol["training_cohorts"]
        for model in protocol["models"]
        for seed in protocol["initialization_seeds"]
    }
    assert set(runs[["cohort", "model", "seed"]].itertuples(index=False, name=None)) == expected
    assert len(runs) == len(expected)
    for cohort in protocol["training_cohorts"]:
        partitions = pd.read_csv(REPORTS / f"{cohort}__partitions.csv")
        assert partitions["precursor_id"].is_unique
        assert partitions.groupby("pair_group")["split"].nunique().max() == 1
        assert set(partitions["split"]) == {"train", "validation", "test"}
        test_ids = partitions.loc[partitions["split"] == "test", "precursor_id"].tolist()
        for run in runs.loc[runs["cohort"] == cohort].itertuples(index=False):
            tag = f"{cohort}__{run.model}__{run.seed}"
            history = pd.read_csv(REPORTS / f"{tag}__history.csv")
            assert len(history) == run.epochs <= protocol["training"]["maximum_epochs"]
            assert history["val_loss"].argmin() + 1 == run.best_epoch
            assert np.isfinite(history).all().all()
            predictions = pd.read_csv(REPORTS / f"{tag}__test.csv")
            assert predictions["precursor_id"].tolist() == test_ids
            details = json.loads((REPORTS / f"{tag}__run.json").read_text())
            assert details["checkpoint_sha256"] == run.checkpoint_sha256
            assert details["params"]["learning"]["seed"] == run.seed
            assert details["params"]["_faims"]["units"] == "V"
            if run.model == "xirt_precursor":
                assert details["params"]["numeric_features"] == protocol["numeric_features"]
                assert details["params"]["_numeric_scaling"]["training_rows"] == int(
                    (partitions["split"] == "train").sum()
                )
            else:
                assert details["params"]["numeric_features"] == []
                assert "_numeric_scaling" not in details["params"]


def test_all_reported_metrics_and_transfer_coverage_recompute_from_predictions():
    metrics = pd.read_csv(REPORTS / "metrics.csv")
    source_pairs = set(pd.read_csv(REPORTS / "pooled_4pm_8pm__partitions.csv")["pair_group"])
    for keys, group in metrics.groupby(["cohort", "model", "seed", "evaluation"], sort=False):
        cohort, model, seed, evaluation = keys
        predictions = pd.read_csv(REPORTS / f"{cohort}__{model}__{seed}__{evaluation}.csv")
        assert predictions["precursor_id"].is_unique
        if evaluation.startswith("transfer_"):
            target = evaluation.removeprefix("transfer_")
            partitions = pd.read_csv(REPORTS / f"{target}__partitions.csv")
            eligible = partitions.loc[~partitions["pair_group"].isin(source_pairs)]
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
            score = cv_metrics(subset["faims_cv"], subset["faims_cv_prediction_v"])
            assert score["n"] == row.n
            for metric in ("rmse_v", "mae_v", "r2"):
                if score[metric] is None:
                    assert pd.isna(getattr(row, metric))
                else:
                    # Float32 predictions are serialized at CSV precision;
                    # near-zero R² requires absolute as well as relative tolerance.
                    np.testing.assert_allclose(
                        getattr(row, metric), score[metric], rtol=1e-6, atol=1e-6
                    )
            assert row.unknown_precursors == (subset["unknown_token_count"] > 0).sum()
