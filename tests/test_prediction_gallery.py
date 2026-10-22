"""The illustrated examples must be deterministic and exclude labels from inference."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.generate_prediction_gallery import choose_example, raw_inputs
from examples.train_chromatography import sha256

ROOT = Path(__file__).resolve().parents[1]


def test_example_selection_uses_only_ids_and_lengths():
    table = pd.DataFrame(
        {
            "PSMID": [3, 1, 2],
            "Peptide1": ["AK", "AAAAAAAAK", "AAAAAAAAAK"],
            "Peptide2": ["AAAAAAAAK"] * 3,
            "rp": [0, 10, 20],
        }
    )
    assert choose_example(table)["PSMID"].item() == 1
    table["rp"] = [999, -999, 1000]
    assert choose_example(table)["PSMID"].item() == 1
    with pytest.raises(ValueError, match="8–22"):
        choose_example(table.iloc[:1])


def test_raw_prediction_inputs_never_contain_observed_targets_or_scores():
    table = pd.DataFrame(
        {
            "Peptide1": ["AAAAAAAAK"],
            "Peptide2": ["AAAAAAAAAK"],
            "LinkPos1": [9],
            "LinkPos2": [10],
            "rp": [2000],
            "scx": [20],
            "hsax": [4],
            "score": [100],
        }
    )
    predictor = SimpleNamespace(model=SimpleNamespace(numeric_features=[]))
    inputs = raw_inputs(table, predictor)
    assert list(inputs) == ["Peptide1", "Peptide2", "LinkPos1", "LinkPos2"]


def test_committed_gallery_has_genuine_full_data_provenance_and_four_explanations():
    directory = ROOT / "docs/assets/prediction-gallery"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["training"]["rows"] == 11072
    assert manifest["training"]["pair_groups"] == 5698
    assert manifest["training"]["config"]["network"]["learning"]["epochs"] == 300
    assert manifest["training"]["partitions"]["test"]["csms"] == 1055
    assert set(manifest["models"]) == {"rp", "scx", "hsax", "multitask"}
    for name, record in manifest["models"].items():
        assert 2 < record["epochs"] <= 300
        assert 1 <= record["best_epoch"] <= record["epochs"]
        assert record["checkpoint_sha256"] == manifest["checkpoints"][name]
        assert record["checkpoint_bytes"] == manifest["sizes"][name]
    assert set(manifest["examples"]) == {"rp", "scx", "hsax", "faims_cv"}
    ids = {manifest["examples"][task]["PSMID"] for task in ("rp", "scx", "hsax")}
    assert len(ids) == 1
    for task, example in manifest["examples"].items():
        assert example["crosslink_tokens"] == 2
        assert example["unit"] == {"rp": "s", "faims_cv": "V"}.get(task, "fraction")
        assert example["steps"] in (256, 512)
        delta = example["prediction"] - example["baseline_prediction"]
        assert abs(example["completeness_error"]) <= max(0.05, 0.01 * abs(delta))
    for metric in manifest["metrics"]:
        assert metric["n"] == (484 if metric["task"] == "faims_cv" else 1055)
        assert np.isfinite(
            [metric[key] for key in ("rmse", "mae", "r2", "pearson_r", "spearman_rho")]
        ).all()
    assert manifest["generator_sha256"] == sha256(ROOT / "examples/generate_prediction_gallery.py")
    for relative, digest in manifest["figures"].items():
        assert sha256(directory / relative) == digest
