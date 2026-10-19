"""Execute the user notebooks and check their saved prediction contracts."""

import json
import runpy
import subprocess

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from examples.tutorial_data import ROOT, encode_pairs, network_params, prepare_data
from xirt.xirtnet import RetentionModel, xiRTNET


def test_tutorial_data_preserves_pairs_and_training_only_scaling():
    table, features, targets, metadata = prepare_data()
    assert table["PSMID"].is_unique
    assert table.groupby("split").size().to_dict() == {
        "test": 256,
        "train": 1536,
        "validation": 256,
    }
    assert (table["fdr"] <= 0.01).all()
    assert table["isTT"].all()
    pairs = table.apply(lambda row: tuple(sorted([row.Peptide1, row.Peptide2])), axis=1)
    assert pairs.is_unique
    assert all(array.shape == (2048, 50) for array in features)
    training = (table["split"] == "train").to_numpy()
    np.testing.assert_allclose(targets[training].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(targets[training].std(), 1, atol=1e-6)
    np.testing.assert_allclose(
        targets[:, 0] * metadata["target_std"] + metadata["target_mean"],
        table["rp"],
        rtol=1e-6,
    )


def test_default_training_budget_and_validation_controls():
    params = network_params()
    assert params["learning"]["epochs"] >= 100
    assert params["learning"]["batch_size"] >= 32
    assert params["callbacks"]["early_stopping"]
    assert params["callbacks"]["early_stopping_patience"] >= 10
    assert params["callbacks"]["reduce_lr"]


@pytest.mark.parametrize("notebook", ["train_predict", "fine_tune"])
def test_notebooks_execute_and_save_reusable_checkpoints(tmp_path, notebook, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    fit = RetentionModel.fit
    fit_options = []

    def record_fit(model, *args, **kwargs):
        fit_options.append(kwargs)
        return fit(model, *args, **kwargs)

    monkeypatch.setattr(RetentionModel, "fit", record_fit)
    app = runpy.run_path(str(ROOT / f"examples/{notebook}.py"))["app"]
    budgets = (
        {"epochs": 2}
        if notebook == "train_predict"
        else {"base_epochs": 2, "fine_tuning_epochs": 2}
    )
    _, definitions = app.run(defs={"output": tmp_path, "batch_size": 128, **budgets})
    assert len(fit_options) == (1 if notebook == "train_predict" else 5)
    assert all(options["callbacks"]["early_stopping"] for options in fit_options)
    assert all(options["callbacks"]["reduce_lr"] for options in fit_options)
    assert all(options["epochs"] == 2 for options in fit_options)
    metadata = json.loads((tmp_path / "preprocessing.json").read_text())
    table, _, _, _ = prepare_data()
    test_rows = table.loc[table["split"] == "test"]
    inputs = encode_pairs(test_rows, metadata)
    strategies = (
        ["model"]
        if notebook == "train_predict"
        else ["recurrent_full", "recurrent_frozen", "recurrent_lokr", "recurrent_lora"]
    )
    for strategy in strategies:
        restored = xiRTNET(
            network_params(), metadata["sequence_width"], len(metadata["vocabulary"]) + 1
        )
        restored.load_model(tmp_path / f"{strategy}.pt")
        if strategy != "model":
            assert restored.model.params["learning"]["epochs"] == 2
            assert restored.model.params["learning"]["learningrate"] == 0.0005
        prediction = np.asarray(restored.model.predict(inputs)).reshape(-1)
        seconds = prediction * metadata["target_std"] + metadata["target_mean"]
        filename = "test-predictions.csv" if strategy == "model" else f"{strategy}-predictions.csv"
        saved = pd.read_csv(tmp_path / filename)
        assert saved["PSMID"].tolist() == test_rows["PSMID"].tolist()
        np.testing.assert_allclose(seconds, saved["predicted_seconds"], rtol=1e-6)
        assert np.isfinite(seconds).all()
    if notebook == "fine_tune":
        summary = definitions["summary"].set_index("strategy")
        assert set(summary.index) == set(strategies)
        assert np.isfinite(summary["test_rmse_seconds"]).all()
        assert summary["epochs_trained"].between(1, 2).all()
        assert summary["best_epoch"].between(1, summary["epochs_trained"]).all()
        counts = summary["trainable_parameters"]
        assert counts["recurrent_full"] > counts["recurrent_frozen"] > 0
        assert counts["recurrent_lokr"] > counts["recurrent_frozen"]
        assert counts["recurrent_lora"] > counts["recurrent_frozen"]
        pd.testing.assert_frame_equal(
            summary, pd.read_csv(tmp_path / "summary.csv").set_index("strategy")
        )
    else:
        assert np.isfinite(definitions["rmse_seconds"])
        history = pd.read_csv(tmp_path / "history.csv")
        assert len(history) == 2
        assert history["epoch"].tolist() == [1, 2]
        scores = pd.read_csv(tmp_path / "metrics.csv").set_index("model")
        assert set(scores.index) == {"GRU", "Training mean"}
        assert np.isfinite(scores).all().all()
    plots = (
        ["training-loss.png", "test-predictions.png"]
        if notebook == "train_predict"
        else [
            "trainable-parameters.png",
            "test-rmse.png",
            "test-predictions.png",
            "base-training-loss.png",
            "training-loss.png",
        ]
    )
    for filename in plots:
        image = mpimg.imread(tmp_path / filename)
        assert image.ndim == 3
        assert min(image.shape[:2]) >= 300
        assert np.isfinite(image).all()
        assert image.std() > 0


@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        ("examples/train_predict.py", False),
        ("examples/fine_tune.py", False),
        ("examples/data/crosslinks.csv", False),
        ("examples/results/model.pt", True),
        ("examples/results/summary.csv", True),
        ("examples/results/train-predict/test-predictions.png", True),
        ("examples/__marimo__/session.json", True),
        (".python-version", False),
        ("uv.lock", False),
        ("pixi.lock", False),
        ("docs/assets/example.pdf", False),
        ("data/private.csv", True),
        ("local-data/private.csv", True),
        (".env.credentials", True),
        ("mlflow-artifacts/checkpoint.pt", True),
        ("embedding-cache/features.npy", True),
        ("benchmarks/results/local/metrics.csv", True),
        ("benchmark-tf-torch/cohort/results/data/inputs.npy", False),
        ("benchmark-tf-torch/cohort/results/models/model.pt", False),
        ("benchmark-tf-torch/fine-tune-torch/results/metrics.csv", False),
        ("benchmark-tf-torch/fine-tune-torch/runs/cohort/recurrent_full/s17/model.pt", False),
    ],
)
def test_ignore_rules_keep_sources_and_published_evidence(path, ignored):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    assert (result.returncode == 0) is ignored
