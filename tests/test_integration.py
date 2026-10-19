"""CLI/API provenance, checkpoint and prediction round-trip regression tests."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from xirt.__main__ import xirt_runner
from xirt.config import NetworkConfig
from xirt.xirtnet import read_checkpoint, xiRTNET

FIXTURES = Path(__file__).parent / "fixtures"


def settings(task="rp"):
    """Load small, complete network/training settings from repository fixtures."""
    with open(FIXTURES / f"xirt_params_{task}.yaml") as stream:
        params = yaml.safe_load(stream)
    with open(FIXTURES / "learning_params_training_nocv.yaml") as stream:
        learning = yaml.safe_load(stream)
    return params, learning


@pytest.mark.parametrize("task", ["rp", "3RT"])
@pytest.mark.parametrize("mode", ["pretrained_weights", "pretrained_model"])
def test_train_predict_roundtrip_and_metadata(tmp_path, task, mode):
    params, learning = settings(task)
    original = copy.deepcopy(params)
    train = xirt_runner(
        str(FIXTURES / "DSS_xisearch_fdr_CSM50percent.csv"),
        str(tmp_path / "train"),
        params,
        learning,
        nrows=500,
        perform_qc=False,
        write=False,
    )
    checkpoint_path = tmp_path / "train/xirt_model.pt"
    data = read_checkpoint(checkpoint_path)
    assert data["params"]["_target_scales"]["rp"] == 60
    assert data["params"]["_alphabet"] == train.le.classes_.tolist()
    assert "_split" in data["params"]
    with open(tmp_path / "train/epoch_history.csv") as stream:
        assert stream.readline().split(",").count("epoch\n") == 1
    learned = xiRTNET(params, data["input_dim"], data["alphabet"])
    learned.load_model(checkpoint_path)
    # A subset can omit observed fraction classes; saved mappings must not shrink.
    subset = train.psms.iloc[:15].copy()
    subset["rp"] *= 60
    learning["train"]["mode"] = "predict"
    learning["train"][mode] = str(checkpoint_path)
    predicted = xirt_runner(
        "", str(tmp_path / "pred"), params, learning, api_df=subset, perform_qc=False, write=False
    )
    assert len(predicted.le.classes_) == len(train.le.classes_)
    expected = learned.model.predict(predicted.get_features(predicted.psms.index))
    outputs = expected if isinstance(expected, list) else [expected]
    rp = outputs[-1].ravel()
    np.testing.assert_allclose(predicted.prediction_df["rp-prediction"], rp)
    assert predicted.prediction_df.filter(regex="-prediction$").notna().all().all()
    assert params == original
    assert json.loads((tmp_path / "pred/configuration.json").read_text())


def test_training_without_checkpoint_callbacks(tmp_path):
    params, learning = settings()
    params["callbacks"].update(check_point=False, log_csv=False)
    result = xirt_runner(
        str(FIXTURES / "DSS_xisearch_fdr_CSM50percent.csv"),
        str(tmp_path),
        params,
        learning,
        nrows=150,
        perform_qc=False,
        write=False,
    )
    assert (tmp_path / "xirt_model.pt").exists()
    assert result.prediction_df["rp-prediction"].notna().all()
    assert set(result.prediction_df["cv"]) == {-1, 0, 1}
    history = pd.read_csv(tmp_path / "epoch_history.csv")
    assert not any(column.startswith("epoch.") for column in history)


def test_generated_default_config_and_version(tmp_path):
    path = tmp_path / "params.yaml"
    subprocess.run([sys.executable, "-m", "xirt", "-p", str(path)], check=True)
    config = NetworkConfig.model_validate(yaml.safe_load(path.read_text()))
    assert config.learning["optimizer"] == "adamw"
    checked_in = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "default_parameters/xirt_params.yaml").read_text()
    )
    assert NetworkConfig.model_validate(checked_in).model_dump() == config.model_dump()
    process = subprocess.run(
        [sys.executable, "-m", "xirt", "--version"], check=True, text=True, capture_output=True
    )
    assert process.stdout.strip().endswith("2.0.0")
    learning_path = tmp_path / "learning.yaml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "xirt",
            "--sample_learning_params",
            str(learning_path),
            "--sample_xirt_params",
            str(path),
        ],
        check=True,
    )
    assert yaml.safe_load(learning_path.read_text())["train"]["mode"] == "train"


def test_qc_pipeline_does_not_mutate_config(tmp_path):
    params, learning = settings("3RT")
    original = copy.deepcopy(params)
    result = xirt_runner(
        str(FIXTURES / "DSS_xisearch_fdr_CSM50percent.csv"),
        str(tmp_path),
        params,
        learning,
        nrows=500,
        perform_qc=True,
        write=False,
    )
    assert result.prediction_df.filter(regex="-prediction$").notna().all().all()
    assert len(list(tmp_path.glob("*.svg"))) >= 6
    assert params == original


def test_embedding_runner_checkpoint_and_lokr(tmp_path, monkeypatch):
    """Verify runner wiring without mistaking a deterministic test encoder for ESM-C."""
    import xirt.embeddings

    class TestEncoder:
        """Return small composition features to test embedding-route orchestration."""

        embedding_dim = 4

        def __init__(self, **kwargs):
            pass

        def encode(self, sequences):
            assert all(set(sequence) <= set("ACDEFGHIKLMNPQRSTVWYX") for sequence in sequences)
            return np.array(
                [[sequence.count(aa) for aa in "AKLM"] for sequence in sequences], dtype=np.float32
            )

    monkeypatch.setattr(xirt.embeddings, "ESMCEncoder", TestEncoder)
    params, learning = settings()
    params["encoder"] = {"route": "esm-c", "model": "esmc_300m", "embedding_dim": 4}
    params["adapter"] = {"rank": 2, "factor": 2, "alpha": 2}
    trained = xirt_runner(
        str(FIXTURES / "DSS_xisearch_fdr_CSM50percent.csv"),
        str(tmp_path / "trained"),
        params,
        learning,
        nrows=100,
        perform_qc=False,
        write=False,
    )
    assert trained.features1.shape[1] == 4
    learning["train"]["mode"] = "predict"
    learning["train"]["pretrained_model"] = str(tmp_path / "trained/xirt_model.pt")
    predicted = xirt_runner(
        str(FIXTURES / "DSS_xisearch_fdr_CSM50percent.csv"),
        str(tmp_path / "predicted"),
        params,
        learning,
        nrows=20,
        perform_qc=False,
        write=False,
    )
    assert predicted.prediction_df["rp-prediction"].notna().all()
