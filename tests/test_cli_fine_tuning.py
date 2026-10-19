"""Fine-tuning through the production runner preserves token and target semantics."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from xirt.__main__ import xirt_runner
from xirt.xirtnet import read_checkpoint

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fine-tuning-base")
    params = yaml.safe_load((FIXTURES / "xirt_params_rp.yaml").read_text())
    learning = yaml.safe_load((FIXTURES / "learning_params_training_nocv.yaml").read_text())
    params["LSTM"].update(units=4, nlayers=1)
    params["embedding"]["length"] = 3
    params["learning"].update(epochs=1, batch_size=32, verbose=0)
    params["callbacks"].update(check_point=False, log_csv=False, tensor_board=False)
    learning["train"]["refit"] = False
    result = xirt_runner(
        str(FIXTURES / "DSS_xisearch_fdr_CSM50percent.csv"),
        str(directory),
        params,
        learning,
        nrows=150,
        perform_qc=False,
        write=False,
    )
    return params, learning, result, directory / "xirt_model.pt"


@pytest.mark.parametrize(
    "strategy",
    [
        "recurrent_full",
        "recurrent_frozen",
        "recurrent_lokr",
        "recurrent_lora",
    ],
)
def test_runner_fine_tunes_from_yaml_and_reuses_preprocessing(trained, tmp_path, strategy):
    params, learning, original, checkpoint = copy.deepcopy(trained)
    params["fine_tuning"] = {"checkpoint": str(checkpoint), "strategy": strategy}
    subset = original.psms.iloc[:80].copy()
    subset["rp"] = 200.0
    result = xirt_runner(
        "",
        str(tmp_path),
        params,
        learning,
        api_df=subset,
        perform_qc=False,
        write=False,
    )
    saved = read_checkpoint(tmp_path / "xirt_model.pt")
    base = read_checkpoint(checkpoint)
    assert saved["params"]["fine_tuning"]["strategy"] == strategy
    assert saved["params"]["_alphabet"] == base["params"]["_alphabet"]
    assert saved["params"]["_sequence_length"] == base["params"]["_sequence_length"]
    assert saved["params"]["_target_scales"] == base["params"]["_target_scales"]
    assert (
        saved["params"]["_provenance"]["input_sha256"]
        != base["params"]["_provenance"]["input_sha256"]
    )
    np.testing.assert_allclose(result.psms["rp"], 200 / 60)
    assert result.prediction_df["rp-prediction"].notna().all()


def test_runner_rejects_conflicting_checkpoints(trained, tmp_path):
    params, learning, result, checkpoint = copy.deepcopy(trained)
    params["fine_tuning"] = {"checkpoint": str(checkpoint), "strategy": "recurrent_frozen"}
    learning["train"]["pretrained_model"] = str(checkpoint)
    with pytest.raises(ValueError, match="not both"):
        xirt_runner("", str(tmp_path), params, learning, api_df=result.psms, perform_qc=False)


def test_refit_fine_tuning_uses_current_run_provenance(trained, tmp_path):
    params, learning, result, checkpoint = copy.deepcopy(trained)
    params["fine_tuning"] = {"checkpoint": str(checkpoint), "strategy": "recurrent_frozen"}
    params["tracking"] = {"enabled": False, "experiment": "new-fine-tuning"}
    learning["train"]["refit"] = True
    subset = result.psms.iloc[:80].copy()
    subset["rp"] *= 60
    xirt_runner("", str(tmp_path), params, learning, api_df=subset, perform_qc=False, write=False)
    saved = read_checkpoint(tmp_path / "xirt_model.pt")["params"]
    base = read_checkpoint(checkpoint)["params"]
    assert saved["_provenance"]["input_sha256"] != base["_provenance"]["input_sha256"]
    assert saved["_split"]["test"] == []
    assert len(saved["_split"]["train"]) < len(base["_split"]["train"])
    assert saved["tracking"] == params["tracking"]
    assert saved["fine_tuning"]["strategy"] == "recurrent_frozen"


def test_prediction_does_not_construct_optional_training_optimizer(trained, tmp_path, monkeypatch):
    params, learning, result, checkpoint = copy.deepcopy(trained)
    data = read_checkpoint(checkpoint)
    data["params"]["learning"]["optimizer"] = "lion"
    lion = tmp_path / "lion.pt"
    torch.save(data, lion)
    learning["train"].update(mode="predict", pretrained_model=str(lion))
    subset = result.psms.iloc[:15].copy()
    subset["rp"] *= 60

    def no_optimizer(*args, **kwargs):
        raise AssertionError("Inference must not construct a training optimizer.")

    monkeypatch.setattr("xirt.xirtnet.make_optimizers", no_optimizer)
    predicted = xirt_runner(
        "",
        str(tmp_path / "predictions"),
        params,
        learning,
        api_df=subset,
        perform_qc=False,
        write=False,
    )
    assert predicted.prediction_df["rp-prediction"].notna().all()
