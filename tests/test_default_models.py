"""Bundled defaults must work without training, downloads or checkout-relative paths."""

import hashlib
import json
import re
import zipfile
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from xirt import assets
from xirt.__main__ import arg_parser
from xirt.faims import FAIMSPredictor, run_faims_command
from xirt.predictor import Predictor, run_prediction_command

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def peptides():
    torch.set_num_threads(2)
    return pd.DataFrame(
        {
            "Peptide1": ["PEPTIDEK", "AKLVK"],
            "Peptide2": ["AAKPEPTIDER", "MPEPTIDEK"],
            "LinkPos1": [8, 2],
            "LinkPos2": [3, 9],
        },
        index=[7, 2],
    )


def faims_inputs(peptides):
    frame = peptides.rename(
        columns={
            "Peptide1": "peptide1",
            "Peptide2": "peptide2",
            "LinkPos1": "link_pos1",
            "LinkPos2": "link_pos2",
        }
    )
    frame["precursor_charge"] = [3, 4]
    frame["precursor_mz"] = [700.0, 900.0]
    return frame


@pytest.mark.parametrize("name", assets.MODEL_NAMES)
def test_bundled_models_match_final_training_checksums_and_predict(
    name, peptides, tmp_path, monkeypatch
):
    manifest = json.loads(files("xirt.assets").joinpath("manifest.json").read_text())
    gallery = json.loads((ROOT / "docs/assets/prediction-gallery/manifest.json").read_text())
    assert set(manifest["models"]) == set(assets.MODEL_NAMES)
    record = manifest["models"][name]
    resource = files("xirt.assets").joinpath(record["filename"]).read_bytes()
    assert len(resource) == record["bytes"]
    assert hashlib.sha256(resource).hexdigest() == record["sha256"]
    expected = (
        gallery["faims_checkpoint_sha256"] if name == "faims" else gallery["checkpoints"][name]
    )
    assert record["sha256"] == expected
    monkeypatch.chdir(tmp_path)
    predictor = Predictor.load(model=name)
    inputs = faims_inputs(peptides) if name == "faims" else peptides
    result = predictor.predict(inputs)
    assert result.index.equals(inputs.index)
    assert set(predictor.model.tasks) == set(record["outputs"])
    for task, unit in record["outputs"].items():
        assert predictor.model.target_transform(task)[2] == unit
        assert np.isfinite(result[f"{task}-prediction"]).all()


def test_no_argument_defaults_are_joint_chromatography_and_charge_mz_faims(peptides):
    joint = Predictor.load()
    pd.testing.assert_frame_equal(
        joint.predict(peptides), Predictor.load(model="multitask").predict(peptides)
    )
    cv = FAIMSPredictor.load()
    assert cv.model.numeric_features == ["precursor_charge", "precursor_mz"]
    inputs = faims_inputs(peptides)
    pd.testing.assert_frame_equal(cv.predict(inputs), Predictor.load(model="faims").predict(inputs))


def test_packaged_defaults_reproduce_the_documented_held_out_examples(peptides):
    gallery = json.loads((ROOT / "docs/assets/prediction-gallery/manifest.json").read_text())
    rt = pd.DataFrame(
        {
            "Peptide1": ["KSGVLTGLPDAYGR"],
            "Peptide2": ["LKMQVGPK"],
            "LinkPos1": [1],
            "LinkPos2": [2],
        }
    )
    predictions = Predictor.load().predict(rt)
    for task in ("rp", "scx", "hsax"):
        np.testing.assert_allclose(
            predictions[f"{task}-prediction"], gallery["examples"][task]["prediction"], rtol=1e-6
        )
    cv = pd.DataFrame(
        {
            "peptide1": ["AVEKLSVEALNSLTGEFK"],
            "peptide2": ["LNYKSEEEYPDLSK"],
            "link_pos1": [4],
            "link_pos2": [4],
            "precursor_charge": [3],
            "precursor_mz": [1263.111461],
        }
    )
    np.testing.assert_allclose(
        FAIMSPredictor.load().predict(cv)["faims_cv-prediction"],
        gallery["examples"]["faims_cv"]["prediction"],
        rtol=1e-6,
    )


@pytest.mark.parametrize("name", ["", "../rp", "unknown"])
def test_unknown_models_and_ambiguous_sources_fail_explicitly(name):
    with pytest.raises(ValueError, match="Unknown bundled model"):
        Predictor.load(model=name)
    with pytest.raises(ValueError, match="not both"):
        Predictor.load("custom.pt", model="rp")


def test_missing_default_asset_is_not_silently_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "files", lambda _anchor: tmp_path)
    with pytest.raises(FileNotFoundError):
        Predictor.load()


@pytest.mark.parametrize("name", ["multitask", "faims"])
def test_zip_resources_remain_alive_for_complete_loading(name, peptides, tmp_path, monkeypatch):
    archive_path = tmp_path / "resources.zip"
    data = files("xirt.assets").joinpath(f"{name}.pt").read_bytes()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"assets/{name}.pt", data)
    extracted = []
    original_as_file = assets.as_file

    @contextmanager
    def record_extraction(resource):
        with original_as_file(resource) as path:
            extracted.append(path)
            assert path.is_file()
            yield path

    with zipfile.ZipFile(archive_path) as archive:
        monkeypatch.setattr(assets, "files", lambda _anchor: zipfile.Path(archive, "assets/"))
        monkeypatch.setattr(assets, "as_file", record_extraction)
        predictor = Predictor.load(model=name)
    assert extracted and all(not path.exists() for path in extracted)
    inputs = faims_inputs(peptides) if name == "faims" else peptides
    assert np.isfinite(predictor.predict(inputs).to_numpy()).all()


@pytest.mark.parametrize("name", assets.MODEL_NAMES)
def test_cli_named_models_do_not_require_checkpoint_paths(name, peptides, tmp_path):
    frame = faims_inputs(peptides) if name == "faims" else peptides
    source, output = tmp_path / "inputs.csv", tmp_path / "predictions.csv"
    frame.to_csv(source, index=False)
    args = arg_parser().parse_args(
        ["predict", "--model", name, "--input", str(source), "--output", str(output)]
    )
    run_prediction_command(args)
    expected = Predictor.load(model=name).predict(frame.reset_index(drop=True))
    actual = pd.read_csv(output)[expected.columns]
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_cli_default_prediction_and_explanation(peptides, tmp_path):
    source, output = tmp_path / "inputs.csv", tmp_path / "predictions.csv"
    peptides.to_csv(source, index=False)
    parser = arg_parser()
    args = parser.parse_args(["predict", "--input", str(source), "--output", str(output)])
    run_prediction_command(args)
    assert {"rp-prediction", "scx-prediction", "hsax-prediction"} <= set(pd.read_csv(output))
    args = parser.parse_args(
        [
            "explain",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "explanation"),
            "--task",
            "scx",
            "--limit",
            "1",
            "--steps",
            "16",
        ]
    )
    run_prediction_command(args)
    details = json.loads((args.output / "explanation.json").read_text())
    assert details["task"] == "scx" and details["unit"] == "fraction"
    assert (args.output / "residues.png").is_file()


@pytest.mark.parametrize("custom", [False, True])
def test_compatibility_faims_cli_defaults_and_custom_paths(peptides, tmp_path, custom):
    frame = faims_inputs(peptides)
    source, output = tmp_path / "inputs.csv", tmp_path / "predictions.csv"
    frame.to_csv(source, index=False)
    arguments = ["faims", "predict", "--input", str(source), "--output", str(output)]
    with assets.checkpoint_path("faims") as path:
        if custom:
            arguments += ["--checkpoint", str(path)]
        args = arg_parser().parse_args(arguments)
        run_faims_command(args)
    actual = pd.read_csv(output)["faims_cv-prediction"]
    np.testing.assert_allclose(
        actual, FAIMSPredictor.load().predict(frame)["faims_cv-prediction"], rtol=1e-6
    )


@pytest.mark.parametrize("command", ["predict", "explain"])
def test_cli_model_and_checkpoint_are_mutually_exclusive(command):
    with pytest.raises(SystemExit) as error:
        arg_parser().parse_args(
            [
                command,
                "--input",
                "inputs.csv",
                "--output",
                "outputs",
                "--model",
                "rp",
                "--checkpoint",
                "custom.pt",
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("document", ["docs/index.md", "README.md"])
def test_landing_page_api_examples_execute_verbatim(document, peptides):
    text = (ROOT / document).read_text()
    match = re.search(r"## Python API.*?```python\n(.*?)```", text, re.DOTALL)
    assert match is not None
    namespace = {}
    exec(compile(match.group(1), document, "exec"), namespace)
    predictions = namespace["predictions"]
    explanation = namespace["explanation"]
    assert predictions.shape == (2, 4)
    assert np.isfinite(predictions.to_numpy()).all()
    assert explanation.unit == "s"
    assert explanation.residues["crosslinked"].sum() == 2
    np.testing.assert_allclose(
        explanation.summary["prediction"], predictions["rp-prediction"].iloc[:1], rtol=1e-6
    )
