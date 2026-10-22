"""One public prediction/QC/explanation contract for native RT and FAIMS."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from xirt import qc
from xirt.__main__ import arg_parser
from xirt.faims import FEATURE_SETS, FAIMSPredictor, default_faims_params
from xirt.predictor import Predictor, preprocess, run_prediction_command
from xirt.xirtnet import xiRTNET

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frame():
    torch.set_num_threads(2)
    return pd.DataFrame(
        {
            "peptide1": ["AK", "CK", "DK", "EK", "FK", "GK", "HK", "IK"],
            "peptide2": ["PEPK"] * 8,
            "link_pos1": [2] * 8,
            "link_pos2": [4] * 8,
            "precursor_charge": [3, 4] * 4,
            "precursor_mz": np.arange(8) * 10 + 700.0,
            "precursor_mass": np.arange(8) * 30 + 2100.0,
            "faims_cv": [-45.0, -50.0, -40.0, -55.0, -45.0, -60.0, -50.0, -40.0],
        }
    )


@pytest.mark.parametrize("features", ["sequence", "charge", "charge-mz", "precursor"])
def test_faims_variants_only_require_their_declared_features(frame, features, tmp_path):
    params = default_faims_params(features=features)
    params["learning"]["epochs"] = 2
    required = ["peptide1", "peptide2", "link_pos1", "link_pos2", *FEATURE_SETS[features]]
    model = FAIMSPredictor(params).fit(frame.iloc[:4], frame.iloc[4:6])
    path = tmp_path / "faims.pt"
    model.save(path)
    predictor = Predictor.load(path)
    minimal = frame[required].iloc[6:]
    prediction = predictor.predict(minimal)
    assert prediction.index.equals(minimal.index)
    np.testing.assert_allclose(
        prediction["faims_cv-prediction"], prediction["faims_cv_prediction_v"]
    )
    np.testing.assert_allclose(
        prediction["faims_cv-prediction"], model.predict(minimal)["faims_cv-prediction"]
    )
    explanation = predictor.explain(minimal, steps=32)
    assert set(explanation.numeric["feature"]) == set(FEATURE_SETS[features])
    np.testing.assert_allclose(
        explanation.summary["prediction"], prediction["faims_cv-prediction"], atol=1e-5
    )
    figure = qc.plot_residue_attributions(explanation)
    assert len(figure.axes) == 2 + bool(FEATURE_SETS[features])
    figure.canvas.draw()
    title = figure.texts[0].get_window_extent(figure.canvas.get_renderer())
    assert title.x0 >= figure.bbox.x0
    assert title.x1 <= figure.bbox.x1
    plt.close(figure)


def test_raw_rt_prediction_matches_existing_preprocessing_and_restores_units(tmp_path):
    frame = pd.read_csv(ROOT / "examples/data/crosslinks.csv").head(8)
    frame["score"] = np.arange(len(frame), dtype=float)
    data = preprocess(frame.copy(), max_length=50, cl_residue=True)
    params = yaml.safe_load((ROOT / "docs/examples/xirt-rp.yaml").read_text())
    params["siamese"]["use"] = True
    params["_alphabet"] = data.le.classes_.tolist()
    params["_sequence_length"] = 50
    params["_provenance"] = {
        "preprocessing": {"cl_residue": True},
        "training": {"sequence_type": "crosslink"},
    }
    params["_target_scales"] = {"rp": 60.0}
    params["target_units"] = {"rp": "s"}
    model = xiRTNET(params, 50, len(data.le.classes_) + 1)
    model.build_model(siamese=True)
    path = tmp_path / "rt.pt"
    model.model.save(path)
    minimal = frame[["Peptide1", "Peptide2", "LinkPos1", "LinkPos2"]]
    predictor = Predictor.load(path)
    expected = model.model.predict(data.get_features(frame.index)).reshape(-1) * 60
    np.testing.assert_allclose(predictor.predict(minimal)["rp-prediction"], expected, atol=1e-5)
    explanation = predictor.explain(minimal.head(1), steps=64)
    assert explanation.unit == "s"
    np.testing.assert_allclose(explanation.summary["prediction"], expected[:1], atol=1e-5)


def test_tutorial_rt_checkpoint_is_self_contained_for_prediction_and_explanation(tmp_path):
    from examples.tutorial_data import encode_pairs, network_params, prepare_data

    frame, _, _, metadata = prepare_data()
    params = network_params()
    params["_preprocessing"] = metadata
    model = xiRTNET(params, metadata["sequence_width"], len(metadata["vocabulary"]) + 1)
    model.build_model(siamese=True)
    path = tmp_path / "rt.pt"
    model.model.save(path)
    inputs = frame.iloc[:3]
    expected = (
        model.model.predict(encode_pairs(inputs, metadata)).reshape(-1) * metadata["target_std"]
        + metadata["target_mean"]
    )
    predictor = Predictor.load(path)
    actual = predictor.predict(inputs.drop(columns=["rp", "fdr", "isTT"]))
    np.testing.assert_allclose(actual["rp-prediction"], expected, atol=1e-4)
    assert predictor.explain(inputs.head(1), steps=16).unit == "seconds"


def test_shared_cli_prediction_qc_and_residue_explanations(frame, tmp_path):
    params = default_faims_params(features="charge")
    params["learning"]["epochs"] = 1
    model = FAIMSPredictor(params).fit(frame.iloc[:4], frame.iloc[4:6])
    path = tmp_path / "faims.pt"
    model.save(path)
    inputs = tmp_path / "input.csv"
    frame.iloc[6:].to_csv(inputs, index=False)
    output, plots = tmp_path / "predictions.csv", tmp_path / "qc"
    args = arg_parser().parse_args(
        [
            "predict",
            "--input",
            str(inputs),
            "--checkpoint",
            str(path),
            "--output",
            str(output),
            "--qc",
            str(plots),
        ]
    )
    run_prediction_command(args)
    assert pd.read_csv(output)["faims_cv-prediction"].notna().all()
    scores = json.loads((plots / "metrics.json").read_text())
    assert scores["unit"] == "V"
    assert {"pearson_r", "spearman_rho", "r2"} <= set(scores)
    assert (plots / "predictions.png").is_file()
    args = arg_parser().parse_args(
        [
            "explain",
            "--input",
            str(inputs),
            "--checkpoint",
            str(path),
            "--output",
            str(tmp_path / "explanations"),
            "--limit",
            "1",
            "--steps",
            "16",
        ]
    )
    run_prediction_command(args)
    assert (args.output / "residues.png").is_file()
    residues = pd.read_csv(args.output / "residues.csv")
    assert set(residues["peptide"]) == {1, 2}
    assert residues["crosslinked"].sum() == 2
    assert len(pd.read_csv(args.output / "summary.csv")) == 1


def test_shared_metrics_distinguish_correlation_and_predictive_r2():
    scores = qc.prediction_metrics([1, 2, 3], [2, 3, 4])
    assert scores["pearson_r"] == pytest.approx(1)
    assert scores["spearman_rho"] == pytest.approx(1)
    assert scores["r2"] == pytest.approx(-0.5)
    for values in ([1.0], [0.1] * 9):
        scores = qc.prediction_metrics(values, values)
        assert scores["pearson_r"] is None
        assert scores["spearman_rho"] is None
        assert scores["r2"] is None
    axis = qc.plot_predictions([-60, -50, -40], [-58, -52, -42], name="FAIMS CV", unit="V")
    assert "Pearson r" in axis.texts[0].get_text()
    assert "Spearman rho" in axis.texts[0].get_text()
    assert axis.get_xlim()[0] < -60
    plt.close(axis.figure)
    with pytest.raises(ValueError, match="aligned"):
        qc.prediction_metrics([1, 2], [1])
