"""FAIMS targets, pair isolation, train-only preprocessing and native inference."""

import argparse

import numpy as np
import pandas as pd
import pytest
import torch

from xirt.faims import (
    FAIMSPredictor,
    cv_metrics,
    default_faims_params,
    parse_compensation_voltage,
    precursor_descriptors,
    precursor_inputs,
    prepare_faims_observations,
    run_faims_command,
    split_precursors,
)


@pytest.fixture
def raw():
    return pd.DataFrame(
        {
            "PepSeq1": ["AK", "KK", "CK", "DK", "EK", "FK", "GK", "HK"] * 2,
            "PepSeq2": ["PEPK"] * 16,
            "LinkPos1": [2] * 16,
            "LinkPos2": [4] * 16,
            "exp charge": [3] * 8 + [4] * 8,
            "exp m/z": [800.0] * 16,
            "exp mass": [2400.0] * 16,
            "isTT": [True] * 16,
            "fdr": [0.001] * 16,
            "run": ["test_CV045_rep01"] * 8 + ["8PM_DSS_A_55_rep1"] * 8,
        }
    )


@pytest.fixture
def partitions(raw):
    prepared, _ = prepare_faims_observations(raw, "test")
    return split_precursors(prepared)


@pytest.fixture
def params():
    torch.set_num_threads(2)
    config = default_faims_params()
    config["learning"]["epochs"] = 2
    return config


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        ("prefix_CV045_rep01", -45),
        ("CV-55", -55),
        ("CV+25", 25),
        ("8PM_DSS_A_50_rep1", -50),
        ("CV0", 0),
    ],
)
def test_explicit_cv_parsing(run, expected):
    assert parse_compensation_voltage(run) == expected


@pytest.mark.parametrize(
    "run", ["B200412_09_sample_40_rep01", "sample_CV40_CV45", "sample", None, "CVNaN"]
)
def test_ambiguous_or_unlabelled_runs_raise(run):
    with pytest.raises(ValueError, match="CV|voltage"):
        parse_compensation_voltage(run)


def test_prepare_filters_and_canonicalizes_repeated_pairs(raw):
    swapped = raw.iloc[[0]].copy()
    swapped[["PepSeq1", "PepSeq2"]] = swapped[["PepSeq2", "PepSeq1"]].to_numpy()
    swapped[["LinkPos1", "LinkPos2"]] = swapped[["LinkPos2", "LinkPos1"]].to_numpy()
    swapped["run"] = "CV055"
    excluded = raw.iloc[[0, 0]].copy()
    excluded["isTT"] = [False, True]
    excluded["fdr"] = [0, 0.5]
    frame, audit = prepare_faims_observations(pd.concat([raw, swapped, excluded]), "sample")
    first = frame.loc[(frame["peptide1"] == "AK") & (frame["precursor_charge"] == 3)].iloc[0]
    assert first["faims_cv"] == -50
    assert first["observation_count"] == 2
    assert first["cv_spread_v"] == 10
    assert audit["removed_observations"] == 2
    assert audit["retained_observations"] == 17
    assert not frame["precursor_id"].duplicated().any()
    other, _ = prepare_faims_observations(raw.sample(frac=1, random_state=4), "sample")
    assert set(other["precursor_id"]) == set(frame["precursor_id"])


def test_explicit_target_and_strict_filter_labels(raw):
    raw["voltage"] = -42.0
    frame, _ = prepare_faims_observations(raw.drop(columns="run"), "test", cv_column="voltage")
    assert (frame["faims_cv"] == -42).all()
    raw["isTT"] = "False"
    with pytest.raises(ValueError, match="boolean"):
        prepare_faims_observations(raw, "test")


def test_charge_only_observations_need_no_mass_or_mz(raw):
    prepared, audit = prepare_faims_observations(
        raw.drop(columns=["exp mass", "exp m/z"]), "charge-only"
    )
    assert "precursor_mass" not in prepared
    assert "precursor_mz" not in prepared
    assert audit["retained_observations"] == len(raw)
    assert set(split_precursors(prepared)["split"]) == {"train", "validation", "test"}


def test_group_split_is_stable_and_isolates_modification_charge_and_sites(raw):
    modified = raw.iloc[[0]].copy()
    modified["PepSeq1"] = "AKox"
    modified["LinkPos1"] = 1
    data, _ = prepare_faims_observations(pd.concat([raw, modified]), "test")
    split = split_precursors(data)
    assert split.groupby("pair_group")["split"].nunique().max() == 1
    shuffled = split_precursors(data.sample(frac=1, random_state=3))
    assert dict(zip(split["precursor_id"], split["split"])) == dict(
        zip(shuffled["precursor_id"], shuffled["split"])
    )
    assert set(split["split"]) == {"train", "validation", "test"}
    with pytest.raises(ValueError, match="Too few"):
        split_precursors(data.iloc[:1])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("exp charge", 2.5, "integers"),
        ("exp mass", np.nan, "finite"),
        ("LinkPos1", 0, "positive"),
        ("LinkPos2", 5, "exceeds"),
        ("PepSeq1", "acAK", "N-terminal"),
        ("PepSeq1", "[ac]AK", "N-terminal"),
        ("PepSeq1", "(ac)AK", "N-terminal"),
        ("PepSeq1", "[+42]AK", "N-terminal"),
        ("PepSeq1", "AX", "unsupported"),
        ("PepSeq1", "R.AK.R", "bare"),
    ],
)
def test_invalid_inputs_fail_explicitly(raw, column, value, message):
    raw[column] = value
    with pytest.raises(ValueError, match=message):
        precursor_inputs(raw)


def test_training_and_checkpoint_roundtrip(partitions, params, tmp_path):
    train = partitions.loc[partitions["split"] == "train"].copy()
    validation = partitions.loc[partitions["split"] == "validation"].copy()
    train["faims_cv"] = np.arange(len(train)) - 70.0
    validation["faims_cv"] = -500.0
    predictor = FAIMSPredictor(params).fit(train, validation)
    assert predictor.metadata["target_mean"] == train["faims_cv"].mean()
    np.testing.assert_allclose(
        predictor.model.numeric_mean.numpy(), train[params["numeric_features"]].mean()
    )
    minimal = validation.drop(columns=["faims_cv", "source", "pair_group"])
    prediction = predictor.predict(minimal)
    assert prediction.index.equals(validation.index)
    assert prediction["unknown_token_count"].eq(0).all()
    path = tmp_path / "faims.pt"
    predictor.save(path)
    restored = FAIMSPredictor.load(path)
    pd.testing.assert_frame_equal(prediction, restored.predict(minimal))
    swapped = minimal.copy()
    swapped[["peptide1", "peptide2"]] = minimal[["peptide2", "peptide1"]].to_numpy()
    swapped[["link_pos1", "link_pos2"]] = minimal[["link_pos2", "link_pos1"]].to_numpy()
    pd.testing.assert_frame_equal(prediction, restored.predict(swapped))
    unknown = minimal.copy()
    unknown["peptide1"] = unknown["peptide1"] + "ox"
    assert (restored.predict(unknown)["unknown_token_count"] > 0).all()
    assert all("ox" not in token for token in predictor.metadata["vocabulary"])
    with pytest.raises(ValueError, match="disjoint"):
        FAIMSPredictor(params).fit(train, train)


def test_sequence_only_needs_no_numeric_context(partitions, params):
    params["numeric_features"] = []
    frame = partitions.drop(columns=["precursor_charge", "precursor_mz", "precursor_mass"])
    model = FAIMSPredictor(params).fit(
        frame.loc[frame["split"] == "train"], frame.loc[frame["split"] == "validation"]
    )
    assert np.isfinite(model.predict(frame)["faims_cv_prediction_v"]).all()
    with pytest.raises(ValueError, match="no truncation"):
        FAIMSPredictor(params, maximum_residues=1).fit(
            frame.loc[frame["split"] == "train"], frame.loc[frame["split"] == "validation"]
        )


def test_descriptors_and_metrics(partitions):
    descriptors = precursor_descriptors(partitions)
    assert "faims_cv" not in descriptors
    np.testing.assert_allclose(
        descriptors.filter(like="fraction_").sum(axis=1), np.ones(len(descriptors))
    )
    assert cv_metrics([-50, -40], [-50, -40]) == {
        "n": 2,
        "rmse_v": 0,
        "mae_v": 0,
        "r2": 1,
    }
    assert cv_metrics([-50], [-40])["r2"] is None
    assert cv_metrics([-50, -50], [-40, -45])["r2"] is None
    with pytest.raises(ValueError, match="aligned"):
        cv_metrics([1, 2], [1])


def test_cli_training_and_label_free_prediction(raw, tmp_path):
    from xirt.__main__ import arg_parser

    path = tmp_path / "source.csv"
    raw.to_csv(path, index=False)
    output = tmp_path / "run"
    args = arg_parser().parse_args(
        ["faims", "train", "--input", str(path), "--output", str(output), "--epochs", "1"]
    )
    run_faims_command(args)
    assert (output / "metrics.json").exists()
    assert (output / "history.csv").exists()
    inputs = tmp_path / "unlabelled.csv"
    raw.drop(columns=["run", "isTT", "fdr"]).to_csv(inputs, index=False)
    args = argparse.Namespace(
        faims_action="predict",
        input=inputs,
        checkpoint=output / "faims.pt",
        output=tmp_path / "predicted.csv",
    )
    run_faims_command(args)
    result = pd.read_csv(args.output)
    assert len(result) == len(raw)
    assert np.isfinite(result["faims_cv_prediction_v"]).all()
