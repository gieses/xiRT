"""Full-cohort training example preserves groups, native units and reusable checkpoints."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from examples.train_chromatography import MODELS, fit_model, prepare_table
from xirt.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]


def observations():
    return pd.DataFrame(
        {
            "PSMID": range(40),
            "Peptide1": ["A" * (i + 1) + "K" for i in range(40)],
            "Peptide2": ["AK"] * 40,
            "LinkPos1": np.arange(40) + 2,
            "LinkPos2": [2] * 40,
            "score": np.arange(40, dtype=float),
            "isTT": [True] * 40,
            "fdr": [0.001] * 40,
            "rp": np.arange(40) * 20.0 + 500,
            "scx": np.arange(40) % 9 + 17,
            "hsax": np.arange(40) % 10 + 1,
        }
    )


def test_grouped_splits_keep_all_csms_and_do_not_separate_modifications_or_sites():
    frame = observations()
    extra = frame.iloc[[0]].copy()
    extra["PSMID"] = 100
    extra["LinkPos1"] = 1
    frame = pd.concat([frame, extra], ignore_index=True)
    result = prepare_table(frame)
    assert len(result) == len(frame)
    assert result.groupby("pair_group")["split"].nunique().eq(1).all()
    assert result.loc[result["PSMID"].isin([0, 100]), "split"].nunique() == 1
    assert result["pair_group"].nunique() == 40
    pd.testing.assert_frame_equal(result, prepare_table(frame))
    frame.loc[0, "rp"] = np.inf
    with pytest.raises(ValueError, match="Nonfinite"):
        prepare_table(frame)


@pytest.mark.parametrize("name", MODELS)
def test_single_and_multitask_checkpoints_predict_and_explain_original_units(tmp_path, name):
    torch.set_num_threads(2)
    table = prepare_table(observations())
    config = copy.deepcopy(
        yaml.safe_load((ROOT / "examples/chromatography_models.yaml").read_text())
    )
    config["network"]["LSTM"].update(units=4, nlayers=1, bidirectional=False)
    config["network"]["dense"]["neurons"] = [6, 4]
    config["network"]["embedding"]["length"] = 3
    record = fit_model(table, MODELS[name], config, tmp_path, epochs=2)
    predictor = Predictor.load(tmp_path / "model.pt")
    test = table.loc[table["split"] == "test"]
    predicted = predictor.predict(test)
    saved = pd.read_csv(tmp_path / "predictions.csv")
    train = table.loc[table["split"] == "train"]
    assert record["epochs"] == 2
    assert record["checkpoint_bytes"] == (tmp_path / "model.pt").stat().st_size
    assert set(record["tasks"]) == set(MODELS[name])
    for task in MODELS[name]:
        transform = record["target_transforms"][task]
        assert transform["offset"] == pytest.approx(train[task].mean())
        assert transform["scale"] == pytest.approx(train[task].std(ddof=0))
        np.testing.assert_allclose(
            predicted[f"{task}-prediction"],
            saved.loc[saved["split"] == "test", f"{task}-prediction"],
            rtol=1e-6,
        )
        result = predictor.explain(test.iloc[:1], task=task, steps=8)
        assert result.unit == ("s" if task == "rp" else "fraction")
        assert set(result.residues["peptide"]) == {1, 2}
        assert np.isfinite(result.summary["completeness_error"]).all()
