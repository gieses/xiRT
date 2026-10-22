"""Execute the bundled FAIMS tutorial without network access."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from xirt.faims import split_precursors
from xirt.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_faims_pairs_and_provenance():
    path = ROOT / "examples/data/faims.csv"
    provenance = json.loads((ROOT / "examples/data/faims-provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["sha256"]
    table = split_precursors(pd.read_csv(path), seed=42)
    assert len(table) == 915 and table["pair_group"].nunique() == 256
    assert table.groupby("pair_group")["split"].nunique().eq(1).all()
    assert set(table["split"]) == {"train", "validation", "test"}
    assert "precursor_mass" not in table


def test_faims_training_notebook_saves_shared_checkpoint(tmp_path):
    app = runpy.run_path(str(ROOT / "examples/faims.py"))["app"]
    _, definitions = app.run(defs={"output": tmp_path, "epochs": 2, "feature_set": "charge"})
    fitted = definitions["fitted"]
    assert fitted.params["numeric_features"] == ["precursor_charge"]
    assert len(fitted.history["loss"]) == 2
    assert fitted.params["callbacks"]["early_stopping"]
    assert fitted.params["callbacks"]["reduce_lr"]
    test = pd.read_csv(tmp_path / "test-predictions.csv")
    model = Predictor.load(tmp_path / "model.pt")
    np.testing.assert_allclose(
        model.predict(test)["faims_cv-prediction"], test["faims_cv-prediction"], rtol=1e-6
    )
    assert set(definitions["metrics"]["model"]) == {"xiRT", "Training mean"}
    for filename in ("training-loss.png", "test-predictions.png", "history.csv"):
        assert (tmp_path / filename).stat().st_size > 100
