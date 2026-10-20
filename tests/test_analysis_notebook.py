"""Exercise shared QC and both residue-level notebook tasks on small fits."""

import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from examples.tutorial_data import ROOT


@pytest.fixture(scope="module")
def tutorial_results(tmp_path_factory):
    result_root = tmp_path_factory.mktemp("analysis-inputs")
    import sys

    sys.path.insert(0, str(ROOT / "examples"))
    try:
        for notebook, directory in (("train_predict", "train-predict"), ("faims", "faims")):
            output = result_root / directory
            output.mkdir()
            app = runpy.run_path(str(ROOT / f"examples/{notebook}.py"))["app"]
            settings = (
                {"batch_size": 128} if notebook == "train_predict" else {"feature_set": "charge"}
            )
            app.run(defs={"output": output, "epochs": 2, **settings})
    finally:
        sys.path.remove(str(ROOT / "examples"))
    return result_root


@pytest.mark.parametrize("task", ["RT", "FAIMS"])
def test_analysis_notebook_resolves_residues_and_units(tutorial_results, tmp_path, task):
    app = runpy.run_path(str(ROOT / "examples/analyze_predictions.py"))["app"]
    _, definitions = app.run(
        defs={
            "result_root": tutorial_results,
            "analysis_output": tmp_path,
            "task_picker": SimpleNamespace(value=task),
            "sample_picker": SimpleNamespace(value=0),
        }
    )
    explanation = definitions["explanation"]
    assert explanation.unit == ("seconds" if task == "RT" else "V")
    assert set(explanation.residues["peptide"]) == {1, 2}
    assert explanation.residues["crosslinked"].sum() == 2
    assert np.isfinite(explanation.residues["attribution"]).all()
    assert np.isfinite(explanation.summary["completeness_error"]).all()
    assert explanation.numeric.empty == (task == "RT")
    assert set(definitions["score_table"]["task"]) == {"RT", "FAIMS"}
    assert (tmp_path / f"{task.lower()}-0-residues.png").stat().st_size > 1000


def test_analysis_notebook_reports_missing_training(tmp_path):
    app = runpy.run_path(str(ROOT / "examples/analyze_predictions.py"))["app"]
    with pytest.raises(FileNotFoundError, match="Run examples/train_predict.py first"):
        app.run(defs={"result_root": tmp_path, "analysis_output": tmp_path})
