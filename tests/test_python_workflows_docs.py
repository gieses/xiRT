"""Execute the copyable documentation blocks, without replacing data or training budgets."""

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from xirt.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/python-workflows.md"
BLOCKS = {
    match["title"]: textwrap.dedent(match["code"])
    for match in re.finditer(
        r'^(?P<indent> *)```python title="(?P<title>[^"]+)"\n'
        r"(?P<code>.*?)^(?P=indent)```",
        GUIDE.read_text(),
        re.MULTILINE | re.DOTALL,
    )
}
RECIPES = {
    "Train RP": ("rp_model", {"rp"}, "rp.pt", "trained-rp-residues.png"),
    "Train RP and SCX": (
        "joint_model",
        {"rp", "scx"},
        "rp-scx.pt",
        "trained-scx-residues.png",
    ),
    "Bundled RP inference": ("predictor", {"rp"}, None, "bundled-rp-residues.png"),
    "Bundled RP and SCX inference": (
        "predictor",
        {"rp", "scx", "hsax"},
        None,
        "bundled-scx-residues.png",
    ),
    "Train and predict FAIMS": (
        "cv_model",
        {"faims_cv"},
        "faims.pt",
        "trained-faims-residues.png",
    ),
}


def execute(title, namespace):
    """Run the actual code a reader copies, including full 25-epoch ceilings and XAI."""
    exec(compile(BLOCKS[title], f"{GUIDE}:{title}", "exec"), namespace)


@pytest.mark.parametrize("recipe", RECIPES)
def test_five_python_workflows_execute_verbatim(recipe, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    try:
        execute("Shared setup", namespace)
        if recipe.startswith("Train"):
            execute("Mock training data", namespace)
            table = namespace["table"]
            pairs = table.apply(
                lambda row: "|".join(sorted((row["Peptide1"], row["Peptide2"]))), axis=1
            )
            assert pairs.is_unique
            assert [len(namespace[name]) for name in ("train", "validation", "held_out")] == [
                8,
                2,
                2,
            ]
            assert set(namespace["inputs"]) == {"Peptide1", "Peptide2", "LinkPos1", "LinkPos2"}
        if recipe in ("Train RP", "Train RP and SCX"):
            execute("RT training helper", namespace)
        execute(recipe, namespace)
        variable, tasks, filename, image = RECIPES[recipe]
        predictor = namespace[variable]
        assert set(predictor.model.tasks) == tasks
        if recipe == "Train and predict FAIMS":
            inputs = namespace["cv_inputs"]
            assert "faims_cv" not in inputs
            assert predictor.faims.metadata["training_rows"] == 8
            assert predictor.faims.metadata["validation_rows"] == 2
            assert 0 < len(namespace["fitted"].history["loss"]) <= 25
        else:
            inputs = namespace["inputs"] if filename else namespace["precursors"]
        predicted = predictor.predict(inputs)
        assert predicted.index.equals(inputs.index)
        assert np.isfinite(predicted.to_numpy()).all()
        for task in tasks:
            scale, offset, unit = predictor.model.target_transform(task)
            assert unit == {"rp": "s", "faims_cv": "V"}.get(task, "fraction")
            if filename and task != "faims_cv":
                train = namespace["train"][task]
                assert offset == pytest.approx(train.mean())
                assert scale == pytest.approx(train.std(ddof=0))
                assert predictor.model.params["LSTM"]["units"] == 16
        if filename:
            path = namespace["output"] / filename
            assert path.is_file()
            pd.testing.assert_frame_equal(Predictor.load(path).predict(inputs), predicted)
        figure = namespace["output"] / image
        assert figure.is_file()
        pixels = plt.imread(figure)
        assert np.isfinite(pixels).all() and pixels.std() > 0.01
    finally:
        plt.close("all")


def test_python_workflows_is_visible_under_both_navigation_sections():
    nav = yaml.safe_load((ROOT / "mkdocs.yml").read_text())["nav"]
    sections = {name: pages for entry in nav for name, pages in entry.items()}
    for name in ("Get started", "User guide"):
        assert {"Python workflows": "python-workflows.md"} in sections[name]


def test_bundled_faims_example_runs_without_any_training_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    execute("Shared setup", namespace)
    execute("Bundled FAIMS inference", namespace)
    assert set(namespace["cv_precursors"]) == {
        "peptide1",
        "peptide2",
        "link_pos1",
        "link_pos2",
        "precursor_charge",
        "precursor_mz",
    }
    assert np.isfinite(namespace["bundled_cv_predictions"].to_numpy()).all()
