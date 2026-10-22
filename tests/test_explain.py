"""Integrated gradients must resolve residues, context, units and completeness."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from xirt.explain import integrated_gradients
from xirt.faims import FAIMSPredictor, default_faims_params
from xirt.predictor import Predictor


class LinearPeptideModel(nn.Module):
    """Analytically checkable embedding/context model with xiRT's public interface."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2, padding_idx=0)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )
        self.route = "recurrent"
        self.siamese = True
        self.tasks = ["rt"]
        self.numeric_features = ["charge"]
        self.register_buffer("numeric_mean", torch.tensor([3.0]))
        self.register_buffer("numeric_fitted", torch.tensor(True))
        self.activity = []
        self.params = {
            "output": {"rt-dimension": 1, "rt-activation": "linear"},
            "_alphabet": ["A", "K", "clK"],
        }

    @property
    def device(self):
        return self.embedding.weight.device

    def target_transform(self, task):
        return 2.0, 10.0, "s"

    def forward(self, first, second=None, *, numeric=None):
        values = []
        for branch in (first, second):
            compact = torch.zeros_like(branch)
            for row in range(len(branch)):
                tokens = branch[row][branch[row] != 0]
                compact[row, : len(tokens)] = tokens
            values.append(self.embedding(compact).sum((1, 2)))
        return {"rt": (values[0] + values[1] + 3 * (numeric[:, 0] - 3) + 5)[:, None]}


def test_exact_residue_numeric_and_original_unit_contributions_preserve_state():
    model = LinearPeptideModel()
    model.embedding.eval()
    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)
    original_gradient = model.embedding.weight.grad.clone()
    inputs = (
        torch.tensor([[0, 1, 3], [2, 1, 0]]),
        torch.tensor([[0, 0, 2], [0, 3, 0]]),
        torch.tensor([[4.0], [2.0]]),
    )
    expected = model(*inputs[:2], numeric=inputs[-1])["rt"].detach().numpy().ravel() * 2 + 10
    result = integrated_gradients(model, inputs, task="rt", index=pd.Index(["a", "b"]), steps=8)
    np.testing.assert_allclose(result.summary["prediction"], expected)
    np.testing.assert_allclose(result.summary["baseline_prediction"], [20, 20])
    np.testing.assert_allclose(result.summary["completeness_error"], 0, atol=1e-5)
    np.testing.assert_allclose(result.numeric["attribution"], [6, -6])
    assert result.summary.index.tolist() == ["a", "b"]
    assert result.residues["position"].tolist() == [1, 2, 1, 2, 1, 1]
    assert result.residues["crosslinked"].sum() == 2
    assert result.unit == "s"
    assert model.training and not model.embedding.training
    np.testing.assert_allclose(
        model(*inputs[:2], numeric=inputs[-1])["rt"].detach().numpy().ravel() * 2 + 10,
        expected,
    )
    torch.testing.assert_close(model.embedding.weight.grad, original_gradient)
    assert result.residues.groupby("sample")["attribution"].sum().tolist() == [42, 42]


@pytest.fixture
def faims(tmp_path):
    torch.set_num_threads(2)
    frame = pd.DataFrame(
        {
            "peptide1": ["AK", "CK", "DK", "EK", "FK", "GK", "HK", "IK"],
            "peptide2": ["PEPK"] * 8,
            "link_pos1": [2] * 8,
            "link_pos2": [4] * 8,
            "precursor_charge": [3, 4, 3, 4, 3, 4, 3, 4],
            "precursor_mz": np.arange(8) * 10 + 700.0,
            "faims_cv": [-45.0, -50.0, -40.0, -55.0, -45.0, -60.0, -50.0, -40.0],
        }
    )
    params = default_faims_params(features="charge-mz")
    params["learning"]["epochs"] = 2
    model = FAIMSPredictor(params).fit(frame.iloc[:4], frame.iloc[4:6])
    path = tmp_path / "model.pt"
    model.save(path)
    return model, Predictor.load(path), frame.iloc[6:].copy()


def test_native_gru_explanations_match_predictions_and_original_peptide_order(faims):
    faims_model, predictor, frame = faims
    result = predictor.explain(frame, steps=128)
    np.testing.assert_allclose(
        result.summary["prediction"], predictor.predict(frame)["faims_cv-prediction"], atol=1e-5
    )
    assert np.abs(result.summary["completeness_error"]).max() < 0.1
    assert set(result.numeric["feature"]) == {"precursor_charge", "precursor_mz"}
    assert result.residues.loc[result.residues["kind"] == "terminus"].shape[0] == 8
    first = result.residues.query("sample == 0 and peptide == 1 and kind == 'residue'")
    assert first["token"].tolist() == ["H", "clK"]
    np.testing.assert_allclose(
        faims_model.explain(frame, steps=128).residues["attribution"],
        result.residues["attribution"],
        atol=1e-5,
    )
    swapped = frame.copy()
    swapped[["peptide1", "peptide2"]] = frame[["peptide2", "peptide1"]].to_numpy()
    swapped[["link_pos1", "link_pos2"]] = frame[["link_pos2", "link_pos1"]].to_numpy()
    other = predictor.explain(swapped, steps=128)
    np.testing.assert_allclose(other.summary["prediction"], result.summary["prediction"], atol=1e-5)
    np.testing.assert_allclose(
        other.residues.query("peptide == 2")["attribution"],
        result.residues.query("peptide == 1")["attribution"],
        atol=1e-5,
    )


def test_explanation_validation_and_no_grad_context(faims):
    _, predictor, frame = faims
    with torch.inference_mode():
        result = predictor.explain(frame.head(1), steps=16)
    assert np.isfinite(result.summary.to_numpy()).all()
    for steps in (0, 1, 513, True):
        with pytest.raises(ValueError, match="steps"):
            predictor.explain(frame, steps=steps)
    with pytest.raises(ValueError, match="Unknown"):
        predictor.explain(frame, task="not_a_task")
