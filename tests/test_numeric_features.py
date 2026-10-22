"""Numeric precursor inputs must survive fitting, checkpoints and branch handling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from xirt.config import NetworkConfig
from xirt.xirtnet import xiRTNET


@pytest.fixture
def params():
    params = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docs/examples/xirt-rp.yaml").read_text()
    )
    params["numeric_features"] = ["charge", "mass"]
    params["callbacks"]["progressbar"] = False
    return params


def build(params, siamese):
    net = xiRTNET(params, 6, 10)
    net.build_model(siamese=siamese)
    net.compile()
    return net


def features(siamese, size=8):
    seq = np.tile(np.array([[0, 0, 1, 2, 3, 4]]), (size, 1))
    numeric = pd.DataFrame({"charge": np.arange(size) + 2.0, "mass": np.arange(size) * 10 + 100.0})
    return (seq, seq.copy(), numeric) if siamese else (seq, numeric)


@pytest.mark.parametrize("siamese", [False, True])
def test_fit_only_normalizes_training_partition(params, siamese):
    model = build(params, siamese).model
    inputs = features(siamese)
    inputs[-1].iloc[-2:] = [10000.0, 20000.0]
    train = inputs[-1].iloc[:-2].to_numpy()
    model.fit(inputs, np.ones((8, 1)), validation_split=0.25, epochs=2, verbose=0)
    np.testing.assert_allclose(model.numeric_mean.numpy(), train.mean(0))
    np.testing.assert_allclose(model.numeric_scale.numpy(), train.std(0), rtol=1e-6)
    assert model.params["_numeric_scaling"]["training_rows"] == 6
    assert np.isfinite(model.predict(inputs)).all()


@pytest.mark.parametrize("siamese", [False, True])
def test_checkpoint_roundtrip_retains_numeric_normalization(params, siamese, tmp_path):
    net = build(params, siamese)
    inputs = features(siamese)
    net.model.fit(inputs, np.ones((8, 1)), epochs=1, verbose=0)
    path = tmp_path / "precursor.pt"
    net.model.save(path)
    loaded = xiRTNET(params, 6, 10)
    loaded.load_model(path)
    np.testing.assert_allclose(net.model.predict(inputs), loaded.model.predict(inputs))
    original_mean = loaded.model.numeric_mean.clone()
    loaded.compile()
    loaded.model.fit((*inputs[:-1], inputs[-1] * 100), np.ones((8, 1)), epochs=1, verbose=0)
    torch.testing.assert_close(loaded.model.numeric_mean, original_mean)
    changed = dict(params, numeric_features=["mass", "charge"])
    with pytest.raises(ValueError, match="schema"):
        build(changed, siamese).model.load_weights(path)


def test_numeric_inputs_change_predictions_and_preserve_pair_symmetry(params):
    model = build(params, True).model
    inputs = features(True)
    model.fit(inputs, np.ones((8, 1)), epochs=1, verbose=0)
    baseline = model.predict(inputs)
    changed = (*inputs[:-1], inputs[-1] + [10, 100])
    assert not np.allclose(baseline, model.predict(changed))
    np.testing.assert_allclose(
        model.predict((inputs[0], inputs[1][::-1].copy(), inputs[-1])),
        model.predict((inputs[1][::-1].copy(), inputs[0], inputs[-1])),
    )


def test_context_validation_is_explicit(params):
    model = build(params, True).model
    inputs = features(True)
    with pytest.raises(ValueError, match="Input count"):
        model.predict(inputs[:-1])
    with pytest.raises(ValueError, match="configured order"):
        model.predict((*inputs[:-1], inputs[-1][["mass", "charge"]]))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(inputs)
    inputs[-1].iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.predict(inputs)


def test_numeric_feature_config_rejects_ambiguous_semantics(params):
    for names in (["mass", "mass"], [""]):
        with pytest.raises(ValueError, match="nonempty and unique"):
            NetworkConfig.model_validate(dict(params, numeric_features=names))
    params["siamese"]["single_predictions"] = True
    with pytest.raises(ValueError, match="attribution"):
        NetworkConfig.model_validate(params)


@pytest.mark.parametrize("siamese", [False, True])
def test_model_data_keeps_numeric_rows_aligned_with_sequence_branches(siamese):
    from xirt.predictor import ModelData

    index = pd.Index([7, 3, 9])
    sequence = pd.DataFrame({"rnn_00": [1, 2, 3], "rnn_01": [2, 3, 4]}, index=index)
    numeric = pd.DataFrame({"charge": [3, 4, 5], "mass": [100, 200, 300]}, index=index)
    second = sequence + 1 if siamese else pd.DataFrame()
    data = ModelData(pd.DataFrame(index=index), sequence, second, numeric_features=numeric)
    inputs = data.get_features([9, 7])
    assert len(inputs) == (3 if siamese else 2)
    pd.testing.assert_frame_equal(inputs[0], sequence.loc[[9, 7]])
    pd.testing.assert_frame_equal(inputs[-1], numeric.loc[[9, 7]])
