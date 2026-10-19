"""Native fine-tuning trainability, optimizer routing and checkpoint round trips."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn.utils import parametrize

from xirt.adapters import apply_lora_gru
from xirt.config import FineTuningConfig, NetworkConfig
from xirt.xirtnet import xiRTNET

STRATEGIES = ["recurrent_full", "recurrent_frozen", "recurrent_lokr", "recurrent_lora"]


@pytest.fixture
def base(tmp_path):
    params = yaml.safe_load((Path(__file__).parent / "fixtures/xirt_params_3RT.yaml").read_text())
    params["predictions"] = {"fractions": [], "continues": ["rp"]}
    params["LSTM"].update(units=4, nlayers=1, lstm_bn=True)
    params["embedding"]["length"] = 3
    params["learning"].update(device="cpu", optimizer="adamw", learningrate=0.01)
    params["callbacks"] = {}
    params["_alphabet"] = ["A", "C", "K"]
    network = xiRTNET(params, 6, 8)
    network.build_model(siamese=True, alphabet=8)
    checkpoint = tmp_path / "base.pt"
    network.model.save(checkpoint)
    inputs = np.array([[0, 0, 0, 1, 2, 3], [0, 0, 1, 2, 3, 4], [0, 0, 0, 2, 3, 4]])
    return params, network, checkpoint, (inputs, inputs[::-1].copy())


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_fine_tuning_initial_predictions_freezing_and_checkpoint(base, tmp_path, strategy):
    params, original, checkpoint, inputs = base
    network = xiRTNET(params, 6, 8)
    network.fine_tune(checkpoint, strategy, rank=2, factor=2, alpha=2)
    model = network.model
    np.testing.assert_allclose(model.predict(inputs), original.model.predict(inputs), atol=1e-6)
    frozen = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    buffers = {name: value.clone() for name, value in model.encoder_norm.named_buffers()}
    network.compile()
    assigned = [
        id(parameter)
        for optimizer in model.optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(p) for p in model.parameters() if p.requires_grad}
    before = copy.deepcopy(model.state_dict())
    model.fit(
        inputs,
        np.ones((3, 1), dtype=np.float32),
        epochs=3,
        batch_size=3,
        verbose=0,
    )
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
            assert parameter.grad is None
    if strategy != "recurrent_full":
        for name, value in model.encoder_norm.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        assert not model.embedding.weight.requires_grad
    else:
        assert all(parameter.requires_grad for parameter in model.parameters())
    if strategy in {"recurrent_lokr", "recurrent_lora"}:
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for name, parameter in model.named_parameters()
            if "lokr_" in name or "lora_" in name
        )
        rates = {group["lr"] for group in model.optimizer.param_groups}
        assert rates == {0.01, 0.001}
    path = tmp_path / f"{strategy}.pt"
    model.save(path)
    restored = xiRTNET(params, 6, 8)
    restored.load_model(path)
    np.testing.assert_allclose(model.predict(inputs), restored.model.predict(inputs), atol=1e-6)
    assert restored.model.params["fine_tuning"]["strategy"] == strategy


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_yaml_fine_tuning_checkpoint_builds_a_loaded_model(base, strategy):
    params, original, checkpoint, inputs = base
    params["fine_tuning"] = {"strategy": strategy, "checkpoint": str(checkpoint)}
    network = xiRTNET(params, 6, 8)
    network.build_model(siamese=True, alphabet=8)
    np.testing.assert_allclose(
        network.model.predict(inputs), original.model.predict(inputs), atol=1e-6
    )


def test_fine_tuning_requires_checkpoint_and_matching_vocabulary(base):
    params, _, checkpoint, _ = base
    params["fine_tuning"] = {"strategy": "recurrent_frozen"}
    with pytest.raises(ValueError, match="checkpoint is required"):
        xiRTNET(params, 6, 8).build_model()
    params["_alphabet"] = ["X"]
    with pytest.raises(ValueError, match="vocabulary differs"):
        xiRTNET(params, 6, 8).fine_tune(checkpoint, "recurrent_full")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_recurrent_adapter_adjustment_preserves_effective_encoder(base, strategy):
    params, _, checkpoint, inputs = base
    network = xiRTNET(params, 6, 8)
    network.fine_tune(checkpoint, strategy)
    with torch.no_grad():
        for name, parameter in network.model.named_parameters():
            if "lora_b" in name or "lokr_w2_b" in name:
                parameter.add_(0.01)
    expected = network.model.predict(inputs)
    network.adjust_model()
    np.testing.assert_allclose(network.model.predict(inputs), expected, atol=1e-6)
    assert network.model.params["fine_tuning"]["strategy"] == strategy
    assert network.model.embedding.weight.requires_grad == (strategy == "recurrent_full")
    if strategy in {"recurrent_lokr", "recurrent_lora"}:
        assert parametrize.is_parametrized(network.model.recurrent, "weight_ih_l0")


@pytest.mark.parametrize("layers,bidirectional", [(1, False), (2, True)])
def test_lora_gru_covers_layers_directions_and_preserves_packed_forward(layers, bidirectional):
    gru = nn.GRU(3, 4, num_layers=layers, bidirectional=bidirectional, batch_first=True)
    inputs = torch.randn(5, 6, 3)
    expected = gru(inputs)[0].detach()
    targets = apply_lora_gru(gru, rank=2, alpha=2)
    assert len(targets) == layers * (4 if bidirectional else 2)
    torch.testing.assert_close(gru(inputs)[0], expected)
    gru(inputs)[0].square().mean().backward()
    assert gru.parametrizations.weight_ih_l0[0].lora_b.grad.abs().sum() > 0
    assert not gru.parametrizations.weight_ih_l0.original.requires_grad
    assert parametrize.is_parametrized(gru, "weight_ih_l0")


@pytest.mark.parametrize(
    "values",
    [
        {"strategy": "unknown"},
        {"strategy": "recurrent_lora", "rank": True},
        {"strategy": "recurrent_lora", "alpha": float("nan")},
        {"strategy": "recurrent_lokr", "factor": 0},
        {"strategy": "recurrent_lora", "adapter_learning_rate_ratio": 0},
    ],
)
def test_invalid_fine_tuning_settings(values):
    with pytest.raises(ValueError):
        FineTuningConfig.model_validate(values)


def test_adapter_strategy_rejects_lstm(base):
    params, _, _, _ = base
    params["LSTM"]["type"] = "LSTM"
    params["fine_tuning"] = {"strategy": "recurrent_lora"}
    with pytest.raises(ValueError, match="require a GRU"):
        NetworkConfig.model_validate(params)
