"""Exact LoKr algebra, freezing, optimization and checkpoint reconstruction."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xirt.adapters import LoKrLinear, LoKrWeight, LoRALinear, apply_lokr, apply_lokr_gru, apply_lora


def test_exact_kronecker_and_zero_initialization() -> None:
    base = nn.Linear(12, 8)
    inputs = torch.randn(5, 12)
    expected = base(inputs).detach()
    adapter = LoKrLinear(base, rank=2, factor=2, alpha=3)
    torch.testing.assert_close(adapter(inputs), expected)
    with torch.no_grad():
        adapter.lokr_w2_b.fill_(0.2)
    delta = 1.5 * torch.kron(adapter.lokr_w1, adapter.lokr_w2_a @ adapter.lokr_w2_b)
    torch.testing.assert_close(adapter.delta_weight(), delta)
    torch.testing.assert_close(adapter.weight, base.weight + delta)
    torch.testing.assert_close(
        adapter(inputs), nn.functional.linear(inputs, base.weight + delta, base.bias)
    )
    assert not base.weight.requires_grad and not base.bias.requires_grad


def test_updates_gradients_and_checkpoint() -> None:
    model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    inputs = torch.randn(8, 12)
    expected = model(inputs).detach()
    targets = apply_lokr(model, rank=2, factor=2, alpha=2)
    assert targets == ("0", "2")
    torch.testing.assert_close(model(inputs), expected)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.01
    )
    for _ in range(3):
        optimizer.zero_grad()
        model(inputs).square().mean().backward()
        optimizer.step()
    assert not torch.equal(model(inputs), expected)
    for name, parameter in model.named_parameters():
        if ".base." in name:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, original[name.replace(".base.", ".")])
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    restored = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
    apply_lokr(restored, targets, rank=2, factor=2, alpha=2)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(inputs), model(inputs))
    assert apply_lokr(model, targets, rank=2) == targets
    assert apply_lokr(model) == ()


def test_selected_targets_and_no_hidden_global_state() -> None:
    first = nn.ModuleDict({"a": nn.Linear(4, 4), "b": nn.Linear(4, 4)})
    second = nn.ModuleDict({"a": nn.Linear(4, 4)})
    assert apply_lokr(first, ["a"], rank=1) == ("a",)
    assert isinstance(first["a"], LoKrLinear)
    assert isinstance(first["b"], nn.Linear)
    assert isinstance(second["a"], nn.Linear)
    assert first["b"].weight.requires_grad
    with pytest.raises(ValueError, match="not Linear"):
        apply_lokr(first, ["missing"])
    with pytest.raises(TypeError):
        apply_lokr(second, "a")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"rank": True},
        {"factor": 0},
        {"factor": -2},
        {"alpha": 0},
        {"alpha": float("nan")},
    ],
)
def test_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LoKrLinear(nn.Linear(4, 4), **kwargs)


@pytest.mark.parametrize("dimensions", [(7, 11), (1, 1), (12, 8)])
def test_prime_and_small_dimensions(dimensions: tuple[int, int]) -> None:
    base = nn.Linear(*dimensions, bias=False)
    adapter = LoKrLinear(base, rank=2, factor=-1)
    values = adapter(torch.randn(3, dimensions[0]))
    assert values.shape == (3, dimensions[1]) and torch.isfinite(values).all()


def test_lora_algebra_updates_freezing_and_checkpoint() -> None:
    model = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 1))
    x = torch.randn(9, 6)
    before = model(x).detach()
    assert apply_lora(model, rank=2, alpha=2) == ("0", "2")
    torch.testing.assert_close(before, model(x))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.01, weight_decay=0
    )
    base = {name: p.detach().clone() for name, p in model.named_parameters() if ".base." in name}
    for _ in range(3):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
    assert not torch.equal(before, model(x))
    for name, p in model.named_parameters():
        if name in base:
            torch.testing.assert_close(p, base[name])
            assert p.grad is None
    layer = model[0]
    assert isinstance(layer, LoRALinear)
    torch.testing.assert_close(layer.delta_weight(), layer.lora_b @ layer.lora_a)
    restored = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 1))
    apply_lora(restored, rank=2, alpha=2)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x), restored(x))
    assert apply_lora(model) == ()
    with pytest.raises(ValueError, match="reinterpret"):
        apply_lokr(model, ["0"])


@pytest.mark.parametrize("layers,bidirectional", [(1, False), (2, True)])
def test_gru_lokr_packed_gradient_freezing_and_checkpoint(layers, bidirectional) -> None:
    model = nn.GRU(4, 8, num_layers=layers, bidirectional=bidirectional, batch_first=True)
    model.requires_grad_(False)
    inputs = nn.utils.rnn.pack_padded_sequence(
        torch.randn(3, 5, 4), [5, 3, 2], batch_first=True, enforce_sorted=False
    )
    expected = model(inputs)[1].detach()
    original = {name: p.clone() for name, p in model.named_parameters()}
    targets = apply_lokr_gru(model, rank=2, factor=2, alpha=2)
    assert len(targets) == layers * (2 if bidirectional else 1) * 2
    torch.testing.assert_close(model(inputs)[1], expected)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.01, weight_decay=0
    )
    for _ in range(3):
        optimizer.zero_grad()
        model(inputs)[1].square().mean().backward()
        optimizer.step()
    assert not torch.equal(model(inputs)[1], expected)
    for name, parameter in model.named_parameters():
        if name.startswith("bias_"):
            torch.testing.assert_close(parameter, original[name])
            assert parameter.grad is None
        elif name.endswith(".original"):
            source_name = name.split(".")[1]
            torch.testing.assert_close(parameter, original[source_name])
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    reconstructed = nn.GRU(4, 8, num_layers=layers, bidirectional=bidirectional, batch_first=True)
    apply_lokr_gru(reconstructed, rank=2, factor=2, alpha=2)
    reconstructed.load_state_dict(model.state_dict())
    torch.testing.assert_close(reconstructed(inputs)[1], model(inputs)[1])
    assert set(apply_lokr_gru(model)) == set(targets)
    assert any(isinstance(module, LoKrWeight) for module in model.modules())


def test_adapter_selection_rejects_invalid_and_cross_kind_targets() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(TypeError):
        apply_lora(model, "0")
    with pytest.raises(ValueError, match="not Linear"):
        apply_lora(model, ["missing"])
    apply_lokr(model)
    with pytest.raises(ValueError, match="reinterpret"):
        apply_lora(model, ["0"])
    with pytest.raises(ValueError, match="No GRU"):
        apply_lokr_gru(model)
