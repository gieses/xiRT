"""Offline ESM-C contract tests; real weights require explicit opt-in."""

from __future__ import annotations

import builtins
import logging
import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from xirt.adapters import LoKrLinear, LoRALinear
from xirt.embeddings import ESMCEncoder, ESMCRetentionModel, tokenize_batch


@pytest.fixture
def cache_dir() -> Iterator[Path]:
    directory = Path.cwd() / "tests" / f".embedding-cache-{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(
        self, sequence: str, add_special_tokens: bool = True, truncation: bool = False
    ) -> list[int]:
        assert add_special_tokens and not truncation
        return [1, *[ord(letter) - 60 for letter in sequence], 2]

    def get_vocab(self) -> dict[str, int]:
        return {"pad": 0, "bos": 1, "eos": 2}


class FakeESMC(nn.Module):
    def __init__(self, offset: float = 0) -> None:
        super().__init__()
        self.tokenizer = FakeTokenizer()
        self.embed = nn.Embedding(64, 4)
        self.transformer = nn.Sequential(nn.Linear(4, 4, bias=False))
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.arange(64, dtype=torch.float32)[:, None].expand(64, 4) + offset
            )
            self.embed.weight[0] = 0
            self.embed.weight[1] = 1000
            self.embed.weight[2] = -1000
            self.transformer[0].weight.copy_(torch.eye(4))
        self.calls: list[torch.Tensor] = []

    def forward(self, sequence_tokens: torch.Tensor) -> SimpleNamespace:
        self.calls.append(sequence_tokens.detach().clone())
        return SimpleNamespace(embeddings=self.transformer(self.embed(sequence_tokens)))

    def logits(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("logits inference API must never be used")


@pytest.mark.parametrize("kind", ["none", "lora", "lokr"])
def test_retention_backbone_adapter_variants(kind: str) -> None:
    model = ESMCRetentionModel(model=FakeESMC(), adapter_kind=kind, rank=2, alpha=2)
    tokens, mask = model.tokenize(["AC", "W"])
    pooled = model.encode(tokens, mask)
    assert pooled.shape == (2, 4)
    model(tokens, mask).square().mean().backward()
    assert model.regressor.weight.grad is not None
    base = model.backbone.transformer[0]
    if kind == "none":
        assert model.adapter_targets == ()
        assert all(p.grad is None for p in model.backbone.parameters())
    else:
        assert isinstance(base, LoRALinear if kind == "lora" else LoKrLinear)
        assert any(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)
        assert all(p.grad is None for p in base.base.parameters())


def test_pooling_duplicates_and_frozen_parameters() -> None:
    model = FakeESMC()
    encoder = ESMCEncoder(model=model)
    values = encoder.encode(["AC", "W", "AC"])
    np.testing.assert_array_equal(values[:, 0], [6, 27, 6])
    assert values.shape == (3, 4)
    assert values.dtype == np.float32
    assert len(model.calls) == 1 and model.calls[0].shape[0] == 2
    assert not model.training and all(
        not parameter.requires_grad for parameter in model.parameters()
    )
    values[0] = 999
    np.testing.assert_array_equal(encoder.encode(["AC"])[0], [6] * 4)


def test_batch_invariance_and_empty() -> None:
    sequences = ["AC", "W", "AXC", "AC"]
    first = ESMCEncoder(model=FakeESMC(), batch_size=1).encode(sequences)
    second = ESMCEncoder(model=FakeESMC(), batch_size=3).encode(sequences)
    np.testing.assert_array_equal(first, second)
    assert ESMCEncoder(model=FakeESMC()).encode([]).shape == (0, 4)


def test_nonfinite_special_tokens_are_excluded() -> None:
    model = FakeESMC()
    with torch.no_grad():
        model.embed.weight[:3] = float("nan")
    np.testing.assert_array_equal(ESMCEncoder(model=model).encode(["AC", "W"]), [[6] * 4, [27] * 4])


def test_nonfinite_residues_and_wrong_dimensions_rejected() -> None:
    model = FakeESMC()
    with torch.no_grad():
        model.embed.weight[5] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        ESMCEncoder(model=model).encode(["AC"])
    model = FakeESMC()
    model.transformer = nn.Sequential(nn.Linear(4, 3))
    with pytest.raises(ValueError, match="shape"):
        ESMCEncoder(model=model).encode(["AC"])


def test_public_tokenizer_padding_and_residue_mask() -> None:
    tokens, mask = tokenize_batch(["AC", "W"], FakeESMC())
    assert tokens.tolist() == [[1, 5, 7, 2], [1, 27, 2, 0]]
    assert mask.tolist() == [[False, True, True, False], [False, True, False, False]]
    assert mask.dtype == torch.bool
    assert tokenize_batch([], FakeESMC())[0].shape == (0, 0)


@pytest.mark.parametrize("sequence", ["", "AC(ox)", "ac", "AC B", "U", "A" * 2047, None])
def test_invalid_sequences_are_rejected_before_forward(sequence: str) -> None:
    model = FakeESMC()
    with pytest.raises(ValueError):
        ESMCEncoder(model=model).encode([sequence])
    assert not model.calls


def test_input_and_batch_validation() -> None:
    with pytest.raises(TypeError):
        ESMCEncoder(model=FakeESMC()).encode("AC")
    for batch_size in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            ESMCEncoder(model=FakeESMC(), batch_size=batch_size)
    with pytest.raises(ValueError):
        tokenize_batch(["AC"], FakeESMC(), max_length=1)


def test_disk_cache_and_model_fingerprints(cache_dir: Path) -> None:
    model = FakeESMC()
    expected = ESMCEncoder(model=model, cache_dir=cache_dir).encode(["AC", "W"])
    other = FakeESMC()
    np.testing.assert_array_equal(
        ESMCEncoder(model=other, cache_dir=cache_dir).encode(["AC", "W"]), expected
    )
    assert not other.calls
    changed_name = FakeESMC()
    ESMCEncoder(model=changed_name, model_name="another_model", cache_dir=cache_dir).encode(["AC"])
    assert len(changed_name.calls) == 1
    changed_weights = FakeESMC(offset=1)
    values = ESMCEncoder(model=changed_weights, cache_dir=cache_dir).encode(["AC"])
    np.testing.assert_array_equal(values, expected[:1] + 1)
    assert len(changed_weights.calls) == 1


def test_default_loaded_model_weights_and_vocab_define_cache_identity(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, same, changed, vocabulary_changed = (
        FakeESMC(),
        FakeESMC(),
        FakeESMC(offset=1),
        FakeESMC(offset=1),
    )
    original_vocab = vocabulary_changed.tokenizer.get_vocab()
    monkeypatch.setattr(
        vocabulary_changed.tokenizer, "get_vocab", lambda: {**original_vocab, "reserved": 63}
    )
    pending = iter([first, same, changed, vocabulary_changed])
    monkeypatch.setattr("xirt.embeddings._load_esmc", lambda name, device: next(pending))

    baseline = ESMCEncoder(cache_dir=cache_dir)
    expected = baseline.encode(["AC"])
    identical = ESMCEncoder(cache_dir=cache_dir)
    np.testing.assert_array_equal(identical.encode(["AC"]), expected)
    assert identical.fingerprint == baseline.fingerprint and not same.calls
    updated = ESMCEncoder(cache_dir=cache_dir)
    np.testing.assert_array_equal(updated.encode(["AC"]), expected + 1)
    assert updated.fingerprint != baseline.fingerprint and len(changed.calls) == 1
    retokenized = ESMCEncoder(cache_dir=cache_dir)
    np.testing.assert_array_equal(retokenized.encode(["AC"]), expected + 1)
    assert retokenized.fingerprint != updated.fingerprint and len(vocabulary_changed.calls) == 1


@pytest.mark.parametrize(
    "corruption", ["shape", "fingerprint", "sequence", "nan", "dtype", "bytes", "pickle", "npy"]
)
def test_invalid_cache_recomputed(
    cache_dir: Path, corruption: str, caplog: pytest.LogCaptureFixture
) -> None:
    original = ESMCEncoder(model=FakeESMC(), cache_dir=cache_dir)
    original.encode(["AC"])
    path = next(cache_dir.glob("*.npz"))
    if corruption == "bytes":
        path.write_bytes(b"not a numpy archive")
    elif corruption == "npy":
        with path.open("wb") as stream:
            np.save(stream, np.zeros(4, dtype=np.float32))
    else:
        value = np.full(4, 6, dtype=np.float32)
        fingerprint = original.fingerprint
        sequence = "AC"
        if corruption == "shape":
            value = np.zeros(2, dtype=np.float32)
        elif corruption == "fingerprint":
            fingerprint = "stale"
        elif corruption == "sequence":
            sequence = "W"
        elif corruption == "nan":
            value[:] = np.nan
        elif corruption == "dtype":
            value = value.astype(np.float64)
        elif corruption == "pickle":
            value = np.array([object()] * 4, dtype=object)
        np.savez(path, embedding=value, fingerprint=fingerprint, sequence=sequence)
    fresh = FakeESMC()
    result = ESMCEncoder(model=fresh, cache_dir=cache_dir).encode(["AC"])
    np.testing.assert_array_equal(result, [[6] * 4])
    assert len(fresh.calls) == 1
    assert len(list(cache_dir.iterdir())) == 1
    warnings = [record for record in caplog.records if record.name == "xirt.embeddings"]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert "regenerating" in warnings[0].getMessage()


def test_missing_cache_is_normal_without_warning(
    cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ESMCEncoder(model=FakeESMC(), cache_dir=cache_dir).encode(["AC"])
    assert not [record for record in caplog.records if record.name == "xirt.embeddings"]


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_cache_io_errors_are_not_hidden(
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_type: type[OSError],
) -> None:
    model = FakeESMC()
    encoder = ESMCEncoder(model=model, cache_dir=cache_dir)

    def unreadable_cache(*args: object, **kwargs: object) -> None:
        raise error_type("cache cannot be read")

    monkeypatch.setattr("xirt.embeddings.np.load", unreadable_cache)
    with pytest.raises(error_type, match="cannot be read"):
        encoder.encode(["AC"])
    assert not model.calls
    assert not [record for record in caplog.records if record.name == "xirt.embeddings"]


def test_sdk_version_is_part_of_cache_identity(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("xirt.embeddings.importlib.metadata.version", lambda name: "3.2.3")
    ESMCEncoder(model=FakeESMC(), cache_dir=cache_dir).encode(["AC"])
    monkeypatch.setattr("xirt.embeddings.importlib.metadata.version", lambda name: "3.2.4")
    model = FakeESMC()
    ESMCEncoder(model=model, cache_dir=cache_dir).encode(["AC"])
    assert len(model.calls) == 1


def test_dependency_error_and_local_models_only(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def no_esm(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("esm"):
            raise ImportError("offline missing esm")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_esm)
    with pytest.raises(ImportError, match="xiRT\\[esm\\]"):
        ESMCEncoder()
    with pytest.raises(ValueError, match="local"):
        ESMCEncoder(model_name="esmc-6b-remote")


def test_differentiable_backbone_lokr_and_checkpoint(cache_dir: Path) -> None:
    torch.manual_seed(7)
    model = ESMCRetentionModel(model=FakeESMC(), rank=2, factor=2, alpha=2)
    tokens, mask = model.tokenize(["AC", "W"])
    initial = model(tokens, mask).detach().clone()
    reference = model.regressor(torch.tensor([[6.0] * 4, [27.0] * 4]))
    torch.testing.assert_close(initial, reference)
    model.train()
    assert not model.backbone.training
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.0001
    )
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(tokens, mask).square().mean()
        loss.backward()
        assert model.backbone.embed.weight.grad is None
        assert model.backbone.transformer[0].base.weight.grad is None
        assert model.backbone.transformer[0].lokr_w2_b.grad.abs().sum() > 0
        optimizer.step()
    updated = model(tokens, mask).detach()
    assert torch.isfinite(updated).all() and not torch.equal(updated, initial)
    checkpoint = cache_dir / "retention.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = ESMCRetentionModel(model=FakeESMC(), rank=2, factor=2, alpha=2)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(tokens, mask), updated)


def test_fine_tuning_requires_transformer_targets_and_valid_inputs() -> None:
    model = FakeESMC()
    model.transformer = nn.Identity()
    with pytest.raises(ValueError, match="transformer"):
        ESMCRetentionModel(model=model)
    model = ESMCRetentionModel(model=FakeESMC())
    tokens, mask = model.tokenize(["AC"])
    with pytest.raises(ValueError, match="int64"):
        model(tokens.float(), mask)
    with pytest.raises(ValueError, match="boolean mask"):
        model(tokens, torch.zeros_like(mask))


@pytest.mark.esm
def test_real_sdk_small_architecture_without_downloads() -> None:
    """Exercise SDK tokenization/autograd, not pretrained embedding quality."""
    pytest.importorskip("esm")
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = ESMC(
        d_model=32,
        n_heads=4,
        n_layers=2,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=False,
    )
    encoder = ESMCEncoder(model=backbone, device=device, batch_size=2)
    expected = encoder.encode(["ACDE", "W"])
    assert expected.shape == (2, 32) and np.isfinite(expected).all()
    encoder._cache.clear()
    encoder.batch_size = 1
    np.testing.assert_allclose(encoder.encode(["ACDE", "W"]), expected, rtol=1e-5, atol=1e-5)
    model = ESMCRetentionModel(model=backbone, device=device, rank=2, factor=2, alpha=2)
    tokens, mask = model.tokenize(["ACDE", "W"])
    model(tokens, mask).square().mean().backward()
    assert model.adapter_targets
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in model.named_parameters()
        if "lokr_w2_b" in name
    )


@pytest.mark.esmc
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("XIRT_RUN_ESMC") != "1", reason="set XIRT_RUN_ESMC=1")
def test_real_esmc_local_embeddings_and_lokr() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ESMCEncoder(device=device, batch_size=2)
    expected = encoder.encode(["ACDE", "W"])
    assert expected.shape == (2, 960) and np.isfinite(expected).all()
    encoder.batch_size = 1
    encoder._cache.clear()
    np.testing.assert_allclose(encoder.encode(["ACDE", "W"]), expected, rtol=0.03, atol=0.03)
    model = ESMCRetentionModel(device=device, model=encoder.model, rank=2)
    tokens, mask = model.tokenize(["ACDE", "W"])
    output = model(tokens, mask)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    grads = [parameter.grad for name, parameter in model.named_parameters() if "lokr_w2_b" in name]
    assert grads and any(grad is not None and grad.abs().sum() > 0 for grad in grads)
