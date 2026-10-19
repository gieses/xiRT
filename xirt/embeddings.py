"""Local ESM-C embeddings and differentiable retention-time fine-tuning.

ESM SDK 3.2.x is optional (``uv sync --python 3.12 --extra esm``). Only pretrained
weights are downloaded; peptide sequences never go to a remote inference API.
Frozen caches use safe NumPy files, keyed by model, SDK, pooling and sequence.
Fine-tuning calls ESM-C.forward, never its no-gradient logits inference API.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import uuid
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from xirt.adapters import apply_lokr, apply_lora

MAX_RESIDUES = 2046
_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX")
_LOCAL_MODELS = {"esmc_300m", "esmc_600m"}
_LOGGER = logging.getLogger(__name__)


def _validate_sequences(sequences: Sequence[str], max_length: int) -> list[str]:
    """Materialize peptides, rejecting modifications, empty values and excess length."""
    if isinstance(sequences, (str, bytes)):
        raise TypeError("sequences must be a sequence of peptide strings")
    result = list(sequences)
    for sequence in result:
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("ESM-C requires nonempty unmodified peptide strings")
        if set(sequence) - _ALPHABET:
            raise ValueError("ESM-C accepts uppercase canonical residues and X, not modifications")
        if len(sequence) > max_length:
            raise ValueError(f"Sequence exceeds {max_length} residues; truncation is not permitted")
    return result


def _load_esmc(model_name: str, device: torch.device) -> nn.Module:
    """Load local pretrained weights through the compatible optional ESM 3.2 SDK."""
    if model_name not in _LOCAL_MODELS:
        raise ValueError(f"Only local ESM-C models are supported: {sorted(_LOCAL_MODELS)}")
    try:
        from esm.models.esmc import ESMC
    except ImportError as exc:
        raise ImportError(
            "Local ESM-C requires Python 3.12 and xiRT[esm]: uv sync --python 3.12 --extra esm"
        ) from exc
    version = importlib.metadata.version("esm")
    if version.split(".")[:2] != ["3", "2"]:
        raise ImportError(f"ESM SDK >=3.2,<3.3 is required; found {version}")
    return ESMC.from_pretrained(model_name, device=device)


def _embedding_dim(model: nn.Module) -> int:
    """Read and validate the embedding width exposed by an ESM-compatible model."""
    embed = getattr(model, "embed", None)
    dimension = getattr(embed, "embedding_dim", getattr(model, "embedding_dim", None))
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("ESM-compatible models must expose embed.embedding_dim or embedding_dim")
    return dimension


def tokenize_batch(
    sequences: Sequence[str],
    model: nn.Module,
    device: str | torch.device | None = None,
    *,
    max_length: int = MAX_RESIDUES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize with public ESM tokenizer; return padded IDs and residue-only mask.

    The boolean mask excludes BOS, EOS and PAD, regardless of their token IDs.
    Length checks happen before tokenization; no sequence is silently truncated.

    Given a locally loaded ESM-compatible ``model``:

    >>> tokens, residues = tokenize_batch(["ACDE", "W"], model)  # doctest: +SKIP
    >>> residues.sum(dim=1).tolist()  # doctest: +SKIP
    [4, 1]
    """
    if not isinstance(max_length, int) or not 1 <= max_length <= MAX_RESIDUES:
        raise ValueError(f"max_length must be between 1 and {MAX_RESIDUES}")
    validated = _validate_sequences(sequences, max_length)
    if device is None:
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else "cpu"
    tokenizer: Any = getattr(model, "tokenizer", None)
    if tokenizer is None or tokenizer.pad_token_id is None:
        raise ValueError("ESM-compatible model must expose its public sequence tokenizer")
    rows = [
        tokenizer.encode(sequence, add_special_tokens=True, truncation=False)
        for sequence in validated
    ]
    width = max((len(row) for row in rows), default=0)
    tokens = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for index, (sequence, row) in enumerate(zip(validated, rows, strict=True)):
        if (
            len(row) != len(sequence) + 2
            or row[0] != tokenizer.bos_token_id
            or row[-1] != tokenizer.eos_token_id
            or tokenizer.pad_token_id in row[1:-1]
        ):
            raise ValueError("Tokenizer must produce exactly BOS, one token per residue, EOS")
        tokens[index, : len(row)] = torch.as_tensor(row, dtype=torch.long, device=device)
        mask[index, 1 : len(row) - 1] = True
    return tokens, mask


def _pool(embeddings: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
    """Average only masked residue positions in float32, retaining autograd."""
    if embeddings.ndim != 3 or residue_mask.shape != embeddings.shape[:2]:
        raise ValueError("Embeddings must be (batch,length,dimension) with matching residue mask")
    if residue_mask.dtype != torch.bool or not residue_mask.any(dim=1).all():
        raise ValueError("Each sequence must have a boolean mask with at least one residue")
    # masked_fill also excludes non-finite padded values (NaN * 0 is still NaN).
    values = embeddings.float().masked_fill(~residue_mask.unsqueeze(-1), 0)
    return values.sum(dim=1) / residue_mask.sum(dim=1, keepdim=True)


class ESMCEncoder:
    """Frozen, deduplicated mean-residue ESM-C embeddings with optional disk cache.

    Injected models must implement forward(sequence_tokens=...), returning an
    object with .embeddings, and expose the public tokenizer (including
    get_vocab) and embedding size. Every model's actual initial state_dict and
    tokenizer vocabulary are hashed, including downloaded pretrained weights.
    CPU copies are bounded to one tensor at a time, not the entire backbone.
    Treat the model as immutable after construction.

    Loading this example requires optional ESM dependencies and local model weights:

    >>> encoder = ESMCEncoder(cache_dir="embedding-cache")  # doctest: +SKIP
    >>> encoder.encode(["ACDE", "W"]).shape  # doctest: +SKIP
    (2, 960)
    """

    def __init__(
        self,
        model_name: str = "esmc_300m",
        device: str | torch.device = "cpu",
        cache_dir: str | Path | None = None,
        batch_size: int = 8,
        model: nn.Module | None = None,
    ) -> None:
        """Load or freeze an injected local model and initialize its cache identity."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.model_name, self.device = model_name, torch.device(device)
        self.batch_size = batch_size
        self.model = model if model is not None else _load_esmc(model_name, self.device)
        self.model.to(self.device).eval().requires_grad_(False)
        self.embedding_dim = _embedding_dim(self.model)
        try:
            sdk_version = importlib.metadata.version("esm")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = "injected-no-sdk"
        identity = hashlib.sha256()
        identity.update(f"{type(self.model).__module__}.{type(self.model).__qualname__}".encode())
        for key, tensor in self.model.state_dict().items():
            value = tensor.detach().cpu().contiguous()
            identity.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
            identity.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
            del value
        tokenizer: Any = getattr(self.model, "tokenizer", None)
        if not callable(getattr(tokenizer, "get_vocab", None)):
            raise ValueError(
                "ESM-compatible tokenizers must expose public get_vocab for cache identity"
            )
        identity.update(
            json.dumps(
                {
                    "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                    "vocab": tokenizer.get_vocab(),
                    "special_ids": [
                        getattr(tokenizer, name, None)
                        for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id")
                    ],
                },
                sort_keys=True,
            ).encode()
        )
        parameter = next(self.model.parameters(), None)
        precision = str(parameter.dtype) if parameter is not None else "unknown"
        self.fingerprint = hashlib.sha256(
            json.dumps(
                [
                    model_name,
                    sdk_version,
                    "mean-residues-v1",
                    self.embedding_dim,
                    precision,
                    identity.hexdigest(),
                ]
            ).encode()
        ).hexdigest()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, np.ndarray] = {}

    def _cache_path(self, sequence: str) -> Path:
        """Return the collision-resistant filename for this model and peptide."""
        assert self.cache_dir is not None
        key = hashlib.sha256(f"{self.fingerprint}:{sequence}".encode()).hexdigest()
        return self.cache_dir / f"{key}.npz"

    def _read_cache(self, sequence: str) -> np.ndarray | None:
        """Read a valid safe cache, warning on corruption; propagate I/O failures.

        Disabled caches and missing files are normal misses without warnings.
        Invalid metadata or arrays are warned about before recomputation.
        Permission errors and other operating-system failures are not hidden.
        """
        if self.cache_dir is None:
            return None
        path = self._cache_path(sequence)
        try:
            with np.load(path, allow_pickle=False) as data:
                value = data["embedding"]
                if (
                    str(data["fingerprint"].item()) == self.fingerprint
                    and str(data["sequence"].item()) == sequence
                    and value.shape == (self.embedding_dim,)
                    and value.dtype == np.float32
                    and np.isfinite(value).all()
                ):
                    return value.copy()
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, EOFError, TypeError, zipfile.BadZipFile, zlib.error) as exc:
            _LOGGER.warning("Corrupt embedding cache %s; regenerating: %s", path, exc)
            return None
        _LOGGER.warning("Invalid embedding cache metadata or values in %s; regenerating", path)
        return None

    def _write_cache(self, sequence: str, value: np.ndarray) -> None:
        """Atomically write a pickle-free cache entry, cleaning incomplete writes."""
        if self.cache_dir is None:
            return
        destination = self._cache_path(sequence)
        pending = destination.with_suffix(f".{uuid.uuid4().hex}.pending")
        try:
            with pending.open("wb") as stream:
                np.savez(stream, embedding=value, fingerprint=self.fingerprint, sequence=sequence)
            os.replace(pending, destination)
        finally:
            pending.unlink(missing_ok=True)

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        """Return (number of sequences, embedding_dim) float32 mean embeddings.

        >>> encoder.encode(["ACDE", "W"]).dtype  # doctest: +SKIP
        dtype('float32')
        """
        validated = _validate_sequences(sequences, MAX_RESIDUES)
        missing = []
        for sequence in dict.fromkeys(validated):
            if sequence not in self._cache:
                cached = self._read_cache(sequence)
                if cached is None:
                    missing.append(sequence)
                else:
                    self._cache[sequence] = cached
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                tokens, mask = tokenize_batch(batch, self.model, self.device)
                embeddings = self.model(sequence_tokens=tokens).embeddings
                if not isinstance(embeddings, torch.Tensor):
                    raise ValueError("ESM-C forward must return tensor embeddings")
                values = _pool(embeddings, mask).cpu().numpy()
                if (
                    values.shape != (len(batch), self.embedding_dim)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        "ESM-C returned invalid embedding shape or non-finite residues"
                    )
                for sequence, value in zip(batch, values, strict=True):
                    self._cache[sequence] = value.copy()
                    self._write_cache(sequence, value)
        if not validated:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.stack([self._cache[sequence] for sequence in validated])


class ESMCRetentionModel(nn.Module):
    """Differentiable local ESM-C with optional LoRA/LoKr and a retention head.

    The entire pretrained backbone is frozen before transformer Linear adapters
    are injected. Save full state_dict plus constructor configuration and target
    names; reconstruct the same architecture before load_state_dict.

    Loading this example requires optional ESM dependencies and local model weights:

    >>> model = ESMCRetentionModel(rank=2, factor=2, alpha=2)  # doctest: +SKIP
    >>> tokens, residues = model.tokenize(["ACDE", "W"])  # doctest: +SKIP
    >>> model(tokens, residues).square().mean().backward()  # doctest: +SKIP
    """

    def __init__(
        self,
        model_name: str = "esmc_300m",
        device: str | torch.device = "cpu",
        *,
        model: nn.Module | None = None,
        output_dim: int = 1,
        rank: int = 4,
        factor: int = 4,
        alpha: float = 4.0,
        target_names: Sequence[str] | None = None,
        adapter_kind: str = "lokr",
    ) -> None:
        """Freeze the backbone, inject selected transformer adapters and create a head."""
        super().__init__()
        if not isinstance(output_dim, int) or output_dim < 1:
            raise ValueError("output_dim must be a positive integer")
        if adapter_kind not in {"none", "lora", "lokr"}:
            raise ValueError("adapter_kind must be none, lora or lokr")
        if adapter_kind == "none" and target_names:
            raise ValueError("Frozen ESM-C does not accept adapter target names")
        self.adapter_kind = adapter_kind
        self.backbone = model if model is not None else _load_esmc(model_name, torch.device(device))
        self.backbone.to(device).eval().requires_grad_(False)
        if target_names is None:
            target_names = [
                name
                for name, layer in self.backbone.named_modules()
                if name.startswith("transformer.") and isinstance(layer, nn.Linear)
            ]
        if not target_names and adapter_kind != "none":
            raise ValueError("No transformer Linear layers selected for ESM-C LoKr")
        if adapter_kind == "lokr":
            self.adapter_targets = apply_lokr(
                self.backbone, target_names, rank=rank, factor=factor, alpha=alpha
            )
        elif adapter_kind == "lora":
            self.adapter_targets = apply_lora(self.backbone, target_names, rank=rank, alpha=alpha)
        else:
            self.adapter_targets = ()
        self.regressor = nn.Linear(_embedding_dim(self.backbone), output_dim).to(device)

    def tokenize(self, sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare IDs and residue masks on the backbone's current device.

        >>> tokens, residues = model.tokenize(["ACDE"])  # doctest: +SKIP
        """
        return tokenize_batch(sequences, self.backbone)

    def train(self, mode: bool = True) -> ESMCRetentionModel:
        """Set head training mode while preserving deterministic backbone evaluation."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, token_ids: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        """Predict retention values through differentiable residue pooling and the head."""
        pooled = self.encode(token_ids, residue_mask)
        return self.regressor(pooled.to(self.regressor.weight.dtype))

    def encode(self, token_ids: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        """Return differentiable residue-only mean embeddings for shared Siamese heads."""
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a (batch,length) int64 tensor")
        embeddings = self.backbone(sequence_tokens=token_ids).embeddings
        return _pool(embeddings, residue_mask)
