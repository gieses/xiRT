"""Read fingerprinted RP cohorts with grouped partitions and train-only scaling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xirt.config import NetworkConfig
from xirt.data_pipelines.benchmarks.common import file_hash


class BenchmarkDataset:
    """Load pre-padded modification-aware tokens and independently validate partitions."""

    def __init__(self, directory: Path) -> None:
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if file_hash(directory / "pairs.csv") != self.manifest["pairs_sha256"]:
            raise ValueError("Pair table differs from the measured manifest.")
        self.pairs = pd.read_csv(directory / "pairs.csv", keep_default_na=False)
        self.splits = self.manifest["splits"]
        assignment = [index for split in self.splits.values() for index in split]
        if sorted(assignment) != list(range(len(self.pairs))):
            raise ValueError("Partitions must be disjoint and exhaustive.")
        groups = {
            split: set(self.pairs.iloc[indices]["group_key"])
            for split, indices in self.splits.items()
        }
        for first, values in groups.items():
            for second, other in groups.items():
                if first != second and values & other:
                    raise ValueError("Sequence identities overlap between partitions.")
        self.truth = self.pairs["rp_seconds"].to_numpy(dtype=np.float64)
        self.mean = float(self.manifest["target_normalization"]["mean"])
        self.std = float(self.manifest["target_normalization"]["std"])
        train = self.truth[self.splits["train"]]
        if (
            not np.isfinite(self.truth).all()
            or not self.std > 0
            or not np.isclose(train.mean(), self.mean)
            or not np.isclose(train.std(), self.std)
        ):
            raise ValueError(
                "Targets must be finite and normalization must use training rows only."
            )
        self.targets = ((self.truth - self.mean) / self.std).astype(np.float32).reshape(-1, 1)
        mapping = {token: index + 1 for index, token in enumerate(self.manifest["alphabet"])}
        self.inputs = []
        columns = ("tokens_a", "tokens_b") if self.manifest["siamese"] else ("tokens_a",)
        for column in columns:
            matrix = np.zeros((len(self.pairs), self.manifest["max_length"]), dtype=np.int64)
            for index, tokens in enumerate(self.pairs[column]):
                encoded = [mapping[token] for token in json.loads(tokens)]
                if not encoded or len(encoded) > matrix.shape[1]:
                    raise ValueError("Invalid token sequence length.")
                matrix[index, -len(encoded) :] = encoded
            self.inputs.append(matrix)

    def features(self, indices: list[int]) -> tuple[np.ndarray, ...]:
        """Return matched branches in explicit row order."""
        return tuple(array[indices] for array in self.inputs)

    def prediction_table(self, values: np.ndarray, indices: list[int]) -> pd.DataFrame:
        """Preserve pair identities and convert standardized predictions to seconds."""
        return pd.DataFrame(
            {
                "pair_index": indices,
                "truth_seconds": self.truth[indices],
                "prediction_seconds": np.asarray(values, dtype=np.float64).reshape(-1) * self.std
                + self.mean,
            }
        )


def recurrent_parameters(
    manifest: dict[str, Any], config: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Reconstruct the compact native architecture used for the archived RP comparison."""
    architecture = config["architecture"]
    params = {
        "LSTM": {
            "activation": "tanh",
            "activity_regularization": "l2",
            "activityregularizer_value": 0.0,
            "bidirectional": False,
            "kernel_regularization": "l2",
            "kernelregularizer_value": 0.0,
            "lstm_bn": False,
            "nlayers": 1,
            "type": "GRU",
            "units": architecture["gru_units"],
        },
        "dense": {
            "activation": [architecture["dense_activation"]],
            "dense_bn": [False],
            "dropout": [candidate["dropout"]],
            "kernel_regularizer": ["l2"],
            "regularizer_value": [0.0],
            "regularization": [False],
            "neurons": [architecture["dense_units"]],
            "nlayers": 1,
        },
        "embedding": {"length": architecture["embedding_dim"]},
        "learning": {
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "learningrate": candidate["learning_rate"],
            "optimizer": "adamw",
            "weight_decay": candidate["weight_decay"],
            "clip_norm": 1.0,
            "verbose": 0,
            "device": "cpu",
        },
        "output": {
            "rp-activation": "linear",
            "rp-dimension": 1,
            "rp-loss": "mse",
            "rp-metrics": "mse",
            "rp-column": "rp",
            "rp-weight": 1,
        },
        "siamese": {"use": manifest["siamese"], "merge_type": "add", "single_predictions": False},
        "callbacks": {},
        "predictions": {"fractions": [], "continues": ["rp"]},
        "_alphabet": manifest["alphabet"],
    }
    return NetworkConfig.model_validate(params).model_dump()
