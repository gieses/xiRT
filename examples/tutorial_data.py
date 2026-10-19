"""Shared data preparation for the recurrent API notebooks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "examples/data/crosslinks.csv"
SEQUENCE_WIDTH = 50
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
VOCABULARY = {
    token: index for index, token in enumerate([*RESIDUES, *(f"{aa}*" for aa in RESIDUES)], start=1)
}


def encode_pairs(table: pd.DataFrame, metadata: dict) -> list[np.ndarray]:
    """Encode canonical residues, marking each one-based crosslink position."""
    width = metadata["sequence_width"]
    vocabulary = metadata["vocabulary"]
    branches = []
    for branch in (1, 2):
        encoded = np.zeros((len(table), width), dtype=np.int64)
        for row, (sequence, position) in enumerate(
            zip(table[f"Peptide{branch}"], table[f"LinkPos{branch}"], strict=True)
        ):
            if not 1 <= position <= len(sequence) <= width:
                raise ValueError(
                    "Crosslink position or peptide length is outside the input contract"
                )
            tokens = [
                f"{aa}*" if index == position else aa for index, aa in enumerate(sequence, start=1)
            ]
            encoded[row, : len(tokens)] = [vocabulary[token] for token in tokens]
        branches.append(encoded)
    return branches


def prepare_data() -> tuple[pd.DataFrame, list[np.ndarray], np.ndarray, dict]:
    """Split observed pairs 75/12.5/12.5 and fit scaling on training rows only."""
    table = pd.read_csv(DATA_PATH)
    order = np.random.default_rng(42).permutation(len(table))
    table["split"] = ""
    training_end = 3 * len(table) // 4
    validation_end = 7 * len(table) // 8
    for name, indices in (
        ("train", order[:training_end]),
        ("validation", order[training_end:validation_end]),
        ("test", order[validation_end:]),
    ):
        table.loc[indices, "split"] = name
    training_seconds = table.loc[table["split"] == "train", "rp"]
    metadata = {
        "schema": "xirt-api-tutorial-v1",
        "sequence_width": SEQUENCE_WIDTH,
        "vocabulary": VOCABULARY,
        "target_column": "rp",
        "target_units": "seconds",
        "target_mean": float(training_seconds.mean()),
        "target_std": float(training_seconds.std(ddof=0)),
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "split_ids": {
            split: table.loc[table["split"] == split, "PSMID"].tolist()
            for split in ("train", "validation", "test")
        },
    }
    features = encode_pairs(table, metadata)
    targets = ((table["rp"].to_numpy() - metadata["target_mean"]) / metadata["target_std"]).astype(
        np.float32
    )[:, None]
    return table, features, targets, metadata


def network_params() -> dict:
    """Read the CPU GRU configuration and enable shared pair branches."""
    params = yaml.safe_load((ROOT / "docs/examples/xirt-rp.yaml").read_text())
    params["siamese"]["use"] = True
    return params


def partition(features: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    return [branch[mask] for branch in features]


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Score aligned predictions in the original RT units."""
    return {
        "rmse_seconds": float(root_mean_squared_error(observed, predicted)),
        "mae_seconds": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }


def save_metadata(path: Path, metadata: dict) -> None:
    """Save this tutorial's explicit tokenizer, scale and split contract."""
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
