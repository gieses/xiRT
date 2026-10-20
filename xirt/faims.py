"""Crosslinked-peptide FAIMS compensation voltage prediction (volts, not CCS).

Data conventions follow Rappsilber-Laboratory/xiFAIMS (Apache-2.0), revision
b85482ffc246b2dc669c4592f752568fd9b1baed. This is a native xiRT implementation,
not a reproduction of the upstream descriptor/XGBoost models.
"""

import argparse
import copy
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from xirt import processing, sequences
from xirt.predictor import PeptideInputs
from xirt.xirtnet import RetentionModel, read_checkpoint, xiRTNET

logger = logging.getLogger(__name__)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
NUMERIC_FEATURES = ["precursor_charge", "precursor_mz", "precursor_mass"]
FEATURE_SETS = {
    "sequence": [],
    "charge": ["precursor_charge"],
    "charge-mz": ["precursor_charge", "precursor_mz"],
    "precursor": NUMERIC_FEATURES,
}
ALIASES = {
    "PepSeq1": "peptide1",
    "PepSeq2": "peptide2",
    "LinkPos1": "link_pos1",
    "LinkPos2": "link_pos2",
    "exp charge": "precursor_charge",
    "exp m/z": "precursor_mz",
    "exp mass": "precursor_mass",
}
UPSTREAM_REVISION = "b85482ffc246b2dc669c4592f752568fd9b1baed"


def _digest(values: Any) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def parse_compensation_voltage(run: str) -> float:
    """Parse explicit CV labels or the pinned xiFAIMS Liu run convention.

    Unsigned upstream labels denote negative voltages. Missing or multiple CV
    labels are errors; arbitrary numbers in run names are never used as targets.
    """
    if not isinstance(run, str):
        raise ValueError("FAIMS run must be a string with an explicit CV label.")
    matches = re.findall(r"(?:^|_)CV([+-]?\d+(?:\.\d+)?)(?=_|$)", run, re.IGNORECASE)
    if not matches:
        match = re.fullmatch(r"8PM_DSS_[A-Za-z]+_(\d+(?:\.\d+)?)_rep\d+", run)
        matches = [match.group(1)] if match else []
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous compensation voltage in run: {run!r}")
    value = float(matches[0])
    return value if matches[0].startswith(("+", "-")) else -value


def _unmodified(sequence: str) -> str:
    return "".join(re.findall("[A-Z]", sequence))


def precursor_inputs(
    frame: pd.DataFrame, *, numeric: bool = True, numeric_features: list[str] | None = None
) -> pd.DataFrame:
    """Validate inference fields and canonicalize peptide/site pairs without labels."""
    if frame.empty:
        raise ValueError("FAIMS input cannot be empty.")
    data = frame.copy()
    for old, new in ALIASES.items():
        if old in data:
            if new in data and not data[old].equals(data[new]):
                raise ValueError(f"Conflicting input aliases: {old} and {new}")
            if new not in data:
                data[new] = data[old]
    data = data.drop(columns=list(ALIASES), errors="ignore")
    required = ["peptide1", "peptide2", "link_pos1", "link_pos2"]
    required += (
        numeric_features if numeric_features is not None else (NUMERIC_FEATURES if numeric else [])
    )
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing FAIMS input columns: {sorted(missing)}")
    for column in required:
        if column.startswith("peptide"):
            if not data[column].map(lambda v: isinstance(v, str) and bool(v)).all():
                raise ValueError(f"{column} must contain nonempty peptide strings.")
            # xiRT's legacy parser removes N-terminal modifications. Reject them
            # here instead of silently erasing chemistry from a FAIMS input.
            cleaned = data[column].map(sequences.remove_brackets_underscores)
            if not cleaned.str.match(r"^[A-Z]").all():
                raise ValueError("N-terminal modifications are not supported for FAIMS.")
            unmodified = data[column].map(_unmodified)
            if not unmodified.map(lambda v: bool(v) and set(v) <= set(AMINO_ACIDS)).all():
                raise ValueError(f"{column} contains unsupported amino acids.")
            if data[column].str.contains(r"[.\s]").any():
                raise ValueError("Use bare peptide sequences, not flanking residues or whitespace.")
        else:
            values = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{column} must contain finite positive numbers.")
            if column.startswith("link_pos") or column == "precursor_charge":
                if (values != np.floor(values)).any():
                    raise ValueError(f"{column} must contain integers.")
                data[column] = values.astype(np.int64)
            else:
                data[column] = values
    for i in (1, 2):
        if (data[f"link_pos{i}"] > data[f"peptide{i}"].map(lambda s: len(_unmodified(s)))).any():
            raise ValueError(f"link_pos{i} exceeds peptide length (sites are 1-based).")
    original = list(
        data[["peptide1", "peptide2", "link_pos1", "link_pos2"]].itertuples(index=False, name=None)
    )
    pairs = [sorted(((p1, int(l1)), (p2, int(l2)))) for p1, p2, l1, l2 in original]
    data["peptide_swapped"] = [
        pair[0] != (p1, int(l1)) for pair, (p1, _, l1, _) in zip(pairs, original, strict=True)
    ]
    for i in (1, 2):
        data[f"peptide{i}"] = [pair[i - 1][0] for pair in pairs]
        data[f"link_pos{i}"] = [pair[i - 1][1] for pair in pairs]
    data["pair_group"] = [
        _digest(sorted((_unmodified(p1), _unmodified(p2))))
        for p1, p2 in data[["peptide1", "peptide2"]].itertuples(index=False, name=None)
    ]
    return data


def prepare_faims_observations(
    raw: pd.DataFrame,
    source: str,
    *,
    maximum_fdr: float = 0.01,
    cv_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter TT CSMs and aggregate repeated observations to mean observed CV.

    Each precursor is a source, modified peptide pair, link sites and charge.
    Counts and observed CV spread retain information about label ambiguity.
    This target is not an intensity-calibrated optimum or calibrated CCS.
    """
    if not source or not np.isfinite(maximum_fdr) or not 0 <= maximum_fdr <= 1:
        raise ValueError("Supply a source and maximum_fdr in [0, 1].")
    if not {"isTT", "fdr"} <= set(raw):
        raise ValueError("FAIMS training observations require isTT and fdr columns.")
    if not raw["isTT"].map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("isTT must contain explicit boolean target-target labels.")
    fdr = pd.to_numeric(raw["fdr"], errors="raise")
    if not np.isfinite(fdr).all() or not fdr.between(0, 1).all():
        raise ValueError("fdr must contain finite values in [0, 1].")
    selected = raw.loc[raw["isTT"] & (fdr <= maximum_fdr)].copy()
    optional = [
        feature
        for feature in NUMERIC_FEATURES[1:]
        if feature in selected
        or any(old in selected for old, new in ALIASES.items() if new == feature)
    ]
    data = precursor_inputs(selected, numeric_features=["precursor_charge", *optional])
    if cv_column is None:
        if "run" not in data:
            raise ValueError("Supply run labels or an explicit cv_column.")
        data["faims_cv"] = data["run"].map(parse_compensation_voltage)
    else:
        if cv_column not in data:
            raise ValueError(f"Missing explicit CV column: {cv_column}")
        data["faims_cv"] = pd.to_numeric(data[cv_column], errors="raise")
        if not np.isfinite(data["faims_cv"]).all():
            raise ValueError("Compensation voltages must be finite.")
    data["source"] = source
    keys = ["source", "peptide1", "peptide2", "link_pos1", "link_pos2", "precursor_charge"]
    aggregations = {feature: (feature, "mean") for feature in optional}
    aggregations.update(
        pair_group=("pair_group", "first"),
        faims_cv=("faims_cv", "mean"),
        observation_count=("faims_cv", "size"),
        distinct_cv_count=("faims_cv", "nunique"),
        cv_min_v=("faims_cv", "min"),
        cv_max_v=("faims_cv", "max"),
    )
    precursors = data.groupby(keys, sort=True, as_index=False).agg(**aggregations)
    precursors["cv_spread_v"] = precursors["cv_max_v"] - precursors["cv_min_v"]
    precursors.insert(
        0,
        "precursor_id",
        [_digest(row) for row in precursors[keys].itertuples(index=False, name=None)],
    )
    audit = {
        "source": source,
        "raw_observations": len(raw),
        "retained_observations": len(data),
        "removed_observations": len(raw) - len(data),
        "precursors": len(precursors),
        "unmodified_pair_groups": int(precursors["pair_group"].nunique()),
        "ambiguous_precursors": int((precursors["distinct_cv_count"] > 1).sum()),
        "maximum_fdr": maximum_fdr,
        "target": "CSM-weighted mean observed compensation voltage",
        "units": "V",
        "cv_min_v": float(data["faims_cv"].min()),
        "cv_max_v": float(data["faims_cv"].max()),
    }
    return precursors, audit


def split_precursors(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.75, 0.125, 0.125),
) -> pd.DataFrame:
    """Assign deterministic train/validation/test partitions by unmodified pair."""
    if (
        len(fractions) != 3
        or not np.isfinite(fractions).all()
        or min(fractions) <= 0
        or not np.isclose(sum(fractions), 1)
    ):
        raise ValueError("Split fractions must be three positive values summing to one.")
    data = precursor_inputs(frame, numeric=False)
    groups = np.array(sorted(data["pair_group"].unique()))
    np.random.default_rng(seed).shuffle(groups)
    first, second = int(len(groups) * fractions[0]), int(len(groups) * sum(fractions[:2]))
    if not 0 < first < second < len(groups):
        raise ValueError("Too few peptide pairs for three nonempty group partitions.")
    assignment = {
        group: partition
        for partition, subset in zip(
            ("train", "validation", "test"), (groups[:first], groups[first:second], groups[second:])
        )
        for group in subset
    }
    data["split"] = data["pair_group"].map(assignment)
    return data


def default_faims_params(
    *, seed: int = 42, precursor_features: bool = True, features: str = "charge"
) -> dict[str, Any]:
    """Return a shared CV GRU with sequence, charge, or charge+m/z context.

    ``precursor`` retains the original charge/mass/m/z architecture for old
    checkpoints; the focused comparison uses sequence, charge and charge-mz.
    """
    if features not in FEATURE_SETS:
        raise ValueError(f"Unknown FAIMS feature set: {features}")
    return {
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
            "units": 32,
        },
        "dense": {
            "activation": ["relu", "relu"],
            "dense_bn": [False, False],
            "dropout": [0.1, 0.1],
            "kernel_regularizer": ["l2", "l2"],
            "regularization": [False, False],
            "regularizer_value": [0.0, 0.0],
            "neurons": [64, 32],
            "nlayers": 2,
        },
        "embedding": {"length": 16},
        "learning": {
            "batch_size": 64,
            "epochs": 150,
            "learningrate": 0.003,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "device": "cpu",
            "seed": seed,
            "deterministic": True,
        },
        "output": {
            "faims_cv-activation": "linear",
            "faims_cv-column": "faims_cv",
            "faims_cv-dimension": 1,
            "faims_cv-loss": "mse",
            "faims_cv-metrics": "mse",
            "faims_cv-weight": 1,
        },
        "siamese": {"use": True, "merge_type": "add", "single_predictions": False},
        "callbacks": {
            "check_point": False,
            "log_csv": False,
            "early_stopping": True,
            "early_stopping_patience": 20,
            "tensor_board": False,
            "progressbar": False,
            "reduce_lr": True,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 8,
        },
        "predictions": {"continues": ["faims_cv"], "fractions": []},
        "encoder": {"route": "recurrent"},
        "tracking": {"enabled": False},
        "numeric_features": list(FEATURE_SETS[features]) if precursor_features else [],
    }


def _tokens(data: pd.DataFrame, maximum_residues: int) -> list[list[list[str]]]:
    if any(
        data[c].map(lambda s: len(_unmodified(s))).max() > maximum_residues
        for c in ("peptide1", "peptide2")
    ):
        raise ValueError(f"Peptide exceeds maximum_residues={maximum_residues}; no truncation.")
    prepared = processing.prepare_seqs(data.copy(), ["peptide1", "peptide2"])
    return [
        [
            sequences.convert_seqar(list(tokens), int(site))
            for tokens, site in zip(prepared[f"Seqar_peptide{i}"], data[f"link_pos{i}"])
        ]
        for i in (1, 2)
    ]


class FAIMSPredictor:
    """Train and persist a PTM-aware xiRT predictor of observed FAIMS CV in volts.

    Fit requires explicitly disjoint training and validation peptide pairs.
    All preprocessing is fitted on training data only. Inference needs no run,
    target, FDR or TT columns; unseen PTMs are counted and encoded as ``<unk>``.
    """

    def __init__(self, params: dict[str, Any] | None = None, *, maximum_residues: int = 60) -> None:
        if not isinstance(maximum_residues, int) or maximum_residues < 1:
            raise ValueError("maximum_residues must be a positive integer.")
        self.params = copy.deepcopy(params if params is not None else default_faims_params())
        if (
            self.params["predictions"] != {"continues": ["faims_cv"], "fractions": []}
            or self.params["encoder"]["route"] != "recurrent"
            or self.params["siamese"]["merge_type"] not in {"add", "average", "multiply", "maximum"}
            or self.params.get("numeric_features", []) not in FEATURE_SETS.values()
        ):
            raise ValueError(
                "FAIMS requires a symmetric recurrent CV model and known feature schema."
            )
        if (
            self.params["output"]["faims_cv-dimension"] != 1
            or self.params["output"]["faims_cv-activation"] != "linear"
            or self.params.get("fine_tuning") is not None
        ):
            raise ValueError("FAIMS requires one linear CV output without a fine-tuning override.")
        self.maximum_residues = maximum_residues
        self.model: RetentionModel | None = None
        self.metadata: dict[str, Any] = {}
        self.history: dict[str, list[float]] = {}

    def encode(self, frame: pd.DataFrame) -> PeptideInputs:
        """Encode only the checkpoint's required inputs, retaining original peptide order mapping."""
        if not self.metadata:
            raise RuntimeError("Fit or load a FAIMS checkpoint before encoding peptides.")
        data = precursor_inputs(frame, numeric_features=self.params.get("numeric_features", []))
        vocabulary = {token: i + 1 for i, token in enumerate(self.metadata["vocabulary"])}
        branches = _tokens(data, self.maximum_residues)
        arrays = []
        unknown_counts = np.zeros(len(data), dtype=np.int64)
        for branch in branches:
            encoded = np.zeros((len(data), self.maximum_residues + 2), dtype=np.int64)
            for row, tokens in enumerate(branch):
                unknown_counts[row] += sum(token not in vocabulary for token in tokens)
                encoded[row, -len(tokens) :] = [
                    vocabulary.get(token, vocabulary["<unk>"]) for token in tokens
                ]
            arrays.append(encoded)
        if unknown_counts.any():
            logger.warning(
                "%d/%d FAIMS precursors contain unseen tokens (%d total); "
                "predictions use the reserved unknown embedding.",
                int((unknown_counts > 0).sum()),
                len(data),
                int(unknown_counts.sum()),
            )
        inputs: tuple[Any, ...] = tuple(arrays)
        if self.params.get("numeric_features"):
            inputs += (data[self.params["numeric_features"]],)
        return PeptideInputs(
            inputs, branches, frame.index, data["peptide_swapped"].to_numpy(), unknown_counts
        )

    def fit(
        self, train: pd.DataFrame, validation: pd.DataFrame, *, verbose: int = 0
    ) -> "FAIMSPredictor":
        """Fit vocabulary, target scaling and network without test-set access."""
        features = self.params.get("numeric_features", [])
        train = precursor_inputs(train, numeric_features=features)
        validation = precursor_inputs(validation, numeric_features=features)
        if set(train["pair_group"]) & set(validation["pair_group"]):
            raise ValueError("Training and validation must have disjoint unmodified peptide pairs.")
        targets = []
        for data in (train, validation):
            if "faims_cv" not in data:
                raise ValueError("Training and validation require faims_cv targets in volts.")
            target = pd.to_numeric(data["faims_cv"], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(target).all():
                raise ValueError("CV targets must be finite.")
            targets.append(target)
        tokens = _tokens(train, self.maximum_residues)
        vocabulary = {"<unk>", "H-", "-OH", *AMINO_ACIDS, *("cl" + a for a in AMINO_ACIDS)}
        vocabulary.update(token for branch in tokens for row in branch for token in row)
        mean, scale = float(targets[0].mean()), float(targets[0].std())
        self.metadata = {
            "schema": "xirt-faims-v1",
            "units": "V",
            "target": "mean observed compensation voltage; not CCS or intensity optimum",
            "maximum_residues": self.maximum_residues,
            "vocabulary": sorted(vocabulary),
            "target_mean": mean,
            "target_scale": scale if scale > 0 else 1.0,
            "training_rows": len(train),
            "validation_rows": len(validation),
            "upstream_revision": UPSTREAM_REVISION,
        }
        params = copy.deepcopy(self.params)
        params["_faims"] = copy.deepcopy(self.metadata)
        params["_alphabet"] = self.metadata["vocabulary"]
        network = xiRTNET(params, self.maximum_residues + 2, len(vocabulary) + 1)
        network.build_model(siamese=True)
        network.compile()
        assert network.model is not None
        self.model = network.model
        scale = self.metadata["target_scale"]
        train_inputs = self.encode(train).features
        val_inputs = self.encode(validation).features
        result = self.model.fit(
            train_inputs,
            ((targets[0] - mean) / scale)[:, None],
            validation_data=(val_inputs, ((targets[1] - mean) / scale)[:, None]),
            epochs=params["learning"]["epochs"],
            batch_size=params["learning"]["batch_size"],
            callbacks=params["callbacks"],
            verbose=verbose,
        )
        self.history = result.history
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict volts in input row order and expose unseen-token coverage."""
        if self.model is None:
            raise RuntimeError("Fit or load a FAIMS checkpoint before predicting.")
        encoded = self.encode(frame)
        result = self.model.predict_frame(encoded.features, index=frame.index)
        result["unknown_token_count"] = encoded.unknown_token_count
        result["faims_cv_prediction_v"] = result["faims_cv-prediction"]
        return result

    def explain(self, frame: pd.DataFrame, *, steps: int = 64) -> Any:
        """Explain CV at residue level, including charge/m/z contributions in volts."""
        from xirt.predictor import Predictor

        if self.model is None:
            raise RuntimeError("Fit or load a FAIMS checkpoint before explaining predictions.")
        predictor = Predictor(self.model)
        predictor.faims = self
        return predictor.explain(frame, task="faims_cv", steps=steps)

    def save(self, location: str | Path) -> None:
        """Save native tensor weights, vocabulary, scaling and CV semantics."""
        if self.model is None:
            raise RuntimeError("Fit or load a FAIMS model before saving.")
        self.model.save(location)

    @classmethod
    def load(cls, location: str | Path, *, device: str = "cpu") -> "FAIMSPredictor":
        """Reconstruct a FAIMS checkpoint, rejecting RT-only checkpoints."""
        checkpoint = read_checkpoint(location)
        params = checkpoint["params"]
        metadata = params.get("_faims", {})
        if metadata.get("schema") != "xirt-faims-v1" or metadata.get("units") != "V":
            raise ValueError("Not a supported FAIMS compensation voltage checkpoint.")
        params["learning"]["device"] = device
        predictor = cls(params, maximum_residues=metadata["maximum_residues"])
        predictor.metadata = metadata
        network = xiRTNET(params, checkpoint["input_dim"], checkpoint["alphabet"])
        network.load_model(location)
        predictor.model = network.model
        return predictor


def cv_metrics(observed: Any, predicted: Any) -> dict[str, Any]:
    """Compute voltage errors; R² is explicitly undefined for constant/tiny sets."""
    from xirt.qc import prediction_metrics

    metrics = prediction_metrics(observed, predicted)
    return {
        "n": metrics["n"],
        "rmse_v": metrics["rmse"],
        "mae_v": metrics["mae"],
        "r2": metrics["r2"],
    }


def precursor_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    """Return symmetric, target-free precursor and sequence composition features."""
    data = precursor_inputs(frame)
    plain = [[_unmodified(p) for p in data[f"peptide{i}"]] for i in (1, 2)]
    lengths = np.array([[len(p) for p in branch] for branch in plain])
    result = data[NUMERIC_FEATURES].copy()
    result["length_min"] = lengths.min(axis=0)
    result["length_max"] = lengths.max(axis=0)
    for aa in AMINO_ACIDS:
        result[f"fraction_{aa}"] = [
            (p1.count(aa) + p2.count(aa)) / (len(p1) + len(p2)) for p1, p2 in zip(*plain)
        ]
    return result


def add_faims_subcommands(subparsers: Any) -> None:
    """Register FAIMS commands without changing the historical RT CLI."""
    faims = subparsers.add_parser("faims", help="Train/predict FAIMS CV in volts (not CCS).")
    actions = faims.add_subparsers(dest="faims_action", required=True)
    train = actions.add_parser("train", help="Train with grouped held-out evaluation.")
    train.add_argument("--input", nargs="+", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--split-seed", type=int, default=42)
    train.add_argument("--epochs", type=int)
    train.add_argument("--params", type=Path)
    train.add_argument("--cv-column")
    features = train.add_mutually_exclusive_group()
    features.add_argument("--features", choices=list(FEATURE_SETS), default=None)
    features.add_argument("--no-precursor-features", action="store_true")
    predict = actions.add_parser(
        "predict", help="Predict from sequences, sites and precursor fields."
    )
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument(
        "--qc", type=Path, help="QC output directory; requires observed target values."
    )
    predict.add_argument("--observed", help="Observed target column used only for QC.")


def run_faims_command(args: argparse.Namespace) -> None:
    """Execute explicit FAIMS training or label-free prediction."""
    if args.faims_action == "predict":
        from xirt.predictor import run_prediction_command

        run_prediction_command(args)
        return
    sources = [path.stem for path in args.input]
    if len(set(sources)) != len(sources):
        raise ValueError("Input file stems must identify distinct sources.")
    prepared, audits = [], []
    for path, source in zip(args.input, sources):
        data, audit = prepare_faims_observations(
            pd.read_csv(path), source, cv_column=args.cv_column
        )
        prepared.append(data)
        audits.append(audit)
    partitions = split_precursors(pd.concat(prepared, ignore_index=True), seed=args.split_seed)
    params = (
        yaml.safe_load(args.params.read_text())
        if args.params
        else default_faims_params(
            precursor_features=not args.no_precursor_features, features=args.features or "charge"
        )
    )
    params["learning"]["seed"] = args.seed
    if args.epochs is not None:
        params["learning"]["epochs"] = args.epochs
    if args.no_precursor_features:
        params["numeric_features"] = []
    elif args.features is not None:
        params["numeric_features"] = list(FEATURE_SETS[args.features])
    predictor = FAIMSPredictor(params).fit(
        partitions.loc[partitions["split"] == "train"],
        partitions.loc[partitions["split"] == "validation"],
    )
    predictions = predictor.predict(partitions)
    report = partitions.join(predictions)
    metrics = {
        name: cv_metrics(subset["faims_cv"], subset["faims_cv_prediction_v"])
        for name, subset in report.groupby("split")
    }
    args.output.mkdir(parents=True, exist_ok=True)
    predictor.save(args.output / "faims.pt")
    report.to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame(predictor.history).to_csv(args.output / "history.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (args.output / "audit.json").write_text(json.dumps(audits, indent=2, allow_nan=False) + "\n")
    import matplotlib.pyplot as plt

    from xirt import qc

    test = report.loc[report["split"] == "test"]
    figure = qc.prediction_qc(
        test["faims_cv"], test["faims_cv-prediction"], name="FAIMS CV · held-out", unit="V"
    )
    figure.savefig(args.output / "qc_predictions.png", dpi=150)
    plt.close(figure)
    figure = qc.plot_learning_curve(
        predictor.history, scale=predictor.metadata["target_scale"], unit="V"
    )
    figure.savefig(args.output / "qc_learning_curve.png", dpi=150)
    plt.close(figure)
