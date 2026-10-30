"""Train separate RP/SCX/hSAX and joint regressors on the full Zenodo E. coli table.

Data, partitions, predictions and weights stay in ignored local-data. The native
checkpoints include raw-peptide preprocessing and original-unit transformations.
No training is performed by the documentation build.
"""

import argparse
import copy
import hashlib
import json
import logging
import platform
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from xirt.predictor import Predictor, preprocess
from xirt.qc import prediction_metrics
from xirt.xirtnet import xiRTNET

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("rp", "scx", "hsax")
MODELS = {"rp": ["rp"], "scx": ["scx"], "hsax": ["hsax"], "multitask": list(TASKS)}
logger = logging.getLogger(__name__)


def sha256(path):
    """Fingerprint source files, downloaded tables and native checkpoints."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Persist finite, inspectable metadata without custom Python objects."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_table(frame, *, seed=42, fractions=(0.8, 0.1, 0.1)):
    """Keep all confident CSMs and partition by unmodified unordered peptide pair."""
    required = {
        "PSMID",
        "Peptide1",
        "Peptide2",
        "LinkPos1",
        "LinkPos2",
        "score",
        "isTT",
        "fdr",
        *TASKS,
    }
    if not required <= set(frame):
        raise ValueError(f"Missing chromatography columns: {sorted(required - set(frame))}")
    if not frame["isTT"].isin([True, False]).all():
        raise ValueError("isTT must contain explicit boolean values.")
    fdr = pd.to_numeric(frame["fdr"], errors="raise")
    if not np.isfinite(fdr).all() or not fdr.between(0, 1).all():
        raise ValueError("FDR must be finite and in [0, 1].")
    table = frame.loc[frame["isTT"].eq(True) & fdr.le(0.01)].copy()
    if table.empty or table["PSMID"].isna().any() or not table["PSMID"].is_unique:
        raise ValueError("Confident CSMs require nonempty, unique PSMIDs.")
    for task in TASKS:
        table[task] = pd.to_numeric(table[task], errors="raise")
        if not np.isfinite(table[task]).all():
            raise ValueError(f"Nonfinite observed {task} labels.")
    if not table["rp"].gt(0).all() or any(
        not np.equal(table[task], np.floor(table[task])).all() or not table[task].gt(0).all()
        for task in ("scx", "hsax")
    ):
        raise ValueError("RP must be positive; fractions must be positive integers.")
    pairs = []
    for row in table.itertuples():
        bodies = []
        for sequence, site in ((row.Peptide1, row.LinkPos1), (row.Peptide2, row.LinkPos2)):
            if not isinstance(sequence, str) or not re.fullmatch("[A-Za-z]+", sequence):
                raise ValueError("Expected bare xiRT modified peptide sequences.")
            body = "".join(re.findall("[A-Z]", sequence))
            if (
                not body
                or not set(body) <= set("ACDEFGHIKLMNPQRSTVWY")
                or not isinstance(site, (int, np.integer))
                or not 1 <= site <= len(body)
            ):
                raise ValueError("Invalid peptide or one-based crosslink position.")
            bodies.append(body)
        pairs.append("|".join(sorted(bodies)))
    table["pair_group"] = pairs
    groups = np.array(sorted(set(pairs)))
    if (
        len(groups) < 10
        or len(fractions) != 3
        or any(f <= 0 for f in fractions)
        or not np.isclose(sum(fractions), 1)
    ):
        raise ValueError(
            "Require at least ten pair groups and three positive split fractions summing to one."
        )
    groups = np.random.default_rng(seed).permutation(groups)
    first, second = int(len(groups) * fractions[0]), int(len(groups) * sum(fractions[:2]))
    membership = {
        group: name
        for name, part in zip(
            ("train", "validation", "test"), np.split(groups, [first, second]), strict=True
        )
        for group in part
    }
    table["split"] = table["pair_group"].map(membership)
    return table.reset_index(drop=True)


def fit_model(table, tasks, config, output, *, device="cpu", epochs=None):
    """Fit one native model with training-only vocabulary/scaling and validation stopping."""
    train = table.loc[table["split"] == "train"]
    validation = table.loc[table["split"] == "validation"]
    params = copy.deepcopy(config["network"])
    params["learning"]["device"] = device
    if epochs is not None:
        params["learning"]["epochs"] = epochs
    width = config["sequence_width"]
    if (
        table[["Peptide1", "Peptide2"]]
        .map(lambda s: len(re.findall("[A-Z]", s)) + 2 > width)
        .any()
        .any()
    ):
        raise ValueError(
            "A peptide exceeds the fixed checkpoint width; no rows may be truncated or dropped."
        )
    encoded = preprocess(train.copy(), max_length=width, cl_residue=True)
    if len(encoded.psms) != len(train):
        raise ValueError("Preprocessing dropped training CSMs.")
    params["_alphabet"] = encoded.le.classes_.tolist()
    params["_provenance"] = {
        "preprocessing": {"cl_residue": True},
        "training": {
            "sequence_type": "crosslink",
            "source": config["source"],
            "split": config["split"],
        },
    }
    params["_target_transforms"] = {
        task: {
            "scale": float(train[task].std(ddof=0)),
            "offset": float(train[task].mean()),
            "unit": "s" if task == "rp" else "fraction",
        }
        for task in tasks
    }
    if any(value["scale"] <= 0 for value in params["_target_transforms"].values()):
        raise ValueError("Training targets must vary in every selected dimension.")
    params["predictions"] = {"continues": list(tasks), "fractions": []}
    params["output"] = {
        f"{task}-{key}": value
        for task in tasks
        for key, value in {
            "activation": "linear",
            "column": task,
            "dimension": 1,
            "loss": "mse",
            "metrics": "mse",
            "weight": 1,
        }.items()
    }
    network = xiRTNET(params, width, len(params["_alphabet"]) + 1)
    network.build_model(siamese=True)
    network.compile()
    model = network.model
    predictor = Predictor(model)
    inputs = encoded.get_features(encoded.psms.index)
    validation_inputs = predictor.encode(validation).features
    # Strict raw inference also checks held-out token coverage before fitting.
    predictor.encode(table.loc[table["split"] == "test"])

    def targets(data):
        return [
            (
                (data[task].to_numpy() - params["_target_transforms"][task]["offset"])
                / params["_target_transforms"][task]["scale"]
            )[:, None]
            for task in model.tasks
        ]

    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = model.fit(
        inputs,
        targets(encoded.psms),
        validation_data=(validation_inputs, targets(validation)),
        epochs=params["learning"]["epochs"],
        batch_size=params["learning"]["batch_size"],
        callbacks=params["callbacks"],
    )
    model.save(output / "model.pt")
    history = pd.DataFrame(result.history)
    history.to_csv(output / "history.csv", index=False)
    restored = Predictor.load(output / "model.pt")
    report = table.copy()
    predictions = restored.predict(table)
    for column in predictions:
        report[column] = predictions[column]
    report.to_csv(output / "predictions.csv", index=False)
    metrics = [
        {
            "task": task,
            "split": split,
            **prediction_metrics(part[task], part[f"{task}-prediction"]),
            "training_mean_rmse": float(
                np.sqrt(np.mean((part[task] - params["_target_transforms"][task]["offset"]) ** 2))
            ),
        }
        for split, part in report.groupby("split", sort=False)
        for task in model.tasks
    ]
    write_json(output / "metrics.json", metrics)
    record = {
        "tasks": model.tasks,
        "epochs": len(history),
        "best_epoch": int(history["val_loss"].argmin()) + 1,
        "seconds": time.monotonic() - started,
        "checkpoint_sha256": sha256(output / "model.pt"),
        "checkpoint_bytes": (output / "model.pt").stat().st_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "target_transforms": params["_target_transforms"],
    }
    write_json(output / "record.json", record)
    return record


def run(args):
    """Freeze the full-data protocol, train four models and report actual artifact sizes."""
    config = yaml.safe_load(args.config.read_text())
    if sha256(args.data) != config["source"]["sha256"]:
        raise ValueError("Source checksum differs from the pinned Zenodo table.")
    table = prepare_table(
        pd.read_csv(args.data, low_memory=False),
        seed=config["split"]["seed"],
        fractions=config["split"]["fractions"],
    )
    torch.set_num_threads(2)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training requested but this Python environment has no CUDA support."
        )
    if args.epochs is not None:
        config["network"]["learning"]["epochs"] = args.epochs
    sources = [
        Path(__file__),
        args.config,
        ROOT / "xirt/predictor.py",
        ROOT / "xirt/xirtnet.py",
        ROOT / "xirt/processing.py",
        ROOT / "xirt/sequences.py",
    ]
    manifest = {
        "config": config,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "rows": len(table),
        "pair_groups": table["pair_group"].nunique(),
        "partitions": table.groupby("split")
        .agg(csms=("PSMID", "size"), pairs=("pair_group", "nunique"))
        .to_dict("index"),
        "objective": "training-standardized scalar regression; fractions are not ordinal classifiers",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a fresh output directory.")
    write_json(manifest_path, manifest)
    table.to_csv(args.output / "partitions.csv", index=False)
    records = {}
    for name, tasks in MODELS.items():
        output = args.output / name
        record_path = output / "record.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["checkpoint_sha256"] != sha256(output / "model.pt"):
                raise ValueError(f"Changed checkpoint: {output}")
        else:
            logger.info("Training %s on %d confident CSMs", name, len(table))
            record = fit_model(table, tasks, config, output, device=args.device)
        records[name] = record
        logger.info("%s: %s", name, record)
    write_json(args.output / "models.json", records)
    with tarfile.open(args.output / "chromatography-models.tar.gz", "w:gz") as bundle:
        for name in MODELS:
            for filename in ("model.pt", "record.json", "metrics.json"):
                bundle.add(args.output / name / filename, arcname=f"{name}/{filename}")
        bundle.add(manifest_path, arcname="manifest.json")
    sizes = {name: record["checkpoint_bytes"] for name, record in records.items()}
    sizes["total_checkpoint_bytes"] = sum(sizes.values())
    sizes["compressed_bundle_bytes"] = (args.output / "chromatography-models.tar.gz").stat().st_size
    write_json(args.output / "sizes.json", sizes)
    print(json.dumps(sizes, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "local-data/zenodo-4720760/ecoli/ecoli/DSS_CSM_anpxirt.csv",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "examples/chromatography_models.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "local-data/chromatography-models")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
