"""Regenerate the illustrated docs from trained local models; never train or download."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from examples.train_chromatography import ROOT, TASKS, sha256, write_json
from xirt.predictor import Predictor
from xirt.qc import plot_predictions, plot_residue_attributions, prediction_metrics, prediction_qc

LABELS = {"rp": "RP", "scx": "SCX", "hsax": "hSAX", "faims_cv": "FAIMS CV"}
COLORS = {"rp": "#287b8e", "scx": "#966733", "hsax": "#785694"}


def choose_example(table, *, faims=False):
    """Choose a short held-out precursor by ID, never by prediction or error."""
    columns = ("peptide1", "peptide2") if faims else ("Peptide1", "Peptide2")
    lengths = table[list(columns)].map(lambda value: sum(c.isupper() for c in value))
    eligible = table.loc[lengths.ge(8).all(axis=1) & lengths.le(22).all(axis=1)]
    if eligible.empty:
        raise ValueError("No held-out precursor meets the declared 8–22-residue example rule.")
    identifier = "precursor_id" if faims else "PSMID"
    return eligible.sort_values(identifier, kind="stable").iloc[:1]


def raw_inputs(table, predictor, *, faims=False):
    """Pass peptide/site/context inputs only, excluding all observed targets and scores."""
    columns = (
        ["peptide1", "peptide2", "link_pos1", "link_pos2", *predictor.model.numeric_features]
        if faims
        else ["Peptide1", "Peptide2", "LinkPos1", "LinkPos2"]
    )
    return table[columns]


def explain_example(predictor, row, task, *, faims=False):
    """Check quadrature convergence at the fixed example without choosing a nicer case."""
    inputs = raw_inputs(row, predictor, faims=faims)
    previous = predictor.explain(inputs, task=task, steps=128)
    for steps in (256, 512):
        explanation = predictor.explain(inputs, task=task, steps=steps)
        summary = explanation.summary.iloc[0]
        delta = abs(summary["prediction"] - summary["baseline_prediction"])
        error = abs(summary["completeness_error"])
        if error <= max(0.05, 0.01 * delta):
            return explanation, {
                "steps": steps,
                "previous_steps": steps // 2,
                "previous_completeness_error": float(
                    previous.summary.iloc[0]["completeness_error"]
                ),
                "max_token_change": float(
                    np.max(
                        np.abs(
                            explanation.residues["attribution"].to_numpy()
                            - previous.residues["attribution"].to_numpy()
                        )
                    )
                ),
            }
        previous = explanation
    raise ArithmeticError(f"{task} example has material completeness error even at 512 steps.")


def run(args):
    """Execute the shared prediction/XAI APIs and write figures plus aggregate provenance."""
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.models / "manifest.json").read_text())
    records = json.loads((args.models / "models.json").read_text())
    table = pd.read_csv(args.models / "partitions.csv")
    test = table.loc[table["split"] == "test"].copy()
    example = choose_example(test)
    models = {}
    scores = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), layout="constrained")
    for column, name in enumerate(("separate", "multitask")):
        for row, task in enumerate(TASKS):
            key = task if name == "separate" else "multitask"
            if key not in models:
                if sha256(args.models / key / "model.pt") != records[key]["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint no longer matches its training record: {key}")
                models[key] = Predictor.load(args.models / key / "model.pt")
            predictor = models[key]
            prediction = predictor.predict(raw_inputs(test, predictor))[f"{task}-prediction"]
            unit = predictor.model.target_transform(task)[2]
            plot_predictions(
                test[task],
                prediction,
                name=f"{LABELS[task]} · {name}",
                unit=unit,
                ax=axes[row, column],
                color=COLORS[task],
            )
            axes[row, column].collections[0].set_alpha(0.3)
            scores.append(
                {
                    "model": key,
                    "task": task,
                    "unit": unit,
                    **prediction_metrics(test[task], prediction),
                }
            )
    for pair in axes:
        limits = (
            min(axis.get_xlim()[0] for axis in pair),
            max(axis.get_xlim()[1] for axis in pair),
        )
        for axis in pair:
            axis.set(xlim=limits, ylim=limits)
    fig.savefig(args.output / "retention-predictions.png", dpi=170)
    plt.close(fig)
    cases = {}
    for task in TASKS:
        explanation, convergence = explain_example(models["multitask"], example, task)
        figure = plot_residue_attributions(explanation)
        figure.savefig(args.output / f"{task}-residues.png", dpi=170, bbox_inches="tight")
        plt.close(figure)
        cases[task] = {
            "model": "multitask",
            "PSMID": str(example["PSMID"].iloc[0]),
            "observed": float(example[task].iloc[0]),
            "unit": explanation.unit,
            **{key: float(value) for key, value in explanation.summary.iloc[0].items()},
            **convergence,
            "crosslink_tokens": int(explanation.residues["crosslinked"].sum()),
            "terminus_contribution": float(
                explanation.residues.loc[
                    explanation.residues["kind"] == "terminus", "attribution"
                ].sum()
            ),
        }
    faims = Predictor.load(args.faims_checkpoint)
    cv_table = pd.read_csv(args.faims_data)
    cv_test = cv_table.loc[cv_table["split"] == "test"].copy()
    cv_prediction = faims.predict(raw_inputs(cv_test, faims, faims=True))
    figure = prediction_qc(
        cv_test["faims_cv"],
        cv_prediction["faims_cv-prediction"],
        name="FAIMS CV · charge + m/z",
        unit="V",
    )
    figure.savefig(args.output / "faims-predictions.png", dpi=170)
    plt.close(figure)
    cv_example = choose_example(cv_test, faims=True)
    explanation, convergence = explain_example(faims, cv_example, "faims_cv", faims=True)
    figure = plot_residue_attributions(explanation)
    figure.savefig(args.output / "faims-residues.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    cases["faims_cv"] = {
        "model": "pooled_4pm_8pm__xirt_charge_mz__42",
        "precursor_id": cv_example["precursor_id"].iloc[0],
        "observed": float(cv_example["faims_cv"].iloc[0]),
        "unit": explanation.unit,
        **{key: float(value) for key, value in explanation.summary.iloc[0].items()},
        **convergence,
        "crosslink_tokens": int(explanation.residues["crosslinked"].sum()),
        "terminus_contribution": float(
            explanation.residues.loc[
                explanation.residues["kind"] == "terminus", "attribution"
            ].sum()
        ),
        "numeric": explanation.numeric.to_dict("records"),
    }
    scores.append(
        {
            "model": cases["faims_cv"]["model"],
            "task": "faims_cv",
            "unit": "V",
            **prediction_metrics(cv_test["faims_cv"], cv_prediction["faims_cv-prediction"]),
        }
    )
    provenance = {
        "training": manifest,
        "models": records,
        "sizes": json.loads((args.models / "sizes.json").read_text()),
        "checkpoints": {name: sha256(args.models / name / "model.pt") for name in models},
        "faims_checkpoint_sha256": sha256(args.faims_checkpoint),
        "faims_checkpoint_bytes": args.faims_checkpoint.stat().st_size,
        "faims_table_sha256": sha256(args.faims_data),
        "generator_sha256": sha256(Path(__file__)),
        "selection": "First held-out ID with both peptides 8–22 residues; independent of predictions/errors",
        "metrics": scores,
        "examples": cases,
        "figures": {path.name: sha256(path) for path in sorted(args.output.glob("*.png"))},
        "interpretation": "Model attributions relative to zero embeddings/fixed lengths and training-mean numeric context; not causal effects.",
    }
    write_json(args.output / "manifest.json", provenance)
    print(pd.DataFrame(scores).to_string(index=False))
    print(json.dumps(cases, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "local-data/chromatography-models")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/prediction-gallery")
    parser.add_argument(
        "--faims-checkpoint",
        type=Path,
        default=ROOT / "local-data/faims-charge/pooled_4pm_8pm__xirt_charge_mz__42.pt",
    )
    parser.add_argument(
        "--faims-data",
        type=Path,
        default=ROOT / "local-data/faims-charge/pooled_4pm_8pm__precursors.csv",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
