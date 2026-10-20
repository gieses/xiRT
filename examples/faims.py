"""Train a charge-aware xiRT model of observed FAIMS compensation voltage."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
    # Predict FAIMS compensation voltage

    Train xiRT's shared GRU with peptide sequences, crosslink sites and charge.
    The target is the **mean observed compensation voltage in volts**, not CCS
    or an intensity-calibrated optimal voltage. The bundled example contains
    915 precursors from 256 unmodified peptide pairs in the original 4PM/8PM data.

    All charge states, modifications and sites of a pair stay in one partition.
    Vocabulary and scaling are fitted on training rows only; validation selects
    the checkpoint. Proteins, individual peptides and studies are not held out.
    This small example overlaps the full benchmark and is not independent evidence.
    """)
    return (mo,)


@app.cell
def data():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch

    from xirt.faims import FAIMSPredictor, default_faims_params, split_precursors
    from xirt.predictor import Predictor
    from xirt.qc import plot_learning_curve, prediction_metrics, prediction_qc

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    table = split_precursors(pd.read_csv(root / "examples/data/faims.csv"), seed=42)
    table.groupby("split").agg(precursors=("pair_group", "size"), pairs=("pair_group", "nunique"))
    return (
        FAIMSPredictor,
        Predictor,
        default_faims_params,
        np,
        pd,
        plot_learning_curve,
        prediction_metrics,
        prediction_qc,
        root,
        table,
    )


@app.cell
def settings():
    epochs = 150
    feature_set = "charge"  # Alternatives: "sequence" and "charge-mz".
    return epochs, feature_set


@app.cell
def output_directory(root):
    output = root / "examples/results/faims"
    output.mkdir(parents=True, exist_ok=True)
    return (output,)


@app.cell
def training(FAIMSPredictor, default_faims_params, epochs, feature_set, table):
    params = default_faims_params(seed=42, features=feature_set)
    params["learning"]["epochs"] = epochs
    fitted = FAIMSPredictor(params).fit(
        table.loc[table["split"] == "train"],
        table.loc[table["split"] == "validation"],
    )
    return (fitted,)


@app.cell
def checkpoint(Predictor, fitted, output):
    checkpoint_path = output / "model.pt"
    fitted.save(checkpoint_path)
    predictor = Predictor.load(checkpoint_path)
    return (predictor,)


@app.cell
def evaluation(fitted, np, output, pd, prediction_metrics, predictor, table):
    test_table = table.loc[table["split"] == "test"].copy()
    prediction = predictor.predict(test_table)
    test_table["faims_cv-prediction"] = prediction["faims_cv-prediction"]
    np.testing.assert_allclose(
        prediction["faims_cv-prediction"],
        fitted.predict(test_table)["faims_cv-prediction"],
        rtol=1e-6,
    )
    train_mean = table.loc[table["split"] == "train", "faims_cv"].mean()
    metrics = pd.DataFrame(
        [
            {
                "model": "xiRT",
                **prediction_metrics(test_table["faims_cv"], prediction["faims_cv-prediction"]),
            },
            {
                "model": "Training mean",
                **prediction_metrics(test_table["faims_cv"], np.full(len(test_table), train_mean)),
            },
        ]
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    test_table.to_csv(output / "test-predictions.csv", index=False)
    pd.DataFrame(fitted.history).to_csv(output / "history.csv", index=False)
    return metrics, test_table


@app.cell
def plots(
    fitted,
    metrics,
    mo,
    output,
    plot_learning_curve,
    prediction_qc,
    test_table,
):
    learning_figure = plot_learning_curve(
        fitted.history, scale=fitted.metadata["target_scale"], unit="V"
    )
    prediction_figure = prediction_qc(
        test_table["faims_cv"],
        test_table["faims_cv-prediction"],
        name="Held-out FAIMS CV",
        unit="V",
    )
    learning_figure.savefig(output / "training-loss.png", dpi=150)
    prediction_figure.savefig(output / "test-predictions.png", dpi=150)
    mo.vstack([learning_figure, prediction_figure, mo.ui.table(metrics, selection=None)])
    return


@app.cell
def reuse(mo):
    mo.md("""
    ## Use the same prediction interface for RT and FAIMS

    ```python
    from xirt.predictor import Predictor

    predictor = Predictor.load("examples/results/faims/model.pt")
    predictions = predictor.predict(precursors)
    explanation = predictor.explain(precursors.iloc[:1], steps=64)
    ```

    ```bash
    xirt predict --checkpoint examples/results/faims/model.pt     --input precursors.csv --output predictions.csv
    ```

    Inference needs only sequences, link sites and the selected precursor
    features, not CV labels or FDR columns. Unseen tokens are reported rather
    than silently discarded. Open `examples/analyze_predictions.py` for shared
    RT/FAIMS QC and residue-level interpretation.

    Dataset hashes and sampling details are in `examples/data/faims-provenance.json`.
    The controlled three-model comparison is in `benchmark-faims/`; the earlier
    precursor/baseline experiment is preserved under its `archive/` directory.
    """)
    return


if __name__ == "__main__":
    app.run()
