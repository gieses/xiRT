"""Compare RT/FAIMS predictions and explain individual crosslinks at residue level."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
    # Understand a prediction

    Run `examples/train_predict.py` and `examples/faims.py` first. This notebook
    loads their checkpoints and held-out rows; it does not train models.
    The same public `Predictor` and QC tools work for retention time and FAIMS CV.

    Correlations describe association, whereas RMSE and predictive R-squared
    measure agreement. None establishes performance on a new instrument or study.
    """)
    return (mo,)


@app.cell
def imports():
    from pathlib import Path

    import pandas as pd
    import torch

    from xirt.predictor import Predictor
    from xirt.qc import plot_residue_attributions, prediction_metrics, prediction_qc

    torch.set_num_threads(2)
    return (
        Path,
        Predictor,
        pd,
        plot_residue_attributions,
        prediction_metrics,
        prediction_qc,
    )


@app.cell
def directories(Path):
    result_root = Path(__file__).resolve().parent / "results"
    analysis_output = result_root / "analysis"
    analysis_output.mkdir(parents=True, exist_ok=True)
    return analysis_output, result_root


@app.cell
def saved_models(Predictor, pd, result_root):
    experiments = {
        "RT": {"directory": "train-predict", "task": "rp", "observed": "rp", "unit": "s"},
        "FAIMS": {
            "directory": "faims",
            "task": "faims_cv",
            "observed": "faims_cv",
            "unit": "V",
        },
    }
    models, tables = {}, {}
    for _name, _spec in experiments.items():
        _folder = result_root / _spec["directory"]
        if not (_folder / "model.pt").is_file() or not (_folder / "test-predictions.csv").is_file():
            raise FileNotFoundError(
                f"Run examples/{'train_predict' if _name == 'RT' else 'faims'}.py first; "
                f"missing checkpoint or held-out predictions in {_folder}."
            )
        models[_name] = Predictor.load(_folder / "model.pt")
        tables[_name] = pd.read_csv(_folder / "test-predictions.csv")
    return experiments, models, tables


@app.cell
def comparisons(
    analysis_output,
    experiments,
    mo,
    models,
    pd,
    prediction_metrics,
    prediction_qc,
    tables,
):
    figures, scores = [], []
    for _name, _spec in experiments.items():
        _table = tables[_name]
        _prediction = models[_name].predict(_table)[f"{_spec['task']}-prediction"]
        _figure = prediction_qc(
            _table[_spec["observed"]], _prediction, name=_name, unit=_spec["unit"]
        )
        _figure.savefig(analysis_output / f"{_name.lower()}-qc.png", dpi=150)
        figures.append(_figure)
        scores.append(
            {
                "task": _name,
                "unit": _spec["unit"],
                **prediction_metrics(_table[_spec["observed"]], _prediction),
            }
        )
    score_table = pd.DataFrame(scores)
    score_table.to_csv(analysis_output / "metrics.csv", index=False)
    mo.vstack([*figures, mo.ui.table(score_table, selection=None)])
    return


@app.cell
def controls(experiments, mo):
    task_picker = mo.ui.dropdown(options=list(experiments), value="FAIMS", label="Prediction task")
    task_picker
    return (task_picker,)


@app.cell
def samples(mo, tables, task_picker):
    sample_picker = mo.ui.slider(
        start=0,
        stop=len(tables[task_picker.value]) - 1,
        value=0,
        label="Held-out row",
        show_value=True,
    )
    sample_picker
    return (sample_picker,)


@app.cell
def attribution(experiments, models, sample_picker, tables, task_picker):
    selected = tables[task_picker.value].iloc[[sample_picker.value]]
    explanation = models[task_picker.value].explain(
        selected, task=experiments[task_picker.value]["task"], steps=64
    )
    return explanation, selected


@app.cell
def residue_plot(
    analysis_output,
    explanation,
    mo,
    plot_residue_attributions,
    sample_picker,
    selected,
    task_picker,
):
    residue_figure = plot_residue_attributions(explanation)
    prefix = f"{task_picker.value.lower()}-{sample_picker.value}"
    residue_figure.savefig(analysis_output / f"{prefix}-residues.png", dpi=150)
    explanation.residues.to_csv(analysis_output / f"{prefix}-residues.csv", index=False)
    explanation.numeric.to_csv(analysis_output / f"{prefix}-context.csv", index=False)
    explanation.summary.to_csv(analysis_output / f"{prefix}-completeness.csv", index=False)
    mo.vstack(
        [
            mo.ui.table(selected, selection=None),
            residue_figure,
            mo.ui.table(explanation.summary, selection=None),
            mo.ui.table(explanation.residues, selection=None),
            mo.ui.table(explanation.numeric, selection=None),
        ]
    )
    return


@app.cell
def interpretation(mo):
    mo.md("""
    ## Read the contributions, not a causal mechanism

    Integrated gradients follows a joint path from **zero token embeddings at
    fixed peptide lengths** and **training-mean precursor features** to the input.
    This baseline is not an empty peptide or a chemically realizable reference.
    Positive contributions raise the prediction relative to that baseline;
    negative contributions lower it. Values are in seconds (RT) or volts (CV).

    Each residue keeps its peptide, one-based position and modification token.
    Crosslinked residues are bold; terminus contributions are listed separately
    and included in the total. Charge and m/z appear only if the model uses them.
    Contributions describe the fitted model, **not causal residue effects**.

    The completeness error is prediction minus baseline minus all token and
    numeric contributions. Increase `steps` if this numerical integration error
    is appreciable; a small error does not make an explanation biologically true.
    Unknown-token counts flag coverage limits; unknown PTMs share one embedding.
    """)
    return


if __name__ == "__main__":
    app.run()
