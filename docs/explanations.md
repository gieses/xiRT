# Shared prediction and residue-level explanations

For executed predictions, annotated plots and matched RP/SCX/hSAX and FAIMS
residue examples, see the [illustrated walkthrough](prediction-gallery.md).

Self-contained recurrent RT and FAIMS checkpoints use the same public interface:

```python
from xirt.predictor import Predictor
from xirt.qc import prediction_qc, plot_residue_attributions

predictor = Predictor.load()  # Bundled joint RP/SCX/hSAX model.
predictions = predictor.predict(peptides)
explanation = predictor.explain(peptides.iloc[:1], task="rp", steps=64)
figure = plot_residue_attributions(explanation)
```

Use task `faims_cv` for compensation voltage; omit `task` for a single-head
checkpoint. Raw inference requires the checkpoint's saved preprocessing metadata.
With no path, `Predictor.load()` uses the bundled joint RP/SCX/hSAX weights.
Select `model="rp"`, `"scx"`, `"hsax"` or `"faims"` for the separate models;
`Predictor.load("model.pt")` preserves custom checkpoint loading. Named models
and custom paths are mutually exclusive. Packaged files are resolved with
`importlib.resources.files()` and `as_file()`, so installed resources need no
checkout-relative path or network download. See the [model cards](chromatography-models.md).

The bundled RT and FAIMS tutorials save it automatically. Older RT checkpoints
without this metadata still support encoded-array prediction, not raw-peptide
inference through this adapter. Frozen ESM and pseudolinear checkpoints retain
their existing workflows; this residue explanation method requires native
recurrent embeddings and a scalar continuous output.

## Command line

```bash
xirt predict --input precursors.csv \
  --output predictions.csv --qc qc --observed rp --task rp
xirt explain --input precursors.csv \
  --output explanations --task rp --limit 1 --steps 64
```

Both commands default to the joint model. Use `--model rp`, `--model scx`,
`--model hsax` or `--model faims` to select bundled weights, or `--checkpoint
model.pt` for a custom file. The two source options cannot be combined. A joint
model needs `--task` for explanations and optional QC, but predicts all three
outputs without it.

Prediction preserves input row order and adds `<task>-prediction` in the original
target units. Labels are needed only for optional QC. FAIMS also retains
`faims_cv_prediction_v` as a compatibility alias. The explanation command writes
`residues.csv`, `numeric.csv`, `summary.csv`, `explanation.json` and `residues.png`.
`--limit` bounds computation; explanations otherwise cover all input rows.
The plotted example is the first explained row.

## Baseline and interpretation

Embedding integrated gradients integrates a joint straight-line path from
zero embeddings at unchanged peptide lengths and padding masks, plus
training-mean numeric context, to the input. It differentiates embeddings,
not integer token IDs. Gauss-Legendre quadrature uses 64 integration points
by default (accepted range 2–512).
The method follows [Sundararajan, Taly and Yan (2017)](https://proceedings.mlr.press/v70/sundararajan17a.html).

`Explanation.residues` retains input index, sample, peptide branch, one-based
residue position, token, crosslink flag, token kind and signed attribution.
Terminus tokens appear separately when the tokenizer uses them. Canonicalized
FAIMS peptide branches are mapped back to the original input order.
`Explanation.numeric` records feature values, training-mean baselines and
contributions. `Explanation.summary` reports prediction, baseline prediction,
total attribution, completeness error and unknown-token count.

Positive values raise the prediction relative to the baseline; negative values
lower it. Contributions are in the task's original units (seconds for tutorial
RT, volts for FAIMS). The baseline is neither an empty peptide nor a physical
counterfactual. These are **model explanations, not causal chemical effects**.
Interaction effects depend on the joint path and baseline.

The completeness error is prediction minus baseline minus the sum of all token
and numeric contributions, including termini. Increase `steps` when the
numerical integration error is material relative to the prediction change.
A small completeness error establishes numerical consistency, not biological
validity. Unknown PTMs share an unknown embedding; inspect reported coverage
before interpreting them.

## Common QC

`prediction_metrics` returns RMSE, MAE, predictive R-squared, Pearson r and
Spearman rho; undefined statistics are `None`, not fabricated zeroes.
`prediction_qc` adds an identity-line scatterplot with annotated correlations
and a residual plot. The same tools serve RT, FAIMS, notebooks and the CLI.
`examples/analyze_predictions.py` applies them to both tutorial checkpoints.
