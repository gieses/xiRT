# FAIMS compensation-voltage prediction

xiRT predicts crosslinked-peptide **compensation voltage (CV) in volts, not
collision cross section (CCS)**. These data do not establish a calibrated
CV-to-CCS conversion or an intensity-calibrated optimal voltage.

The integration follows the data conventions of
[xiFAIMS](https://github.com/Rappsilber-Laboratory/xiFAIMS), pinned at
`b85482ffc246b2dc669c4592f752568fd9b1baed` (Apache-2.0). It trains xiRT's
native shared GRU, not the upstream XGBoost model.

## Train from observations

```bash
xirt faims train --input 4PM_DSS_LS_nonunique1pCSM.csv \
  8PM_DSS_Liu_nonunique1pCSM.csv --output results/faims --features charge
```

| Canonical column | xiFAIMS alias | Meaning |
| --- | --- | --- |
| `peptide1`, `peptide2` | `PepSeq1`, `PepSeq2` | Bare modified peptide sequences |
| `link_pos1`, `link_pos2` | `LinkPos1`, `LinkPos2` | One-based crosslink sites |
| `precursor_charge` | `exp charge` | Positive integer charge |
| `precursor_mz` | `exp m/z` | Positive finite m/z, needed for `charge-mz` |
| `precursor_mass` | `exp mass` | Optional legacy precursor context |
| `isTT` | same | Explicit boolean target-target identification |
| `fdr` | same | Identification FDR in [0, 1] |
| `run` | same | Explicit CV label, unless `--cv-column` is supplied |

`--features` selects `sequence` (no numeric context), `charge` (default), or
`charge-mz`. Legacy `precursor` uses charge, m/z and mass for compatibility
with the archived first experiment. `--no-precursor-features` is an alias for
the sequence-only ablation. Charge is required during observation aggregation
even for sequence-only training, because it defines a precursor.

Unsigned `CV045` run labels mean negative voltages. Only explicit `CV` labels
and the pinned `8PM_DSS_A_50_rep1`-style Liu convention are parsed; arbitrary
filename numbers are not voltages. Supply a finite numeric voltage column
with `--cv-column your_column` for another convention.

Training retains explicit TT observations with `fdr <= 0.01`, canonicalizes
peptide/site pairs and averages observed CV per modified pair, sites, charge
and source. Source IDs are distinct input filename stems. Do not combine an
aggregate CSV with its constituent original observations.

The 75/12.5/12.5% split groups unordered **unmodified peptide pairs**, keeping
charges, PTMs and sites together. `--split-seed` controls partitions and
`--seed` initialization. Vocabulary, numeric normalization and target scaling
use training rows only. Validation controls early stopping and restores the
best weights. `--epochs` changes the 150-epoch ceiling; `--params` accepts a
full mapping from `default_faims_params()`. Outputs include `faims.pt`,
`predictions.csv`, `history.csv`, `metrics.json`, `audit.json` and shared
prediction/learning-curve QC PNGs.

For a complete copyable Python train/save/reload/predict example, see
[Python workflows: FAIMS](python-workflows.md#5-train-and-run-faims).

## Predict and explain

```bash
xirt predict --input unlabelled_precursors.csv \
  --model faims --output predictions.csv
xirt explain --input unlabelled_precursors.csv \
  --model faims --output explanations --limit 1
```

Inference requires only sequences, sites and the checkpoint's selected numeric
fields. CV labels, FDR, TT and run columns are unnecessary.
These commands use the bundled pooled 4PM/8PM, seed-42 sequence + charge + m/z
checkpoint. Both charge and m/z are required for these weights.
Use `--checkpoint results/faims/faims.pt` instead of `--model faims` for your own
trained model. `Predictor.load(model="faims")` and `FAIMSPredictor.load()` load
the same bundled weights; explicit checkpoint paths still work.
`xirt faims predict` remains a compatibility alias and defaults to the bundled
FAIMS model when `--checkpoint` is omitted. Outputs preserve row order
and add `faims_cv-prediction`, the legacy `faims_cv_prediction_v` alias and
`unknown_token_count`. Add `--qc qc --observed faims_cv` when labels are
available to generate the same annotated QC used for RT.

Unseen PTMs are counted and warned about rather than silently stripped; their
reserved unknown embedding may be untrained. N-terminal modifications,
noncanonical residues, invalid sites and peptides longer than the saved limit
are rejected explicitly. The default maximum is 60 residues, without truncation.

## Python API

```python
from xirt.faims import FAIMSPredictor, prepare_faims_observations, split_precursors
from xirt.predictor import Predictor

precursors, audit = prepare_faims_observations(raw, source="experiment")
partitions = split_precursors(precursors)
fitted = FAIMSPredictor().fit(
    partitions.loc[partitions["split"] == "train"],
    partitions.loc[partitions["split"] == "validation"],
)
fitted.save("faims.pt")
predictor = Predictor.load("faims.pt")
predictions = predictor.predict(unlabelled_precursors)
explanation = predictor.explain(unlabelled_precursors.iloc[:1], steps=64)
```

`fit` checks disjoint unmodified-pair groups and accepts no test set. Checkpoints
contain all preprocessing and unit semantics. Both prediction APIs return
volts. [Residue explanations](explanations.md) separate residue, terminus and
precursor contributions and report a completeness diagnostic.

## Examples and measured comparison

`make notebook-faims` opens a focused train/save/load/predict example using
915 bundled precursors. `make notebook-analysis` opens shared RT/FAIMS QC
and residue-level explanations after running both training tutorials.
See [notebooks](tutorials.md) for execution and output paths.

The source checkout's `benchmark-faims/README.md`, `protocol.json` and reports
document the controlled comparison of **sequence only, sequence + charge and
sequence + charge + m/z** across pooled original 4PM/8PM, 26S and 293T cohorts.
Three initializations share one fixed pair-group partition and training budget.
`make benchmark-faims` runs or resumes the experiment using verified local input
CSVs. Raw observations and benchmark run outputs stay local. Only the selected
pooled charge/m/z seed-42 checkpoint is bundled for default inference; the other
benchmark weights are not distributed.

Held-out RMSE in volts (mean ± sample SD across the three initializations):

| Cohort | Sequence only | + Charge | + Charge and m/z |
| --- | ---: | ---: | ---: |
| Pooled original 4PM+8PM | 9.98 ± 0.30 | 8.77 ± 0.19 | 8.28 ± 0.13 |
| 26S | 7.50 ± 0.44 | 7.03 ± 0.10 | 6.99 ± 0.17 |
| 293T | 11.23 ± 0.11 | 11.14 ± 0.13 | 9.25 ± 0.19 |

Charge helps pooled 4PM/8PM and 26S but adds little on the 56-precursor 293T
holdout. Charge+m/z has the lowest mean RMSE, although its 0.04 V advantage
over charge-only on 26S is small relative to seed variability.
Full predictions, transfer scores and annotated correlations are in the reports.

The earlier precursor/baseline results and exact source snapshot are preserved
unchanged under `benchmark-faims/archive/2026-10-20-precursor-baselines/`.
Its test results were already inspected before designing the new ablation:
the repeated holdout is **not a fresh blinded test**.

Seed variability describes optimization, not biological replication or a
confidence interval. Individual peptides, proteins, runs and instruments are
not held out. External transfer excludes shared source/target peptide pairs
and reports unknown-token coverage separately. The small 293T holdout warrants
particular caution; this comparison does not establish universal generalization.
