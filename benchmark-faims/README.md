# Crosslink FAIMS: what does precursor charge add?

This comparison trains **xiRT only**, with three input variants: peptide
sequence/site tokens, those tokens plus charge, and tokens plus charge and
m/z. All variants use the same grouped partitions, initialization seeds and
training budget. The target is mean observed compensation voltage in **volts**,
not collision cross section or an intensity-calibrated optimum.

The original precursor/baseline experiment, reports and exact source snapshot
are preserved unchanged in
[`archive/2026-10-20-precursor-baselines/`](archive/2026-10-20-precursor-baselines/).
`sha256.json` authenticates the archived files. The new ablation reuses those
partitions **after the earlier test results were inspected**; it is descriptive,
not a fresh blinded evaluation. No test scores control fitting or early stopping.

## Data and controls

The original xiFAIMS 4PM, 8PM, 26S and 293T CSVs are pinned at upstream revision
`b85482ffc246b2dc669c4592f752568fd9b1baed` (Apache-2.0).
`protocol.json` contains their SHA-256 values and all training/evaluation settings.
The overlapping combined 4PM/8PM CSV is excluded.

Only explicit TT CSMs at FDR <= 0.01 are retained. CV is averaged per source,
modified peptide pair, link sites and charge, with CSM counts as the observational
weight. Repeated CV settings therefore represent mean detected CV rather than
an abundance-weighted transmission optimum.

The fixed seed-42 75/12.5/12.5% split groups unordered **unmodified pairs** across
charge, PTM and site variants. Vocabulary, target scaling and numeric normalization
use training rows only. Individual peptides, proteins, runs and instruments
are not held out. Pooled 4PM+8PM to 26S/293T transfer excludes any shared source
pairs and uses no target labels during fitting.

Each variant receives three initializations (17, 42, 73), a shared GRU with
32 hidden units, 16-dimensional embeddings, additive branch merge, 64/32-unit
heads and 0.1 dropout. AdamW starts at 0.003 with weight decay 0.01 and batches
of 64. Training has a 150-epoch ceiling, early-stopping patience 20 and
learning-rate reduction by 0.5 after 8 unimproved validation epochs.
The best validation weights are restored. Runs use CPU with two PyTorch threads.
No tree models, descriptor baselines or mass features enter this comparison.

## Held-out results

RMSE in volts, mean ± sample SD across three initializations on the fixed
pair-group split. Lower is better.

| Cohort | Test precursors | Sequence only | Sequence + charge | Sequence + charge + m/z |
| --- | ---: | ---: | ---: | ---: |
| Original 4PM+8PM pooled | 484 | 9.98 ± 0.30 | 8.77 ± 0.19 | **8.28 ± 0.13** |
| 26S | 346 | 7.50 ± 0.44 | 7.03 ± 0.10 | **6.99 ± 0.17** |
| 293T | 56 | 11.23 ± 0.11 | 11.14 ± 0.13 | **9.25 ± 0.19** |

Charge reduces mean error relative to sequence-only on pooled 4PM/8PM and
26S, but contributes little on the small 293T holdout. Adding m/z has the
lowest mean RMSE on all three cohorts; the 0.04 V difference from charge-only
on 26S is small relative to initialization variability. These are descriptive
comparisons, not significance tests.

Zero-shot transfer from pooled 4PM/8PM uses all eligible target precursors,
not just their internal test partitions:

| Target cohort | Precursors | Sequence only | Sequence + charge | Sequence + charge + m/z |
| --- | ---: | ---: | ---: | ---: |
| 26S | 2,827 | 9.34 ± 0.08 | 9.11 ± 0.37 | **8.81 ± 0.31** |
| 293T | 465 | 11.22 ± 0.22 | 10.61 ± 0.36 | **10.02 ± 0.24** |

No shared peptide pairs or unknown tokens occur in these two transfer sets.
This does not guarantee coverage for new modifications or instruments.
The selected seed-42 plots annotate Pearson r, Spearman rho, predictive
R-squared, RMSE and sample count; `summary.csv` contains all seed aggregates.
The original tree/baseline comparison is archived, not silently replaced by
these more focused neural ablations.

## Reproduce

Use the repository's locked CPU environment:

```bash
uv sync --locked --extra cpu
revision=b85482ffc246b2dc669c4592f752568fd9b1baed
mkdir -p "local-data/xifaims/$revision"
for file in 4PM_DSS_LS_nonunique1pCSM.csv 8PM_DSS_Liu_nonunique1pCSM.csv \
  26S_BS3_LS_nonunique1pCSM.csv 293T_DSSO_nonunique1pCSM.csv; do
  curl --fail --location \
    "https://raw.githubusercontent.com/Rappsilber-Laboratory/xiFAIMS/$revision/data/$file" \
    --output "local-data/xifaims/$revision/$file"
done
make benchmark-faims
```

The runner verifies every input checksum before fitting, checks exact archived
partition identity and writes a source snapshot plus hash manifest before the
first fit. Completed checkpoints are resumed only when source, protocol,
environment and checkpoint hashes agree. To run separately, pass fresh `--work`
and `--reports` paths to `python -m xirt.data_pipelines.benchmarks.faims.run`.
Do not mix partially completed runs from different manifests.

Raw observations and final weights stay in ignored `local-data/`; reports are
checked in. Checkpoints default to `local-data/faims-charge/`. The old checkpoints
remain under `local-data/faims/` and are not overwritten.

## Read the reports

| Artifact | Contents |
| --- | --- |
| `manifest.json`, `source.tar.gz` | Frozen protocol, source hashes/snapshot and environment |
| `data_audit.json`, `*__partitions.csv` | Filtering, precursor counts and pair-group partitions |
| `runs.csv`, `*__run.json`, `*__history.csv` | Epochs, selected weights, settings and checkpoint hashes |
| `*__test.csv`, `*__transfer_*.csv` | Row-level predictions, targets, IDs and unknown-token counts |
| `metrics.csv`, `summary.csv` | Errors/correlations and seed mean/sample SD |
| `heldout_rmse.png`, `heldout_predictions.png`, `learning_curves.png` | Three-variant comparisons |
| `*__qc.png` | Seed-42 prediction/residual plots with r, rho, R-squared and RMSE |
| `completion.json` | Completed 27-run inventory and protocol checksum |

Seed SD measures initialization variability on one split, **not** biological
replication or a confidence interval. The 293T test partition has only 56
precursors. Transfer performance and known-token coverage should be read
alongside within-cohort scores; do not claim cross-instrument generalization.

For a compact runnable example rather than a benchmark dashboard, open
`examples/faims.py`. It trains on bundled data and uses the same public
`Predictor`, QC and checkpoint format as RT. `examples/analyze_predictions.py`
adds shared RT/FAIMS QC and signed residue/precursor explanations with explicit
baseline and completeness checks. The example subset overlaps this benchmark.
