# FAIMS compensation-voltage integration and experiment plan

**Target: FAIMS compensation voltage (CV), in volts, not collision cross
section (CCS).** No calibrated CCS labels or CV-to-CCS conversion are available.
The target is the CSM-weighted mean observed CV per source, modified peptide
pair, link-site pair and charge. It is not an intensity-calibrated optimum.

## Predeclared plan

1. Pin and audit the original xiFAIMS CSVs, their provenance, target semantics,
   modifications, repeat observations and overlap.
2. Integrate PTM-aware xiRT sequence encoding with optional charge/mass/m/z
   context; persist all train-only preprocessing in a native checkpoint.
3. Freeze pair-grouped 75/12.5/12.5% partitions before training. An unordered
   unmodified pair is indivisible across charges, modifications and link sites.
4. Train charge-aware and sequence-only shared GRUs for initialization seeds
   17, 42 and 73 on pooled original 4PM+8PM, 26S and 293T cohorts. Compare a
   training-mean predictor, charge-conditioned means, and a fixed descriptor
   HistGradientBoosting regressor. Restore validation-best weights.
5. Evaluate held-out RMSE/MAE in volts and predictive R², stratified by charge
   and source. Measure zero-shot pooled-to-26S/293T transfer after excluding
   shared pairs; report unknown-token coverage and known-only metrics.
6. Prepare an inspectable notebook and keep implementation, models and results
   local for review before any push or remote PR creation.

`protocol.json` was frozen before benchmark training. All trainable recurrent
models have a 150-epoch ceiling, AdamW learning rate 0.003, batch size 64,
validation early stopping patience 20 and LR-reduction patience 8. Two CPU
threads, a shared 32-unit GRU, 16-wide embeddings and 64/32-unit ReLU heads are
used. Numeric and target normalization and the PTM vocabulary use training
rows only. The native normalization buffers survive checkpoint reload.

All variants of an unmodified pair share one partition. This does not hold out
individual peptides, proteins, runs or instruments. Seed variability on a
fixed partition is not biological replication or a confidence interval.

## Attribution and input provenance

Source: [Rappsilber-Laboratory/xiFAIMS](https://github.com/Rappsilber-Laboratory/xiFAIMS),
revision `b85482ffc246b2dc669c4592f752568fd9b1baed`, Apache-2.0.
The upstream repository credits Sven Giese and Ludwig Sinn. Its
[license](https://github.com/Rappsilber-Laboratory/xiFAIMS/blob/b85482ffc246b2dc669c4592f752568fd9b1baed/LICENCE)
and dataset checksums are retained by reference. Raw source tables are not
redistributed here.

The integration reuses xiRT's existing sequence helpers and implements strict
xiFAIMS run-label parsing, precursor grouping, charge context and CV evaluation
natively. No upstream TensorFlow, pickle, modlamp or XGBoost runtime is required.
The descriptor baseline uses symmetric length/composition and precursor
features; **it is not an exact reproduction of the upstream XGBoost study**.

Only boolean `isTT == True` and `fdr <= 0.01` observations are included. Source
CSV row counts are 9,179 (4PM), 3,254 (8PM), 10,843 (26S) and 1,149 (293T).
The combined 4PM/8PM table contains all original 4PM observations plus a subset
of 8PM and is deliberately excluded from learning and independent evaluation.

## Reproduce

From the repository root, create the locked CPU environment:

```bash
uv sync --locked --extra cpu --group dev
revision=b85482ffc246b2dc669c4592f752568fd9b1baed
mkdir -p "local-data/xifaims/$revision"
for file in 4PM_DSS_LS_nonunique1pCSM.csv 8PM_DSS_Liu_nonunique1pCSM.csv \
    26S_BS3_LS_nonunique1pCSM.csv 293T_DSSO_nonunique1pCSM.csv; do
    curl --fail --location --retry 3 \
      "https://raw.githubusercontent.com/Rappsilber-Laboratory/xiFAIMS/$revision/data/$file" \
      --output "local-data/xifaims/$revision/$file"
done
uv run --no-sync python -m xirt.data_pipelines.benchmarks.faims.run
uv run --no-sync marimo edit examples/faims.py --host 127.0.0.1
```

The runner validates SHA-256 checksums **before training** and writes a manifest
of the protocol, production/runner source hashes and package versions before
fitting. It refuses to reuse a differing manifest. Native checkpoints,
aggregated precursor tables and raw CSVs stay under ignored `local-data/`.
Use fresh `--work` and `--reports` directories for intentional changed protocols.
The default notebook inspects the canonical reports directory.

Opening the notebook does not download data or train models. It displays
completed reports even without local weights. Where weights exist it reloads
a selected checkpoint and predicts from label-free inputs. Its optional
two-epoch training example requires an explicit button click.

## Artifacts

| Path | Contents |
| --- | --- |
| `protocol.json` | Predeclared data hashes, splits, architectures and training budget |
| `reports/manifest.json` | Frozen protocol, source hashes, environment and parent commit |
| `reports/data_audit.json` | Filtering, precursor aggregation and label-ambiguity counts |
| `reports/*__partitions.csv` | Stable precursor IDs, unmodified-pair hashes and fixed splits |
| `reports/*__run.json` | Full parameters, elapsed time and checkpoint hash for each run |
| `reports/*__history.csv` | Every training/validation epoch; standardized losses and target scale |
| `reports/*__test.csv` | Per-precursor held-out labels/predictions, charge and coverage |
| `reports/*__transfer_*.csv` | Zero-shot predictions over full eligible target cohorts |
| `reports/metrics.csv` | All metrics, source/charge strata and known-token subsets |
| `reports/runs.csv`, `summary.csv` | Run inventory and across-seed summaries |
| `reports/*.png` | Learning curves, held-out comparisons and residual plots |
| `local-data/faims/*.pt` | Local native xiRT FAIMS checkpoints |
| `examples/faims.py` | Interactive inspection notebook |

Undefined R² (constant labels or fewer than two rows) is recorded as null in
JSON or an empty CSV cell, rather than forced to a success-shaped score.
Charge-mean predictions fall back explicitly to the training global mean for
unseen charges. Unknown PTM tokens generate warnings and per-row counts;
their embedding may be untrained. External transfer performance must be
interpreted alongside coverage and crosslinker/acquisition differences.

## Measured results (October 20, 2026)

All **18 recurrent runs** completed, stopping after 22–44 epochs and restoring
the validation-best weights. All checkpoints were reloaded and reproduced
their archived held-out and transfer predictions. The table shows held-out
RMSE in volts; recurrent values are mean ± sample SD over three initialization
seeds, while the fixed baselines run once.

| Cohort | Test precursors / pairs | Charge-aware xiRT | Sequence-only xiRT | Descriptor tree | Training mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 4PM+8PM pooled | 484 / 194 | **8.17 ± 0.15** | 9.98 ± 0.30 | 9.10 | 12.17 |
| 26S | 346 / 162 | 7.26 ± 0.17 | 7.50 ± 0.44 | **6.97** | 9.10 |
| 293T | 56 / 43 | 9.59 ± 1.13 | 11.23 ± 0.11 | **8.35** | 11.67 |

Charge/mass/m/z context reduces mean recurrent held-out RMSE on all three
cohorts relative to the sequence-only ablation. xiRT outperforms the tree on
the pooled cohort, **not** on 26S or 293T. The 293T result has particularly
large initialization variation and only 56 test precursors. Predictive R²
for the charge-aware model averages 0.525, 0.360 and 0.279, respectively.
These descriptive comparisons are not significance tests or biological
replication.

Zero-shot transfer from pooled 4PM+8PM uses all eligible target precursors,
not the target cohort's within-cohort test subset:

| Target | Precursors | Charge-aware xiRT RMSE (V) | Descriptor tree RMSE (V) |
| --- | ---: | ---: | ---: |
| 26S | 2,827 | 8.66 ± 0.10 | **7.90** |
| 293T | 465 | 9.75 ± 0.28 | **9.43** |

No shared pairs needed removal, and no unknown tokens occurred in these
held-out/transfer evaluations: known-only and all-row metrics coincide here.
This does not establish coverage of arbitrary new PTMs. The tree is stronger
in both transfer comparisons. Residual plots show compression toward the
central CV range, so these models should not be described as calibrated
physical predictors or as optimal-CV selectors.

Inspect `reports/metrics.csv` for MAE, charge/source strata and individual
seeds. No models were retuned or selected using these test/transfer results.
The notebook defaults to initialization seed 42 rather than selecting the
best test seed. A standalone local HTML export is generated with:

```bash
mkdir -p examples/results/faims
uv run --no-sync marimo export html examples/faims.py \
  -o examples/results/faims/inspection.html
```
