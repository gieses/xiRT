# Chromatography model cards

These four native PyTorch checkpoints support the
[illustrated prediction walkthrough](prediction-gallery.md). They are trained
models, not two-epoch demo weights. **The four checkpoints and the selected FAIMS
checkpoint are bundled in `xirt/assets/`, in Git, wheels and source distributions.**
Training tables, split membership and row-level predictions are not distributed.

## Models and intended use

| Bundled checkpoint | Outputs | Trainable parameters |
| --- | --- | ---: |
| `xirt/assets/rp.pt` | RP, seconds | 138,113 |
| `xirt/assets/scx.pt` | SCX, fraction index | 138,113 |
| `xirt/assets/hsax.pt` | hSAX, fraction index | 138,113 |
| `xirt/assets/multitask.pt` | RP + SCX + hSAX | 187,779 |

`Predictor.load()` defaults to `multitask`. Select the others with
`Predictor.load(model="rp")`, `"scx"` or `"hsax"`. The additional
`xirt/assets/faims.pt` is the pooled 4PM/8PM, seed-42 charge/m/z model:
load it with `Predictor.load(model="faims")` or `FAIMSPredictor.load()`.
It uses the separate [FAIMS input schema and target](faims.md).

Use these models to investigate predictions for crosslinked peptides under the
source study's separation protocol and to reproduce the explanation examples.
They are not universal retention-time calibrators. SCX/hSAX are **scalar
regressions** with linear outputs, not ordinal classifiers: fractional and
out-of-range predictions are possible. Do not treat their outputs as calibrated
class probabilities or round them before evaluating regression metrics.

The joint model shares an embedding and sequence encoder but has separate
prediction heads. It has more parameters than each single-task model; the
single-versus-joint comparison is not capacity-matched and uses one initialization.

## Training data and attribution

- **Source:** Giese, Sinn, Wegner and Rappsilber, *Retention Time Prediction Using
  Neural Networks Increases Identifications in Crosslinking Mass Spectrometry*,
  [Zenodo record 4720760](https://zenodo.org/records/4720760), version 1.0.0;
  related article DOI [10.1038/s41467-021-23441-0](https://doi.org/10.1038/s41467-021-23441-0).
- **Table:** `ecoli.zip` → `ecoli/DSS_CSM_anpxirt.csv`.
- **Data license:** CC-BY-4.0 as recorded in the Zenodo metadata. The project
  license is Apache-2.0; the source-data license and attribution remain separate.
- **CSV SHA-256:** `3833575aa3ec45d55b1c3f3af2556b97314b26dc5d1880301c18514a0295b72f`.
- **Integrity:** the retrieved member's size/CRC was checked against the archive
  catalogue; CRC32 is `1470239631`. The published whole-ZIP MD5 is recorded,
  but a range-only retrieval does **not** verify the whole archive's MD5.

The full source table has 17,059 CSMs; all 11,072 explicit target-target CSMs
with FDR ≤ 0.01 and measured RP/SCX/hSAX labels are retained. No peptide-sequence
duplicates are discarded. Observations are not averaged across fractions.
Each CSM receives equal loss weight, so repeated peptide pairs have more influence.

This E. coli table supplies all three measured targets in the same separation
scheme. The pLink2 table is an alternate analysis of overlapping acquisitions,
not independent replication. Figure exports and historical xiRT-rescored
identifications are not added as training data. The FA-complex study is RP-only
under a different acquisition scheme and is not pooled into this three-task
model. The existing FAIMS predictor uses its own separately documented
[FAIMS data and target](faims.md).

## Training protocol

The complete configuration is `examples/chromatography_models.yaml`.

| Setting | Value |
| --- | --- |
| Encoder | Shared two-layer bidirectional GRU, 64 units per direction |
| Embedding | 32 dimensions, PTM and crosslink-site tokens |
| Branch merge | Addition, invariant to swapping the paired inputs |
| Heads | 128/64-unit ReLU layers, dropout 0.1, scalar linear output |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.01, gradient clipping 5 |
| Batch / budget | 256 CSMs; maximum 300 epochs per model |
| Validation controls | Early stopping after 35 unimproved epochs; LR × 0.5 after 8 |
| Model selection | Restore the validation-best weights |
| Loss | Training-standardized MSE; equal task weights for the joint model |
| Seeds | Initialization 42; pair-group split 42 |
| Runtime | FP32, NVIDIA A100 80 GB; two CPU threads |
| Sequence width | 128 tokens including termini; oversize sequences cause an error |

The 80/10/10% split is by **unordered unmodified peptide pairs** across PTMs and
link sites. It yields 8,924/1,093/1,055 CSMs and 4,558/570/570 groups for
training/validation/test. Vocabulary and per-target mean/standard deviation are
fitted on training rows only. The native checkpoint stores the vocabulary,
crosslink encoding, target transformations and unit names.

## Evaluation and limitations

The [walkthrough](prediction-gallery.md) shows held-out predictions from the
saved checkpoints, not training predictions. The
[generated manifest](assets/prediction-gallery/manifest.json) records errors,
correlations, executed/best epochs, exact checkpoint hashes/sizes and the fixed
example selection rule. Individual local model directories also contain
`metrics.json`, `history.csv`, `record.json` and row-level `predictions.csv`.
These local CSVs are not committed.

Each chromatography result below uses the same **1,055 test CSMs from 570
held-out pair groups**. RMSE and MAE have the output's native units.

| Model | Output | RMSE | MAE | R-squared | Pearson r | Spearman rho |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Separate | RP, s | 136.84 | 96.39 | 0.9729 | 0.9867 | 0.9869 |
| Joint | RP, s | 152.87 | 112.88 | 0.9662 | 0.9834 | 0.9850 |
| Separate | SCX, fraction | 1.0552 | 0.6553 | 0.7714 | 0.8840 | 0.8892 |
| Joint | SCX, fraction | 1.0185 | 0.6436 | 0.7871 | 0.8889 | 0.8960 |
| Separate | hSAX, fraction | 0.7280 | 0.4675 | 0.8856 | 0.9412 | 0.9411 |
| Joint | hSAX, fraction | 0.7034 | 0.4683 | 0.8932 | 0.9455 | 0.9474 |

The split does not hold out individual peptides, proteins, instruments or
studies. This source study was used in earlier analyses, so the evaluation is
not a newly blinded benchmark. One seed does not estimate optimization
uncertainty. No confidence intervals, external transfer guarantees, probability
calibration or post-hoc test-set hyperparameter tuning are claimed.

All held-out token sequences must be encodable by the training vocabulary;
the training script checks this before fitting and raises on unsupported
tokens rather than dropping rows. For future inputs, unknown modifications,
new crosslinkers and changed chromatographic protocols remain unsupported
generalization cases. Residue explanations describe the fitted model and its
declared baseline, not causal chemistry.

## Checkpoint format and packaged defaults

Checkpoints contain native model weights and primitive preprocessing metadata,
**not optimizer states or raw observations**. Loading is the same for separate
and joint models:

```python
from xirt.predictor import Predictor

model = Predictor.load()  # Bundled joint model.
prediction = model.predict(precursors)
explanation = model.explain(precursors.iloc[:1], task="hsax", steps=256)
```

The completed full-data run produced these **measured final checkpoint sizes**,
after restoring validation-best weights:

| Model | Executed / best epoch | Exact bytes | MiB |
| --- | ---: | ---: | ---: |
| RP | 139 / 104 | 560,433 | 0.534 |
| SCX | 58 / 23 | 560,433 | 0.534 |
| hSAX | 101 / 66 | 560,433 | 0.534 |
| Joint RP + SCX + hSAX | 91 / 56 | 762,995 | 0.728 |
| **Four checkpoints combined** | — | **2,444,294** | **2.331** |
| **Compressed bundle, including metadata** | — | **2,251,797** | **2.147** |

MiB means 1,048,576 bytes. These are complete native checkpoint files, not
estimates from parameter counts. The reused FAIMS checkpoint is a separate
**49,621-byte** file; it is not included in the four-model totals or bundle.
`local-data/chromatography-models/sizes.json` records the measured counts, and
the [gallery manifest](assets/prediction-gallery/manifest.json) preserves them
alongside all checkpoint SHA-256 hashes. All five packaged checkpoints together
are **2,493,915 bytes (2.378 MiB)**.

`chromatography-models.tar.gz` is prepared **locally only** with the four models,
their records/metrics and the frozen training manifest. The download/source
table, partitions and row-level predictions are not included in the bundle.

The copies in `xirt/assets/` are byte-identical to these final training outputs.
Their filenames, exact sizes, SHA-256 hashes and source attribution are recorded
in the packaged `xirt/assets/manifest.json`; detailed training/evaluation
provenance remains in the gallery manifest. No separate model download or
release bundle is needed for inference.

`importlib.resources.files()` locates the bundled resources and `as_file()`
keeps extracted resources alive until loading finishes, including ZIP imports.
The working directory does not affect model resolution. Custom paths are still
supported with `Predictor.load("my_model.pt")`; they cannot be combined with a
named bundled model. Loading does not train or download anything, and fails
explicitly for an unknown model or missing checkpoint. Load custom weights
only from a trusted source.
