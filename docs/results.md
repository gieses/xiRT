# Outputs and prediction provenance

Keep the model, effective configuration, preprocessing metadata and split
identifiers together. Predictions alone are not enough to reproduce a run or
determine which rows were held out.

## Main artifacts

| Artifact | Purpose |
|---|---|
| `xirt_model.pt` | Final PyTorch model checkpoint for compatible 2.0 inference |
| `configuration.json` | Effective network and learning configuration |
| `split_NN.json` | Recorded training, validation, and prediction IDs for fold `NN` |
| `error_features.csv` | Predictions and per-task prediction-error features when table writing is enabled |
| `error_features_interactions.csv` | Derived interactions between error features when table writing is enabled |
| `encoder.json` | Exported alphabet, padded sequence length, and fraction-level mapping |
| Callback logs/checkpoints | Optional fold-level training history and checkpoints according to callback settings |
| `mlflow.db`, `mlflow-artifacts/` | Output-local tracking store and artifacts when tracking is enabled without overriding paths |

Some artifacts depend on mode, task, preprocessing route, and the `--write`
setting. ESM feature caches are separate from recurrent label-encoder
metadata. The exact model architecture and feature/preprocessing contract
must be compatible when reusing a checkpoint.

Checkpoints retain `_sequence_length`, `_target_scales`, the vocabulary and
fraction mappings, input-content provenance, and the selected model's
per-fold `_split`. Prediction on a smaller subset reuses the saved sequence
width and target scale rather than shrinking padding or choosing a new scale
from that subset.

The source-content fingerprint hashes a documented serialization of the input
table; it is not necessarily the raw CSV file checksum. Keep the original
file checksum as an additional provenance record when needed.

`callbacks.check_point` controls writing optional fold checkpoints. Best-epoch
selection/restoration also uses an in-memory state and does not require those
files to be written. The final `xirt_model.pt` is not a TensorFlow HDF5 model;
legacy HDF5 files require isolated 1.x inference or retraining, not renaming.

## Interpret the `cv` column

| Value | Interpretation |
|---|---|
| Positive fold number | Predictions assigned to that fold's held-out prediction IDs; inspect the recorded split and run mode |
| `0` | Final-model predictions populated for training/validation rows; **not out-of-fold estimates** |
| `-1` | Predictions for the separate non-training candidate set |

For an out-of-fold comparison, use only genuinely held-out fold predictions
and verify the sample IDs against `split_NN.json`. Positive fold numbers alone
do not prove a leakage-free scientific split. A single train/test run is not
equivalent to full crossvalidation, and rows used for validation are not an
independent final test set.

Do not combine `cv=0` predictions with out-of-fold predictions and report the
combined error as crossvalidated accuracy. Likewise, candidate-set predictions
do not establish accuracy without appropriate independent observed labels.

## Reproducibility and privacy

Archive input checksums/identifiers, task labels and units, transformation
settings, source revision, package versions, seeds, and hardware with the
outputs. Keep paired-peptide reorderings traceable through identification IDs.

MLflow fingerprints training features and targets and logs configuration/metrics/model
artifacts; it does not upload feature arrays or labels. A fingerprint does not
replace the data/split provenance needed to reproduce a result. Model weights
and configuration remain potentially sensitive and should not be added to
public CI artifact paths indiscriminately.

Consult the [benchmark protocol](benchmarks.md) for matched comparisons and the
[migration guide](migration.md) before reusing legacy models.
