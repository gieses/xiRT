"""Default column names, YAML templates and run-output descriptions."""

from xirt import __version__

default_column_names = {
    "peptide1_sequence": "Peptide1",
    "peptide2_sequence": "Peptide2",
    "link_pos_basename": "LinkPos",
    "peptide1_unmod_sequence": "Peptide1",
    "peptide2_unmod_sequence": "Peptide2",
    "score": "score",
}

learning_params = f"""# xiRT {__version__} preprocessing and training
preprocessing:
  # Reuse encoder.json for checkpoint vocabulary and sequence width.
  le: None
  # -1 retains all sequence lengths.
  max_length: -1
  cl_residue: true
  filter: ""
train:
  fdr: 0.01
  ncv: 1
  refit: false
  # train, crossvalidation or predict
  mode: train
  # crosslink, linear or pseudolinear
  sequence_type: crosslink
  pretrained_weights: None
  pretrained_model: None
  # Validation fraction within each training fold.
  test_frac: 0.10
  sample_frac: 1
  sample_state: 21
"""

xirt_params = f"""# xiRT {__version__} PyTorch network
LSTM:
  activation: tanh
  activity_regularization: l2
  activityregularizer_value: 0.001
  bidirectional: true
  kernel_regularization: l2
  kernelregularizer_value: 0.001
  lstm_bn: true
  nlayers: 1
  type: GRU
  units: 50
dense:
  # Lists must contain nlayers entries.
  activation: [relu, relu, relu]
  dense_bn: [true, true, true]
  dropout: [0.1, 0.1, 0.1]
  kernel_regularizer: [l2, l2, l2]
  regularization: [true, true, true]
  regularizer_value: [0.001, 0.001, 0.001]
  neurons: [300, 150, 50]
  nlayers: 3
embedding:
  length: 50
learning:
  batch_size: 512
  epochs: 50
  learningrate: 0.01
  verbose: 1
  optimizer: adamw
  seed: 42
  device: cpu
  deterministic: true
  clip_norm: 5.0
output:
  # Set task columns and dimensions for the input table.
  hsax-activation: sigmoid
  hsax-column: hsax_ordinal
  hsax-dimension: 10
  hsax-loss: binary_crossentropy
  hsax-metrics: mse
  hsax-weight: 50
  rp-activation: linear
  rp-column: rp
  rp-dimension: 1
  rp-loss: mse
  rp-metrics: mse
  rp-weight: 1
siamese:
  use: true
  merge_type: add
  single_predictions: true
callbacks:
  check_point: true
  log_csv: true
  early_stopping: true
  early_stopping_patience: 15
  tensor_board: false
  progressbar: true
  reduce_lr: true
  reduce_lr_factor: 0.5
  reduce_lr_patience: 15
predictions:
  # The historical key continues denotes continuous targets.
  continues: [rp]
  fractions: [hsax]
"""

readme = f"""xiRT {__version__} outputs

xirt_model.pt: PyTorch model state and architecture/preprocessing metadata.
configuration.json: Effective network, learning settings and input provenance.
encoder.json: Token vocabulary, padded sequence width and fraction levels.
split_NN.json: Training, validation and prediction row identifiers for each fold.
epoch_history.csv: Per-epoch training and validation metrics.
model_summary.csv: Metrics by fold and partition.
xirt_logger.log: CLI execution log.

With prediction-table output enabled:
processed_psms.csv.gz: Input identifications with preprocessing columns.
error_features.csv: Predictions, errors and fold assignments.
error_features_interactions.csv: Derived error features for rescoring.
callbacks/: Optional fold checkpoints/logs and legacy Python data exports.

cv=0 rows are training/validation predictions, not out-of-fold estimates.
Checkpoint files do not replace a documented scientific partition.
Treat pickle data exports as trusted-source-only files; use encoder.json for inference.

https://gieses.github.io/xiRT/results/
"""
