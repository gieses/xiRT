# xiRT 2.0

xiRT predicts peptide separation behavior across one or more chromatographic
dimensions. Linear peptides use one encoder branch; crosslinked peptides can use
two branches with shared parameters and a configurable merge operation. Task
heads model continuous retention time, categorical fractions, or ordinal
fractions. Predictions may become features in downstream identification
rescoring, but xiRT does not itself establish identification confidence.

Version 2.0 replaces TensorFlow/Keras with PyTorch. The CLI and legacy-shaped
YAML sections remain, while validation, checkpoints, optimizer implementation,
and optional embedding routes change. It is a new training implementation, not
a numerically identical loader for 1.x models.

## Start here

1. [Install and run the CLI](getting-started.md).
2. [Check input columns, target encoding, and scientific split design](configuration.md).
   [Interpret output files and prediction provenance](results.md).
3. [Migrate existing workflows and models](migration.md).
4. Run the [training and fine-tuning notebooks](tutorials.md), consult the
   [Python API](api.md), or evaluate [ESM-C and LoKr](embeddings.md).
5. Consult the [benchmark protocol and measured-output locations](benchmarks.md)
   before making performance claims.

!!! warning "Legacy model files are incompatible"
    TensorFlow/Keras HDF5 (`.h5`/`.hdf5`) files are not loadable in 2.0.
    Use archived legacy predictions for comparisons and retrain for 2.0.

## Scientific scope

The original xiRT publication studied retention-time prediction for
crosslinking mass spectrometry. Optional protein-language-model features in 2.0
are experimental in this application: model size, protein pretraining, and
parameter-efficient adaptation are not evidence of improved peptide
retention-time prediction. Short peptides, modifications, chromatography
changes, and peptide-pair leakage require separate evaluation.

The [references](references.md) preserve the xiRT citation and link the primary
upstream documentation used for packaging, ESM, CI, and publishing.
