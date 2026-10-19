# Scientific configuration

## Input contract

The default column convention includes the following fields. Which metadata
are required depends on preprocessing and mode; validate your own search-engine
export before starting an expensive run.

| Field | Meaning |
|---|---|
| `Peptide1`, `Peptide2` | Peptide sequences; the second branch is used for crosslinks |
| `Fasta1`, `Fasta2` | Protein descriptions used by applicable filtering |
| `LinkPos1`, `LinkPos2` | Crosslink residue positions, one-based within each peptide |
| `PSMID` | Unique identification identifier; preserve through joins |
| `score` | Search-engine score |
| `isTT`, `isTD`, `isDD` | Target/decoy status flags where applicable |
| `fdr` | Identification FDR estimate for selecting training data |
| Task-specific RT columns | Continuous RT or observed fraction labels |

xiRT sorts crosslink branches so the first peptide is longer or
lexicographically larger. Associated columns ending in `1` and `2` are swapped
as pairs. Do not use that naming convention for unrelated or unpaired columns.
Keep original identification IDs so predictions can be traced back to input.
For example, `LinkPos1: 1` refers to the first residue, not the second.
The processed token list includes the N-terminal `H-` token at index zero,
so the one-based residue position indexes that termini-inclusive list directly.
Do not subtract one from input `LinkPos` values.

## Network YAML (`-x`)

The legacy section names are retained:

| Section | Function and consistency checks |
|---|---|
| `LSTM` | GRU/LSTM units, layers, bidirectionality, normalization, regularization |
| `embedding` | Trainable recurrent residue embedding width (`length`) |
| `dense` | Per-task head widths, activations, dropout, normalization and regularizers |
| `learning` | Optimizer, learning rate, epochs, batch size, seed, device |
| `output` | Each task's column, dimension, activation, loss, metric, loss weight |
| `siamese` | Shared-branch use and merge operation |
| `callbacks` | Checkpoint/CSV logging, early stopping, learning-rate reduction |
| `predictions` | `continues` and `fractions` lists identifying the tasks |
| `encoder` | `route: recurrent` or `route: esm-c` with a positive `embedding_dim` |
| `adapter` | Optional LoKr settings; see [embedding guide](embeddings.md) |
| `fine_tuning` | Native checkpoint and full/frozen/LoKr/LoRA strategy; see [fine-tuning](fine-tuning.md) |
| `tracking` | Local-first MLflow experiment/store/run settings; see [tracking](tracking.md) |
| `target_scales` | Optional explicit positive divisors for continuous target columns |

`callbacks.progressbar` enables tqdm training progress. TensorBoard logging is
optional: run `uv sync --locked --extra cpu --extra tensorboard` before setting
`callbacks.tensor_board: true`. It is not required for normal training or
MLflow tracking.

`continues` is the historical spelling of the continuous-task key. Do not
rename it to `continuous`. Task names must be unique and appear consistently
in `predictions`, all corresponding `output` keys, and the input columns.

Dense option lists must match `dense.nlayers`. Dropout must be in `[0, 1)`;
dimensions, recurrent units/layers, learning rate, batch size, and epochs must
be positive. Pydantic's `NetworkConfig` rejects inconsistent architecture
settings instead of silently creating an unrelated network.
The [small RP configuration](examples/xirt-rp.yaml) supplies the architecture
used in the [API notebooks](tutorials.md); it is not a tuned chromatographic model.

### Target types

- **Continuous RT:** typically dimension `1`, activation `linear`, loss `mse`
  (or explicitly selected supported `mae`/`huber`).
- **Ordinal fractions:** multiple sigmoid outputs with binary cross-entropy
  and cumulative fraction encoding. A cumulative ordinal target is not a
  one-hot categorical target.
- **Categorical fractions:** softmax with at least two outputs and an
  appropriate categorical loss. Class encoding and dimension must agree.

For multi-task fitting, loss weights change optimization and can dominate the
continuous head. Select them on validation data, not on the held-out test set.
Evaluate fraction accuracy separately from RT error; neither is interchangeable
with identification FDR.

### Reproducibility and optimizers

```yaml
learning:
  optimizer: adam
  learningrate: 0.001
  batch_size: 128
  epochs: 50
  device: cpu
  seed: 42
  deterministic: true
```

This is a **section fragment**, not a complete network configuration. Merge it
into a freshly generated YAML. Optimizer names and options are checked by the
implementation; consult `make_optimizers` in the [API reference](api.md).
Experimental Lion or Muon configurations require a separate controlled
comparison, not a substitution presented as a guaranteed improvement.

Deterministic initialization does not guarantee bitwise-identical results
across devices, PyTorch versions, BLAS implementations, or numerical precision.
Record these together with the seed, effective YAML, training split, and
checkpoint.

## Learning YAML (`-l`)

`preprocessing` controls sequence filtering/encoding, `max_length`, crosslink
residue labels, and optional saved label encoders. `preprocessing.le` accepts
the exported xiRT 2.0 `encoder.json`, containing the alphabet, sequence length,
and fraction levels; it does not accept a legacy pickled encoder. `train` controls FDR
filtering, `mode`, `ncv`, `test_frac`, `sample_frac`, `sample_state`,
refitting, and pretrained-model paths. Use the exact keys
in your generated config; do not assume a historical example is current.

## Leakage and validation

### Preserve target scaling

Continuous targets are divided by their recorded scale. Set an explicit
top-level mapping when the input units are known:

```yaml
target_scales:
  rp: 60.0
```

For input RP times in seconds, this divisor expresses the modeled target in
minutes. Without an explicit value, the legacy heuristic selects `60.0` if
the target maximum exceeds 1500, otherwise `1.0`; this is not a reliable
semantic unit detector. The effective `_target_scales` mapping is saved once
and reused for checkpoint-based inference, including smaller subsets. Do not
choose a new scale from the prediction subset or compare scaled errors to
unscaled times.

### Keep scientific partitions distinct

For scientific comparisons, fix the split before fitting encoders, target
scalers, or models. Group related peptide pairs and duplicates where needed;
random row splits can put the same peptide in both train and test. Consider
protein, acquisition, chromatographic run, and modification overlap according
to the intended generalization question.

Keep training, validation, and test data roles separate. Crossvalidation is a
tool for out-of-fold predictions, but your grouping and preprocessing determine
whether those predictions answer the intended question. Fit normalization on
training data and retain the fitted transformations for inference.
Rows marked `cv=0` include training/validation predictions populated by the
final model; they are not out-of-fold estimates and must not be included in
an out-of-fold accuracy claim. Interpret fold-assigned rows using
`split_NN.json` and the [output provenance guide](results.md).
