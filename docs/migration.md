# Migrating TensorFlow xiRT 1.x to 2.0

## Compatibility boundary

| Component | 2.0 behavior |
|---|---|
| CLI | Retains `xirt -i ... -o ... -x ... -l ...` and config-generation flags |
| YAML | Retains section names, adds validation and modern optimizer/encoder settings |
| Network | `xiRTNET(params, input_dim, embedding_dim)` builds a PyTorch module |
| Training API | `build_model`, `compile`, module `fit`/`predict`/`evaluate` adapters remain |
| Dependencies | Python 3.12–3.14, PyTorch; no TensorFlow/Keras/TensorFlow Addons |
| Distribution | Install `xirt-torch`; continue to import `xirt` and run `xirt` |
| Checkpoints | PyTorch state and architecture metadata, not Keras HDF5 |
| Numerical behavior | Retraining required; no promise of identical weights or predictions |

!!! danger "H5 is not loadable"
    Keras `.h5` and `.hdf5` architectures/weights cannot be loaded by xiRT 2.0.
    Renaming a file to `.pt` does not convert it. There is no automatic
    TensorFlow-to-PyTorch weight converter.

## Recommended migration sequence

1. Archive the exact 1.x commit/release, environment lock, YAML files, trained
   model, label encoder, target transformations, and input-data identifiers.
2. Use archived predictions where available. The completed comparison preserves
   train/validation/test predictions and source/environment provenance under
   `benchmark-tf-torch/<cohort>/results/`; no TensorFlow installation is needed
   to inspect or recompute their metrics.
3. Create a clean Python 3.12–3.14 environment and install the 2.0 checkout.
4. Generate fresh YAML with `xirt -p` and `xirt -s`. Copy scientifically relevant
   choices from old files only after checking dimensions, loss/activation
   combinations, paired columns, and target encodings.
5. Remove legacy pretrained-model/weight paths. Retrain with the original data
   and a documented split. Save the new model and preprocessing metadata.
6. Compare held-out metrics with the [benchmark protocol](benchmarks.md).
   Evaluate whether the new model meets your application criteria; do not infer
   equivalence from a passing smoke test. Legacy inference is not maintained.

## Changes that affect model meaning

The old backend did not consistently honor configured dense activations.
The 2.0 implementation uses the configured activation, so the same YAML can
represent a different effective model. A matched-architecture benchmark must
account for this, for example using explicitly linear head activations when
matching the legacy effective network.

Use `GRU` or `LSTM` on either CPU or GPU. Historical `CuDNNGRU`/`CuDNNLSTM`
names may be accepted as aliases, but they do not control device placement;
use `learning.device`. Recurrent activation is `tanh`.

Callbacks are explicit training-loop settings rather than arbitrary Keras
callback objects. Use the adapter's documented settings and returned history
mapping. Do not assume Keras layer objects, graph plotting, callback classes,
or optimizer objects are compatible.

### QC R² definition changed

The 2.0 QC value uses predictive `sklearn.metrics.r2_score`, not the legacy
squared Pearson correlation. These quantities are not interchangeable:
predictions shifted by a constant can have perfect squared correlation while
having poor predictive R². Predictive R² can be negative.

Constant observed targets do not define a meaningful predictive R²; xiRT
records `NaN` rather than presenting an undefined result as perfect accuracy.
Historical QC numbers labeled R² must therefore not be compared directly with
the corrected 2.0 metric. Recompute both models' predictions with the same
metric and held-out samples when reporting a migration comparison.

## Checkpoints and transfer learning

Keep the architecture, alphabet, sequence width, target task ordering, and
preprocessing metadata with each new checkpoint. Loading weights requires
compatible tensors; changing task dimensions requires intentional output-head
replacement. Check shapes and task labels before trusting predictions.

The CLI's final model is `xirt_model.pt`, accompanied by
`configuration.json` and recorded `split_NN.json` partitions. Fold checkpoint
writing is optional: the best epoch can be restored from memory without a
saved fold checkpoint. Rows populated as `cv=0` are not out-of-fold
predictions; see [outputs and provenance](results.md).

The 2.0 CLI exports `encoder.json` with the vocabulary, padded sequence width,
and fraction levels. Set `preprocessing.le` to that exported JSON when
reusing encoding metadata. Legacy pickled label encoders are not accepted:
retain them only in the isolated legacy environment and generate new metadata
when retraining. Checkpoint-based prediction restores the saved vocabulary,
fraction mapping, sequence length, and target scales even for smaller subsets.

Load only files produced by a trusted source. A state dictionary is
not a complete scientific provenance record: include the source revision,
environment, seeds, split IDs, scaling, and model-weight provenance.

## ESM migration is a distinct experiment

The frozen ESM-C route operates on unmodified residue sequences and mean-pools
residue embeddings. It does not reproduce the trainable modification-aware
alphabet of a recurrent xiRT model. Do not silently strip modifications and
claim equivalent inputs. Record any deliberate loss of chemical information
and evaluate whether that is acceptable for your task.
