# Retired implementation and examples

These bundles preserve material removed from the maintained PyTorch tree.
They are historical references, not supported runtime code or dependency
installation instructions.

| Bundle | Manifest | Retained material |
|---|---|---|
| `retired-benchmark-code.tar.gz` | `manifest.json` | Original benchmark runners, workflow Python/YAML, three environment locks, TF backend, old runner and workflow tests |
| `retired-support-files.tar.gz` | `support-manifest.json` | Remaining workflow/requirements, checkpoint replay helper and Sphinx sources/assets |
| `retired-examples.tar.gz` | `examples-manifest.json` | Legacy HDF5 models, output/configuration examples, old plotting/grid utilities and remaining documentation assets |

Each manifest records the bundle SHA256 and the original paths and hashes of
its files. Some documentation assets occur in more than one bundle because
retirement was performed in separate verified stages.

Use the adjacent prediction archives to inspect the historical model outputs
without installing TensorFlow. Original training configurations and the
immutable source snapshot used for the completed comparison are retained
within each cohort's `results/provenance/` directory.

Legacy pickle/HDF5 files are not xiRT 2.0 checkpoints. Do not load untrusted
pickle files or rename HDF5 models to `.pt`.
