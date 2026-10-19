# Installation and CLI

## Supported runtime

The distribution name is `xirt-torch`; imports and commands still use `xirt`.
After a release is published, install it with `pip install xirt-torch`.
The checkout commands below use the locked development environment instead.

Use CPython 3.12, 3.13, or 3.14. Core dependencies include PyTorch
`>=2.9,<3`, NumPy, pandas `<3`, SciPy, scikit-learn, Biopython, Pyteomics,
PyYAML, Matplotlib, seaborn `<0.14`, palettable, statannotations, Pydantic 2+,
and tqdm for training progress bars.
The core package does not depend on TensorFlow, TensorFlow Addons, or Keras.

Create an isolated environment in your project:

```bash
uv sync --locked --python 3.13 --extra cpu
uv run --no-sync xirt --version
```

Install uv with its standalone installer. The `cpu` extra selects matching
CPU-only PyTorch wheels; `cuda` selects the CUDA 13.0 backend.
These extras are mutually exclusive. On a compatible NVIDIA host, use
`uv sync --locked --python 3.12 --extra cuda`. For other backends, use uv's
PyTorch backend support with the standard package metadata, rather than mixing
CPU and CUDA installations. Set
`learning.device: cuda` in the network YAML; do not select the device by
choosing a CuDNN-prefixed recurrent layer type.

## Optional extras

| Extra | Installation from a checkout | Purpose |
|---|---|---|
| `esm` | `uv sync --locked --python 3.12 --extra cpu --extra esm` | Local ESM-C SDK 3.2.3–3.2.x, Python 3.12 only |
| `experiments` | `uv sync --locked --extra cpu --extra experiments` | Optional Lion optimizer |
| `tracking` | `uv sync --locked --extra cpu --extra tracking` | Local-first MLflow 3 run tracking |
| `tensorboard` | `uv sync --locked --extra cpu --extra tensorboard` | TensorBoard logging when `callbacks.tensor_board` is enabled |

Development tools use `--group dev`; documentation tools use `--group docs`.
They are dependency groups, not runtime extras.

Include all desired extras in each sync, replacing `cpu` with `cuda` as needed.
`uv.lock` owns the project resolution. After syncing, use `uv run --no-sync`
before the CLI commands below, or activate `.venv` (`.venv\Scripts\activate`
on Windows) to invoke the installed commands directly.

The ESM SDK is deliberately bounded below 3.3 because this implementation
validates the SDK 3.2 `ESMC` interface. Do not independently upgrade ESM without
checking xiRT compatibility. Installing extras does not obtain, license, or
validate pretrained model weights.

## Stable command line

Generate fresh configurations rather than copying historical environment
lockfiles:

```bash
xirt -p xirt_params.yaml
xirt -s learning_params.yaml
xirt -i psms.csv -o out_dir -x xirt_params.yaml -l learning_params.yaml
```

Both templates can also be generated together:

```bash
xirt -p xirt_params.yaml -s learning_params.yaml
# Equivalent long options:
xirt --sample_xirt_params xirt_params.yaml \
  --sample_learning_params learning_params.yaml
```

`-x` supplies network architecture, task, and optimizer options. `-l` supplies
preprocessing, input filtering, and training mode. `-i` is the peptide table and
`-o` is the result directory. Modes in the learning file are `train`, `predict`,
and `crossvalidation`.
For tracked runs, install `.[tracking]` and consult the
[local tracking guide](tracking.md).

The runner writes `xirt_model.pt`, `encoder.json`, effective `configuration.json`, and
`split_NN.json` files. Per-fold checkpoint files are optional; best-epoch
selection/restoration also works in memory. Training/validation predictions
marked `cv=0` are not out-of-fold estimates. See
[outputs and provenance](results.md) before assessing prediction quality.
For inference on a subset, reuse the saved checkpoint/encoder metadata:
sequence width, vocabulary, fraction levels, and target scales must remain
consistent with training.

Configure target names and table columns before running. The generated defaults
are examples, not proof that arbitrary input tables meet the data contract.
See [configuration](configuration.md) for schema and split cautions.

For a small example with supplied data, use the
[training and fine-tuning notebooks](tutorials.md).

## Containers

Both `Dockerfile` and the compatibility filename `Dockerfile-slim` build a
Python 3.13 slim CPU image, with no TensorFlow and a non-root runtime user:

```bash
docker build -t xirt:2.0 .
docker run --rm -v "$PWD:/data" xirt:2.0 \
  -i /data/psms.csv -o /data/out_dir \
  -x /data/xirt_params.yaml -l /data/learning_params.yaml
```

Ensure the container user can write the mounted output directory. The image
does not include sample data, ESM weights, or CUDA. CPU CI validates the package
separately; building or running the image requires Docker.
