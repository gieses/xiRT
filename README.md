# xiRT 2.0

![xiRT](docs/assets/xirt-logo.png)

Retention-time prediction for linear and crosslinked peptides using PyTorch.
The default model learns modification-aware peptide embeddings, shares a
recurrent encoder between crosslink branches, and predicts continuous RTs or
fraction labels. It runs on CPU or CUDA.

## Install and run

Requires Python 3.12–3.14; the default development interpreter is 3.13.
The PyPI distribution is **`xirt-torch`**; the Python import and CLI remain
`xirt`.

```bash
uv sync --locked --extra cpu
uv run --no-sync xirt -p xirt_params.yaml -s learning_params.yaml
# Edit the generated configurations for your input columns and targets.
uv run --no-sync xirt -i psms.csv -o results -x xirt_params.yaml -l learning_params.yaml
```

Replace `--extra cpu` with `--extra cuda` for the locked CUDA backend.
Optional extras provide local MLflow tracking (`tracking`), TensorBoard
(`tensorboard`), Lion (`experiments`), and local ESM-C embeddings
(`esm`, Python 3.12 only). ESM model access and licensing are separate.

## Fine-tuning

Native recurrent models support full fine-tuning, frozen-encoder head training,
and GRU-weight LoKr or LoRA. Checkpoints retain architecture and adapter settings.
See [fine-tuning](docs/fine-tuning.md) for the API, YAML and measured comparison.

TensorFlow is no longer an application dependency or supported backend.
Historical TF and PyTorch predictions, checkpoint provenance, tables and plots
are preserved under `benchmark-tf-torch/<cohort>/results/`.
PyTorch had lower mean held-out RMSE on two of five evaluated cohorts, not all
five. See [benchmark results](docs/benchmarks.md).
Keras `.h5` models cannot be loaded by xiRT 2.0.

## Examples

[CPU notebooks](docs/tutorials.md) cover training, checkpoint-based prediction
and all four fine-tuning strategies with 2,048 observed peptide pairs.
Training uses up to 150 epochs with validation-based early stopping:

```bash
uv run --no-sync marimo edit examples/train_predict.py --host 127.0.0.1
uv run --no-sync marimo edit examples/fine_tune.py --host 127.0.0.1
# Run both without a notebook server:
make examples
```

## Development

```bash
make sync
make check
make build
```

`pixi run sync` bootstraps the same locked uv environment. `make help` lists
documentation, notebook, test and release commands. Benchmark code lives in
`xirt/data_pipelines/benchmarks/` and is excluded from the production wheel.

[Documentation](https://gieses.github.io/xiRT/) ·
[Configuration](docs/configuration.md) · [Migration](docs/migration.md) ·
[Development](docs/development.md)

## Citation

Giese, S.H., Sinn, L.R., Wegner, F. et al. Retention time prediction using neural
networks increases identifications in crosslinking mass spectrometry.
*Nature Communications* **12**, 3237 (2021).
https://doi.org/10.1038/s41467-021-23441-0

Contributors: Sven Giese and Ludwig Sinn.
