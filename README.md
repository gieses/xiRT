<p align="center">
  <img src="https://gieses.github.io/xiRT/assets/xirt-logo.svg" alt="xiRT — retention-time prediction" width="360">
</p>

<p align="center">
  <strong>Retention-time prediction for linear and crosslinked peptides.</strong><br>
  Modification-aware sequences. Shared recurrent encoders. PyTorch on CPU or CUDA.
</p>

<p align="center">
  <a href="https://gieses.github.io/xiRT/">Documentation</a> ·
  <a href="https://gieses.github.io/xiRT/getting-started/">Quick start</a> ·
  <a href="https://gieses.github.io/xiRT/tutorials/">Notebooks</a> ·
  <a href="https://gieses.github.io/xiRT/benchmarks/">Benchmarks</a> ·
  <a href="https://pypi.org/project/xirt-torch/">PyPI</a>
</p>

## What xiRT does

xiRT learns peptide embeddings and predicts continuous retention times or
fraction labels. Crosslinked peptides share a recurrent encoder between their
two sequence branches; linear peptides use a single input. Train a model for
your experiment, use a saved checkpoint for prediction, or adapt a recurrent
model with full fine-tuning, a frozen encoder, LoKr, or LoRA.

## Install

Requires **Python 3.12–3.14**. The PyPI distribution is `xirt-torch`; the Python
import and command remain `xirt`.

```bash
pip install xirt-torch
```

Generate the two configuration files, then edit their input columns and targets
before training:

```bash
xirt -p xirt_params.yaml -s learning_params.yaml
xirt -i psms.csv -o results -x xirt_params.yaml -l learning_params.yaml
```

For a locked source installation, clone this repository and run
`uv sync --locked --extra cpu` (or `--extra cuda`), then prefix the commands with
`uv run --no-sync`. See [installation and CLI](https://gieses.github.io/xiRT/getting-started/) for
backend selection and optional tracking, TensorBoard, Lion, and ESM-C extras.

## Learn with real peptide data

Two [CPU notebooks](https://gieses.github.io/xiRT/tutorials/) use 2,048 observed peptide pairs:

| Notebook | Covers |
| --- | --- |
| [Train and predict](https://github.com/gieses/xiRT/blob/master/examples/train_predict.py) | Prepare inputs, train a model, save a checkpoint, and predict retention times |
| [Fine-tune a model](https://github.com/gieses/xiRT/blob/master/examples/fine_tune.py) | Compare full, frozen-encoder, GRU-LoKr, and GRU-LoRA fine-tuning |

From the source checkout:

```bash
uv run --no-sync marimo edit examples/train_predict.py --host 127.0.0.1
uv run --no-sync marimo edit examples/fine_tune.py --host 127.0.0.1
```

`make examples` executes these plus the FAIMS and interpretation examples without a notebook server. Training uses up to
150 epochs with validation-based early stopping. The
[fine-tuning guide](https://gieses.github.io/xiRT/fine-tuning/) covers checkpoint compatibility, the
Python API, and YAML settings.

## Crosslink FAIMS prediction

The source checkout also includes `xirt faims train` and shared `xirt predict`
for charge-aware **FAIMS compensation voltage (V), not collision cross section**.
`examples/faims.py` trains and evaluates a charge-aware model on bundled data.
`examples/analyze_predictions.py` compares RT and FAIMS predictions with annotated
correlations and explains residue, crosslink-site and precursor contributions.
Neither requires data downloads.

```bash
make notebook-faims
make notebook-analysis
```

`xirt predict` and `xirt explain` accept RT and FAIMS checkpoints through the same
interface. See the [FAIMS guide](https://gieses.github.io/xiRT/faims/) and
`benchmark-faims/README.md` for the sequence-only / charge / charge+m/z comparison,
pinned inputs and reproduction commands. The earlier precursor/baseline experiment
and its source snapshot are preserved under `benchmark-faims/archive/`.

## Results and compatibility

The [benchmark documentation](https://gieses.github.io/xiRT/benchmarks/) links protocols, predictions,
checkpoint provenance, and plots. PyTorch had lower mean held-out RMSE than
TensorFlow on **two of five** evaluated cohorts, not all five. Full fine-tuning
had the lowest mean RMSE among the four native fine-tuning strategies on all
five cohorts under the fixed comparison protocol.

**Upgrading from 1.x:** TensorFlow is no longer an application dependency or
supported backend, and xiRT 2.0 cannot load Keras `.h5` models. See the
[migration guide](https://gieses.github.io/xiRT/migration/). Historical artifacts remain in
`benchmark-tf-torch/<cohort>/results/`.

## Development

```bash
make sync
make check
make build
```

`pixi run sync` bootstraps the same locked uv environment. `make help` lists
documentation, notebook, test and release commands. Benchmark code lives in
`xirt/data_pipelines/benchmarks/` and is excluded from the production wheel.

[Configuration](https://gieses.github.io/xiRT/configuration/) ·
[Migration](https://gieses.github.io/xiRT/migration/) ·
[Python API](https://gieses.github.io/xiRT/api/) ·
[Development](https://gieses.github.io/xiRT/development/)

## Citation

Giese, S.H., Sinn, L.R., Wegner, F. et al. Retention time prediction using neural
networks increases identifications in crosslinking mass spectrometry.
*Nature Communications* **12**, 3237 (2021).
[DOI: 10.1038/s41467-021-23441-0](https://doi.org/10.1038/s41467-021-23441-0)

Contributors: Sven Giese and Ludwig Sinn.
