# Training and fine-tuning notebooks

Two CPU notebooks cover the recurrent Python API. Both include observed
data, run without downloads and save outputs under `examples/results/`.
They are marimo notebooks stored as Python source, so their cells can also run
as scripts.

Use a source checkout with the locked CPU environment:

```bash
uv sync --locked --extra cpu
uv run --no-sync marimo edit examples/train_predict.py --host 127.0.0.1
uv run --no-sync marimo edit examples/fine_tune.py --host 127.0.0.1
```

Run the examples without opening a notebook server:

```bash
make examples
```

Or run one directly:

```bash
uv run --no-sync python examples/train_predict.py
uv run --no-sync python examples/fine_tune.py
```

For a static report, run `uv run --no-sync marimo export html
examples/train_predict.py -o examples/results/train-predict.html`.
The source files and data are also included in the source distribution,
not the production wheel.

## Train, save and predict

`examples/train_predict.py` builds a shared GRU for crosslinked peptide pairs,
fits it for up to **150 epochs** and plots training and validation loss. It saves
`model.pt` and `preprocessing.json`, reloads both, re-encodes test rows and checks
that checkpoint predictions match the in-memory model. A scatterplot and
`test-predictions.csv` show observed and predicted RP times in seconds.
The figures are also saved as `training-loss.png` and `test-predictions.png`
under `examples/results/train-predict/`.
`history.csv` contains the per-epoch scores; `metrics.csv` compares held-out
RMSE, MAE and predictive R² against a constant training-mean predictor.

The GRU has 32 hidden units, a 16-dimensional embedding and a 32-unit ReLU
head. AdamW starts at a learning rate of 0.003 with batches of 64.
Validation loss controls learning-rate reduction (factor 0.5, patience 8)
and early stopping (patience 20). The best validation weights are restored
before saving and testing, so the executed epoch count can be below 150.
The complete settings are in `docs/examples/xirt-rp.yaml`.

On the supplied partition, the default CPU run stops after 52 epochs and
restores epoch 32. Its 256 held-out pairs give the following scores:

| Predictor | RMSE (seconds) | MAE (seconds) | Predictive R² |
|---|---:|---:|---:|
| GRU | 208.8 | 146.8 | 0.936 |
| Training mean | 830.2 | 667.1 | -0.004 |

These are example results for one seed and split, not a cross-study benchmark.
Your environment may give slightly different values. Read `metrics.csv` and
`history.csv` for your run rather than treating the table as an accuracy target.

The example uses a fixed vocabulary of canonical residues plus marked
crosslink-residue tokens, zero padding and a fixed width of 50. Target
normalization is fitted only on training rows. Its metadata file records that
contract, the data checksum and partition IDs.

This deliberately restricted tokenizer excludes modified peptides. Its
`preprocessing.json` is an example-specific API format, **not** a CLI
`encoder.json`. For production CLI use, follow the
[configuration guide](configuration.md) and retain the exported encoder and
checkpoint metadata together.

## Fine-tune from a checkpoint

`examples/fine_tune.py` trains its own starting checkpoint, so it does not
depend on running the other notebook first. Each strategy then loads that
same checkpoint:

| Strategy | Updated parameters |
|---|---|
| `recurrent_full` | Entire model |
| `recurrent_frozen` | Prediction heads |
| `recurrent_lokr` | GRU LoKr adapters and prediction heads |
| `recurrent_lora` | GRU LoRA adapters and prediction heads |

The starting checkpoint uses the same 150-epoch training budget as the first
notebook. Each strategy then gets up to **75 additional epochs**, with a base
learning rate of 0.0005 and the same validation-based controls. Adapter updates
use 0.1 times that learning rate. Every strategy starts from the same checkpoint
and uses the same partitions and target transformation.
Change `learning.epochs` in the example YAML to adjust the base-training limit,
or `fine_tuning_epochs` in the fine-tuning notebook to adjust adaptation.

The notebook saves a checkpoint, training history and held-out predictions
for each strategy. `summary.csv` records trainable parameter counts, executed
and best epochs, validation RMSE, test metrics and training-mean baseline RMSE.
`base-history.csv` contains the starting model's training history. The parameter
plot illustrates what each strategy changes. PNG files under
`examples/results/fine-tune/` contain the parameter counts
(`trainable-parameters.png`), held-out errors (`test-rmse.png`) and
observed-versus-predicted RP times for all four strategies
(`test-predictions.png`). These figures are also displayed in the notebook.
`base-training-loss.png` and `training-loss.png` show the starting model's and
the four strategies' learning curves.
See the
[fine-tuning guide](fine-tuning.md) for configuration and compatibility details.

## What the examples do not measure

Both notebooks use 2,048 distinct observed E. coli peptide pairs: 1,536 training,
256 validation and 256 test pairs. Only target-target identifications at
`fdr <= 0.01` are retained. Modified sequences and repeated sequence pairs
are excluded. The seeded data selection and checksums are documented in
`examples/data/README.md`.

Individual peptides, proteins and acquisition runs can overlap between
partitions. Fine-tuning reuses the base model's training partition;
it is staged training, not external transfer learning. Do not select methods
using the tutorial test scores or present them as benchmark results.
The [measured fine-tuning comparison](benchmarks/fine-tune-torch.md) uses a
separate fixed protocol.

The notebook tests override the epoch budgets with two-epoch smoke runs to
check execution and checkpoint round trips. `make examples` uses the full
defaults above and regenerates the CSVs and PNGs.
