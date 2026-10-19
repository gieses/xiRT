# Recurrent fine-tuning

Fine-tuning starts from a native xiRT checkpoint rather than a new random
model. **Recurrent full** updates the entire model: peptide embeddings, GRU or
LSTM, normalization layers and prediction heads. It does not mean training
only the recurrent weights.

| Strategy | Trainable components | Encoder normalization |
|---|---|---|
| `recurrent_full` | All model parameters | Updated during training |
| `recurrent_frozen` | Prediction heads | Frozen running statistics |
| `recurrent_lokr` | GRU gate-weight LoKr adapters and heads | Frozen running statistics |
| `recurrent_lora` | GRU gate-weight LoRA adapters and heads | Frozen running statistics |

LoKr and LoRA keep the pretrained embedding and base GRU weights frozen.
Adapters are zero-update initialized, so initial predictions match the base
checkpoint. These are GRU-weight adapters, not transformer adapters or
head-only adapters. GRU adaptation supports bidirectional/multilayer encoders;
LSTM supports full/frozen strategies, but not gate-weight adaptation.

## Python API

The [fine-tuning notebook](tutorials.md#fine-tune-from-a-checkpoint) includes
data, a starting-checkpoint training step and all four strategies. The
following snippet assumes prepared arrays and a compatible checkpoint:

```python
from xirt.xirtnet import xiRTNET

# params supplies learning settings; checkpoint architecture is preserved.
network = xiRTNET(params, input_dim=sequence_width, embedding_dim=alphabet_size)
network.fine_tune(
    "base.pt",
    "recurrent_lora",
    rank=4,
    alpha=4.0,
    adapter_learning_rate_ratio=0.1,
)
network.compile()
history = network.model.fit(
    train_features,
    train_targets,
    validation_data=(validation_features, validation_targets),
    epochs=30,
    batch_size=64,
)
network.model.save("adapted.pt")
```

Use `recurrent_full`, `recurrent_frozen` or `recurrent_lokr` for the other
strategies. LoKr also accepts `factor` (default `4`). Adapter learning rate is
the configured base learning rate multiplied by
`adapter_learning_rate_ratio`; heads use the base learning rate.

## Network YAML

Add this section to a complete network configuration:

```yaml
fine_tuning:
  checkpoint: base.pt
  strategy: recurrent_lokr
  rank: 4
  factor: 4
  alpha: 4.0
  adapter_learning_rate_ratio: 0.1
```

The checkpoint is required when building from YAML. The model uses the
checkpoint architecture, task definitions and vocabulary size; learning
settings come from the requested configuration. Do not combine `fine_tuning`
with the separate `adapter` section or an ESM encoder route.

Save and reuse preprocessing metadata with the checkpoint. Equal vocabulary
sizes do not prove equal token meanings: use the same `encoder.json`, sequence
width, modification encoding, target definitions and scaling. For a different
prediction task, replace the output heads intentionally before fitting rather
than assuming checkpoint loading remaps targets.

## Choosing a strategy

Start with full fine-tuning when enough representative training data and
compute are available. Frozen-encoder training is a useful low-cost control.
Adapters constrain trainable encoder parameters, but fewer parameters do not
guarantee lower error or faster training.

The [measured comparison](benchmarks/fine-tune-torch.md) uses common train-only
starting checkpoints and the same fixed budget for all four strategies.
It measures staged within-cohort training, not transfer to a new study.
