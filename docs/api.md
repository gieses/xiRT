# Python API

For complete, runnable code with input tables, start with
[Python workflows: train, predict and explain](python-workflows.md).
This page is the lower-level API reference.

## Training and prediction

`xiRTNET(params, input_dim, embedding_dim)` validates the network configuration.
For recurrent input, `input_dim` is the padded residue-sequence width;
`build_model(alphabet=...)` supplies the vocabulary size including padding
index zero. The recurrent trainable embedding width is configured by
`embedding.length`, not inferred from the constructor's compatibility argument.

```python
import yaml

from xirt.xirtnet import xiRTNET

with open("xirt_params.yaml") as stream:
    params = yaml.safe_load(stream)

network = xiRTNET(params, input_dim=30, embedding_dim=25)
network.build_model(siamese=True, alphabet=25)
network.compile()

# Supply your encoded arrays and targets in consistent sample/task order.
# x1, x2: integer arrays (samples, 30), zero-padded.
# targets: one target array per network.tasks entry.
# train/validation partitions must be selected before fitting preprocessing.
history = network.model.fit(
    [x1, x2], targets,
    validation_data=([validation_x1, validation_x2], validation_targets),
    epochs=10, batch_size=128,
)
predictions = network.model.predict([test_x1, test_x2])
metrics = network.model.evaluate([test_x1, test_x2], test_targets)
```

The arrays in this snippet are placeholders; the
[training notebook](tutorials.md) includes data preparation and a runnable
example. A non-Siamese model
accepts one array. Fraction tasks precede continuous tasks, with names sorted
within each group; use `network.tasks` rather than guessing head order.

`network.model` is a `torch.nn.Module` with xiRT's compatibility
`fit`/`predict`/`evaluate` methods. `network.compile()` configures its
optimizer(s). `fit` returns an object with `.history` holding epoch metric
lists. These adapters are not Keras objects and do not accept every Keras
extension. Direct `forward` is for tensor-based PyTorch usage.

## Network and checkpoints

::: xirt.predictor.Predictor

::: xirt.predictor.PeptideInputs

::: xirt.predictor.prediction_frame

::: xirt.xirtnet.xiRTNET

::: xirt.xirtnet.RetentionModel

::: xirt.xirtnet.read_checkpoint

## Configuration and training utilities

::: xirt.config.NetworkConfig
    options:
      show_docstring_description: false

::: xirt.config.FineTuningConfig

::: xirt.training.make_optimizers

::: xirt.training.seed_everything

::: xirt.training.task_loss

## Frozen features and differentiable ESM-C

The optional ESM SDK is imported when needed, not to run a core recurrent
model. Consult the [ESM guide](embeddings.md) for licensing, modification
limitations, local weights, and cache semantics.

::: xirt.embeddings.ESMCEncoder

::: xirt.embeddings.tokenize_batch

::: xirt.embeddings.ESMCRetentionModel

## Recurrent adapters and linear LoKr

::: xirt.adapters.LoKrLinear

::: xirt.adapters.apply_lokr

::: xirt.adapters.apply_lora_gru

::: xirt.adapters.apply_lokr_gru

## Local experiment tracking

::: xirt.tracking.TrainingRun

## FAIMS compensation voltage

These APIs predict compensation voltage in volts, **not CCS**. See the
[FAIMS guide](faims.md) for inputs, grouping, units and coverage limitations.

::: xirt.faims.FAIMSPredictor

::: xirt.faims.prepare_faims_observations

::: xirt.faims.split_precursors

::: xirt.faims.default_faims_params

::: xirt.faims.cv_metrics

::: xirt.faims.precursor_descriptors

## Explanations and shared QC

See [prediction explanations](explanations.md) for baselines and limitations.

::: xirt.explain.Explanation

::: xirt.explain.integrated_gradients

::: xirt.qc.prediction_metrics

::: xirt.qc.prediction_qc

::: xirt.qc.plot_predictions

::: xirt.qc.plot_learning_curve

::: xirt.qc.plot_residue_attributions
