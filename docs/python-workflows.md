# Train, predict and explain in Python

Start with a peptide table, choose trained or new weights, and use the same
`Predictor` interface for predictions and residue explanations.

| What you want to do | Recipe |
| --- | --- |
| Train a new RP model, save it and predict | [1. Train RP](#1-train-rp-and-run-inference) |
| Train one model for RP and SCX | [2. Train RP + SCX](#2-train-rp-scx-and-run-inference) |
| Predict RP without training | [3. Bundled RP](#3-run-rp-inference-with-the-bundled-model) |
| Predict RP and SCX without training | [4. Bundled joint model](#4-run-rp-scx-inference-with-the-bundled-model) |
| Train a FAIMS CV model and predict | [5. FAIMS](#5-train-and-run-faims) |

**Run the shared setup first, then your chosen recipe.** Recipes 3 and 4 need
no training data. Recipes 1, 2 and 5 use the small mock table below so you can
try the complete workflow without downloading a dataset. None of these mock
training runs estimates real-world accuracy.

## Environment and shared setup

Use Python 3.12-3.14 and the [current source checkout](getting-started.md) to ensure the bundled
checkpoints and their loading API are available:

```bash
git clone https://github.com/gieses/xiRT.git
cd xiRT
uv sync --locked --extra cpu
uv run --no-sync python
```

Paste the Python blocks into that interpreter, a notebook, or a `.py` file in
the same order. They import only installed `xirt` APIs and dependencies, not
checkout-only example helpers. All outputs go to `results/python-workflows/`
relative to your working directory.

```python title="Shared setup"
from pathlib import Path

import pandas as pd
import torch

from xirt.predictor import Predictor
from xirt.qc import plot_residue_attributions

torch.set_num_threads(2)
output = Path("results/python-workflows")
output.mkdir(parents=True, exist_ok=True)

precursors = pd.DataFrame({
    "Peptide1": ["KSGVLTGLPDAYGR"],
    "Peptide2": ["LKMQVGPK"],
    "LinkPos1": [1],
    "LinkPos2": [2],
})
```

The link positions are **one-based** residue positions, not character offsets
in a modified sequence. Prediction needs only the two sequences and sites.
No observed RP/SCX, FDR or identification score is passed to the predictor.
The two-thread setting keeps these small CPU examples lightweight; tune it
for your own workload.

| Output | Meaning |
| --- | --- |
| `rp-prediction` | RP retention time, **seconds** in these recipes |
| `scx-prediction` | Continuous **SCX fraction index**, not seconds or a probability |
| `faims_cv-prediction` | Mean observed compensation voltage, **volts**, not CCS |
| `unknown_token_count` | Input coverage diagnostic; inspect warnings as well |

## Mock data for training

**Only recipes 1, 2 and 5 need this block.** Each row is a different unmodified
peptide pair. The first eight pairs are training data, the next two validation
data, and the final two held-out inputs. Labels are invented, not measurements.

```python title="Mock training data"
table = pd.DataFrame({
    "PSMID": range(12),
    "Peptide1": ["A" * n + "K" for n in range(1, 13)],
    "Peptide2": ["PEPTIDEK"] * 12,
    "LinkPos1": list(range(2, 14)),
    "LinkPos2": [8] * 12,
    "score": [1.0] * 12,
    "rp": [900.0 + 40 * n for n in range(12)],
    "scx": [17 + n % 4 for n in range(12)],
})
train = table.iloc[:8].copy()
validation = table.iloc[8:10].copy()
held_out = table.iloc[10:].copy()
inputs = held_out[["Peptide1", "Peptide2", "LinkPos1", "LinkPos2"]]
```

`PSMID` and `score` are preprocessing bookkeeping, not model features.
With real data, use measured labels and quality-filtered identifications.
Split **unordered unmodified peptide-pair groups before fitting**, keeping
repeated observations, PTMs and site variants together. Do not copy this
row-based slicing for a table with repeated pairs. The held-out labels must
not influence training, preprocessing or model selection.

## 1. Train RP and run inference

The helper below handles the native RT training steps once, for both recipes
1 and 2. Run its block before either training call.
`Predictor` is the inference interface; RT training uses `preprocess` and
`xiRTNET`. The helper keeps that preparation in one place.

It copies the bundled RP model's **architecture configuration**, then builds
a **fresh randomly initialized network**. It does not fine-tune or reuse its
learned weights. Vocabulary and target means/standard deviations are fitted on
training rows only; the saved metadata makes raw-peptide inference self-contained.

??? example "Run once: RP / RP + SCX training helper"

    ```python title="RT training helper"
    from copy import deepcopy

    from xirt.predictor import preprocess
    from xirt.xirtnet import xiRTNET

    def train_rt(train, validation, *, tasks, checkpoint):
        template = Predictor.load(model="rp").model.params
        params = {
            key: deepcopy(template[key])
            for key in (
                "LSTM", "dense", "embedding", "learning",
                "siamese", "callbacks", "encoder",
            )
        }
        params["LSTM"].update(units=16, nlayers=1, bidirectional=False)
        params["learning"].update(epochs=25, batch_size=8, device="cpu")
        params["callbacks"].update(
            early_stopping_patience=5, reduce_lr_patience=3, progressbar=False,
        )
        params["predictions"] = {"continues": list(tasks), "fractions": []}
        params["output"] = {
            f"{task}-{key}": value
            for task in tasks
            for key, value in {
                "activation": "linear", "column": task, "dimension": 1,
                "loss": "mse", "metrics": "mse", "weight": 1,
            }.items()
        }

        encoded = preprocess(train.copy(), max_length=32, cl_residue=True)
        if len(encoded.psms) != len(train):
            raise ValueError("Check unique PSMIDs and peptide lengths: rows were removed.")
        params["_alphabet"] = encoded.le.classes_.tolist()
        params["_provenance"] = {
            "preprocessing": {"cl_residue": True},
            "training": {"sequence_type": "crosslink", "example": "mock Python workflow"},
        }
        params["_target_transforms"] = {
            task: {
                "offset": float(train[task].mean()),
                "scale": float(train[task].std(ddof=0)),
                "unit": "s" if task == "rp" else "fraction",
            }
            for task in tasks
        }
        if any(t["scale"] <= 0 for t in params["_target_transforms"].values()):
            raise ValueError("Each training target must vary.")

        network = xiRTNET(params, input_dim=32, embedding_dim=len(encoded.le.classes_) + 1)
        network.build_model(siamese=True)
        network.compile()
        validation_inputs = Predictor(network.model).encode(validation).features

        def targets(frame):
            return [
                (
                    (frame[task].to_numpy() - params["_target_transforms"][task]["offset"])
                    / params["_target_transforms"][task]["scale"]
                )[:, None]
                for task in network.model.tasks
            ]

        network.model.fit(
            encoded.get_features(encoded.psms.index),
            targets(encoded.psms),
            validation_data=(validation_inputs, targets(validation)),
            epochs=params["learning"]["epochs"],
            batch_size=params["learning"]["batch_size"],
            callbacks=params["callbacks"],
        )
        network.model.save(checkpoint)
        return Predictor.load(checkpoint)
    ```

The helper uses scalar linear heads and standardized MSE. It takes labels
from `encoded.psms` because preprocessing can reorder training rows, and uses
`network.model.tasks` to match target arrays to head order. The width is
**32 tokens including termini** for this small example; choose a sufficient
width for real inputs rather than dropping or truncating them.

Now fit, save, reload and predict:

```python title="Train RP"
rp_model = train_rt(
    train, validation, tasks=("rp",), checkpoint=output / "rp.pt",
)
rp_predictions = rp_model.predict(inputs)
print(rp_predictions[["rp-prediction"]])

rp_explanation = rp_model.explain(inputs.iloc[:1], task="rp", steps=256)
rp_figure = plot_residue_attributions(rp_explanation)
rp_figure.savefig(output / "trained-rp-residues.png", bbox_inches="tight")
```

`rp_model` is loaded from the newly written `rp.pt`, not the bundled model.
The checkpoint carries the vocabulary, site encoding and inverse target
transform, so predictions and contributions are in seconds automatically.
Training has a 25-epoch ceiling, early stopping and restoration of validation-best
weights. This short mock run is a wiring example, not a recommended training budget.

## 2. Train RP + SCX and run inference

Reuse the mock data and training helper above, changing only the task list:

```python title="Train RP and SCX"
joint_model = train_rt(
    train, validation, tasks=("rp", "scx"), checkpoint=output / "rp-scx.pt",
)
joint_predictions = joint_model.predict(inputs)
print(joint_predictions[["rp-prediction", "scx-prediction"]])

scx_explanation = joint_model.explain(inputs.iloc[:1], task="scx", steps=256)
scx_figure = plot_residue_attributions(scx_explanation)
scx_figure.savefig(output / "trained-scx-residues.png", bbox_inches="tight")
```

This trains **one shared encoder with exactly two heads**, not two independent
models. Each head receives equal weight on standardized MSE, so the numerical
scale of RP seconds does not dominate SCX fractions. SCX is scalar regression
here: predictions can be fractional or outside the measured fraction range.
This is not the ordinal-classification workflow.

## 3. Run RP inference with the bundled model

Run only the shared setup and this block. No training or dataset download is
needed:

```python title="Bundled RP inference"
predictor = Predictor.load(model="rp")
predictions = predictor.predict(precursors)
print(predictions[["rp-prediction"]])

explanation = predictor.explain(precursors.iloc[:1], task="rp", steps=256)
figure = plot_residue_attributions(explanation)
figure.savefig(output / "bundled-rp-residues.png", bbox_inches="tight")
```

`model="rp"` explicitly selects the separate pretrained RP model. The same
input columns work with your own weights using `Predictor.load("my_model.pt")`.
Bundled weights are resolved from package resources, not relative to the
repository or your working directory.

## 4. Run RP + SCX inference with the bundled model

The package has **no RP + SCX-only pretrained checkpoint**. Its joint checkpoint
predicts **RP, SCX and hSAX**; select the two columns you need:

```python title="Bundled RP and SCX inference"
predictor = Predictor.load()  # Equivalent to model="multitask".
predictions = predictor.predict(precursors)
print(predictions[["rp-prediction", "scx-prediction"]])

explanation = predictor.explain(precursors.iloc[:1], task="scx", steps=256)
figure = plot_residue_attributions(explanation)
figure.savefig(output / "bundled-scx-residues.png", bbox_inches="tight")
```

This follows the same prediction, explanation and plotting workflow, but `task="scx"`
chooses **only the explanation head**, not which outputs `predict()` computes.
Use `task="rp"` to explain the joint model's RP head. Its RP predictions need
not match the separate `model="rp"` checkpoint.

## 5. Train and run FAIMS

This is xiRT's PyTorch **FAIMS/xiFAIMS-style compensation-voltage workflow**,
using `xirt.faims`; it does not require the separate historical xiFAIMS checkout.
Run the shared setup and mock-data block, but not the RT training helper.

FAIMS uses **lowercase sequence/site columns** plus the selected numeric
precursor features. Here the feature set is **charge + m/z**:

```python title="Train and predict FAIMS"
from xirt.faims import FAIMSPredictor, default_faims_params

faims_table = table.rename(columns={
    "Peptide1": "peptide1", "Peptide2": "peptide2",
    "LinkPos1": "link_pos1", "LinkPos2": "link_pos2",
})[["peptide1", "peptide2", "link_pos1", "link_pos2"]].copy()
faims_table["precursor_charge"] = [3 + n % 3 for n in range(12)]
faims_table["precursor_mz"] = [700.0 + 10 * n for n in range(12)]
faims_table["faims_cv"] = [-60.0 + 2 * n for n in range(12)]

params = default_faims_params(features="charge-mz")
params["learning"]["epochs"] = 25
fitted = FAIMSPredictor(params, maximum_residues=32).fit(
    faims_table.iloc[:8].copy(),
    faims_table.iloc[8:10].copy(),
)
fitted.save(output / "faims.pt")

cv_model = Predictor.load(output / "faims.pt")
cv_inputs = faims_table.iloc[10:].drop(columns="faims_cv")
cv_predictions = cv_model.predict(cv_inputs)
print(cv_predictions[["faims_cv-prediction"]])

cv_explanation = cv_model.explain(cv_inputs.iloc[:1], task="faims_cv", steps=256)
cv_figure = plot_residue_attributions(cv_explanation)
cv_figure.savefig(output / "trained-faims-residues.png", bbox_inches="tight")
```

`fit()` requires disjoint unmodified-pair groups and fits vocabulary, numeric
normalization and target scaling on training rows only. Predictions and
residue/precursor contributions are in **volts**. These synthetic CV labels are
already one target per precursor; for real CSM tables, use the
[FAIMS data preparation and target definition](faims.md) instead of assigning
an arbitrary optimum or mixing aggregate and original observations.

To skip FAIMS training, run only the shared setup and this block. It uses
the packaged charge/m/z checkpoint and a hypothetical precursor context:

```python title="Bundled FAIMS inference"
cv_precursors = precursors.rename(columns={
    "Peptide1": "peptide1", "Peptide2": "peptide2",
    "LinkPos1": "link_pos1", "LinkPos2": "link_pos2",
}).copy()
cv_precursors["precursor_charge"] = [3]
cv_precursors["precursor_mz"] = [900.0]

bundled_cv_model = Predictor.load(model="faims")
bundled_cv_predictions = bundled_cv_model.predict(cv_precursors)
print(bundled_cv_predictions[["faims_cv-prediction"]])
```

Replace charge and m/z with the measured values for your precursor, not
retention labels or a CV setting. `FAIMSPredictor.load()` is the equivalent
FAIMS-only loading interface.

## Read the result before using it

- **These toy fits are not accuracy demonstrations.** Use representative measured
  data, validation-based model selection and an untouched test set for evaluation.
- **Pretrained does not mean protocol-independent.** The bundled chromatography
  and FAIMS models are specific to their source studies. Read the
  [model cards](chromatography-models.md) and [measured gallery](prediction-gallery.md).
- **Attributions explain the model, not causal chemistry.** Positive bars raise
  the output relative to the zero-embedding/training-mean-context baseline.
  Keep terminus and numeric contributions in the accounting; inspect
  `explanation.summary` for completeness error. See
  [explanations and convergence](explanations.md) for interpretation.
- **Coverage errors are not fixed by relabelling columns.** RT checkpoints reject
  unseen modification/site tokens and oversized inputs. FAIMS warns and counts
  unknown tokens; inspect the coverage diagnostic before trusting predictions.

Continue with the [API reference](api.md), [training notebooks](tutorials.md),
[scientific configuration](configuration.md), or [recurrent fine-tuning](fine-tuning.md).
