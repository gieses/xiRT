"""PyTorch sequence/embedding networks with the historical xiRT training API."""

import copy
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm.auto import trange

from xirt.config import FineTuningConfig, NetworkConfig
from xirt.tracking import TrainingRun
from xirt.training import make_optimizers, seed_everything, task_loss

logger = logging.getLogger(__name__)


class SafeBatchNorm(nn.BatchNorm1d):
    """Use running statistics for singleton training batches instead of dropping samples."""

    def forward(self, x: Tensor) -> Tensor:
        """Normalize features while preserving singleton batches."""
        if self.training and x.shape[0] == 1:
            return nn.functional.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps
            )
        return super().forward(x)


def _activation(name: str) -> nn.Module:
    """Resolve supported dense activations or report an invalid configuration."""
    functions = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "linear": nn.Identity,
        "sigmoid": nn.Sigmoid,
    }
    if name not in functions:
        raise ValueError(f"Unsupported dense activation: {name}")
    return functions[name]()


class RetentionModel(nn.Module):
    """Shared peptide encoder and task-specific heads, with NumPy/Pandas adapters.

    Parameters:
        params: Validated architecture mapping.
        input_dim: Padded sequence length or embedding width.
        alphabet: Vocabulary size including padding index zero.
        siamese: Whether two peptides share the encoder.

    Example:
        >>> model = RetentionModel(params, 30, 25, True)  # doctest: +SKIP
        >>> outputs = model(torch.ones(2, 30, dtype=torch.long),
        ...                 torch.ones(2, 30, dtype=torch.long))  # doctest: +SKIP
    """

    numeric_mean: Tensor
    numeric_scale: Tensor
    numeric_fitted: Tensor

    def __init__(
        self, params: dict[str, Any], input_dim: int, alphabet: int, siamese: bool
    ) -> None:
        super().__init__()
        self.params = copy.deepcopy(params)
        self.input_dim, self.alphabet, self.siamese = input_dim, alphabet, siamese
        self.tasks = sorted(params["predictions"]["fractions"]) + sorted(
            params["predictions"]["continues"]
        )
        self.route = params["encoder"]["route"]
        self.optimizers: list[torch.optim.Optimizer] = []
        self.metrics_names = (
            ["loss"]
            + [f"{t}_loss" for t in self.tasks]
            + [f"{t}_{params['output'][f'{t}-metrics']}" for t in self.tasks]
        )
        self._is_compiled = False
        self.regularized: list[tuple[nn.Module, str, float]] = []
        self.activity: list[Tensor] = []
        if self.route == "recurrent":
            self.embedding = nn.Embedding(alphabet, params["embedding"]["length"], padding_idx=0)
            r = params["LSTM"]
            cls = nn.GRU if "GRU" in r["type"] else nn.LSTM
            self.recurrent = cls(
                params["embedding"]["length"],
                r["units"],
                num_layers=r["nlayers"],
                bidirectional=r["bidirectional"],
                batch_first=True,
            )
            width = r["units"] * (2 if r["bidirectional"] else 1)
            self.encoder_norm = SafeBatchNorm(width) if r["lstm_bn"] else nn.Identity()
            self.regularized.append(
                (self.recurrent, r["kernel_regularization"], r["kernelregularizer_value"])
            )
        else:
            width = params["encoder"]["embedding_dim"]
        if siamese and params["siamese"]["merge_type"] == "concatenate":
            width *= 2
        self.numeric_features = list(params.get("numeric_features", []))
        if self.numeric_features:
            count = len(self.numeric_features)
            width += count
            self.register_buffer("numeric_mean", torch.zeros(count))
            self.register_buffer("numeric_scale", torch.ones(count))
            self.register_buffer("numeric_fitted", torch.tensor(False))
        self.heads = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        for task in self.tasks:
            dim = width
            layers: list[nn.Module] = []
            d = params["dense"]
            for i in range(d["nlayers"]):
                layer = nn.Linear(dim, d["neurons"][i])
                layers.append(layer)
                if d.get("regularization", [True] * d["nlayers"])[i]:
                    self.regularized.append(
                        (layer, d["kernel_regularizer"][i], d["regularizer_value"][i])
                    )
                if d["dense_bn"][i]:
                    layers.append(SafeBatchNorm(d["neurons"][i]))
                # Legacy dense layers omitted their configured activations. 2.0 honors them.
                layers.extend([_activation(d["activation"][i]), nn.Dropout(d["dropout"][i])])
                dim = d["neurons"][i]
            self.heads[task] = nn.Sequential(*layers)
            self.outputs[task] = nn.Linear(dim, params["output"][f"{task}-dimension"])
        if params.get("adapter"):
            from xirt.adapters import apply_lokr

            apply_lokr(self.heads, **params["adapter"])
        self.to(torch.device(params["learning"].get("device", "cpu")))
        if params.get("fine_tuning"):
            self.configure_fine_tuning(params["fine_tuning"])

    def configure_fine_tuning(self, options: dict[str, Any]) -> None:
        """Set trainability after loading base weights; recompile before training."""
        from torch.nn.utils import parametrize

        from xirt.adapters import apply_lokr_gru, apply_lora_gru

        settings = FineTuningConfig.model_validate(options)
        if self.route != "recurrent":
            raise ValueError("Fine-tuning strategies require a recurrent encoder.")
        if any(parametrize.is_parametrized(module) for module in self.modules()):
            raise ValueError("Load an unadapted base checkpoint before changing strategies.")
        self.requires_grad_(settings.strategy == "recurrent_full")
        self.heads.requires_grad_(True)
        self.outputs.requires_grad_(True)
        if settings.strategy in {"recurrent_lokr", "recurrent_lora"}:
            if not isinstance(self.recurrent, nn.GRU):
                raise ValueError("Recurrent adapters require a GRU.")
            if settings.strategy == "recurrent_lokr":
                apply_lokr_gru(
                    self.recurrent, rank=settings.rank, factor=settings.factor, alpha=settings.alpha
                )
            else:
                apply_lora_gru(self.recurrent, rank=settings.rank, alpha=settings.alpha)
        self.params["fine_tuning"] = settings.model_dump()
        self.params["learning"]["adapter_learning_rate_ratio"] = (
            settings.adapter_learning_rate_ratio
        )
        self.optimizers = []
        self._is_compiled = False
        self.train(self.training)

    def train(self, mode: bool = True) -> "RetentionModel":
        """Keep frozen encoder statistics unchanged while training heads/adapters."""
        super().train(mode)
        settings = self.params.get("fine_tuning")
        if settings and settings["strategy"] != "recurrent_full":
            self.embedding.eval()
            self.encoder_norm.eval()
            if settings["strategy"] == "recurrent_frozen":
                self.recurrent.eval()
        return self

    @property
    def device(self) -> torch.device:
        """Return the device holding model parameters."""
        return next(self.parameters()).device

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Return the primary optimizer; Muon also has an AdamW companion."""
        return self.optimizers[0]

    def encode(self, x: Tensor) -> Tensor:
        """Encode pre-padded tokens, ignoring padding, or accept frozen embeddings."""
        if self.route == "esm-c":
            return x.float()
        mask = x != 0
        lengths = mask.sum(1)
        # Historical preprocessing pads on the left. Compact real residues to the right
        # padding convention required by pack_padded_sequence.
        positions = mask.long().cumsum(1) - 1
        compact = torch.zeros_like(x)
        rows = torch.arange(len(x), device=x.device)[:, None].expand_as(x)
        compact[rows[mask], positions[mask]] = x[mask]
        embedded = self.embedding(compact)
        packed = pack_padded_sequence(
            embedded, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.recurrent(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        directions = 2 if self.params["LSTM"]["bidirectional"] else 1
        features = hidden[-directions:].transpose(0, 1).reshape(len(x), -1)
        features = self.encoder_norm(features)
        self.activity.append(features)
        return features * (lengths > 0).unsqueeze(1)

    def forward(
        self, first: Tensor, second: Tensor | None = None, *, numeric: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Return task logits; probabilistic activation is applied only for prediction."""
        if self.numeric_features:
            if numeric is None or numeric.shape != (len(first), len(self.numeric_features)):
                raise ValueError("Configured numeric features are required for every sample.")
            if not bool(self.numeric_fitted):
                raise RuntimeError("Numeric normalization is not fitted; train or load a model.")
            if not torch.isfinite(numeric).all():
                raise ValueError("Numeric features must be finite.")
        elif numeric is not None:
            raise ValueError("This model was not configured with numeric features.")
        self.activity = []
        features = self.encode(first)
        if self.siamese:
            if second is None:
                raise ValueError("Siamese networks require two peptide inputs.")
            other = self.encode(second)
            merge = self.params["siamese"]["merge_type"]
            if merge == "concatenate":
                features = torch.cat((features, other), dim=1)
            elif merge == "add":
                features = features + other
            elif merge == "average":
                features = (features + other) / 2
            elif merge == "multiply":
                features = features * other
            else:
                features = torch.maximum(features, other)
        elif second is not None:
            raise ValueError("Single-peptide networks accept only one input.")
        if numeric is not None:
            normalized = (numeric - self.numeric_mean) / self.numeric_scale
            features = torch.cat((features, normalized), dim=1)
        outputs = {t: self.outputs[t](self.heads[t](features)) for t in self.tasks}
        for t in self.tasks:
            if self.params["output"][f"{t}-activation"] == "relu":
                outputs[t] = outputs[t].relu()
        return outputs

    def _forward_inputs(self, inputs: tuple[Tensor, ...]) -> dict[str, Tensor]:
        branches = 2 if self.siamese else 1
        return self(
            inputs[0],
            inputs[1] if self.siamese else None,
            numeric=inputs[branches] if self.numeric_features else None,
        )

    def _fit_numeric_scaling(self, inputs: tuple[Tensor, ...]) -> None:
        if not self.numeric_features or bool(self.numeric_fitted):
            return
        values = inputs[-1].to(self.device)
        self.numeric_mean.copy_(values.mean(0))
        scale = values.std(0, unbiased=False)
        self.numeric_scale.copy_(torch.where(scale > 0, scale, torch.ones_like(scale)))
        if not (
            torch.isfinite(self.numeric_mean).all() and torch.isfinite(self.numeric_scale).all()
        ):
            raise FloatingPointError("Nonfinite numeric-feature normalization.")
        self.numeric_fitted.fill_(True)
        self.params["_numeric_scaling"] = {
            "columns": self.numeric_features,
            "mean": self.numeric_mean.cpu().tolist(),
            "scale": self.numeric_scale.cpu().tolist(),
            "training_rows": len(values),
        }

    def _inputs(self, x: Any) -> tuple[Tensor, ...]:
        """Validate and convert tabular features without losing sample order."""
        is_branches = isinstance(x, tuple) or (
            (self.siamese or self.numeric_features) and isinstance(x, list)
        )
        values = list(x) if is_branches else [x]
        branches = 2 if self.siamese else 1
        if len(values) != branches + bool(self.numeric_features):
            raise ValueError(
                "Input count does not match the peptide branches and numeric features."
            )
        result = []
        for index, value in enumerate(values):
            is_numeric = index == branches
            if (
                is_numeric
                and isinstance(value, pd.DataFrame)
                and value.columns.tolist() != self.numeric_features
            ):
                raise ValueError("Numeric feature columns must match the configured order.")
            array = value.to_numpy() if isinstance(value, pd.DataFrame) else np.asarray(value)
            width = (
                len(self.numeric_features)
                if is_numeric
                else (
                    self.input_dim
                    if self.route == "recurrent"
                    else self.params["encoder"]["embedding_dim"]
                )
            )
            if (
                array.ndim != 2
                or array.shape[1] != width
                or array.dtype.kind not in "biuf"
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"Expected finite inputs of shape (samples, {width}).")
            if (
                not is_numeric
                and self.route == "recurrent"
                and (
                    (array < 0).any()
                    or (array >= self.alphabet).any()
                    or not np.equal(array, np.floor(array)).all()
                )
            ):
                raise ValueError("Sequence tokens must be integer indices within the vocabulary.")
            tensor = torch.as_tensor(
                array,
                dtype=(
                    torch.long if self.route == "recurrent" and not is_numeric else torch.float32
                ),
            )
            if not torch.isfinite(tensor).all():
                raise ValueError("Inputs must be representable as finite model tensors.")
            result.append(tensor)
        if len({len(v) for v in result}) != 1:
            raise ValueError("All inputs must have matching sample counts.")
        return tuple(result)

    def _targets(self, y: Any, size: int) -> tuple[Tensor, ...]:
        """Validate labels for each task before training or evaluation."""
        values = y if isinstance(y, (list, tuple)) else [y]
        if len(values) != len(self.tasks):
            raise ValueError("Target count does not match prediction tasks.")
        result = tuple(
            torch.as_tensor(np.asarray(v).reshape(size, -1), dtype=torch.float32) for v in values
        )
        for t, value in zip(self.tasks, result):
            if value.shape != (size, self.params["output"][f"{t}-dimension"]):
                raise ValueError(f"Incorrect target dimensions for {t}.")
            if not torch.isfinite(value).all():
                raise ValueError("Targets must be finite.")
        return result

    def _regularization(self) -> Tensor:
        """Compute configured kernel penalties, excluding biases and frozen parameters."""
        penalty = torch.zeros((), device=self.device)
        for layer, kind, value in self.regularized:
            if kind not in {"l1", "l2", "l1l2"}:
                raise ValueError(f"Unsupported regularizer: {kind}")
            for name, weight in layer.named_parameters():
                if "weight" not in name or not weight.requires_grad:
                    continue
                if kind in {"l1", "l1l2"}:
                    penalty = penalty + value * weight.abs().sum()
                if kind in {"l2", "l1l2"}:
                    penalty = penalty + value * weight.square().sum()
        return penalty

    def _losses(
        self, predictions: dict[str, Tensor], targets: tuple[Tensor, ...]
    ) -> tuple[Tensor, list[Tensor]]:
        """Compute weighted task objectives and optional recurrent activity penalty."""
        out = self.params["output"]
        losses = [
            task_loss(
                predictions[t],
                y,
                out[f"{t}-loss"],
                self.params["learning"].get("focal_gamma", 2.0),
                self.params["learning"].get("focal_alpha"),
            )
            for t, y in zip(self.tasks, targets)
        ]
        total = sum(out[f"{t}-weight"] * loss for t, loss in zip(self.tasks, losses))
        if self.route == "recurrent" and self.activity:
            kind = self.params["LSTM"]["activity_regularization"]
            scale = self.params["LSTM"]["activityregularizer_value"]
            for value in self.activity:
                if kind in {"l1", "l1l2"}:
                    total = total + scale * value.abs().sum() / len(value)
                if kind in {"l2", "l1l2"}:
                    total = total + scale * value.square().sum() / len(value)
        return total + self._regularization(), losses

    def compile(self) -> None:
        """Initialize fresh optimizers; call after replacing heads or trainability."""
        self.optimizers = make_optimizers(self, self.params["learning"])
        self._is_compiled = True

    def _probability(self, task: str, logits: Tensor) -> Tensor:
        """Convert raw classification logits to public probability outputs."""
        activation = self.params["output"][f"{task}-activation"]
        if activation == "sigmoid":
            return logits.sigmoid()
        if activation == "softmax":
            return logits.softmax(-1)
        return logits

    def evaluate(self, x: Any, y: Any, batch_size: int = 512, verbose: int = 0) -> list[float]:
        """Return total loss, task losses and configured metrics, in metrics_names order."""
        inputs = self._inputs(x)
        size = len(inputs[0])
        if size == 0:
            raise ValueError("Cannot evaluate an empty dataset.")
        targets = self._targets(y, size)
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        sums = np.zeros(len(self.metrics_names))
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for start in range(0, size, batch_size):
                    sl = slice(start, start + batch_size)
                    preds = self._forward_inputs(tuple(v[sl].to(self.device) for v in inputs))
                    labels = tuple(v[sl].to(self.device) for v in targets)
                    loss, losses = self._losses(preds, labels)
                    metrics = []
                    for task, label in zip(self.tasks, labels):
                        pred = self._probability(task, preds[task])
                        metric = self.params["output"][f"{task}-metrics"]
                        if metric == "accuracy":
                            if self.params["output"][f"{task}-activation"] == "sigmoid":
                                value = ((pred > 0.5) == (label > 0.5)).float().mean()
                            else:
                                value = (pred.argmax(-1) == label.argmax(-1)).float().mean()
                        elif metric == "mae":
                            value = (pred - label).abs().mean()
                        else:
                            value = (pred - label).square().mean()
                        metrics.append(value)
                    sums += np.array([v.item() for v in [loss, *losses, *metrics]]) * len(labels[0])
        finally:
            self.train(was_training)
        return (sums / size).tolist()

    def fit(
        self,
        x: Any,
        y: Any,
        validation_data: tuple[Any, Any] | None = None,
        validation_split: float = 0.0,
        epochs: int = 1,
        batch_size: int = 128,
        verbose: int = 1,
        callbacks: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        """Train shuffled minibatches and restore the best validation checkpoint.

        Explicit validation data is preferred; validation_split reserves the tail
        before shuffling. No sample is discarded, including singleton batches.
        Returns an object with ``history``, a mapping of epoch metric lists.

        Example:
            >>> history = model.fit((first, second), [observed_rt],
            ...                     validation_data=(val_features, val_targets),
            ...                     epochs=10, batch_size=64)  # doctest: +SKIP
        """
        if not self._is_compiled:
            raise RuntimeError("Call compile before fit.")
        if epochs < 1 or batch_size < 1:
            raise ValueError("Epochs and batch size must be positive.")
        inputs = self._inputs(x)
        size = len(inputs[0])
        if size < 1:
            raise ValueError("Cannot train on an empty dataset.")
        targets = self._targets(y, size)
        if validation_data is not None and validation_split:
            raise ValueError("Choose validation_data or validation_split, not both.")
        if validation_split:
            if not 0 < validation_split < 1:
                raise ValueError("Validation split must be in (0, 1).")
            split = int(size * (1 - validation_split))
            if not 0 < split < size:
                raise ValueError("Validation split leaves an empty partition.")
            val_x = tuple(v[split:].numpy() for v in inputs)
            validation_data = (
                val_x if self.siamese or self.numeric_features else val_x[0],
                [v[split:].numpy() for v in targets],
            )
            inputs, targets = tuple(v[:split] for v in inputs), tuple(v[:split] for v in targets)
            size = split
        self._fit_numeric_scaling(inputs)
        options = callbacks or {}
        monitor = "val_loss" if validation_data is not None else "loss"
        history: dict[str, list[float]] = {k: [] for k in self.metrics_names}
        if validation_data is not None:
            history.update({f"val_{k}": [] for k in self.metrics_names})
        schedulers = (
            [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    factor=options.get("reduce_lr_factor", 0.5),
                    patience=options.get("reduce_lr_patience", 15),
                )
                for opt in self.optimizers
            ]
            if options.get("reduce_lr")
            else []
        )
        tracking = TrainingRun(self.params.get("tracking", {}), self.params, inputs, targets)
        with tracking:
            return self._fit_epochs(
                inputs,
                targets,
                validation_data,
                epochs,
                batch_size,
                verbose,
                options,
                history,
                monitor,
                schedulers,
                tracking,
            )

    def _fit_epochs(
        self,
        inputs: tuple[Tensor, ...],
        targets: tuple[Tensor, ...],
        validation_data: tuple[Any, Any] | None,
        epochs: int,
        batch_size: int,
        verbose: int,
        options: dict[str, Any],
        history: dict[str, list[float]],
        monitor: str,
        schedulers: list[Any],
        tracking: TrainingRun,
    ) -> SimpleNamespace:
        """Run epochs within the tracking context and always close the summary writer."""
        size = len(inputs[0])
        best, stale, best_state = float("inf"), 0, None
        writer = None
        try:
            if options.get("tensor_board"):
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except ImportError as exc:
                    raise ImportError(
                        "TensorBoard requires the tensorboard extra: uv sync --extra tensorboard."
                    ) from exc

                writer = SummaryWriter(str(Path(options["callback_path"]) / "tensorboard"))
            for epoch in trange(epochs, disable=not options.get("progressbar", False), desc="xiRT"):
                self.train()
                for indices in torch.randperm(size).split(batch_size):
                    for optimizer in self.optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    predictions = self._forward_inputs(
                        tuple(v[indices].to(self.device) for v in inputs)
                    )
                    loss, _ = self._losses(
                        predictions, tuple(v[indices].to(self.device) for v in targets)
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss.")
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.parameters(),
                        self.params["learning"].get("clip_norm", 5.0),
                        error_if_nonfinite=True,
                    )
                    for optimizer in self.optimizers:
                        optimizer.step()
                train_x = tuple(v.numpy() for v in inputs)
                metrics = self.evaluate(
                    train_x if self.siamese or self.numeric_features else train_x[0],
                    [v.numpy() for v in targets],
                    batch_size,
                )
                for key, value in zip(self.metrics_names, metrics):
                    history[key].append(value)
                if validation_data is not None:
                    val_metrics = self.evaluate(*validation_data, batch_size=batch_size)
                    for key, value in zip(self.metrics_names, val_metrics):
                        history[f"val_{key}"].append(value)
                score = history[monitor][-1]
                tracking.log_metrics({key: values[-1] for key, values in history.items()}, epoch)
                if not np.isfinite(score):
                    raise FloatingPointError("Nonfinite validation loss.")
                for scheduler in schedulers:
                    scheduler.step(score)
                if score < best:
                    best, stale = score, 0
                    best_state = copy.deepcopy(self.state_dict())
                    if options.get("check_point"):
                        path = Path(options["callback_path"])
                        suffix = options.get("suffix", "")
                        self.save(path / f"xirt_model_{suffix}.pt")
                        self.save(path / f"xirt_weights_{suffix}.pt")
                else:
                    stale += 1
                if options.get("log_csv"):
                    pd.DataFrame(history).assign(epoch=np.arange(epoch + 1)).to_csv(
                        Path(options["callback_path"])
                        / f"xirt_epochlog_{options.get('suffix', '')}.log",
                        index=False,
                    )
                if writer is not None:
                    for key, values in history.items():
                        writer.add_scalar(key, values[-1], epoch)
                if verbose:
                    logger.info("Epoch %d/%d: %s=%.6g", epoch + 1, epochs, monitor, score)
                if (
                    options.get("early_stopping")
                    and stale >= options.get("early_stopping_patience", 15)
                    and stale > 0
                ):
                    break
            if best_state is not None:
                self.load_state_dict(best_state)
            if tracking.run_id is not None:
                import tempfile

                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pt"
                    self.save(path)
                    tracking.log_artifact(path)
        finally:
            if writer is not None:
                writer.close()
        return SimpleNamespace(history=history)

    def predict(self, x: Any, batch_size: int = 512, verbose: int = 0) -> Any:
        """Predict in bounded batches; return an array for one task, otherwise a list.

        Example:
            >>> predicted_rt = model.predict((first, second))  # doctest: +SKIP
        """
        inputs = self._inputs(x)
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        outputs: dict[str, list[np.ndarray]] = {t: [] for t in self.tasks}
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(inputs[0]), batch_size):
                    preds = self._forward_inputs(
                        tuple(v[start : start + batch_size].to(self.device) for v in inputs)
                    )
                    for t in self.tasks:
                        outputs[t].append(self._probability(t, preds[t]).cpu().numpy())
        finally:
            self.train(was_training)
        arrays = [
            np.concatenate(outputs[t])
            if outputs[t]
            else np.empty((0, self.params["output"][f"{t}-dimension"]), dtype=np.float32)
            for t in self.tasks
        ]
        return arrays[0] if len(arrays) == 1 else arrays

    def count_params(self) -> int:
        """Return total parameter count, excluding running-statistics buffers."""
        return sum(p.numel() for p in self.parameters())

    def target_transform(self, task: str) -> tuple[float, float, str]:
        """Return multiplier, offset and units mapping a scalar head to reported values."""
        if task not in self.tasks:
            raise ValueError(f"Unknown prediction task: {task}")
        transform = self.params.get("_target_transforms", {}).get(task)
        if transform is not None:
            return float(transform["scale"]), float(transform["offset"]), transform["unit"]
        faims = self.params.get("_faims")
        if faims is not None and task == "faims_cv":
            return float(faims["target_scale"]), float(faims["target_mean"]), faims["units"]
        tutorial = self.params.get("_preprocessing", {})
        if tutorial.get("schema") == "xirt-api-tutorial-v1" and task == tutorial["target_column"]:
            return (
                float(tutorial["target_std"]),
                float(tutorial["target_mean"]),
                tutorial["target_units"],
            )
        return (
            float(self.params.get("_target_scales", {}).get(task, 1.0)),
            0.0,
            self.params.get("target_units", {}).get(task, "input units"),
        )

    def predict_frame(self, x: Any, *, index: Any = None, batch_size: int = 512) -> pd.DataFrame:
        """Return named predictions in original units, preserving an optional input index."""
        from xirt.predictor import prediction_frame

        return prediction_frame(self, self.predict(x, batch_size=batch_size), index=index)

    def explain(
        self,
        x: Any,
        *,
        task: str | None = None,
        index: Any = None,
        tokens: Any = None,
        steps: int = 64,
    ) -> Any:
        """Explain a continuous prediction at residue level using integrated gradients.

        Embeddings are integrated from zero with sequence lengths held fixed;
        numeric features are integrated from their fitted training means.
        Returned contributions and completeness errors use the target's units.
        """
        from xirt.explain import integrated_gradients

        return integrated_gradients(
            self, self._inputs(x), task=task, index=index, tokens=tokens, steps=steps
        )

    @classmethod
    def load(cls, location: str | Path, *, device: str = "cpu") -> "RetentionModel":
        """Reconstruct a native checkpoint through the public tensor-model API."""
        data = read_checkpoint(location)
        params = data["params"]
        params["learning"]["device"] = device
        model = cls(params, data["input_dim"], data["alphabet"], data["siamese"])
        model.load_state_dict(data["state_dict"])
        return model

    def save(self, location: str | Path) -> None:
        """Save a versioned tensor checkpoint with reconstruction and vocabulary metadata.

        Example:
            >>> model.save("retention.pt")  # doctest: +SKIP
        """
        path = Path(location)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "params": self.params,
                "input_dim": self.input_dim,
                "alphabet": self.alphabet,
                "siamese": self.siamese,
                "state_dict": self.state_dict(),
            },
            path,
        )

    def load_weights(self, location: str | Path) -> None:
        """Load strictly compatible 2.0 tensor weights; legacy HDF5 is not accepted."""
        checkpoint = read_checkpoint(location)
        if checkpoint["params"].get("numeric_features", []) != self.numeric_features:
            raise ValueError("Checkpoint numeric feature schema differs from the model.")
        if self.route == "recurrent" and checkpoint["params"].get("_alphabet") != self.params.get(
            "_alphabet"
        ):
            raise ValueError("Checkpoint vocabulary differs from model vocabulary.")
        if any(
            checkpoint["params"]["encoder"].get(key) != self.params["encoder"].get(key)
            for key in ("route", "model", "embedding_dim")
        ):
            raise ValueError("Checkpoint encoder configuration differs from model encoder.")
        self.load_state_dict(checkpoint["state_dict"])
        if self.numeric_features:
            self.params["_numeric_scaling"] = copy.deepcopy(
                checkpoint["params"].get("_numeric_scaling")
            )


def read_checkpoint(location: str | Path) -> dict[str, Any]:
    """Load safe tensors/primitive metadata and reject legacy checkpoint formats.

    Example:
        >>> data = read_checkpoint("xirt_model_01.pt")  # doctest: +SKIP
        >>> data["format_version"]  # doctest: +SKIP
        2
    """
    if Path(location).suffix.lower() in {".h5", ".hdf5"}:
        raise ValueError("xiRT 1.x HDF5 models require the legacy environment; retrain for 2.0.")
    checkpoint = torch.load(location, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise ValueError("Not a xiRT 2.0 checkpoint.")
    return checkpoint


class xiRTNET:
    """Build, train and transfer xiRT PyTorch models using legacy YAML sections.

    Example:
        >>> network = xiRTNET(params, input_dim=30, embedding_dim=25)  # doctest: +SKIP
        >>> network.build_model(siamese=True, alphabet=25)  # doctest: +SKIP
        >>> network.compile()  # doctest: +SKIP
    """

    def __init__(self, params: dict[str, Any], input_dim: int, embedding_dim: int) -> None:
        if input_dim < 1 or embedding_dim < 1:
            raise ValueError("Input width and vocabulary size must be positive.")
        self.params = NetworkConfig.model_validate(copy.deepcopy(params)).model_dump()
        self.model: RetentionModel | None = None
        self.input_dim, self.embedding_dim = input_dim, embedding_dim
        self.LSTM_p = self.params["LSTM"]
        self.dense_p = self.params["dense"]
        self.embedding_p = self.params["embedding"]
        self.learning_p = self.params["learning"]
        self.output_p = self.params["output"]
        self.siamese_p = self.params["siamese"]
        self.callback_p = self.params["callbacks"]
        self.tasks = sorted(self.params["predictions"]["fractions"]) + sorted(
            self.params["predictions"]["continues"]
        )

    def build_model(self, siamese: bool = False, alphabet: int | None = None) -> None:
        """Create a new model with shared branches and deterministic initialization."""
        seed_everything(self.learning_p.get("seed", 42), self.learning_p.get("deterministic", True))
        fine_tuning = self.params.get("fine_tuning")
        if fine_tuning:
            if not fine_tuning["checkpoint"]:
                raise ValueError(
                    "fine_tuning.checkpoint is required when building a fine-tuned model."
                )
            self.fine_tune(
                fine_tuning["checkpoint"],
                **{key: value for key, value in fine_tuning.items() if key != "checkpoint"},
            )
            return
        self.model = RetentionModel(
            self.params,
            self.input_dim,
            self.embedding_dim if alphabet is None else alphabet,
            siamese,
        )

    def compile(self) -> None:
        """Initialize optimizers for the current model."""
        if self.model is None:
            raise RuntimeError("Build or load a model before compile.")
        self.model.compile()

    def get_callbacks(self, suffix: str = "") -> dict[str, Any]:
        """Return explicit training options and create the callback output directory."""
        options = dict(self.callback_p, suffix=suffix)
        Path(options["callback_path"]).mkdir(parents=True, exist_ok=True)
        return options

    def print_layers(self) -> None:
        """Print the actual PyTorch module tree."""
        if self.model is None:
            raise RuntimeError("Build or load a model first.")
        print(f"siamese={self.model.siamese}\n{self.model}")

    def get_param_overview(self) -> None:
        """Print total, trainable and frozen parameter counts."""
        if self.model is None:
            raise RuntimeError("Build or load a model first.")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = self.model.count_params()
        print(
            f"Total params: {total:,}\nTrainable params: {trainable:,}\n"
            f"Non-trainable params: {total - trainable:,}"
        )

    def load_weights(self, location: str | Path) -> None:
        """Load weights into an existing architecture."""
        if self.model is None:
            raise RuntimeError("Build or load a model first.")
        self.model.load_weights(location)

    def load_model(self, location: str | Path) -> None:
        """Reconstruct a saved model on the configured device.

        Example:
            >>> network.load_model("retention.pt")  # doctest: +SKIP
            >>> predicted_rt = network.model.predict(features)  # doctest: +SKIP
        """
        self.model = RetentionModel.load(location, device=self.learning_p.get("device", "cpu"))

    def fine_tune(self, location: str | Path, strategy: str, **options: Any) -> None:
        """Load a base checkpoint and configure native full, frozen, LoKr or LoRA training."""
        settings = FineTuningConfig.model_validate(
            {"strategy": strategy, **options, "checkpoint": str(location)}
        )
        self.load_model(location)
        if self.model is None:
            raise RuntimeError("Checkpoint did not construct a model.")
        if self.model.params.get("fine_tuning"):
            raise ValueError("Fine-tuning requires an unadapted base checkpoint.")
        if self.model.numeric_features != self.params["numeric_features"]:
            raise ValueError("Fine-tuning numeric feature schema differs from the checkpoint.")
        self.model.params["learning"].update(self.learning_p)
        if self.params.get("_alphabet") is not None and (
            self.model.params.get("_alphabet") != self.params["_alphabet"]
        ):
            raise ValueError("Fine-tuning checkpoint vocabulary differs from the training data.")
        self.model.params.update(
            {
                key: copy.deepcopy(value)
                for key, value in self.params.items()
                if (key.startswith("_") and key != "_numeric_scaling") or key == "callbacks"
            }
        )
        self.model.params["tracking"] = copy.deepcopy(self.params.get("tracking", {}))
        with Path(location).open("rb") as stream:
            self.model.params["_base_checkpoint_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
        self.model.configure_fine_tuning(settings.model_dump())
        self.params["fine_tuning"] = settings.model_dump()
        self.params["_base_checkpoint_sha256"] = self.model.params["_base_checkpoint_sha256"]

    def adjust_model(self) -> None:
        """Replace changed task outputs while retaining compatible encoder and head weights."""
        old = self.model
        if old is None:
            raise RuntimeError("Build or load a model before adjusting outputs.")
        if old.numeric_features != self.params["numeric_features"]:
            raise ValueError("Cannot change numeric feature semantics when adjusting outputs.")
        if old.params.get("fine_tuning"):
            from torch.nn.utils import parametrize

            old = copy.deepcopy(old)
            if old.route == "recurrent":
                for name in tuple(old.recurrent._flat_weights_names):
                    if parametrize.is_parametrized(old.recurrent, name):
                        parametrize.remove_parametrizations(
                            old.recurrent, name, leave_parametrized=True
                        )
        params = copy.deepcopy(self.params)
        fine_tuning = params.pop("fine_tuning", None)
        updated = RetentionModel(params, old.input_dim, old.alphabet, old.siamese)
        weights = old.state_dict()
        compatible = {
            k: v
            for k, v in weights.items()
            if k in updated.state_dict() and v.shape == updated.state_dict()[k].shape
        }
        updated.load_state_dict(compatible, strict=False)
        if fine_tuning:
            updated.configure_fine_tuning(fine_tuning)
        self.model = updated


def params_to_df(yaml_file: str | Path, out_file: str | Path) -> pd.DataFrame:
    """Flatten YAML parameters into a CSV table and return the table.

    Example:
        >>> table = params_to_df("params.yaml", "params.csv")  # doctest: +SKIP
    """
    with open(yaml_file) as stream:
        table = pd.json_normalize(yaml.safe_load(stream)).transpose()
    table.to_csv(out_file)
    return table


def reshapey(values: Any) -> np.ndarray:
    """Stack per-row vector labels, preserving an empty input as an empty matrix.

    Example:
        >>> reshapey([np.array([1, 0]), np.array([0, 1])]).shape
        (2, 2)
    """
    return np.stack(values) if len(values) else np.empty((0, 0))
