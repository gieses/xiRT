"""Validate the YAML architecture shared by the CLI and Python API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class FineTuningConfig(BaseModel):
    """Train all recurrent weights, only heads, or GRU adapters plus heads."""

    model_config = ConfigDict(extra="forbid")
    strategy: Literal["recurrent_full", "recurrent_frozen", "recurrent_lokr", "recurrent_lora"]
    checkpoint: str | None = None
    rank: int = Field(default=4, gt=0, strict=True)
    factor: int = Field(default=4, strict=True)
    alpha: FiniteFloat = Field(default=4.0, gt=0)
    adapter_learning_rate_ratio: FiniteFloat = Field(default=0.1, gt=0)

    @model_validator(mode="after")
    def validate_factor(self) -> "FineTuningConfig":
        """Accept positive factor bounds or balanced factorization."""
        if self.factor != -1 and self.factor < 1:
            raise ValueError("factor must be positive or -1")
        return self


class NetworkConfig(BaseModel):
    """Validate legacy-shaped architecture options without silently ignoring errors.

    Parameters are the sections of a xiRT YAML architecture file. Additional
    metadata is retained for checkpoint provenance.

    Example:
        >>> config = NetworkConfig.model_validate(params)  # doctest: +SKIP
        >>> config.model_dump()["learning"]["optimizer"]  # doctest: +SKIP
        'adamw'
    """

    model_config = ConfigDict(extra="allow")
    LSTM: dict[str, Any]
    dense: dict[str, Any]
    embedding: dict[str, Any]
    learning: dict[str, Any]
    output: dict[str, Any]
    siamese: dict[str, Any]
    callbacks: dict[str, Any]
    predictions: dict[str, list[str]]
    encoder: dict[str, Any] = {"route": "recurrent"}
    adapter: dict[str, Any] | None = None
    fine_tuning: FineTuningConfig | None = None
    numeric_features: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_architecture(self) -> "NetworkConfig":
        """Reject inconsistent dimensions, loss/activation pairs and training options."""
        tasks = self.predictions["fractions"] + self.predictions["continues"]
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("Prediction tasks must be nonempty and unique.")
        if len(self.numeric_features) != len(set(self.numeric_features)) or any(
            not name.strip() for name in self.numeric_features
        ):
            raise ValueError("Numeric feature names must be nonempty and unique.")
        if self.numeric_features and self.siamese.get("single_predictions"):
            raise ValueError("Single-peptide attribution is undefined with precursor features.")
        if self.encoder.get("route") not in {"recurrent", "esm-c"}:
            raise ValueError("encoder.route must be recurrent or esm-c.")
        if self.encoder["route"] == "esm-c" and self.encoder.get("embedding_dim", 0) < 1:
            raise ValueError("ESM-C requires a positive encoder.embedding_dim.")
        if self.fine_tuning is not None:
            if self.encoder["route"] != "recurrent":
                raise ValueError("fine_tuning applies to the native recurrent model.")
            if self.adapter is not None:
                raise ValueError("Use fine_tuning or adapter, not both.")
            if (
                self.fine_tuning.strategy in {"recurrent_lokr", "recurrent_lora"}
                and "GRU" not in self.LSTM["type"]
            ):
                raise ValueError("Recurrent LoKr/LoRA require a GRU architecture.")
        if self.LSTM["type"] not in {"GRU", "LSTM", "CuDNNGRU", "CuDNNLSTM"}:
            raise ValueError("Recurrent type must be GRU or LSTM.")
        if self.LSTM["activation"] != "tanh":
            raise ValueError("PyTorch recurrent layers require tanh activation.")
        for key in ("units", "nlayers"):
            if self.LSTM[key] < 1:
                raise ValueError(f"LSTM.{key} must be positive.")
        n = self.dense["nlayers"]
        if n < 0:
            raise ValueError("dense.nlayers must be nonnegative.")
        for key in (
            "neurons",
            "dense_bn",
            "dropout",
            "activation",
            "kernel_regularizer",
            "regularizer_value",
        ):
            if len(self.dense[key]) != n:
                raise ValueError(f"dense.{key} must contain {n} entries.")
        if any(v < 1 for v in self.dense["neurons"]):
            raise ValueError("Dense dimensions must be positive.")
        if any(not 0 <= v < 1 for v in self.dense["dropout"]):
            raise ValueError("Dropout must be in [0, 1).")
        regularizers = {"l1", "l2", "l1l2"}
        if not (
            set(self.dense["kernel_regularizer"]) <= regularizers
            and self.LSTM["kernel_regularization"] in regularizers
            and self.LSTM["activity_regularization"] in regularizers
        ):
            raise ValueError("Regularizers must be l1, l2 or l1l2.")
        penalties = self.dense["regularizer_value"] + [
            self.LSTM["kernelregularizer_value"],
            self.LSTM["activityregularizer_value"],
        ]
        if any(value < 0 for value in penalties):
            raise ValueError("Regularization coefficients must be nonnegative.")
        if self.learning.get("clip_norm", 5.0) <= 0:
            raise ValueError("learning.clip_norm must be positive.")
        if self.siamese["merge_type"] not in {
            "add",
            "multiply",
            "average",
            "concatenate",
            "maximum",
        }:
            raise ValueError("Unsupported siamese merge_type.")
        for task in tasks:
            dim = self.output[f"{task}-dimension"]
            activation = self.output[f"{task}-activation"]
            loss = self.output[f"{task}-loss"]
            allowed = {
                "linear": {"mse", "mae", "huber"},
                "relu": {"mse", "mae", "huber"},
                "sigmoid": {"binary_crossentropy", "binary_focal"},
                "softmax": {"categorical_crossentropy", "categorical_focal", "ordered"},
            }
            if dim < 1 or activation not in allowed or loss not in allowed[activation]:
                raise ValueError(f"Invalid dimension/activation/loss for {task}.")
            if activation == "softmax" and dim < 2:
                raise ValueError("Softmax needs at least two classes.")
            if self.output[f"{task}-weight"] < 0:
                raise ValueError("Loss weights must be nonnegative.")
            if self.output[f"{task}-metrics"] not in {"mse", "mae", "accuracy"}:
                raise ValueError(f"Unsupported metric for {task}.")
        if self.learning["learningrate"] <= 0:
            raise ValueError("learning.learningrate must be positive.")
        if self.learning["batch_size"] < 1 or self.learning["epochs"] < 1:
            raise ValueError("Epochs and batch size must be positive.")
        return self
