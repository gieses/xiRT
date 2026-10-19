"""Optional, local-first MLflow tracking without global active-run side effects."""

import hashlib
import json
import platform
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

from xirt import __version__


def _tensor_digest(values: tuple[torch.Tensor, ...]) -> str:
    """Hash ordered tensor shapes, dtypes and bytes without storing training data."""
    digest = hashlib.sha256()
    for tensor in values:
        array = tensor.cpu().numpy()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class TrainingRun:
    """Record configuration, data fingerprints, epoch metrics and model artifacts.

    Parameters:
        options: Tracking configuration with ``enabled``, ``uri``, ``experiment``
            and optional ``run_name``. Only local stores/localhost are accepted.
        params: Architecture and training configuration.
        inputs: Training features, used only for SHA256 fingerprints (not uploaded).
        targets: Optional training labels, fingerprinted without uploading them.

    Example:
        >>> with TrainingRun({"enabled": False}, {}, ()) as run:
        ...     run.log_metrics({"loss": 1.0}, step=0)
    """

    def __init__(
        self,
        options: dict[str, Any],
        params: dict[str, Any],
        inputs: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, ...] = (),
    ) -> None:
        self.client: Any = None
        self.run_id: str | None = None
        self.options, self.params, self.inputs, self.targets = options, params, inputs, targets

    def __enter__(self) -> "TrainingRun":
        """Create an independent run when tracking is enabled."""
        if not self.options.get("enabled", False):
            return self
        try:
            from mlflow import MlflowClient
        except ImportError as exc:
            raise ImportError(
                "Tracking requires the tracking extra: uv sync --extra tracking."
            ) from exc
        uri = self.options.get("uri", "sqlite:///mlflow.db")
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file", "sqlite", "http", "https"} or (
            parsed.scheme in {"http", "https"}
            and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("MLflow tracking must use a local store or localhost server.")
        self.client = MlflowClient(tracking_uri=uri)
        experiment_name = self.options.get("experiment", "xiRT")
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            artifacts = Path(self.options.get("artifact_dir", "mlflow-artifacts")).resolve()
            artifacts.mkdir(parents=True, exist_ok=True)
            experiment_id = self.client.create_experiment(
                experiment_name, artifact_location=artifacts.as_uri()
            )
        else:
            experiment_id = experiment.experiment_id
            artifact_uri = urlparse(experiment.artifact_location)
            if artifact_uri.scheme not in {"", "file"}:
                raise ValueError("Use an MLflow experiment with local artifact storage.")
        run = self.client.create_run(
            experiment_id,
            tags={
                "mlflow.runName": self.options.get("run_name", "xiRT-training"),
                "xirt.version": __version__,
                "python.version": platform.python_version(),
                "torch.version": str(torch.__version__),
            },
        )
        self.run_id = run.info.run_id
        try:
            config = json.dumps(self.params, sort_keys=True)
            self.client.log_dict(self.run_id, self.params, "configuration.json")
            self.client.set_tag(
                self.run_id, "config.sha256", hashlib.sha256(config.encode()).hexdigest()
            )
            for section in ("learning", "encoder"):
                for key, value in self.params.get(section, {}).items():
                    self.client.log_param(self.run_id, f"{section}.{key}", value)
            self.client.set_tag(
                self.run_id, "training.features.sha256", _tensor_digest(self.inputs)
            )
            self.client.set_tag(
                self.run_id, "training.targets.sha256", _tensor_digest(self.targets)
            )
        except Exception:
            self.client.set_terminated(self.run_id, status="FAILED")
            raise
        return self

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics at their epoch index, without changing global MLflow state."""
        if self.client is not None:
            for key, value in metrics.items():
                self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, path: str | Path) -> None:
        """Log a safe xiRT checkpoint or reproducibility artifact."""
        if self.client is not None:
            self.client.log_artifact(self.run_id, str(path))

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Mark failed training runs as failed; never swallow training/logging errors."""
        if self.client is not None:
            self.client.set_terminated(
                self.run_id, status="FINISHED" if exc_type is None else "FAILED"
            )
