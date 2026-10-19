"""Real local MLflow tracking lifecycle and failure tests."""

import numpy as np
import pytest
import torch

from xirt.tracking import TrainingRun
from xirt.xirtnet import xiRTNET


def test_disabled_tracking():
    with TrainingRun({"enabled": False}, {}, ()) as run:
        run.log_metrics({"loss": 1.0}, 0)
    assert run.run_id is None


def test_local_training_tracking(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    import yaml

    with open("tests/fixtures/xirt_params_rp.yaml") as stream:
        params = yaml.safe_load(stream)
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    params["tracking"] = {
        "enabled": True,
        "uri": uri,
        "experiment": "test",
        "artifact_dir": str(tmp_path / "artifacts"),
    }
    net = xiRTNET(params, 6, 25)
    net.build_model(siamese=True, alphabet=25)
    net.compile()
    x = np.ones((4, 6), dtype=int)
    net.model.fit(
        (x, x),
        [np.ones(4)],
        validation_data=((x, x), [np.ones(4)]),
        batch_size=3,
        epochs=2,
        verbose=0,
    )
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("test")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1 and runs[0].info.status == "FINISHED"
    assert "val_loss" in runs[0].data.metrics
    assert runs[0].data.params["learning.optimizer"] == "adam"
    assert len(runs[0].data.tags["training.features.sha256"]) == 64
    assert len(runs[0].data.tags["training.targets.sha256"]) == 64
    assert len(client.get_metric_history(runs[0].info.run_id, "loss")) == 2
    files = {a.path for a in client.list_artifacts(runs[0].info.run_id)}
    assert {"model.pt", "configuration.json"} <= files
    assert mlflow.active_run() is None


def test_failed_run_and_remote_rejection(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    options = {
        "enabled": True,
        "uri": f"sqlite:///{tmp_path / 'runs.db'}",
        "artifact_dir": str(tmp_path / "artifacts"),
    }
    with pytest.raises(RuntimeError, match="training failure"):
        with TrainingRun(options, {}, (torch.ones(2, 3),)) as run:
            raise RuntimeError("training failure")
    client = mlflow.MlflowClient(tracking_uri=options["uri"])
    assert client.get_run(run.run_id).info.status == "FAILED"
    with pytest.raises(ValueError, match="local"):
        with TrainingRun({"enabled": True, "uri": "https://example.com"}, {}, ()):
            pass
