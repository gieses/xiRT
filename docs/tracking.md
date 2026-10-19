# Local MLflow experiment tracking

Install the optional dependency:

```bash
uv sync --locked --extra cpu --extra tracking
```

xiRT uses MLflow 3's explicit `MlflowClient` API, not a global active run or
automatic monkey-patching. Runs record configuration, runtime versions,
training-feature/target/configuration fingerprints, epoch metrics, and model
checkpoints. Training and logging errors mark a started run as `FAILED`
instead of silently reporting success.

## Configuration

Add this **top-level** section to the network YAML, or the same dictionary to
the API's `params`:

```yaml
tracking:
  enabled: true
  uri: sqlite:///xirt-mlflow.db
  experiment: xiRT
  run_name: recurrent-seed42
  artifact_dir: mlflow-artifacts
```

Relative SQLite and artifact paths resolve from the working directory.
The CLI can initialize output-local tracking settings: database
`<out_dir>/mlflow.db`, artifacts `<out_dir>/mlflow-artifacts`, and fold-specific
run names. Inspect the effective configuration for the exact settings.
Set `tracking.enabled: false` explicitly to disable tracking; disabled
tracking does not import MLflow.

Only local file/SQLite stores or localhost tracking endpoints are supported.
An existing experiment must also have local artifact storage. xiRT does not
accept a remote cloud tracking URI as an accidental destination for peptide
data or model artifacts.

## View runs

From the same working directory used with the example configuration, start the
local UI/server:

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///xirt-mlflow.db
```

Open `http://127.0.0.1:5000` in your browser. For CLI output-local tracking,
substitute `sqlite:///out_dir/mlflow.db`. Keep the server bound to localhost;
do not expose an unauthenticated tracker on a shared network.

Track configuration, target preprocessing, split IDs, seed, device, and software
revision alongside the reported metrics. Feature and target fingerprints aid auditing
but do not reconstruct your dataset, labels, or scientific split.

## Privacy and comparison boundaries

The tracker fingerprints feature and target arrays; it does not upload those arrays,
training tables, or labels. Configuration and model checkpoints are still scientific artifacts:
keep local tracking directories out of public commits and CI artifact uploads,
and review paths/configuration for sensitive information.

Separate per-fold runs prevent crossvalidation folds from overwriting one
another's metrics. Local tracking and GPU/ESM integration do not establish
scientific superiority; interpret results using the
[benchmark protocol](benchmarks.md). An MLflow checkpoint artifact is a xiRT
checkpoint, not a promise of a registered MLflow pyfunc deployment model.
