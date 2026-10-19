# Changelog

## [2.0.1] - 2026-10-19

### Changed

- Refresh the Material documentation with responsive layouts, light and dark
  palettes, clearer navigation, and an original crosslinked-peptide logo.
- Update the README with installation commands, notebook links, and measured
  benchmark results.
- Use absolute documentation and image URLs in the package description so
  links work on PyPI as well as GitHub.
- Check generated documentation links, anchors, and package-description URLs
  before deployment.

The scientific models, dependencies, and benchmark results are unchanged.

## [2.0.0] - 2026-10-19

### Changed

- Publish the PyTorch implementation as `xirt-torch`, retaining the `xirt`
  Python import and command.
- Replace the TensorFlow application runtime with a PyTorch implementation.
  The default model retains learned peptide embeddings, a shared recurrent
  encoder and retention-time heads; AdamW is the default optimizer.
- Preserve train, prediction and cross-validation workflows with reproducible
  checkpoints, vocabulary/target metadata and prediction preprocessing.
- Manage dependencies with uv, PEP 735 development groups and a committed lock.
  Default to Python 3.13, support 3.12–3.14, and add a locked Pixi bootstrap.
- Remove TensorFlow dependencies and active legacy runners after archiving
  matched predictions, checkpoints, source, environment and metric provenance.
- Separate benchmark scripts from the production wheel; retire duplicate
  Sphinx documentation in favor of MkDocs.
- Preserve fine-tuning strategy and current-run provenance through head
  replacement/refitting; include activity regularization in evaluation loss.
- Avoid constructing optional training optimizers during prediction.
- Accept list and tuple containers for Siamese inputs while retaining
  row-list input for single-peptide models.

### Added

- Optional local ESM-C representations, genuine transformer LoRA/LoKr and
  GRU/linear-weight LoKr adapters; optional Lion/Muon training and focal loss.
- Optional local MLflow tracking with configuration and model provenance.
- Native recurrent full, frozen-encoder, GRU-LoKr and GRU-LoRA fine-tuning,
  with adapter learning-rate groups and reconstructible checkpoints.
- SHA256-indexed TF/PyTorch prediction archives and a native fine-tuning
  comparison of 60 runs using shared train-only starting checkpoints.
- A read-only marimo notebook for completion, quality, tuning, cost,
  training-curve and prediction analysis. Marimo is a default dependency.
- Material documentation, CPU/GPU CI, protected publishing workflows,
  curated release notes, a developer Makefile and uv-backed pre-commit hooks.
- Automatic PyPI publication when a GitHub release is published, using the
  `PYPI_TOKEN` repository or `pypi` environment secret.
- Small CPU notebooks for training, checkpoint-based prediction and all four
  recurrent fine-tuning strategies, with observed data and saved preprocessing.

### Scientific limitations

- Historical full-cohort comparisons improve RT RMSE in two of five correlated
  cohorts, not universally; those views represent two biological studies.
- The completed AdamW comparison contains 1,080 tuning runs and 135 final
  evaluations. Three initialization seeds on one partition do not establish
  independent biological replication or external transfer performance.
