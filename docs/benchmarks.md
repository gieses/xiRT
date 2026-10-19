# Benchmarks

The application backend is PyTorch. TensorFlow is retained only as archived
scientific evidence, not an installable dependency or an active runner.

| Comparison | Evidence | Results |
|---|---|---|
| TF versus native PyTorch | Five modification-aware cohorts, three seeds; regenerated train/validation/test predictions | [Matched comparison](benchmarks/tf-torch.md) |
| Native recurrent fine-tuning | Full, frozen, LoKr and LoRA; common train-only starting checkpoints | [Fine-tuning comparison](benchmarks/fine-tune-torch.md) |

## Archive layout

```text
benchmark-tf-torch/
  <cohort>/results/
    archive.json
    data/
    models/
    provenance/
    tables/
    plots/
  fine-tune-torch/
    protocol.json
    environment.json
    source.tar.gz
    runs/<cohort>/<strategy>/s<seed>/
    results/
  legacy-reference/
```

Each cohort archive contains aligned predictions, metrics in seconds, the
partition manifest and observations, original model records, native PyTorch
checkpoint exports, plots, and frozen source/configuration provenance.
SHA256 indexes detect changed or missing artifacts.

The original nine-variant experiment completed 1,080 validation-led tuning jobs
and 135 final evaluations. Its raw records are local historical output at
`benchmarks/results/scheduled-v2/`, excluded from Git and distributions.
The selected comparison is preserved in the indexed cohort archives above.
Older pilot reports are superseded by that run.
Retired runners, TensorFlow dependency locks and Sphinx sources are preserved
in `benchmark-tf-torch/legacy-reference/`, with file hashes.

## Commands

Run from a source checkout, not the production wheel:

```bash
bash run.sh verify
bash run.sh plan
# A new experiment needs its own output when code or dependencies differ.
bash run.sh fine-tune --output benchmarks/results/fine-tune-local
```

`verify` recomputes prediction metrics without training. `plan` validates inputs
and prints the number of requested jobs. `fine-tune` resumes only when code,
configuration, environment and input identities match. Published output is
immutable; changing the Python environment requires a new output directory.

`make notebook` opens a read-only marimo dashboard when the historical scheduled
output is available locally. It does not execute the retired TensorFlow
workflow. The published results pages and indexed archives do not require
that local output.

## Interpretation

The five cohorts are correlated views of two published studies, not five
independent replications. Three initialization seeds share one grouped
partition. Seed standard deviation is not a confidence interval. Grouped
peptide-pair splits reduce duplicate/PTM-state leakage but do not establish
cross-study or new-acquisition performance.

The source-table census and retrieval tools are under
`xirt/data_pipelines/benchmarks/datasets/`. Manifests retain source identities,
observed RT units, exclusion rules and training-only normalization. ESM results
use a PTM-blind residue projection and are not chemically equivalent inputs.
No ESM model is needed for the recurrent fine-tuning comparison.
