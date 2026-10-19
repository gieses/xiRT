# xiRT comparison archives

| Directory | Contents |
|---|---|
| `full_ecoli_modified/results/` | E. coli crosslink TF/PyTorch predictions |
| `full_plink_modified/results/` | E. coli pLink crosslink TF/PyTorch predictions |
| `full_fa_cross_modified/results/` | FA crosslink TF/PyTorch predictions |
| `full_fa_linear_modified/results/` | FA linear-peptide TF/PyTorch predictions |
| `full_ecoli_linear_modified/results/` | E. coli linear-peptide TF/PyTorch predictions |
| `fine-tune-torch/` | Native recurrent full/frozen/LoKr/LoRA comparison |
| `legacy-reference/` | Retired code, environment locks and example-model bundles |

Each prediction archive contains observations and partitions, aligned
train/validation/test predictions, per-seed metrics, plots, original model
records, native checkpoint exports and source/configuration provenance.
`archive.json` records SHA256 hashes. Do not edit indexed files.

From the repository root:

```bash
bash run.sh verify
```

Maintained benchmark code is under `xirt/data_pipelines/benchmarks/`.
Scientific interpretation and figures are in `docs/benchmarks/tf-torch.md`
and `docs/benchmarks/fine-tune-torch.md`. The production wheel contains neither
these archives nor benchmark scripts, and does not depend on TensorFlow.
