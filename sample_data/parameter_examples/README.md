# Current configuration examples

Generate complete defaults with:

```bash
uv run --no-sync xirt -p xirt_params.yaml -s learning_params.yaml
```

The maintained small RP network is `docs/examples/xirt-rp.yaml`.
Fine-tuning configuration is documented in `docs/fine-tuning.md`.
Runnable training, prediction and fine-tuning notebooks with a small dataset
are in `examples/`; see `docs/tutorials.md`.
The input peptide tables remain in `sample_data/`; legacy TensorFlow models,
output examples and configurations are archived in
`benchmark-tf-torch/legacy-reference/retired-examples.tar.gz`.
