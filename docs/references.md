# References

## Cite xiRT

Giese, S.H., Sinn, L.R., Wegner, F. et al. **Retention time prediction using neural
networks increases identifications in crosslinking mass spectrometry.**
*Nature Communications* **12**, 3237 (2021).
[DOI: 10.1038/s41467-021-23441-0](https://doi.org/10.1038/s41467-021-23441-0).

The publication describes the original scientific work; its existence does not
validate every optional 2.0 model route or optimizer. Cite the exact software
version, benchmark revision, and upstream embedding model when applicable.

## Primary upstream documentation

- [PyTorch installation](https://pytorch.org/get-started/locally/)
  and [reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- [Setuptools pyproject configuration](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html):
  static project metadata, modern build backend, license expressions.
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/creating-your-site/)
  and [Python mkdocstrings](https://mkdocstrings.github.io/python/usage/).
- [Ruff configuration](https://docs.astral.sh/ruff/configuration/).
- [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/),
  [MlflowClient](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html),
  and [local server CLI](https://mlflow.org/docs/latest/api_reference/cli.html).
- [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
  and [security model](https://docs.pypi.org/trusted-publishers/security-model/).
- [GitHub build provenance action](https://github.com/actions/attest-build-provenance).
- [Official GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
- [Predictive R² definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html):
  this is not squared Pearson correlation.
- [ESM upstream implementation and model documentation](https://github.com/evolutionaryscale/esm).
  xiRT currently supports SDK 3.2.x; the upstream main branch may describe newer,
  incompatible APIs and different model terms. Consult the precise downloaded
  model card and versioned SDK sources before using or redistributing weights.

Review upstream documentation and license terms for the exact versions used in
your environment. Package compatibility bounds are not a claim that arbitrary
future SDK/model revisions are interchangeable.
