# Development and release

## Environment

Python 3.13 is the default; CPU CI covers 3.12, 3.13 and 3.14.
`pyproject.toml` owns package metadata and dependency ranges. `uv.lock` owns the
resolved Python environment. Development and documentation tools are PEP 735
dependency groups rather than published package extras.

```bash
uv sync --locked --extra cpu --group dev --group docs --extra tracking
make check
make build
```

Use `--extra cuda` instead of `--extra cpu` on a compatible NVIDIA host.
ESM-C uses a separate Python 3.12 environment with `--extra esm`; do not
change an environment while an experiment is using it.

Pixi is an optional bootstrap for Python and uv on Linux x86-64 and Apple
Silicon. Its `pixi.lock` pins those tools; it deliberately delegates scientific
Python dependencies to `uv.lock` instead of maintaining a second resolution.

```bash
pixi install --locked
pixi run sync
pixi run check
```

Update dependency metadata and locks together. Keep measured environments
separate from development upgrades. Historical environment versions are part
of archived provenance, not constraints on current package development.

## Commands and layout

| Command | Purpose |
|---|---|
| `make sync` | Install the locked CPU development environment |
| `make lint` / `make format` | Ruff, formatting, mypy, notebook and shell checks / formatting |
| `make test` / `make check` | CPU regression tests / lint, tests and strict docs |
| `make docs` / `make serve-docs` | Strict MkDocs build / localhost preview |
| `make install-hooks` / `make pre-commit` | Install / run pinned Git hooks |
| `make build` | Build and validate wheel and sdist |
| `make notebook` | Historical results dashboard on localhost |
| `make examples` | Small CPU training/prediction and fine-tuning notebooks |
| `make benchmark-plan` / `make benchmark-smoke` | List jobs / run four short native jobs |

Production modules live in `xirt/`. Benchmark-only code is under
`xirt/data_pipelines/benchmarks/<use_case>/`: it is included in the source
distribution for reproducibility, but excluded from the production wheel.
Data, results, checkpoint archives and retired code bundles are excluded from
both distributions. TensorFlow and Snakemake are not runtime dependencies.
The user notebooks and their small dataset live in `examples/` and are
included only in the source distribution. Their generated results are ignored.
Local benchmark reruns, model weights, embedding caches and tracking databases
are ignored too. Published evidence under `benchmark-tf-torch/` has explicit
exceptions; do not use that directory for scratch runs.

Ruff covers all maintained Python source. Retained legacy tests have explicit
lint/format exceptions; fatal syntax checks still cover them. Mypy checks the
model, training, adapters, configuration, embeddings and tracking interfaces.
Pre-commit adds TOML/YAML, merge-conflict, whitespace and large-file checks.
Checksum-indexed benchmark artifacts are excluded from automatic rewriting.

## CI and deployment

CPU CI runs tests without ESM downloads, builds distributions, installs the wheel
in an isolated environment, checks CLI/version metadata and builds strict docs.
Only coverage reports, distributions and the static documentation site are
uploaded.

The manual GPU/ESM workflow requires the default branch, an approved self-hosted
runner and protected `gpu-integration` environment. ESM weights must already be
licensed and cached locally. No sequence, model-cache or credential artifacts
are uploaded.

For Pages, configure the repository's **Settings → Pages → Source: GitHub
Actions** and restrict the `github-pages` environment to the default branch.
`pages.yml` builds the static site on relevant default-branch pushes. Only its
deployment job receives Pages write and OIDC permissions.

## Release

Keep `project.version` and `xirt.__version__` equal. Before publication, date the
matching `CHANGELOG.md` section using `YYYY-MM-DD`. The release validator checks
source, tag and distribution versions, and rejects benchmark/archive code or
TensorFlow requirements in the wheel.

After reviewing and committing changes:

```bash
make release-check
make release
```

`make release` runs checks, creates an annotated version tag, pushes it to
`origin` and publishes a GitHub release with the matching changelog notes.
It refuses a dirty tree, an existing local tag or undated release notes.
This is not a local preview and does not create a commit. If this checkout
uses the repository-local GitHub account alias, run `make release GH="git gh"`.

Publishing a GitHub release triggers `release.yml`, which runs CPU checks,
builds and validates the distributions, records GitHub build provenance,
attaches distributions to that release and uploads **`xirt-torch`** to PyPI.
Creating a draft or pushing a tag alone does not publish. A release created
by another workflow using its default `GITHUB_TOKEN` does not trigger this
event; publish through the GitHub UI or an authenticated user such as the
`make release` command above.

Store the production API token as **`PYPI_TOKEN`**, either in repository
Actions secrets or the protected **`pypi`** environment. The token must allow
publishing `xirt-torch`; a token scoped only to the old `xiRT` project will
not work. For the first upload of a new project, use a token permitted to
create it, then replace it with a project-scoped token. A missing token fails
the publishing job explicitly. Production publishing does not require a
PyPI trusted publisher.

TestPyPI remains a separate manual route: configure a trusted publisher on
TestPyPI for owner `gieses`, repository `xiRT`, workflow `release.yml` and
environment `testpypi`. Do not reuse the production PyPI token there.
Manual dispatches must select the matching version tag; branch dispatches
do not publish. A manual production dispatch requires an existing GitHub
release for that tag. Protect both environments and allow only intended
release tags. Build, provenance, asset upload and publication jobs have
separate permissions.
