.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

UV ?= uv
GH ?= gh
RUN = $(UV) run --no-sync
PYTHON ?= 3.13
BACKEND ?= cpu
BENCHMARK_OUTPUT ?= benchmarks/results/fine-tune-local
EXTRAS ?= --group dev --group docs --extra tracking
HOST ?= 127.0.0.1
DOCS_PORT ?= 8000
NOTEBOOK_PORT ?= 2718
REMOTE ?= origin
VERSION = $(shell $(RUN) python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
TAG = v$(VERSION)

.PHONY: help sync build docs serve-docs test lint format check examples pre-commit install-hooks \
	notebook notebook-edit benchmark-plan benchmark-smoke benchmark changelog release-check release publish

help: ## Show available commands (default)
	@printf '\nxiRT - developer commands\n\n'
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\nDefaults: Python $(PYTHON), backend $(BACKEND), localhost-only servers.\n'

sync: ## Create/update the locked uv development environment
	$(UV) sync --locked --python "$(PYTHON)" --extra "$(BACKEND)" $(EXTRAS)

build: ## Build wheel + sdist and validate metadata
	$(UV) build
	$(RUN) twine check --strict dist/*.whl dist/*.tar.gz
	$(RUN) python .github/scripts/validate_release.py

docs: ## Build documentation with strict validation
	$(RUN) mkdocs build --strict
	$(RUN) python .github/scripts/check_docs_links.py

serve-docs: ## Serve live-reloading docs at localhost:8000
	$(RUN) mkdocs serve --dev-addr "$(HOST):$(DOCS_PORT)"

test: ## Run the CPU regression suite (no ESM downloads)
	CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
		$(RUN) pytest -m "not gpu and not esm and not esmc and not integration"

lint: ## Check Python, types, notebook dataflow and Bash syntax
	$(RUN) ruff check .
	$(RUN) ruff check --isolated --select E9,F63,F7,F82 xirt tests
	$(RUN) ruff format --check --quiet .
	$(RUN) mypy
	$(RUN) marimo check --strict xirt/data_pipelines/benchmarks/tf_torch/analyze_results.py examples/train_predict.py examples/fine_tune.py
	bash -n run.sh

format: ## Fix modern-code lint issues and apply formatting
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

check: lint test docs ## Run local checks without publishing or benchmark runs

examples: ## Run the two small CPU training and fine-tuning notebooks
	MPLBACKEND=Agg $(RUN) python examples/train_predict.py
	MPLBACKEND=Agg $(RUN) python examples/fine_tune.py

pre-commit: ## Run configured hooks on all tracked files
	$(RUN) pre-commit run --all-files

install-hooks: ## Install the Git pre-commit hook
	$(RUN) pre-commit install

notebook: ## Open the read-only benchmark dashboard at localhost:2718
	$(RUN) marimo run xirt/data_pipelines/benchmarks/tf_torch/analyze_results.py --host "$(HOST)" --port "$(NOTEBOOK_PORT)" --headless

notebook-edit: ## Open the benchmark notebook editor
	$(RUN) marimo edit xirt/data_pipelines/benchmarks/tf_torch/analyze_results.py --host "$(HOST)" --port "$(NOTEBOOK_PORT)" --headless

benchmark-plan: ## Validate archived inputs and list native fine-tuning jobs
	bash run.sh plan

benchmark-smoke: ## Run four short native fine-tuning jobs in a separate output
	bash run.sh fine-tune --datasets full_fa_cross_modified --seeds 17 --epochs 2 --output benchmarks/.cache/fine-tune-smoke

benchmark: ## Run or resume the archived-input native fine-tuning comparison
	bash run.sh fine-tune --output "$(BENCHMARK_OUTPUT)"

changelog: ## Display curated notes for the current package version
	@$(RUN) python .github/scripts/validate_release.py --notes-only

release-check: ## Require a clean tree, dated changelog and unused version tag
	@test -z "$$(git status --porcelain)" || { printf '%s\n' 'Release blocked: commit all intended changes first.' >&2; exit 1; }
	@RELEASE_TAG="$(TAG)" $(RUN) python .github/scripts/validate_release.py --notes-only >/dev/null
	@! git show-ref --quiet --verify "refs/tags/$(TAG)" || { printf '%s\n' 'Release blocked: local tag $(TAG) already exists.' >&2; exit 1; }
	@git remote get-url "$(REMOTE)" >/dev/null

release: release-check ## Check/build, push a tag and publish a GitHub release to trigger PyPI
	$(MAKE) check build
	RELEASE_TAG="$(TAG)" $(RUN) python .github/scripts/validate_release.py
	git tag --annotate "$(TAG)" --message "xiRT $(VERSION)"
	git push "$(REMOTE)" "refs/tags/$(TAG)"
	$(RUN) python .github/scripts/validate_release.py --notes-only | \
		$(GH) release create "$(TAG)" --verify-tag --title "xiRT $(VERSION)" \
		--repo "$$(git remote get-url "$(REMOTE)")" --notes-file -

publish: release ## Alias for the guarded release command (remote writes)
