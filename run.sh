#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode="${1:-plan}"
usage='Usage: ./run.sh [setup|verify|plan|fine-tune] [pipeline options...]
Historical TensorFlow measurements are archived; no TensorFlow environment is installed.'
[[ $# -eq 0 ]] || shift
case "$mode" in
    -h|--help) printf '%s\n' "$usage"; exit 0 ;;
    setup) exec uv sync --locked --extra cpu --group dev --group docs ;;
    verify) exec uv run --no-sync python -m xirt.data_pipelines.benchmarks.verify "$@" ;;
    plan) exec uv run --no-sync python -m xirt.data_pipelines.benchmarks.fine_tune_torch.pipeline --dry-run "$@" ;;
    fine-tune) exec uv run --no-sync python -m xirt.data_pipelines.benchmarks.fine_tune_torch.pipeline "$@" ;;
    *) printf '%s\n' "$usage" >&2; exit 2 ;;
esac
