"""Read-only analysis of scheduled xiRT benchmarks: uv run --no-sync marimo edit this_file."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")

with app.setup:
    import hashlib
    import json
    from io import BytesIO
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def read_json(path: Path) -> dict:
        with path.open() as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return value

    def digest(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def figure_html(figure):
        """Explicit images survive both nested layouts and headless HTML export."""
        with BytesIO() as buffer:
            figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
            image = mo.image(buffer.getvalue())
        plt.close(figure)
        return image

    def load_run(directory: Path) -> dict:
        """Read published JSON only; missing jobs remain missing, never zero-valued results."""
        provenance = read_json(directory / "provenance.json")
        config = provenance["config"]
        if config["schema_version"] != 1:
            raise ValueError("Unsupported scheduled benchmark schema")
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if config_hash != provenance["protocol_sha256"]:
            raise ValueError("Resolved protocol configuration changed")
        manifests = {
            dataset: read_json(path)
            for dataset in config["datasets"]
            if (path := directory / "data" / dataset / "manifest.json").exists()
        }
        manifest_hashes = {
            dataset: digest(directory / "data" / dataset / "manifest.json") for dataset in manifests
        }
        selections = {
            (dataset, variant): read_json(path)
            for dataset in config["datasets"]
            for variant in config["variants"]
            if (path := directory / "selection" / dataset / f"{variant}.json").exists()
        }
        if any(selection["test_used"] for selection in selections.values()):
            raise ValueError("Candidate selection used test data")
        pretraining = {}
        for path in sorted(directory.glob("pretrain/*/*/result.json")):
            record = read_json(path)
            if (
                record["status"] != "ok"
                or record["protocol_sha256"] != provenance["protocol_sha256"]
                or record["source_snapshot_sha256"] != provenance["source_snapshot_sha256"]
            ):
                raise ValueError(f"Invalid train-only warm-start record: {path}")
            pretraining[record["dataset"], record["seed"]] = record
        records, final_rows, tuning_rows = {}, [], []
        for stage, rows in (("final", final_rows), ("tuning", tuning_rows)):
            for path in sorted(directory.glob(f"{stage}/**/result.json")):
                record = read_json(path)
                if record["status"] != "ok":
                    raise ValueError(f"Non-success result in published result.json: {path}")
                if (
                    record["protocol_sha256"] != provenance["protocol_sha256"]
                    or record["source_snapshot_sha256"] != provenance["source_snapshot_sha256"]
                    or record["scope"] != provenance["scope"]
                ):
                    raise ValueError(f"Mixed protocol/source/scope: {path}")
                dataset, variant, seed = record["dataset"], record["variant"], record["seed"]
                if dataset not in config["datasets"] or variant not in config["variants"]:
                    raise ValueError(f"Unexpected cohort or variant: {path}")
                if seed not in config["seeds" if stage == "final" else "tuning_seeds"]:
                    raise ValueError(f"Unexpected seed: {path}")
                if record["test_evaluated"] != (stage == "final"):
                    raise ValueError(f"Test access differs from the worker stage: {path}")
                manifest_path = directory / "data" / dataset / "manifest.json"
                if manifest_hashes.get(dataset) != record["data_manifest_sha256"]:
                    raise ValueError(f"Dataset manifest changed: {manifest_path}")
                candidate, model = record["candidate"], record["model"]
                if candidate not in config["candidates"]:
                    raise ValueError(f"Unspecified tuning candidate: {path}")
                expected = directory / stage / dataset / variant
                if stage == "tuning":
                    expected /= candidate["id"]
                if path != expected / f"s{seed}" / "result.json":
                    raise ValueError(f"Result identity differs from its path: {path}")
                if stage == "final":
                    selection_path = directory / "selection" / dataset / f"{variant}.json"
                    if (
                        digest(selection_path) != record["selection_sha256"]
                        or candidate["id"] != selections[dataset, variant]["candidate_id"]
                    ):
                        raise ValueError(f"Validation-selected candidate changed: {path}")
                relative = str(path.relative_to(directory))
                records[relative] = record
                warm_sha = None if variant == "legacy_tf" else model["warmstart_sha256"]
                warm = pretraining.get((dataset, seed)) if warm_sha else None
                if warm_sha and warm is None:
                    raise ValueError(f"Missing train-only warm-start record: {path}")
                if warm and warm["checkpoint_sha256"] != warm_sha:
                    raise ValueError(f"Warm-start checkpoint identity differs: {path}")
                row = {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "candidate": candidate["id"],
                    "device": model["device"],
                    "learning_rate": candidate["learning_rate"],
                    "dropout": candidate["dropout"],
                    "weight_decay": candidate["weight_decay"],
                    "scheduler": candidate["scheduler"],
                    "validation_loss": record["validation_loss"],
                    "trainable_parameters": model["trainable_parameters"],
                    "total_parameters": model["total_parameters"],
                    "training_s": record["training_seconds"],
                    "setup_s": record["setup_seconds"],
                    "warmstart_training_s": warm["training_seconds"] if warm else 0.0,
                    "job_s": record["job_seconds"],
                    "best_epoch": record["best_epoch"],
                    "epochs_budget": record["epochs_budget"],
                    "checkpoint_mib": record["checkpoint_bytes"] / 1024**2,
                    "record": relative,
                }
                if stage == "final":
                    row.update(record["test_quality"])
                    row.update(
                        inference_ms=1000 * record["inference_seconds"],
                        initial_adapter_gradient_l2=record["initial_adapter_gradient_l2"],
                        **record["diagnostics"],
                    )
                rows.append(row)
        finals, tuning = pd.DataFrame(final_rows), pd.DataFrame(tuning_rows)
        coverage = []
        for dataset in config["datasets"]:
            for variant in config["variants"]:
                manifest = manifests.get(dataset)
                coverage.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "pairs": manifest["pair_count"] if manifest else None,
                        **{
                            f"{split}_rows": len(manifest["splits"][split]) if manifest else None
                            for split in ("train", "validation", "test")
                        },
                        "tuning_done": int(
                            ((tuning["dataset"] == dataset) & (tuning["variant"] == variant)).sum()
                        )
                        if not tuning.empty
                        else 0,
                        "tuning_expected": len(config["candidates"]) * len(config["tuning_seeds"]),
                        "final_done": int(
                            ((finals["dataset"] == dataset) & (finals["variant"] == variant)).sum()
                        )
                        if not finals.empty
                        else 0,
                        "final_expected": len(config["seeds"]),
                        "selected_candidate": selections.get((dataset, variant), {}).get(
                            "candidate_id"
                        ),
                    }
                )
        verification_path = directory / "verification.json"
        verification = read_json(verification_path) if verification_path.exists() else None
        verification_state = "Not published; results are partial or unverified"
        if verification is not None:
            reported = {item["path"]: item["sha256"] for item in verification["records"]}
            verified = (
                (
                    verification["status"] == "ok"
                    and set(reported) == set(finals["record"])
                    and verification["final_runs_verified"] == len(finals)
                    and len(finals)
                    == len(config["datasets"]) * len(config["variants"]) * len(config["seeds"])
                    and all(
                        digest(directory / name) == expected for name, expected in reported.items()
                    )
                )
                if not finals.empty
                else False
            )
            verification_state = (
                "Published verification; final record hashes still match"
                if verified
                else "STALE or incomplete verification; do not treat results as verified"
            )
        failures = []
        for path in sorted(directory.glob("failures/*.json")):
            failure = read_json(path)
            failures.append(
                {
                    "file": path.name,
                    "error_type": failure["error_type"],
                    "error": failure["error"],
                }
            )
        return {
            "directory": directory,
            "provenance": provenance,
            "records": records,
            "finals": finals,
            "tuning": tuning,
            "coverage": pd.DataFrame(coverage),
            "manifests": manifests,
            "selections": selections,
            "failures": pd.DataFrame(failures),
            "verification_state": verification_state,
        }

    def predictions(run: dict, relative: str, split: str) -> pd.DataFrame:
        """Check row identity, targets and fingerprints before showing prediction diagnostics."""
        if split not in {"validation", "test"}:
            raise ValueError("Predictions must be validation or test")
        record = run["records"][relative]
        directory = run["directory"]
        path = (directory / relative).parent / f"{split}.csv"
        if digest(path) != record[f"{split}_predictions_sha256"]:
            raise ValueError(f"Prediction fingerprint changed: {path}")
        manifest = run["manifests"][record["dataset"]]
        frame = pd.read_csv(path)
        if frame["pair_index"].tolist() != manifest["splits"][split]:
            raise ValueError(f"Prediction row identities changed: {path}")
        pairs_path = directory / "data" / record["dataset"] / "pairs.csv"
        if digest(pairs_path) != record["pairs_sha256"]:
            raise ValueError(f"Dataset rows changed: {pairs_path}")
        truth = pd.read_csv(pairs_path, usecols=["rp_seconds"]).iloc[frame["pair_index"]][
            "rp_seconds"
        ]
        if not np.allclose(frame["truth_seconds"], truth, rtol=1e-7, atol=1e-3):
            raise ValueError(f"Prediction targets changed: {path}")
        frame["residual_seconds"] = frame["prediction_seconds"] - frame["truth_seconds"]
        return frame


@app.cell(hide_code=True)
def _():
    mo.md("""
    # xiRT scheduled benchmark analysis

    Read-only exploration of **one frozen run at a time**. No training, selection,
    downloads, checkpoint loading or report rewriting happens here.
    **Smoke subsets demonstrate execution, not convergence or predictive superiority.**
    Full cohorts are correlated views of two studies; seed dispersion is not biological
    replication. Do not tune configurations after inspecting these test scores without fresh holdouts.
    """)
    return


@app.cell
def _():
    results_root = Path(mo.notebook_dir()).parents[3] / "benchmarks/results"
    return (results_root,)


@app.cell
def _():
    refresh = mo.ui.run_button(label="Refresh artifacts")
    refresh
    return (refresh,)


@app.cell
def _(refresh, results_root):
    _refresh_requested = refresh.value
    run_paths = sorted(
        (path.parent for path in results_root.glob("scheduled-*/provenance.json")),
        key=lambda path: ((path / "verification.json").exists(), path.stat().st_mtime),
        reverse=True,
    )
    mo.stop(
        not run_paths,
        mo.md(
            "**No scheduled runs found.** Run `./run.sh smoke` or `./run.sh full` first; "
            "a dry-run does not produce measured results."
        ),
    )
    run_picker = mo.ui.dropdown(
        {path.name: path for path in run_paths},
        value=run_paths[0].name,
        label="Frozen run",
    )
    run_picker
    return (run_picker,)


@app.cell
def _(refresh, run_picker):
    _refresh_requested = refresh.value
    run = load_run(run_picker.value)
    run_config = run["provenance"]["config"]
    mo.md(f"""
    **Scope: `{run["provenance"]["scope"]}`** · **{run["verification_state"]}**

    Requested: {len(run_config["datasets"])} cohorts, {len(run_config["variants"])} variants,
    {len(run_config["seeds"])} final seeds and {len(run_config["candidates"])} tuning candidates.
    Protocol: `{run["provenance"]["protocol_sha256"]}`.
    Source snapshot: `{run["provenance"]["source_snapshot_sha256"]}`.

    This badge checks final JSON hashes against the recorded verification. It does not rerun
    every pipeline integrity check; prediction fingerprints/targets are checked when inspected below.
    """)
    return (run,)


@app.cell(hide_code=True)
def _(run):
    mo.vstack(
        [
            mo.md(
                "## Completion and data coverage\nMissing jobs are not zero-valued measurements."
            ),
            mo.ui.table(run["coverage"], selection=None),
            mo.md(
                "### Failure events\nThese artifacts may describe earlier attempts that later succeeded."
            ),
            mo.ui.table(run["failures"], selection=None)
            if not run["failures"].empty
            else mo.md("No recorded failure events."),
        ]
    )
    return


@app.cell
def _(run):
    cohort_picker = mo.ui.dropdown(
        run["provenance"]["config"]["datasets"],
        value=run["provenance"]["config"]["datasets"][0],
        label="Cohort",
    )
    cohort_picker
    return (cohort_picker,)


@app.cell
def _(cohort_picker, run):
    cohort = cohort_picker.value
    final_frame = run["finals"]
    cohort_finals = (
        final_frame[final_frame["dataset"] == cohort] if not final_frame.empty else final_frame
    )
    tuning_frame = run["tuning"]
    cohort_tuning = (
        tuning_frame[tuning_frame["dataset"] == cohort] if not tuning_frame.empty else tuning_frame
    )
    return cohort, cohort_finals, cohort_tuning


@app.cell(hide_code=True)
def _(cohort_finals):
    mo.stop(cohort_finals.empty, mo.md("**No final test evaluations yet for this cohort.**"))
    comparison = (
        cohort_finals.groupby(["variant", "device"], sort=False)
        .agg(
            seeds=("seed", "nunique"),
            rmse_s=("rmse_seconds", "mean"),
            rmse_seed_sd_s=("rmse_seconds", "std"),
            mae_s=("mae_seconds", "mean"),
            r2=("r2", "mean"),
            trainable_parameters=("trainable_parameters", "mean"),
            training_s=("training_s", "mean"),
            warmstart_training_s=("warmstart_training_s", "mean"),
            reconstruction_setup_s=("setup_s", "mean"),
            inference_ms=("inference_ms", "mean"),
            checkpoint_mib=("checkpoint_mib", "mean"),
        )
        .reset_index()
    )
    comparison["training_plus_warmstart_s"] = (
        comparison["training_s"] + comparison["warmstart_training_s"]
    )
    mo.vstack(
        [
            mo.md(
                "## Held-out quality and model cost\n"
                "Means cover **completed seeds only**; check coverage before comparisons. "
                "Seed SD is unavailable for one seed—not zero uncertainty. "
                "Reconstruction setup is the final job's setup, not the selected tuning job's setup."
            ),
            mo.ui.table(comparison, selection=None),
            mo.accordion(
                {
                    "Per-seed metrics and adapter diagnostics": mo.ui.table(
                        cohort_finals, selection=None
                    )
                }
            ),
        ]
    )
    return (comparison,)


@app.cell(hide_code=True)
def _(comparison):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    _x = np.arange(len(comparison))
    _axes[0].bar(_x, comparison["rmse_s"])
    _has_sd = comparison["rmse_seed_sd_s"].notna()
    _axes[0].errorbar(
        _x[_has_sd],
        comparison.loc[_has_sd, "rmse_s"],
        yerr=comparison.loc[_has_sd, "rmse_seed_sd_s"],
        fmt="none",
        color="black",
        capsize=3,
    )
    _axes[0].set_xticks(_x, comparison["variant"], rotation=65, ha="right")
    _axes[0].set_ylabel("Test RT RMSE (seconds); error bars = seed SD")
    for _device, _group in comparison.groupby("device"):
        _axes[1].scatter(_group["training_plus_warmstart_s"], _group["rmse_s"], label=_device)
        for _row in _group.itertuples():
            _axes[1].annotate(
                _row.variant, (_row.training_plus_warmstart_s, _row.rmse_s), fontsize=8
            )
    _axes[1].set_xlabel("Selected training + warm-start training (seconds)")
    _axes[1].set_ylabel("Test RT RMSE (seconds)")
    _axes[1].legend(title="Device")
    _display = mo.vstack(
        [
            mo.md(
                "Timing is shared-host and CPU/GPU-dependent; this is **not an end-to-end speedup claim**. "
                "ESM protein-pretraining cost is not included. Warm-start training is shared between "
                "matched recurrent variants, not independently repeated for each."
            ),
            figure_html(_fig),
        ]
    )
    _display
    return


@app.cell(hide_code=True)
def _(cohort, cohort_tuning, run):
    mo.stop(cohort_tuning.empty, mo.md("**No completed tuning jobs yet for this cohort.**"))
    candidate_summary = (
        cohort_tuning.groupby(
            ["variant", "candidate", "learning_rate", "dropout", "weight_decay", "scheduler"],
            sort=False,
        )
        .agg(
            tuning_seeds_done=("seed", "nunique"),
            mean_validation_loss=("validation_loss", "mean"),
            validation_seed_sd=("validation_loss", "std"),
            search_job_s=("job_s", "sum"),
        )
        .reset_index()
    )
    candidate_summary["tuning_seeds_expected"] = len(run["provenance"]["config"]["tuning_seeds"])
    candidate_summary["selected"] = [
        run["selections"].get((cohort, row.variant), {}).get("candidate_id") == row.candidate
        for row in candidate_summary.itertuples()
    ]
    mo.vstack(
        [
            mo.md(
                "## Validation-only hyperparameter search\n"
                "Selection is made by the pipeline, never by this notebook's test plots. "
                "Validation loss is the normalized task objective; do not compare it across "
                "cohorts/task sets. Partial means are explicitly labeled by seed count. "
                "`search_job_s` includes worker setup, training and artifact work, not scheduler overhead."
            ),
            mo.ui.table(candidate_summary, selection=None),
        ]
    )
    return


@app.cell
def _(cohort_tuning):
    mo.stop(cohort_tuning.empty)
    trial_options = {
        f"{row.variant} / {row.candidate} / seed {row.seed}": row.record
        for row in cohort_tuning.itertuples()
    }
    trial_picker = mo.ui.dropdown(
        trial_options, value=next(iter(trial_options)), label="Training curve"
    )
    trial_picker
    return (trial_picker,)


@app.cell(hide_code=True)
def _(run, trial_picker):
    trial = run["records"][trial_picker.value]
    _history = pd.DataFrame(trial["history"])
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    _axes[0].plot(_history["epoch"], _history["train_loss"], label="Training")
    _axes[0].plot(_history["epoch"], _history["validation_loss"], label="Validation")
    _axes[0].axvline(trial["best_epoch"], linestyle="--", color="grey", label="Selected checkpoint")
    _axes[0].set(xlabel="Epoch", ylabel="Normalized task objective")
    _axes[0].legend()
    _axes[1].plot(_history["epoch"], _history["learning_rate_ratio"])
    _axes[1].set(xlabel="Epoch", ylabel="Learning-rate multiplier")
    _display = mo.vstack(
        [
            mo.md(
                f"## Training curve\nBest epoch **{trial['best_epoch']}**; "
                f"configured budget **{trial['epochs_budget']}** epochs."
            ),
            figure_html(_fig),
        ]
    )
    _display
    return


@app.cell
def _(cohort_finals):
    mo.stop(cohort_finals.empty)
    prediction_options = {
        f"{row.variant} / seed {row.seed}": row.record for row in cohort_finals.itertuples()
    }
    prediction_picker = mo.ui.dropdown(
        prediction_options,
        value=next(iter(prediction_options)),
        label="Prediction diagnostics",
    )
    split_picker = mo.ui.dropdown(["validation", "test"], value="test", label="Partition")
    mo.hstack([prediction_picker, split_picker])
    return prediction_picker, split_picker


@app.cell(hide_code=True)
def _(prediction_picker, run, split_picker):
    prediction_frame = predictions(run, prediction_picker.value, split_picker.value)
    _record = run["records"][prediction_picker.value]
    _residual = prediction_frame["residual_seconds"]
    _truth = prediction_frame["truth_seconds"]
    _train_mean = run["manifests"][_record["dataset"]]["target_normalization"]["mean"]
    prediction_metrics = {
        "partition": split_picker.value,
        "rows": len(prediction_frame),
        "rmse_seconds": float(np.sqrt(np.mean(_residual**2))),
        "mae_seconds": float(np.mean(np.abs(_residual))),
        "train_mean_baseline_rmse_seconds": float(np.sqrt(np.mean((_truth - _train_mean) ** 2))),
    }
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    _axes[0].scatter(_truth, prediction_frame["prediction_seconds"], alpha=0.5)
    _bounds = [
        min(_truth.min(), prediction_frame["prediction_seconds"].min()),
        max(_truth.max(), prediction_frame["prediction_seconds"].max()),
    ]
    _axes[0].plot(_bounds, _bounds, linestyle="--", color="grey")
    _axes[0].set(xlabel="Observed RT (seconds)", ylabel="Predicted RT (seconds)")
    _axes[1].scatter(_truth, _residual, alpha=0.5)
    _axes[1].axhline(0, linestyle="--", color="grey")
    _axes[1].set(xlabel="Observed RT (seconds)", ylabel="Predicted minus observed (seconds)")
    _display = mo.vstack(
        [
            mo.md(
                "## Prediction diagnostics\nMetrics below are recomputed from checked CSVs. "
                "The constant baseline uses the **training-only** mean."
            ),
            mo.ui.table([prediction_metrics], selection=None),
            figure_html(_fig),
            mo.accordion(
                {
                    "Row-level predictions (including available fraction probabilities)": mo.ui.table(
                        prediction_frame, selection=None
                    )
                }
            ),
        ]
    )
    _display
    return prediction_frame, prediction_metrics


if __name__ == "__main__":
    app.run()
