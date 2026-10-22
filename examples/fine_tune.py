"""Compare full, frozen, GRU-LoKr and GRU-LoRA fine-tuning on a small CPU example."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
    # Fine-tune a recurrent xiRT model

    This notebook trains a small base checkpoint, then starts each of the four
    strategies from that same checkpoint. Full fine-tuning updates every
    parameter; frozen training updates the heads; LoKr and LoRA update GRU
    weight adapters and heads.

    The base model trains for up to 150 epochs on 1,536 high-confidence pairs;
    each strategy gets up to 75 additional epochs at a lower learning rate.
    Validation-based early stopping restores the best weights at each stage.
    The 256 validation and 256 test pairs stay separate. This is staged
    training on the same cohort, not transfer to a new study.
    """)
    return (mo,)


@app.cell
def data():
    import numpy as np
    import pandas as pd
    import torch
    from tutorial_data import (
        ROOT,
        network_params,
        partition,
        prediction_metrics,
        prepare_data,
        save_metadata,
    )

    from xirt.xirtnet import xiRTNET

    torch.set_num_threads(2)
    table, features, targets, metadata = prepare_data()
    train_mask = (table["split"] == "train").to_numpy()
    validation_mask = (table["split"] == "validation").to_numpy()
    test_mask = (table["split"] == "test").to_numpy()
    return (
        ROOT,
        features,
        metadata,
        network_params,
        np,
        partition,
        pd,
        prediction_metrics,
        save_metadata,
        table,
        targets,
        test_mask,
        train_mask,
        validation_mask,
        xiRTNET,
    )


@app.cell
def training_settings(network_params):
    base_epochs = network_params()["learning"]["epochs"]
    fine_tuning_epochs = 75
    batch_size = network_params()["learning"]["batch_size"]
    return base_epochs, batch_size, fine_tuning_epochs


@app.cell
def output_directory(ROOT):
    output = ROOT / "examples/results/fine-tune"
    output.mkdir(parents=True, exist_ok=True)
    return (output,)


@app.cell
def preprocessing(metadata, output, save_metadata):
    save_metadata(output / "preprocessing.json", metadata)
    return


@app.cell
def base_training(
    base_epochs,
    batch_size,
    features,
    metadata,
    network_params,
    np,
    output,
    partition,
    pd,
    targets,
    train_mask,
    validation_mask,
    xiRTNET,
):
    params = network_params()
    params["_preprocessing"] = metadata
    base = xiRTNET(
        params,
        input_dim=metadata["sequence_width"],
        embedding_dim=len(metadata["vocabulary"]) + 1,
    )
    base.build_model(siamese=True, alphabet=len(metadata["vocabulary"]) + 1)
    base.compile()
    base_history = base.model.fit(
        partition(features, train_mask),
        [targets[train_mask]],
        validation_data=(partition(features, validation_mask), [targets[validation_mask]]),
        epochs=base_epochs,
        batch_size=batch_size,
        callbacks=base.params["callbacks"],
    )
    pd.DataFrame(base_history.history).assign(
        epoch=np.arange(1, len(base_history.history["loss"]) + 1)
    ).to_csv(output / "base-history.csv", index=False)
    base_path = output / "base.pt"
    base.model.save(base_path)
    return base_history, base_path


@app.cell
def adaptation(
    base_path,
    batch_size,
    features,
    fine_tuning_epochs,
    metadata,
    network_params,
    np,
    output,
    partition,
    pd,
    prediction_metrics,
    table,
    targets,
    test_mask,
    train_mask,
    validation_mask,
    xiRTNET,
):
    records = []
    for strategy in (
        "recurrent_full",
        "recurrent_frozen",
        "recurrent_lokr",
        "recurrent_lora",
    ):
        adapted_params = network_params()
        adapted_params["learning"]["learningrate"] = 0.0005
        adapted_params["learning"]["epochs"] = fine_tuning_epochs
        adapted = xiRTNET(
            adapted_params,
            input_dim=metadata["sequence_width"],
            embedding_dim=len(metadata["vocabulary"]) + 1,
        )
        adapted.fine_tune(
            base_path,
            strategy,
            rank=4,
            factor=4,
            alpha=4.0,
            adapter_learning_rate_ratio=0.1,
        )
        adapted.compile()
        adapted_history = adapted.model.fit(
            partition(features, train_mask),
            [targets[train_mask]],
            validation_data=(partition(features, validation_mask), [targets[validation_mask]]),
            epochs=fine_tuning_epochs,
            batch_size=batch_size,
            callbacks=adapted.params["callbacks"],
        )
        pd.DataFrame(adapted_history.history).assign(
            epoch=np.arange(1, len(adapted_history.history["loss"]) + 1)
        ).to_csv(output / f"{strategy}-history.csv", index=False)
        adapted.model.save(output / f"{strategy}.pt")
        prediction = np.asarray(adapted.model.predict(partition(features, test_mask))).reshape(-1)
        predicted_seconds = prediction * metadata["target_std"] + metadata["target_mean"]
        observed_seconds = table.loc[test_mask, "rp"].to_numpy()
        pd.DataFrame(
            {
                "PSMID": table.loc[test_mask, "PSMID"].to_numpy(),
                "observed_seconds": observed_seconds,
                "predicted_seconds": predicted_seconds,
            }
        ).to_csv(output / f"{strategy}-predictions.csv", index=False)
        records.append(
            {
                "strategy": strategy,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in adapted.model.parameters()
                    if parameter.requires_grad
                ),
                "epochs_trained": len(adapted_history.history["loss"]),
                "best_epoch": int(np.argmin(adapted_history.history["val_loss"])) + 1,
                "validation_rmse_seconds": float(
                    np.sqrt(min(adapted_history.history["val_loss"])) * metadata["target_std"]
                ),
                **{
                    f"test_{name}": value
                    for name, value in prediction_metrics(
                        observed_seconds, predicted_seconds
                    ).items()
                },
                "baseline_rmse_seconds": float(
                    np.sqrt(np.mean((observed_seconds - metadata["target_mean"]) ** 2))
                ),
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(output / "summary.csv", index=False)
    summary
    return (summary,)


@app.cell
def parameter_plot(output, summary):
    import matplotlib.pyplot as plt

    parameter_figure, parameter_axis = plt.subplots(figsize=(8, 3))
    parameter_axis.barh(summary["strategy"], summary["trainable_parameters"])
    parameter_axis.set(xlabel="Trainable parameters")
    parameter_figure.tight_layout()
    parameter_figure.savefig(output / "trainable-parameters.png", dpi=150, bbox_inches="tight")
    parameter_figure
    return (plt,)


@app.cell
def result_plots(base_history, mo, output, pd, plt, summary):
    from xirt.qc import plot_predictions

    rmse_figure, rmse_axis = plt.subplots(figsize=(8, 3))
    rmse_axis.barh(summary["strategy"], summary["test_rmse_seconds"])
    rmse_axis.axvline(
        summary["baseline_rmse_seconds"].iloc[0],
        color="grey",
        linestyle="--",
        label="Training-mean baseline",
    )
    rmse_axis.legend()
    rmse_axis.set(xlabel="Held-out RP RMSE (seconds)", title="Validation-selected fine-tuning")
    rmse_figure.tight_layout()
    rmse_figure.savefig(output / "test-rmse.png", dpi=150, bbox_inches="tight")

    strategy_predictions = {
        strategy: pd.read_csv(output / f"{strategy}-predictions.csv")
        for strategy in summary["strategy"]
    }
    all_predictions = pd.concat(strategy_predictions.values(), ignore_index=True)
    limits = [
        all_predictions[["observed_seconds", "predicted_seconds"]].min().min(),
        all_predictions[["observed_seconds", "predicted_seconds"]].max().max(),
    ]
    prediction_figure, prediction_axes = plt.subplots(
        2, 2, figsize=(8, 8), sharex=True, sharey=True
    )
    for axis, (_strategy, values) in zip(
        prediction_axes.flat, strategy_predictions.items(), strict=True
    ):
        plot_predictions(
            values["observed_seconds"],
            values["predicted_seconds"],
            name=_strategy,
            unit="s",
            ax=axis,
        )
        axis.set(
            xlim=limits,
            ylim=limits,
        )
        axis.set_aspect("equal", adjustable="box")
    prediction_figure.tight_layout()
    prediction_figure.savefig(output / "test-predictions.png", dpi=150, bbox_inches="tight")
    history_figure, history_axes = plt.subplots(2, 2, figsize=(9, 7))
    for _axis, _strategy in zip(history_axes.flat, summary["strategy"], strict=True):
        _history = pd.read_csv(output / f"{_strategy}-history.csv")
        _axis.plot(_history["epoch"], _history["loss"], label="Train")
        _axis.plot(_history["epoch"], _history["val_loss"], label="Validation")
        _axis.set(title=_strategy, xlabel="Adaptation epoch", ylabel="Standardized RP loss")
        _axis.legend()
    history_figure.tight_layout()
    history_figure.savefig(output / "training-loss.png", dpi=150, bbox_inches="tight")

    base_figure, base_axis = plt.subplots()
    _base_history = pd.DataFrame(base_history.history)
    _base_history.index = pd.RangeIndex(1, len(_base_history) + 1, name="epoch")
    _base_history[["loss", "val_loss"]].plot(ax=base_axis)
    base_axis.set(xlabel="Base-training epoch", ylabel="Standardized RP loss")
    base_figure.savefig(output / "base-training-loss.png", dpi=150, bbox_inches="tight")
    mo.vstack([base_figure, history_figure, rmse_figure, prediction_figure])
    return


@app.cell
def interpretation(mo):
    mo.md("""
    ## Applying this to your data

    Keep the base vocabulary, input width and target transformation. A new
    task needs intentional head replacement, not an implicit target remapping.
    GRU adapters support LoKr and LoRA; LSTM supports full and frozen training.
    Adapter learning rates here are 0.1 times the head learning rate.

    Select settings using validation data and inspect test results only after
    selection. Compare against the training-mean baseline, and check the loss
    curves for overfitting. One seed and one within-cohort partition do not
    establish which strategy works best in general. The documentation's
    fine-tuning comparison reports a separate experiment with a fixed protocol.
    """)
    return


if __name__ == "__main__":
    app.run()
