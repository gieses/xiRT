"""Train, save and reload a small crosslink RP model on CPU."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def introduction():
    import marimo as mo

    mo.md("""
    # Train and predict retention times

    This example fits a shared GRU to 2,048 high-confidence E. coli crosslinks.
    It uses 1,536 training pairs, 256 validation pairs and 256 held-out test pairs.
    Each pair occurs once. The data exclude modified peptides; a marked residue
    token records the crosslink position. No weights or data are downloaded.

    Training runs for up to 150 epochs with validation-based early stopping
    and learning-rate reduction. The best validation weights are restored.
    The split does not hold out proteins, individual peptides, runs or studies.
    """)
    return (mo,)


@app.cell
def data():
    import json

    import numpy as np
    import pandas as pd
    import torch
    from tutorial_data import (
        ROOT,
        encode_pairs,
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
    table.groupby("split").size().rename("pairs").to_frame()
    return (
        ROOT,
        encode_pairs,
        features,
        json,
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
    epochs = network_params()["learning"]["epochs"]
    batch_size = network_params()["learning"]["batch_size"]
    return batch_size, epochs


@app.cell
def training(
    batch_size,
    epochs,
    features,
    metadata,
    network_params,
    partition,
    targets,
    train_mask,
    validation_mask,
    xiRTNET,
):
    network = xiRTNET(
        network_params(),
        input_dim=metadata["sequence_width"],
        embedding_dim=len(metadata["vocabulary"]) + 1,
    )
    network.build_model(siamese=True, alphabet=len(metadata["vocabulary"]) + 1)
    network.compile()
    history = network.model.fit(
        partition(features, train_mask),
        [targets[train_mask]],
        validation_data=(
            partition(features, validation_mask),
            [targets[validation_mask]],
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=network.params["callbacks"],
    )
    return history, network


@app.cell
def history_plot(history, output, pd):
    import matplotlib.pyplot as plt

    loss_figure, loss_axis = plt.subplots()
    history_table = pd.DataFrame(history.history)
    history_table.index = pd.RangeIndex(1, len(history_table) + 1, name="epoch")
    history_table.to_csv(output / "history.csv")
    history_table[["loss", "val_loss"]].plot(ax=loss_axis)
    loss_axis.set(xlabel="Epoch", ylabel="Standardized RP loss")
    loss_figure.savefig(output / "training-loss.png", dpi=150, bbox_inches="tight")
    loss_figure
    return (plt,)


@app.cell
def output_directory(ROOT):
    output = ROOT / "examples/results/train-predict"
    output.mkdir(parents=True, exist_ok=True)
    return (output,)


@app.cell
def checkpoint(json, metadata, network, output, save_metadata, xiRTNET):
    checkpoint_path = output / "model.pt"
    network.model.save(checkpoint_path)
    save_metadata(output / "preprocessing.json", metadata)

    restored_metadata = json.loads((output / "preprocessing.json").read_text())
    restored = xiRTNET(
        network.params,
        input_dim=restored_metadata["sequence_width"],
        embedding_dim=len(restored_metadata["vocabulary"]) + 1,
    )
    restored.load_model(checkpoint_path)
    return restored, restored_metadata


@app.cell
def predictions(
    encode_pairs,
    features,
    network,
    np,
    output,
    partition,
    pd,
    prediction_metrics,
    restored,
    restored_metadata,
    table,
    test_mask,
):
    test_table = table.loc[test_mask].copy()
    reencoded = encode_pairs(test_table, restored_metadata)
    predicted = np.asarray(restored.model.predict(reencoded)).reshape(-1)
    np.testing.assert_allclose(
        predicted,
        np.asarray(network.model.predict(partition(features, test_mask))).reshape(-1),
        rtol=1e-6,
        atol=1e-6,
    )
    test_table["predicted_seconds"] = (
        predicted * restored_metadata["target_std"] + restored_metadata["target_mean"]
    )
    rmse_seconds = float(
        np.sqrt(np.mean((test_table["rp"] - test_table["predicted_seconds"]) ** 2))
    )
    metrics = pd.DataFrame(
        [
            {
                "model": "GRU",
                **prediction_metrics(
                    test_table["rp"].to_numpy(), test_table["predicted_seconds"].to_numpy()
                ),
            },
            {
                "model": "Training mean",
                **prediction_metrics(
                    test_table["rp"].to_numpy(),
                    np.full(len(test_table), restored_metadata["target_mean"]),
                ),
            },
        ]
    )
    metrics.to_csv(output / "metrics.csv", index=False)
    test_table.to_csv(output / "test-predictions.csv", index=False)
    test_table[["PSMID", "rp", "predicted_seconds"]]
    return metrics, rmse_seconds, test_table


@app.cell
def held_out_plot(metrics, mo, output, plt, rmse_seconds, test_table):
    prediction_figure, prediction_axis = plt.subplots()
    prediction_axis.scatter(test_table["rp"], test_table["predicted_seconds"], alpha=0.6, s=18)
    limits = [
        min(test_table["rp"].min(), test_table["predicted_seconds"].min()),
        max(test_table["rp"].max(), test_table["predicted_seconds"].max()),
    ]
    prediction_axis.plot(limits, limits, "--", color="grey")
    prediction_axis.set(xlabel="Observed RP (seconds)", ylabel="Predicted RP (seconds)")
    prediction_axis.set_title(f"Held-out RMSE {rmse_seconds:.0f} s; R2 {metrics.iloc[0]['r2']:.2f}")
    prediction_figure.savefig(output / "test-predictions.png", dpi=150, bbox_inches="tight")
    mo.vstack(
        [
            mo.md(
                f"Held-out RMSE: **{rmse_seconds:.1f} seconds**. This is a tutorial, not a benchmark."
            ),
            prediction_figure,
            metrics,
        ]
    )
    return


@app.cell
def preprocessing_notes(mo):
    mo.md("""
    ## Reusing the checkpoint

    Inference needs the saved vocabulary, sequence width, crosslink-token rule
    and training-only target mean and standard deviation. The cell above loads
    both files, re-encodes the test rows and checks predictions against the
    in-memory model. Prediction does not require `compile()`.

    `preprocessing.json` is this example's API format, **not** the CLI's
    `encoder.json`. For modified peptides or CLI inference, use xiRT's
    production preprocessing and exported metadata instead of this restricted
    tokenizer. Do not reuse the tutorial's weights as a scientific baseline.
    """)
    return


if __name__ == "__main__":
    app.run()
