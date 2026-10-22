"""xiRT main module to run the training and prediction."""

import argparse
import copy
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

import xirt.features as xf
import xirt.predictor as xr
from xirt import __version__, const, qc, xirtnet
from xirt.predictor import ModelData
from xirt.training import seed_everything

matplotlib.use("Agg")
logger = logging.getLogger("xirt").getChild(__name__)


def arg_parser() -> argparse.ArgumentParser:  # pragma: not covered
    """
    Parse the arguments from the CLI.

    Returns:
        arguments, from parse_args
    """
    description = f"""
    xiRT is a machine learning tool for the (multidimensional) RT prediction of linear and
    crosslinked peptides. Use --help to see the command line arguments.

    Visit the documentation to get more information:
    https://gieses.github.io/xiRT/

    You can generate example configs by running:
    >xirt -p xirt_params.yaml

    and
    >xirt -s learning_params.yaml
    Current Version: {__version__}
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-i",
        "--in_peptides",
        help="Input peptide table to learn (and predict) the retention times.",
        action="store",
        dest="in_peptides",
    )

    parser.add_argument(
        "-o", "--out_dir", help="Directory to store the results", action="store", dest="out_dir"
    )

    parser.add_argument(
        "-x",
        "--xirt_params",
        help="YAML parameter file to control xiRT's deep learning architecture.",
        action="store",
        dest="xirt_params",
    )

    parser.add_argument(
        "-l",
        "--learning_params",
        help="YAML parameter file to control training and testing splits and data.",
        action="store",
        dest="learning_params",
    )

    parser.add_argument(
        "-s",
        "--sample_learning_params",
        help="If set to an output location, all other options are ignored and an "
        "example config is written for the learning parameter file.",
        required=False,
        action="store",
        dest="sample_learning_params",
        default="",
    )
    parser.add_argument(
        "-p",
        "--sample_xirt_params",
        help="If set to an output location, all other options are ignored and an "
        "example config is written for the xirt parameter file.",
        required=False,
        action="store",
        dest="sample_xirt_params",
        default="",
    )

    parser.add_argument(
        "--write",
        dest="write",
        action="store_true",
        help="Flag for writing result prediction files. If false only summaries "
        "are written (default: --write).",
    )
    parser.add_argument(
        "--no-write",
        dest="write",
        action="store_false",
        help="Flag for writing result prediction files. If false only summaries "
        "are written (default: --write).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.set_defaults(write=True)
    from xirt.faims import add_faims_subcommands
    from xirt.predictor import add_prediction_subcommands

    subparsers = parser.add_subparsers(dest="command")
    add_faims_subcommands(subparsers)
    add_prediction_subcommands(subparsers)
    return parser


def xirt_runner(
    peptides_file: str,
    out_dir: str,
    xirt_params: dict[str, Any],
    learning_params: dict[str, Any],
    nrows: int | None = None,
    perform_qc: bool = True,
    write: bool = True,
    write_dummy: str | bool = "",
    api_df: pd.DataFrame | None = None,
) -> ModelData:
    """
    Execute xiRT, train a model or generate predictions for RT across multiple RT domains.

    Args:
        peptides_file: str, location of the input psm/csm file
        out_dir: str, folder to store the results to
        xirt_params: object, yaml for xirt
        learning_params: object, setup yaml
        nrows: int, number of rows to sample (for quicker testing purposes only)
        perform_qc: bool, indicates if qc plots should be done.
        write: bool,  indicates result predictions should be stored
        write_dummy: str, optional workflow completion-marker path.
        api_df: pandas.DataFrame, Dataframe from API call

    Returns:
        ModelData, processed data and indexed predictions. Continuous outputs use
        the persisted target scale (normally minutes).

    Example:
        >>> result = xirt_runner("peptides.csv", "results", params, learning,
        ...                      perform_qc=False)  # doctest: +SKIP
    """

    start_time = time.time()
    xirt_params = copy.deepcopy(xirt_params)
    learning_params = copy.deepcopy(learning_params)
    xirt_params["siamese"]["use"] = learning_params["train"]["sequence_type"] == "crosslink"
    seed_everything(
        xirt_params["learning"].get("seed", learning_params["train"]["sample_state"]),
        xirt_params["learning"].get("deterministic", True),
    )
    matches_df = None
    if api_df is None:
        if peptides_file.endswith(".parquet"):
            matches_df = pd.read_parquet(peptides_file)
        elif peptides_file.endswith(".tab") or peptides_file.endswith(".tsv"):
            matches_df = pd.read_csv(peptides_file, nrows=nrows, delimiter="\t")
        else:
            matches_df = pd.read_csv(peptides_file, nrows=nrows)
    else:
        matches_df = api_df

    logger.info("Done reading input data.")

    # convenience short cuts
    if learning_params["train"]["mode"] == "train":
        n_splits = 1
    elif learning_params["train"]["mode"] == "predict":
        n_splits = 0
    else:
        n_splits = learning_params["train"]["ncv"]

    test_size = learning_params["train"]["test_frac"]
    outpath = os.path.abspath(out_dir)
    os.makedirs(outpath, exist_ok=True)
    xirt_params["callbacks"]["callback_path"] = os.path.join(outpath, "callbacks")
    os.makedirs(xirt_params["callbacks"]["callback_path"], exist_ok=True)
    tracking = xirt_params.get("tracking", {})
    if tracking.get("enabled"):
        tracking.setdefault("uri", "sqlite:///" + os.path.join(outpath, "mlflow.db"))
        tracking.setdefault("artifact_dir", os.path.join(outpath, "mlflow-artifacts"))
    saved = None
    for key in ("pretrained_model", "pretrained_weights"):
        location = learning_params["train"][key]
        if location and str(location).lower() != "none":
            saved = xirtnet.read_checkpoint(location)
            break
    fine_tuning = xirt_params.get("fine_tuning")
    if fine_tuning:
        if saved is not None:
            raise ValueError(
                "Use fine_tuning.checkpoint or train.pretrained_model/weights, not both."
            )
        if learning_params["train"]["mode"] == "predict":
            raise ValueError(
                "Predict with train.pretrained_model pointing to the adapted checkpoint."
            )
        if not fine_tuning.get("checkpoint"):
            raise ValueError("fine_tuning.checkpoint is required for training.")
        saved = xirtnet.read_checkpoint(fine_tuning["checkpoint"])
    encoder = None
    encoder_path = learning_params["preprocessing"].get("le")
    if encoder_path and str(encoder_path).lower() != "none":
        if not str(encoder_path).endswith(".json"):
            raise ValueError("Use a xiRT 2.0 encoder.json, not a legacy pickled encoder.")
        with open(encoder_path) as stream:
            encoder_data = json.load(stream)
        encoder = LabelEncoder().fit(encoder_data["alphabet"])
        learning_params["preprocessing"]["max_length"] = encoder_data["sequence_length"]
    if saved is not None:
        alphabet = saved["params"].get("_alphabet")
        if alphabet is None:
            raise ValueError("CLI checkpoints require a saved peptide vocabulary.")
        if xirt_params.get("encoder", {}).get("route", "recurrent") == "recurrent":
            if encoder is not None and encoder.classes_.tolist() != alphabet:
                raise ValueError("Supplied encoder vocabulary differs from checkpoint vocabulary.")
            encoder = LabelEncoder().fit(alphabet)
        if saved["params"].get("encoder", {}).get("route", "recurrent") != (
            xirt_params.get("encoder", {}).get("route", "recurrent")
        ):
            raise ValueError("Checkpoint and configured encoder routes differ.")
        if saved["params"].get("numeric_features", []) != xirt_params.get("numeric_features", []):
            raise ValueError("Checkpoint and configured numeric feature schemas differ.")
        # Preserve token semantics and padding width when transferring or predicting.
        learning_params["preprocessing"]["max_length"] = saved["params"].get(
            "_sequence_length", saved["input_dim"]
        )
        if learning_params["train"]["mode"] == "predict" or fine_tuning:
            if saved["params"]["predictions"] != xirt_params["predictions"]:
                raise ValueError("Prediction and fine-tuning require checkpoint prediction tasks.")
            if saved["siamese"] != xirt_params["siamese"]["use"]:
                raise ValueError("Checkpoint and configured peptide branches differ.")
            for task in (
                saved["params"]["predictions"]["fractions"]
                + saved["params"]["predictions"]["continues"]
            ):
                for suffix in ("dimension", "activation", "column"):
                    key = f"{task}-{suffix}"
                    if saved["params"]["output"][key] != xirt_params["output"][key]:
                        raise ValueError(f"Config differs from checkpoint: {key}")
    fraction_levels = (
        saved["params"].get("_fraction_levels")
        if (saved is not None and (learning_params["train"]["mode"] == "predict" or fine_tuning))
        else None
    )

    # preprocess training data
    training_data = xr.preprocess(
        matches_df,
        sequence_type=learning_params["train"]["sequence_type"],
        max_length=learning_params["preprocessing"]["max_length"],
        cl_residue=learning_params["preprocessing"]["cl_residue"],
        fraction_cols=xirt_params["predictions"]["fractions"],
        column_names=xirt_params.get("column_names", const.default_column_names),
        le=encoder,
        fraction_levels=fraction_levels,
        numeric_columns=xirt_params.get("numeric_features", []),
    )
    xirt_params["_alphabet"] = training_data.le.classes_.tolist()
    xirt_params["_sequence_length"] = training_data.features1.shape[1]
    xirt_params["_fraction_levels"] = (
        {
            column: training_data.psms[column].drop_duplicates().sort_values().tolist()
            for column in xirt_params["predictions"]["fractions"]
        }
        if fraction_levels is None
        else fraction_levels
    )
    xirt_params.setdefault("learning", {}).setdefault(
        "seed", learning_params["train"]["sample_state"]
    )
    xirt_params["_provenance"] = {
        "input_sha256": hashlib.sha256(
            matches_df.to_json(orient="split", date_format="iso", double_precision=15).encode()
        ).hexdigest(),
        "serialization": "pandas-split-json-v1",
        "preprocessing": learning_params["preprocessing"],
        "training": learning_params["train"],
    }
    if xirt_params.get("encoder", {}).get("route", "recurrent") == "esm-c":
        from xirt.embeddings import ESMCEncoder

        logger.warning(
            "ESM-C uses unmodified residues; PTM identity and crosslink position "
            "are not represented in its embeddings."
        )
        options = xirt_params["encoder"]
        esmc = ESMCEncoder(
            model_name=options.get("model", "esmc_300m"),
            device=options.get("device", xirt_params["learning"].get("device", "cpu")),
            cache_dir=options.get("cache_dir"),
            batch_size=options.get("batch_size", 8),
        )
        if learning_params["train"]["sequence_type"] == "pseudolinear":
            raise ValueError(
                "ESM-C supports linear or crosslink inputs, not artificial pseudolinears."
            )
        columns = xirt_params.get("column_names", const.default_column_names)
        for branch in (1, 2) if xirt_params["siamese"]["use"] else (1,):
            tokens = training_data.psms["Seqar_" + columns[f"peptide{branch}_sequence"]]
            # PTMs/crosslink labels are deliberately absent from pretrained LM embeddings.
            sequences = tokens.apply(
                lambda seq: "".join(token[-1] for token in seq if token not in {"H-", "-OH"})
            )
            embedded = esmc.encode(sequences.tolist())
            frame = pd.DataFrame(
                embedded,
                index=training_data.psms.index,
                columns=[f"esm_{i}" for i in range(embedded.shape[1])],
            )
            setattr(training_data, f"features{branch}", frame)
        options["embedding_dim"] = esmc.embedding_dim

    # set training index by FDR and duplicates
    training_data.set_fdr_mask(
        fdr_cutoff=learning_params["train"]["fdr"],
        str_filter=learning_params["preprocessing"]["filter"],
    )
    training_data.set_unique_shuffled_sampled_training_idx(
        sample_frac=learning_params["train"]["sample_frac"],
        random_state=learning_params["train"]["sample_state"],
    )

    # adjust RT if necessary to guarantee smooth learning
    # gradient length > 30 minutes (1500 seconds)
    target_scales = (
        saved["params"].get("_target_scales", {})
        if (saved is not None and (learning_params["train"]["mode"] == "predict" or fine_tuning))
        else (xirt_params.get("target_scales", {}))
    )
    if fine_tuning and any(
        target_scales.get(task) != scale
        for task, scale in xirt_params.get("target_scales", {}).items()
    ):
        raise ValueError("Fine-tuning target scales differ from the checkpoint.")
    for cont_col in xirt_params["predictions"]["continues"]:
        scale = target_scales.get(
            cont_col, 60.0 if training_data.psms[cont_col].max() > 1500 else 1.0
        )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Target scales must be finite and positive.")
        training_data.psms[cont_col] = training_data.psms[cont_col] / scale
        target_scales[cont_col] = scale
    xirt_params["_target_scales"] = target_scales

    # init neural network structure
    xirtnetwork = xirtnet.xiRTNET(
        xirt_params,
        input_dim=training_data.features1.shape[1],
        embedding_dim=len(training_data.le.classes_) + 1,
    )

    # get the columns where the RT information is stored
    frac_cols = [
        xirtnetwork.output_p[tt + "-column"]
        for tt in sorted(xirt_params["predictions"]["fractions"])
    ]

    cont_cols = [
        xirtnetwork.output_p[tt + "-column"]
        for tt in sorted(xirt_params["predictions"]["continues"])
    ]

    # init data structures for results
    histories = []
    model_summary = []
    # manual accuracy for ordinal data
    accuracies_all = []
    fold_states = []
    fold_params = []
    if "ordinal" in ";".join([xirtnetwork.output_p[i + "-column"] for i in xirtnetwork.tasks]):
        has_ordinal = True
    else:
        has_ordinal = False

    cv_counter = 1
    # perform crossvalidation
    # train on n-1 fold, use test_size from n-1 folds for validation and test/predict RT
    # on the remaining fold
    logger.info(f"Starting crossvalidation (nfolds={n_splits})")
    start_timecv = time.time()
    for train_idx, val_idx, pred_idx in training_data.iter_splits(
        n_splits=n_splits, test_size=test_size
    ):
        logger.info("---------------------------------------------------------")
        logger.info(f"Starting crossvalidation iteration: {cv_counter}")
        logger.info(f"# Train peptides: {len(train_idx)}")
        logger.info(f"# Validation peptides: {len(val_idx)}")
        logger.info(f"# Prediction peptides: {len(pred_idx)}")
        logger.info(f"# Rescoring Candidates peptides: {len(training_data.predict_idx)}")

        logger.info(f"Train indices: {train_idx[0:10]}")
        logger.info(f"Validation indices: {val_idx[0:10]}")
        logger.info(f"Prediction indices: {pred_idx[0:10]}")
        logger.info(f"Rescoring Candidates indices: {training_data.predict_idx[0:10]}")
        if tracking.get("enabled"):
            xirtnetwork.params["tracking"]["run_name"] = f"xiRT-fold-{cv_counter:02d}"
        split_path = os.path.join(outpath, f"split_{cv_counter:02d}.json")
        with open(split_path, "w") as stream:
            json.dump(
                {
                    "train": train_idx.tolist(),
                    "validation": val_idx.tolist(),
                    "test": pred_idx.tolist(),
                },
                stream,
            )
        xirtnetwork.params["_split"] = {
            "train": train_idx.tolist(),
            "validation": val_idx.tolist(),
            "test": pred_idx.tolist(),
        }

        # init the network model
        # either load from existing or create from config
        if learning_params["train"]["pretrained_model"].lower() != "none":
            logger.info("Loading existing model from reference.")
            xirtnetwork.load_model(learning_params["train"]["pretrained_model"])
        else:
            logger.info("Building new model from config.")
            xirtnetwork.build_model(
                siamese=xirt_params["siamese"]["use"], alphabet=len(training_data.le.classes_) + 1
            )
        # once model architecture is there, check if weights should be loaded
        # loading predefined weights
        if learning_params["train"]["pretrained_weights"].lower() != "none":
            logger.info("Loading pre-trained weights into model.")
            xirtnetwork.model.load_weights(learning_params["train"]["pretrained_weights"])

        # after loading the weights, we need to pop / re-design the layers for transfer learning
        if learning_params["train"]["pretrained_model"].lower() != "none":
            logger.info("Adjusting model architecture.")
            xirtnetwork.adjust_model()

        # finally compile
        xirtnetwork.compile()

        callbacks = xirtnetwork.get_callbacks(suffix=str(cv_counter).zfill(2))
        # assemble training data
        xt_cv = training_data.get_features(train_idx)
        yt_cv = training_data.get_classes(train_idx, frac_cols=frac_cols, cont_cols=cont_cols)
        # validation data
        xv_cv = training_data.get_features(val_idx)
        yv_cv = training_data.get_classes(val_idx, frac_cols=frac_cols, cont_cols=cont_cols)
        # prediction data
        xp_cv = training_data.get_features(pred_idx)
        yp_cv = training_data.get_classes(pred_idx, frac_cols=frac_cols, cont_cols=cont_cols)

        # fit the mode, use the validation split to determine the best
        # epoch for selecting the best weights
        logger.info("Fitting model.")
        history = xirtnetwork.model.fit(
            xt_cv,
            yt_cv,
            validation_data=(xv_cv, yv_cv),
            epochs=xirt_params["learning"]["epochs"],
            batch_size=xirt_params["learning"]["batch_size"],
            verbose=xirt_params["learning"]["verbose"],
            callbacks=callbacks,
        )
        fold_states.append(copy.deepcopy(xirtnetwork.model.state_dict()))
        fold_params.append(copy.deepcopy(xirtnetwork.model.params))
        metric_columns = xirtnetwork.model.metrics_names

        # model evaluation training (t), validation (v), prediction (p)
        model_summary.append(xirtnetwork.model.evaluate(xt_cv, yt_cv, batch_size=512))
        model_summary.append(xirtnetwork.model.evaluate(xv_cv, yv_cv, batch_size=512))
        model_summary.append(xirtnetwork.model.evaluate(xp_cv, yp_cv, batch_size=512))

        # use the training for predicting unseen RTs
        training_data.predict_and_store(xirtnetwork, xp_cv, pred_idx, cv=cv_counter)

        if has_ordinal:
            train_preds = xirtnetwork.model.predict(xt_cv)
            val_preds = xirtnetwork.model.predict(xv_cv)
            pred_preds = xirtnetwork.model.predict(xp_cv)

            accuracies_all.extend(
                xr.compute_accuracy(
                    train_preds,
                    training_data.psms.loc[train_idx],
                    xirtnetwork.tasks,
                    xirtnetwork.output_p,
                )
            )
            accuracies_all.extend(
                xr.compute_accuracy(
                    val_preds,
                    training_data.psms.loc[val_idx],
                    xirtnetwork.tasks,
                    xirtnetwork.output_p,
                )
            )
            accuracies_all.extend(
                xr.compute_accuracy(
                    pred_preds,
                    training_data.psms.loc[pred_idx],
                    xirtnetwork.tasks,
                    xirtnetwork.output_p,
                )
            )

        # store metrics
        # store model? / callback
        df_history = pd.DataFrame(history.history)
        df_history["CV"] = cv_counter
        df_history["epoch"] = np.arange(1, len(df_history) + 1)
        cv_counter += 1
        histories.append(df_history)
    logger.info("Finished CV training model.")

    # CV training done, now deal with the data not used for training
    if learning_params["train"]["mode"] != "predict":
        # store model summary data
        model_summary_df = pd.DataFrame(model_summary, columns=metric_columns)
        model_summary_df["CV"] = np.repeat(np.arange(1, n_splits + 1), 3)
        model_summary_df["Split"] = np.tile(["Train", "Validation", "Prediction"], n_splits)

        # store manual accuray for ordinal data
        if has_ordinal:
            for count, task_i in enumerate(frac_cols):
                taski_short = task_i.rsplit("_", 1)[0]
                if "ordinal" in xirtnetwork.output_p[taski_short + "-column"]:
                    # accuracies_all contains accuracies for train, validation, pred
                    # problem is that accuracies are computed after the loop and stored in a 1d-ar
                    # retrieve information again
                    if len(frac_cols) == 1:
                        model_summary_df[taski_short + "_ordinal-accuracy"] = accuracies_all
                    else:
                        model_summary_df[taski_short + "_ordinal-accuracy"] = accuracies_all[
                            count :: len(frac_cols)
                        ]

        if learning_params["train"]["refit"]:
            logger.info("Refitting model on entire data to predict unseen data.")
            callbacks = xirtnetwork.get_callbacks(suffix="full")
            xrefit = training_data.get_features(training_data.train_idx)
            yrefit = training_data.get_classes(
                training_data.train_idx, frac_cols=frac_cols, cont_cols=cont_cols
            )
            split = int(len(training_data.train_idx) * (1 - test_size))
            xirtnetwork.params["_split"] = {
                "train": training_data.train_idx[:split].tolist(),
                "validation": training_data.train_idx[split:].tolist(),
                "test": [],
            }
            if tracking.get("enabled"):
                xirtnetwork.params["tracking"]["run_name"] = "xiRT-refit"
            xirtnetwork.build_model(
                siamese=xirt_params["siamese"]["use"], alphabet=len(training_data.le.classes_) + 1
            )
            xirtnetwork.compile()

            if learning_params["train"]["pretrained_weights"].lower() != "none":
                logger.info("Loading pretrained weights into model for refit option.")
                xirtnetwork.model.load_weights(learning_params["train"]["pretrained_weights"])

            _ = xirtnetwork.model.fit(
                xrefit,
                yrefit,
                validation_split=test_size,
                epochs=xirt_params["learning"]["epochs"],
                batch_size=xirt_params["learning"]["batch_size"],
                verbose=xirt_params["learning"]["verbose"],
                callbacks=callbacks,
            )
        else:
            logger.info("Selecting best performing CV model to predict unseen data.")

            # load the best performing model across cv from the validation split
            best_model_idx = np.argmin(
                model_summary_df[model_summary_df["Split"] == "Validation"]["loss"].values
            )
            logger.info(f"Best Model: {best_model_idx}")
            logger.info("Loading weights.")
            xirtnetwork.model.load_state_dict(fold_states[best_model_idx])
            xirtnetwork.model.params = fold_params[best_model_idx]
            logger.info("Model Summary:")
            logger.info(model_summary_df.groupby("Split").agg(["mean", "std"]).to_string())
    else:
        logger.info("Loading model weights.")
        pretrained = learning_params["train"]["pretrained_model"]
        if pretrained and str(pretrained).lower() != "none":
            xirtnetwork.load_model(pretrained)
        else:
            xirtnetwork.build_model(
                siamese=xirt_params["siamese"]["use"], alphabet=len(training_data.le.classes_) + 1
            )
        weights = learning_params["train"]["pretrained_weights"]
        if weights and str(weights).lower() != "none":
            xirtnetwork.model.load_weights(weights)
        elif not pretrained or str(pretrained).lower() == "none":
            raise ValueError("Predict mode requires pretrained_model or pretrained_weights.")
        logger.info("Reassigning prediction index for prediction mode.")
        training_data.predict_idx = training_data.psms.index
        model_summary_df = pd.DataFrame()
        df_history_all = pd.DataFrame()

    # get the 'unvalidation data', e.g. data that was not used during training because
    # here the CSMs are that we want to save / rescore later!
    # assign prediction fold to entire data
    if len(training_data.predict_idx):
        xu = training_data.get_features(training_data.predict_idx)
        yu = training_data.get_classes(
            training_data.predict_idx, frac_cols=frac_cols, cont_cols=cont_cols
        )
        training_data.predict_and_store(xirtnetwork, xu, training_data.predict_idx, cv=-1)
        eval_unvalidation = xirtnetwork.model.evaluate(xu, yu, batch_size=512)
        eval_unvalidation.extend([-1, "Unvalidation"])
        if has_ordinal:
            accs_tmp = xr.compute_accuracy(
                xirtnetwork.model.predict(xu),
                training_data.psms.loc[training_data.predict_idx],
                xirtnetwork.tasks,
                xirtnetwork.output_p,
            )
            eval_unvalidation.extend(accs_tmp)
        if learning_params["train"]["mode"] != "predict":
            model_summary_df.loc[len(model_summary_df)] = eval_unvalidation

    # prediction modes dont have any training information
    if learning_params["train"]["mode"] != "predict":
        # collect epoch training data
        df_history_all = pd.concat(histories, ignore_index=True)
        # Training/validation rows are not out-of-fold predictions. Keep their
        # predictions explicit (cv=0) rather than leaking them into CV assessment.
        missing_idx = training_data.prediction_df.index[
            training_data.prediction_df.filter(regex="-prediction$").isna().any(axis=1)
        ]
        if len(missing_idx):
            training_data.predict_and_store(
                xirtnetwork, training_data.get_features(missing_idx), missing_idx, cv=0
            )
    xirtnetwork.model.save(os.path.join(outpath, "xirt_model.pt"))
    with open(os.path.join(outpath, "configuration.json"), "w") as stream:
        json.dump(
            {
                "network": xirtnetwork.model.params,
                "learning": learning_params,
                "input_provenance": xirt_params["_provenance"],
            },
            stream,
            indent=2,
        )
    with open(os.path.join(outpath, "encoder.json"), "w") as stream:
        json.dump(
            {
                "alphabet": xirt_params["_alphabet"],
                "sequence_length": xirt_params["_sequence_length"],
                "fraction_levels": xirt_params["_fraction_levels"],
            },
            stream,
            indent=2,
        )

    # compute features for rescoring
    logger.info("Compute features for rescoring")
    xf.compute_prediction_errors(
        training_data.psms,
        training_data.prediction_df,
        xirtnetwork.tasks,
        frac_cols,
        (xirtnetwork.siamese_p["single_predictions"] & xirtnetwork.siamese_p["use"]),
    )

    # store results
    features_exhaustive = xf.add_rt_features(training_data.prediction_df.filter(regex="error"))
    # qc
    # only training procedure includes qc
    if perform_qc and (learning_params["train"]["mode"] != "predict"):
        logger.info("Generating qc plots.")
        qc.plot_epoch_cv(
            callback_path=xirtnetwork.callback_p["callback_path"],
            tasks=xirtnetwork.tasks,
            xirt_params=xirt_params,
            outpath=outpath,
        )

        qc.plot_summary_strip(
            model_summary_df, tasks=xirtnetwork.tasks, xirt_params=xirt_params, outpath=outpath
        )

        qc.plot_cv_predictions(
            training_data.prediction_df,
            training_data.psms,
            xirt_params=xirt_params,
            outpath=outpath,
        )

        qc.plot_error_characteristics(
            training_data.prediction_df,
            training_data.psms,
            xirtnetwork.tasks,
            xirt_params,
            outpath,
            max_fdr=0.01,
        )

    logger.info("Writing output tables.")

    df_history_all.to_csv(os.path.join(outpath, "epoch_history.csv"))
    model_summary_df.round(2).to_csv(os.path.join(outpath, "model_summary.csv"))

    if write:
        # store data
        training_data.psms.to_csv(os.path.join(outpath, "processed_psms.csv.gz"))
        training_data_Xy = (
            (
                training_data.get_features(training_data.psms.index)
                if xirt_params.get("numeric_features")
                else (training_data.features1, training_data.features2)
            ),
            training_data.get_classes(
                training_data.psms.index, frac_cols=frac_cols, cont_cols=cont_cols
            ),
        )

        with open(os.path.join(xirt_params["callbacks"]["callback_path"], "Xy_data.p"), "wb") as po:
            pickle.dump(training_data_Xy, po, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(xirt_params["callbacks"]["callback_path"], "encoder.p"), "wb") as po:
            pickle.dump(training_data.le, po, protocol=pickle.HIGHEST_PROTOCOL)

        training_data.prediction_df.to_csv(os.path.join(outpath, "error_features.csv"))
        features_exhaustive.to_csv(os.path.join(outpath, "error_features_interactions.csv"))

    # write readme file to results dir
    with open(os.path.join(outpath, "readme.txt"), "w") as of:
        of.write(const.readme)

    # write a text file to indicate xirt is done.
    if isinstance(write_dummy, str) and write_dummy != "":
        with open(write_dummy.replace(".yaml", ".txt"), "w") as of:
            of.write("done.")

    logger.info("Completed xiRT run.")
    logger.info(f"End Time: {datetime.now().strftime('%H:%M:%S')}")
    end_time = time.time()
    logger.info(f"xiRT CV-training took: {(end_time - start_timecv) / 60.0:.2f} minutes")
    logger.info(f"xiRT took: {(end_time - start_time) / 60.0:.2f} minutes")

    return training_data


def main() -> None:  # pragma: no cover
    """Run xiRT main function."""
    parser = arg_parser()

    args = parser.parse_args(sys.argv[1:])
    if args.command in {"predict", "explain"}:
        xr.run_prediction_command(args)
        return
    if args.command == "faims":
        from xirt.faims import run_faims_command

        run_faims_command(args)
        return
    if args.sample_learning_params or args.sample_xirt_params:
        for outfile, contents in (
            (args.sample_learning_params, const.learning_params),
            (args.sample_xirt_params, const.xirt_params),
        ):
            if outfile:
                with open(outfile, "w") as stream:
                    stream.write(contents)
        return
    for key in ("in_peptides", "out_dir", "xirt_params", "learning_params"):
        if getattr(args, key) is None:
            parser.error(f"--{key} is required unless generating sample configurations.")

    # create dir if not there
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # create logger
    logger = logging.getLogger("xirt")
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.FileHandler(os.path.join(args.out_dir, "xirt_logger.log"), "w")
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

    logger.info("command line call:")
    logger.info(
        f"xirt -i {args.in_peptides} "
        f"-o {args.out_dir} -x {args.xirt_params} "
        f"-l {args.learning_params}"
    )
    logger.info("Init logging file.")
    logger.info(f"Starting Time: {datetime.now().strftime('%H:%M:%S')}")
    logger.info("Starting xiRT.")
    logger.info(f"Using xiRT version: {__version__}")

    # call function
    xirt_params = yaml.load(open(args.xirt_params), Loader=yaml.FullLoader)
    learning_params = yaml.load(open(args.learning_params), Loader=yaml.FullLoader)

    logger.info(f"xi params: {args.xirt_params}")
    logger.info(f"learning_params: {args.learning_params}")
    logger.info(f"peptides: {args.in_peptides}")

    xirt_runner(args.in_peptides, args.out_dir, xirt_params, learning_params, write=args.write)


if __name__ == "__main__":  # pragma: no cover
    main()
