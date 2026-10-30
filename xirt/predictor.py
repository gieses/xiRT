"""Module to organize predictions from CLMS data."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from xirt import const, xirtnet
from xirt import processing as xp
from xirt import sequences as xs
from xirt.assets import MODEL_NAMES, checkpoint_path

logger = logging.getLogger("xirt").getChild(__name__)


@dataclass
class PeptideInputs:
    """Encoded branches and their row/token mapping for prediction and explanation."""

    features: tuple[Any, ...]
    tokens: list[list[list[str]]]
    index: pd.Index
    swapped: np.ndarray
    unknown_token_count: np.ndarray


def prediction_frame(model, predictions, *, index=None, original_units=True):
    """Format native task outputs consistently for API, CLI and legacy ModelData."""
    arrays = [predictions] if len(model.tasks) == 1 else predictions
    columns = {}
    for task, values in zip(model.tasks, arrays, strict=True):
        activation = model.params["output"][f"{task}-activation"]
        if activation in {"linear", "relu"}:
            if model.params["output"][f"{task}-dimension"] != 1:
                raise ValueError("Tabular regression predictions require scalar task heads.")
            scale, offset, _ = model.target_transform(task) if original_units else (1.0, 0.0, "")
            columns[f"{task}-prediction"] = np.asarray(values).reshape(-1) * scale + offset
        elif activation == "softmax":
            columns[f"{task}-prediction"] = np.argmax(values, axis=1)
            columns[f"{task}-probability"] = np.max(values, axis=1)
        elif activation == "sigmoid":
            columns[f"{task}-prediction"] = sigmoid_to_class(values)
        else:
            raise ValueError(f"Unsupported prediction activation: {activation}")
    return pd.DataFrame(columns, index=index)


class Predictor:
    """Load a self-contained RT or FAIMS checkpoint for label-free prediction and XAI.

    Raw RT inputs follow the checkpoint's saved columns/preprocessing. Tutorial
    checkpoints embed their explicit tokenizer metadata. FDR, scores and target
    labels are not needed for inference. Frozen ESM encoders use the ESM API.
    """

    def __init__(self, model):
        if model.route != "recurrent":
            raise ValueError(
                "Raw peptide prediction requires a recurrent checkpoint; use the ESM API."
            )
        self.model = model
        self.faims = None

    @classmethod
    def load(cls, location=None, *, model=None, device="cpu"):
        """Load a custom checkpoint or a bundled model (joint chromatography by default).

        Use ``model="rp"``, ``"scx"``, ``"hsax"``, ``"multitask"`` or ``"faims"``
        to select bundled weights. A custom ``location`` and ``model`` are exclusive.
        """
        if location is not None and model is not None:
            raise ValueError("Use either a checkpoint location or a bundled model, not both.")
        if location is None:
            with checkpoint_path("multitask" if model is None else model) as path:
                return cls.load(path, device=device)
        predictor = cls(xirtnet.RetentionModel.load(location, device=device))
        if "_faims" in predictor.model.params:
            from xirt.faims import FAIMSPredictor

            predictor.faims = FAIMSPredictor.load(location, device=device)
            predictor.model = predictor.faims.model
        return predictor

    def encode(self, frame):
        """Encode raw peptides without sorting/dropping rows or fitting preprocessing."""
        if self.faims is not None:
            return self.faims.encode(frame)
        if frame.empty:
            raise ValueError("Peptide input cannot be empty.")
        params = self.model.params
        data = frame.copy().reset_index(drop=True)
        tutorial = params.get("_preprocessing", {})
        branches = (1, 2) if self.model.siamese else (1,)
        swapped = np.zeros(len(data), dtype=bool)
        if tutorial.get("schema") == "xirt-api-tutorial-v1":
            vocabulary = tutorial["vocabulary"]
            tokens = []
            for branch in branches:
                column, site = f"Peptide{branch}", f"LinkPos{branch}"
                if not {column, site} <= set(data):
                    raise ValueError(f"Missing peptide inference columns: {column}, {site}")
                part = []
                for sequence, position in zip(data[column], data[site], strict=True):
                    if (
                        not isinstance(sequence, str)
                        or not sequence
                        or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
                    ):
                        raise ValueError(
                            "Tutorial RT checkpoints require canonical unmodified peptides."
                        )
                    if not isinstance(position, (int, np.integer)) or not 1 <= position <= len(
                        sequence
                    ):
                        raise ValueError(
                            "Crosslink positions must be one-based integers within the peptide."
                        )
                    part.append(
                        [f"{aa}*" if i == position else aa for i, aa in enumerate(sequence, 1)]
                    )
                tokens.append(part)
        else:
            provenance = params.get("_provenance", {})
            preprocessing = provenance.get("preprocessing")
            if preprocessing is None or "_alphabet" not in params:
                raise ValueError("Checkpoint lacks raw peptide preprocessing metadata.")
            columns = params.get("column_names", const.default_column_names)
            seq_cols = [columns[f"peptide{i}_sequence"] for i in branches]
            required = list(seq_cols)
            if preprocessing["cl_residue"]:
                required += [columns["link_pos_basename"] + str(i) for i in branches]
            required += [c for c in self.model.numeric_features if c not in required]
            missing = set(required) - set(data)
            if missing:
                raise ValueError(f"Missing peptide inference columns: {sorted(missing)}")
            data = data[required].copy()
            if any(
                not data[c].map(lambda v: isinstance(v, str) and bool(v)).all() for c in seq_cols
            ):
                raise ValueError("Peptides must be nonempty strings.")
            if self.model.siamese:
                data = xs.reorder_sequences(data, column_names=columns)
                swapped = data["swapped"].to_numpy()
            prepared = xp.prepare_seqs(data, seq_cols)
            if preprocessing["cl_residue"]:
                for branch, column in zip(branches, seq_cols, strict=True):
                    sites = data[columns["link_pos_basename"] + str(branch)]
                    if any(
                        not isinstance(site, (int, np.integer))
                        or not 1 <= site <= len(sequence) - 2
                        for sequence, site in zip(prepared["Seqar_" + column], sites, strict=True)
                    ):
                        raise ValueError(
                            "Crosslink positions must be one-based integers within the peptide."
                        )
                xs.modify_cl_residues(prepared, seq_cols, column_names=columns)
            tokens = [prepared["Seqar_" + column].tolist() for column in seq_cols]
            if provenance.get("training", {}).get("sequence_type") == "pseudolinear":
                raise ValueError("Raw pseudolinear inference requires the legacy xiRT workflow.")
            vocabulary = {token: i + 1 for i, token in enumerate(params["_alphabet"])}
        encoded = []
        unknown = np.zeros(len(data), dtype=np.int64)
        for branch in tokens:
            array = np.zeros((len(data), self.model.input_dim), dtype=np.int64)
            for row, sequence in enumerate(branch):
                if len(sequence) > self.model.input_dim:
                    raise ValueError("Peptide exceeds checkpoint sequence width; no truncation.")
                unseen = [token for token in sequence if token not in vocabulary]
                if unseen:
                    raise ValueError(
                        f"Checkpoint cannot encode unseen residue/modification tokens: {unseen}"
                    )
                array[row, -len(sequence) :] = [vocabulary[token] for token in sequence]
            encoded.append(array)
        features = tuple(encoded)
        if self.model.numeric_features:
            missing = set(self.model.numeric_features) - set(data)
            if missing:
                raise ValueError(f"Missing numeric feature columns: {sorted(missing)}")
            features += (data[self.model.numeric_features],)
        return PeptideInputs(features, tokens, frame.index, swapped, unknown)

    def predict(self, frame):
        """Return task-prediction columns in original target units and input row order."""
        if self.faims is not None:
            return self.faims.predict(frame)
        encoded = self.encode(frame)
        result = self.model.predict_frame(encoded.features, index=frame.index)
        result["unknown_token_count"] = encoded.unknown_token_count
        return result

    def explain(self, frame, *, task=None, steps=64):
        """Return residue/numeric attributions mapped back to the input peptide order."""
        encoded = self.encode(frame)
        result = self.model.explain(
            encoded.features, task=task, index=frame.index, tokens=encoded.tokens, steps=steps
        )
        swapped = encoded.swapped[result.residues["sample"].to_numpy()]
        result.residues.loc[swapped, "peptide"] = 3 - result.residues.loc[swapped, "peptide"]
        result.residues = result.residues.sort_values(
            ["sample", "peptide", "position"], kind="stable"
        ).reset_index(drop=True)
        result.summary["unknown_token_count"] = encoded.unknown_token_count
        return result


def add_prediction_subcommands(subparsers):
    """Register checkpoint-driven commands shared by retention-time and FAIMS models."""
    predict = subparsers.add_parser("predict", help="Predict RT or FAIMS from a saved checkpoint.")
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument(
        "--qc", type=Path, help="QC directory; requires an observed target column."
    )
    predict.add_argument("--observed", help="Observed column for QC; never used as a model input.")
    predict.add_argument("--task", help="Continuous task to use for optional QC.")
    explain = subparsers.add_parser("explain", help="Explain RT or FAIMS at residue level.")
    explain.add_argument("--input", required=True, type=Path)
    explain.add_argument("--output", required=True, type=Path)
    explain.add_argument("--task", help="Continuous task; required for multi-task models.")
    explain.add_argument("--steps", type=int, default=64)
    explain.add_argument("--limit", type=int, help="Explain only the first N input rows.")
    for parser in (predict, explain):
        source = parser.add_mutually_exclusive_group()
        source.add_argument(
            "--checkpoint", type=Path, help="Custom checkpoint instead of bundled weights."
        )
        source.add_argument(
            "--model",
            choices=MODEL_NAMES,
            help="Bundled model (default: multitask, predicting RP/SCX/hSAX).",
        )


def run_prediction_command(args):
    """Use one prediction/explanation/QC path regardless of the checkpoint's target."""
    import json

    import matplotlib.pyplot as plt

    from xirt import qc

    predictor = Predictor.load(args.checkpoint, model=getattr(args, "model", None))
    frame = pd.read_csv(args.input)
    if getattr(args, "command", None) == "explain":
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive.")
            frame = frame.head(args.limit)
        explanation = predictor.explain(frame, task=args.task, steps=args.steps)
        args.output.mkdir(parents=True, exist_ok=True)
        explanation.residues.to_csv(args.output / "residues.csv", index=False)
        explanation.numeric.to_csv(args.output / "numeric.csv", index=False)
        explanation.summary.to_csv(args.output / "summary.csv", index_label="input_index")
        (args.output / "explanation.json").write_text(
            json.dumps(
                {
                    "task": explanation.task,
                    "unit": explanation.unit,
                    "steps": args.steps,
                    "method": "embedding integrated gradients",
                    "embedding_baseline": "zero embedding, fixed sequence lengths and padding masks",
                    "numeric_baseline": "fitted training means",
                    "interpretation": "signed model contributions, not causal chemical effects",
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        figure = qc.plot_residue_attributions(explanation)
        figure.savefig(args.output / "residues.png", dpi=150)
        plt.close(figure)
        return
    predictions = predictor.predict(frame)
    output = frame.copy()
    for column in predictions:
        output[column] = predictions[column]
    directory = getattr(args, "qc", None)
    if directory is not None:
        task = getattr(args, "task", None)
        if task is None:
            if len(predictor.model.tasks) != 1:
                raise ValueError("Select --task for QC of a multi-task checkpoint.")
            task = predictor.model.tasks[0]
        if task not in predictor.model.tasks:
            raise ValueError(f"Unknown prediction task: {task}")
        observed = (
            getattr(args, "observed", None) or predictor.model.params["output"][f"{task}-column"]
        )
        if observed not in frame:
            raise ValueError(f"QC requires an observed target column: {observed}")
        if predictor.model.params["output"][f"{task}-activation"] not in {"linear", "relu"}:
            raise ValueError("Shared regression QC requires a continuous task.")
        _, _, unit = predictor.model.target_transform(task)
        scores = qc.prediction_metrics(frame[observed], predictions[f"{task}-prediction"])
        figure = qc.prediction_qc(
            frame[observed], predictions[f"{task}-prediction"], name=task, unit=unit
        )
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / "predictions.png", dpi=150)
        plt.close(figure)
        (directory / "metrics.json").write_text(
            json.dumps({"task": task, "unit": unit, **scores}, indent=2, allow_nan=False) + "\n"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


class ModelData:
    """Class to keep data together."""

    def __init__(
        self, psms_df, features1, features2=pd.DataFrame(), le=None, numeric_features=None
    ):
        """
        Class to keep the model data at a single place.

        Args:
            psms_df: df, dataframe with identifications
            features1: df, features for the second peptide
            features2: df, features for the second peptide
            le:
        """
        self.psms = psms_df
        self.features1 = features1
        self.features2 = features2
        self.numeric_features = (
            pd.DataFrame(index=psms_df.index) if numeric_features is None else numeric_features
        )
        self.le: LabelEncoder = le
        self.train_idx = []
        self.predict_idx = []
        self.cv_idx = []
        self.shuffled = False
        # init prediction df
        self.prediction_df = pd.DataFrame(index=psms_df.index)
        self.prediction_df["cv"] = 0

    def set_fdr_mask(self, fdr_cutoff, str_filter=""):
        """
        Set the training mask on the input data.

        Args:
            fdr_cutoff: float, fdr cutoff to be used
            str_filter: str, a string filter to include psms only if containg the given string

        Returns:
            None
        """
        logger.info("Setting FDR mask.")
        if str_filter == "":
            fdr_mask = (self.psms["fdr"] <= fdr_cutoff) & self.psms["isTT"]
            logger.info("Removed 0 peptides (str).")
        else:
            str_mask = (self.psms["Fasta1"].str.contains(str_filter)) & (
                self.psms["Fasta2"].str.contains(str_filter)
            )
            fdr_mask = (self.psms["fdr"] <= fdr_cutoff) & (self.psms["isTT"]) & str_mask
            logger.info(f"Removed {np.sum(~str_mask)} peptides (str filter).")

        self.psms["fdr_mask"] = fdr_mask
        logger.info(f"Removed {np.sum(~self.psms['isTT'])} peptides (TD/DD).")
        logger.info(f"Removed {np.sum(~(self.psms['fdr'] <= fdr_cutoff))} peptides (FDR).")
        logger.info(f"Removed {np.sum(~fdr_mask)} peptides (combined).")
        logger.info(f"Setting FDR mask: {np.sum(fdr_mask)} valid entries")

    def set_unique_shuffled_sampled_training_idx(self, sample_frac=1, random_state=42):
        """
        Set the training index based on fdr mask and removing duplicates.

        This function also servers for learning curves (by reducing the sample_frac) and
        CV variations (using different states).

        Args:
            sample_frac: float, percentage of training samples to use
            random_state: int, random state

        Returns:
            None
        """
        logger.info(f"Shuffling data (random_state: {random_state})")
        psms_train_idx = (
            self.psms[(self.psms["fdr_mask"]) & (~self.psms["Duplicate"].astype(bool))]
            .sample(frac=sample_frac, random_state=random_state)
            .index
        )
        self.train_idx = psms_train_idx
        self.predict_idx = self.psms.index.difference(psms_train_idx)
        self.shuffled = True

    def iter_splits(self, n_splits, test_size):
        """
        Return iterator indicies for training, testing, validation based on the training data.

        The returned indices belong to the training data, validation data and prediction data.
        The prediction fold is used in xiRT to predict RTs without using the observations during
        training.

        Args:
            n_splits: int, number of crossvalidation splits
            test_size: float, percentage of validation data to use

        Returns:
            iterator, (train_idx, val_idx, pred_idx)
        """
        # for predict mode, do not iterate
        if n_splits == 0:
            return

        if not self.shuffled:
            msg = "Data must be shuffled to avoid undesired bias in the splits."
            logger.critical(msg)
            raise ValueError(msg)

        if test_size < 0.1:
            msg = "Test split value must be > 0.1. Please set test_size to min 0.1 (10%)."
            logger.critical(msg)
            raise ValueError(msg)

        # note: code with *loc indicates 0 based locations. code with idx indicates pandas index
        # used for train/validation splits
        cv_folds_ar = np.arange(n_splits) + 1
        cv_pattern = ["t"] * (cv_folds_ar[-1] - 1) + ["v"]
        train_df_idx = self.train_idx.values

        if len(cv_pattern) == 1:
            logger.info("Running in train-mode: no cv will be done - only 1 split.")
            # train on entire data set - a fraction for testing/validation is mandatory!
            # syntax a bit confusing:
            # split data into 100% - 2x test size, % test size, % test size
            # e.g. test-size= 10% --> 80%, 10%, 10%
            # prediction set is used for model assessment here
            train_idx, val_idx, pre_idx = np.split(
                train_df_idx,
                [
                    int((1 - test_size * 2) * len(train_df_idx)),
                    int((1 - test_size) * len(train_df_idx)),
                ],
            )
            yield train_idx, val_idx, pre_idx

        else:
            logger.info("Running in crossvalidation-mode: cv will be done.")
            # get n_splits of the training
            train_features1 = self.features1.loc[self.train_idx]
            kf = KFold(n_splits=n_splits, shuffle=False)
            kf.get_n_splits(train_features1)
            cv_slice_idxs = np.array([i[1] for i in kf.split(train_features1)], dtype=object)

            for i in cv_folds_ar:
                # this will get all the slices where "t" indicates the training
                # here the validation data should be dependant on the CV fold sizes
                train_msk = np.array(cv_pattern) == "t"
                split_train = np.concatenate(cv_slice_idxs[train_msk])

                # combine 2 test folds to get a training fold
                # get train folds -> get locations -> get index
                train_init_idx = train_df_idx[split_train.tolist()]
                train_idx, val_idx = np.split(
                    train_init_idx, [int((1 - test_size) * len(train_init_idx))]
                )

                # take a testing fold
                pre_idx = train_df_idx[list(cv_slice_idxs[~train_msk][0])]

                # change the pattern for next iteration
                cv_pattern = cv_pattern[1:] + [cv_pattern[0]]

                yield train_idx, val_idx, pre_idx

    def get_train_psms(self):
        """
        Return the psms used for training.

        Returns:
            df, dataframe with psms passing training conditions
        """
        return self.psms.loc[self.train_idx]

    def get_predict_psms(self):
        """
        Return the psms used for prediction.

        Returns:
            df, dataframe with psms passing training conditions
        """
        return self.psms.loc[self.predict_idx]

    def get_features(self, idx, meta=False):
        """
        Return the feature format for multi-task learning for linear and crosslinked peptides.

        Args:
            idx: ar-like, indices to subset feature data
            meta: bool, if meta features should be included.

        Returns:
            df, feature dataframe
        """
        if meta:
            # TODO
            xfeatures = None
        else:
            # if lienar peptides are used features 2 are empty
            if self.features2.empty:
                xfeatures = self.features1.filter(regex="rnn|esm").loc[idx]
            else:
                xfeatures = (
                    self.features1.filter(regex="rnn|esm").loc[idx],
                    self.features2.filter(regex="rnn|esm").loc[idx],
                )
            if self.numeric_features.shape[1]:
                branches = xfeatures if isinstance(xfeatures, tuple) else (xfeatures,)
                xfeatures = (*branches, self.numeric_features.loc[idx])
        return xfeatures

    def get_classes(self, idx, frac_cols, cont_cols):
        """
        Return the feature format for multi-task learning for linear and crosslinked peptides.

        Args:
            idx: ar-like, indices to subset feature data
            frac_cols: ar-like, strings with categorical variables
            cont_cols: ar-like, strings with continuous variables
        Returns:
            df, feature dataframe
        """
        # if only continous columns, only return these etc.
        if len(frac_cols) == 0:
            return [self.psms.loc[idx, ccol].values for ccol in cont_cols]

        if len(cont_cols) == 0:
            return [xirtnet.reshapey(self.psms.loc[idx, fcol].values) for fcol in frac_cols]

        y_var = [xirtnet.reshapey(self.psms.loc[idx, fcol].values) for fcol in frac_cols]
        y_var.extend([self.psms.loc[idx, ccol].values for ccol in cont_cols])
        return y_var

    def predict_and_store(self, xirtnetwork, xdata, store_idx, cv=0):
        """
        Generate and store predictions for the observations from store_idx.

        Formatting, prediction and indexing is all done via this high-level wrapper function.
        The given network model will be used to predict for all CSMs (store_idx) the respective
        RT dimensions. The predictions are processed so that they can be stored in a table.

        Parameters:
            xirtnetwork: xirnetwork, class object xirt
            xdata: tuple of dataframes, corresponding on the RNN / features of the one/two peptides
            store_idx: ar-like, indices to be used for the prediction process.
            cv: int, cv-index of current iteration

        Returns:
            None
        """
        # store crosslink predictions
        predictions = xirtnetwork.model.predict(xdata)
        self.store_predictions(xirtnetwork, predictions, store_idx, cv=cv, suf="")

        # if single predictions should be included in the df.Not meaningful for linear peptides.
        # For crosslinked peptides, the raw RT time of the two peptides are added.
        if (xirtnetwork.siamese_p["single_predictions"]) & (xirtnetwork.siamese_p["use"]):
            # create dummy input with all zeroes as second peptide
            dummy = np.zeros_like(xdata[0])
            pep1_predictions = xirtnetwork.model.predict((xdata[0], dummy))
            self.store_predictions(xirtnetwork, pep1_predictions, store_idx, cv=cv, suf="peptide1")

            pep2_predictions = xirtnetwork.model.predict((xdata[1], dummy))
            self.store_predictions(xirtnetwork, pep2_predictions, store_idx, cv=cv, suf="peptide2")

    def store_predictions(self, xirtnetwork, predictions, store_idx, cv=0, suf=""):
        """
        Format predictions to store them in a prediction dataframe.

        This function processes linear, softmax, sigmoid predictions differently. For linear
        activations no processing apart from 1d flattening is done. For softmax the class prediction
        and the probability are retrieved. For sigmoid (usually used for ordered regression style
        predictions, the class values are retrieved.

        Args:
            xirtnetwork:    tf obj, trained network model
            predictions: ar-like, array with predictions
            store_idx: ar-like, index belonging to the predictions to the predictions
            cv: int, cv iteration if crossvalidaton was performed
            suf: str, suffix to append to the default column names for the predictions
        Returns:
            None
        """
        # make column name nicer
        if len(suf) > 0:
            suf = "-" + suf

        formatted = prediction_frame(
            xirtnetwork, predictions, index=store_idx, original_units=False
        )
        self.prediction_df.loc[store_idx, "cv"] = cv
        for column in formatted:
            self.prediction_df.loc[store_idx, column + suf] = formatted[column]


def compute_accuracy(predictions, expected, tasks, params):
    """
    Compute accuracy for ordinal predictions.

    Args:
        predictions: ar-like, multi-task predictions from xirt
        expected: dataframe, pandas dataframe with indices matching the predictions
        tasks: ar-like, lists of tasks
        params: dict, parameter
    Returns:
        list, accuracy values for tasks with ordinal scale
    """
    accuracy_tmp = []
    # predictions must have the same shape as the tasks.
    # class predictions with a single task come as an n-dimensional array instead of an nested ar
    # therefore, predictions must be put into a list to match the dimension of tasks
    if len(predictions) > len(tasks):
        predictions = [predictions]
    # compute accuracy
    for task_i, pred_ar in zip(tasks, predictions):
        if "ordinal" in params[task_i + "-column"]:
            accuracy_tmp.append(
                accuracy_score(sigmoid_to_class(pred_ar), expected[task_i + "_0based"])
            )
    return np.round(accuracy_tmp, 3)


def sigmoid_to_class(predictions, t=0.5):
    """
    Transform an array of sigmoid activations to a class prediction.

    The class prediction will be done to the first entry that has a lower probability than t or
    the last value in the array if no value is smaller than t.

    Args:
        predictions: ar-like, predictions from for ordinal regression task
        t: float, score cutoff to determine prediction.

    Returns:
        ar-like, predictions
    """
    # init output
    pred_hats = np.zeros(len(predictions), dtype=np.intp)

    # iterate over prediction df
    for ii, predi in enumerate(predictions):
        tval = np.where(predi <= t)[0]
        # if all values > t, use last class as predicted label
        pred_hats[ii] = len(predi) - 1 if len(tval) == 0 else tval[0]
    return pred_hats


def preprocess(
    matches_df,
    sequence_type="crosslink",
    max_length=-1,
    cl_residue=True,
    fraction_cols=[],
    column_names=const.default_column_names,
    le=None,
    fraction_levels=None,
    numeric_columns=None,
):
    """Prepare peptide identifications to be used with xiRT.

    High-level wrapper performing multiple steps. In processing order this function:
    sets and index, sorts by score, reorders crosslinked peptides for length (alpha longer),
    rewrites sequences to modX, marks duplicates, modifies cl residues, computes RNN features
    and stores everything in the data model.

    Args:
        matches_df: df, dataframe with peptide identifications
        sequence_type: str, either linear, pseudolinear or cross-linked indicating the input
        molecule type (default: crosslink)
        max_length: int, maximal length of peptide sequences to consider. Longer sequences will
        be removed. (default: 1)
        cl_residue: bool, if true handles cross-link sites as additional modifications.
        (default: True)
        fraction_cols: ar-like, list of columsn that encode frationation data. (default: [])
        column_names: Column names of input file.
    Returns:
        model_data, processed feature dataframes and label encoder
    """
    logger.info("Preprocessing peptides.")
    logger.info(f"Input peptides: {len(matches_df)}")
    if matches_df.empty:
        raise ValueError("Input peptide table is empty.")

    # sort to keep only highest scoring peptide from duplicated entries
    matches_df = matches_df.sort_values(by=column_names["score"], ascending=False)

    if len(matches_df["PSMID"].unique()) != matches_df.shape[0]:
        logger.warning("PSMID column was not unique! Redundant PSMIDs were removed")
        matches_df = matches_df.drop_duplicates("PSMID")

    logger.info(f"Reordering peptide sequences. (mode: {sequence_type})")

    # generate columns to handle based on input data type
    if sequence_type in ["crosslink", "pseudolinear"]:
        matches_df = xs.reorder_sequences(matches_df, column_names=column_names).copy()
        seq_in = [column_names["peptide1_sequence"], column_names["peptide2_sequence"]]
    elif sequence_type == "linear":
        matches_df[column_names["peptide2_sequence"]] = ""
        seq_in = [column_names["peptide1_sequence"]]
    else:
        msg = "sequence type not supported. Must be one of (crosslink, pseudolinear, linear)"
        logger.critical(msg)
        sys.exit(msg)

    # dynamically get if linear or crosslinked
    seq_proc = ["Seqar_" + i for i in seq_in]

    # perform the sequence based processing
    matches_df = xp.prepare_seqs_mp(matches_df, seq_cols=seq_in)

    # concat peptide sequences
    matches_df["PepSeq1PepSeq2_str"] = (
        matches_df[column_names["peptide1_sequence"]]
        + matches_df[column_names["peptide2_sequence"]]
    )

    # mark all duplicates
    matches_df["Duplicate"] = matches_df.duplicated(["PepSeq1PepSeq2_str"], keep="first")
    logger.info(
        f"Duplicatad entries (by sequence only): {matches_df['Duplicate'].sum()}/{len(matches_df)}"
    )

    if cl_residue:
        logger.info("Encode crosslinked residues.")
        xs.modify_cl_residues(matches_df, column_names=column_names, seq_in=seq_in)

    # for pseudo linears, simply concat the input data and put a spacer between the two sequences
    if sequence_type == "pseudolinear":
        matches_df["pseudo_spacer"] = [["X"]] * len(matches_df)
        matches_df["Seqar_Pseudo"] = matches_df[[seq_proc[0], "pseudo_spacer", seq_proc[1]]].apply(
            lambda row: np.concatenate(row.to_list()), axis=1
        )
        seq_proc = ["Seqar_Pseudo"]

    # length filter
    if max_length != -1:
        len1 = matches_df[seq_proc[0]].str.len()
        len2 = matches_df[seq_proc[1]].str.len() if len(seq_proc) == 2 else np.zeros_like(len1)
        valid_length = (len1 <= max_length) & (len2 <= max_length)
        matches_df = matches_df[valid_length]

    logger.info(f"Applying length filter: {len(matches_df)} peptides left")
    if matches_df.empty:
        raise ValueError("No peptides remain after length filtering.")
    features_rnn_seq1, features_rnn_seq2, le = xp.featurize_sequences(
        matches_df, seq_cols=seq_proc, max_length=max_length, le=le
    )

    # add the two fraction encoding columns
    # psms_df[col + "_1hot"] and psms_df[col + "_ordinal"]
    if len(fraction_cols) > 0:
        logger.info("Encoding fractions")
        xp.fraction_encoding(matches_df, rt_methods=fraction_cols, levels=fraction_levels)

    # keep all data together in a data class
    missing = set(numeric_columns or []) - set(matches_df.columns)
    if missing:
        raise ValueError(f"Missing configured numeric feature columns: {sorted(missing)}")
    return ModelData(
        matches_df,
        features_rnn_seq1,
        features_rnn_seq2,
        le=le,
        numeric_features=matches_df[numeric_columns or []],
    )
