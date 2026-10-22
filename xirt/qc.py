"""Module to perform QC on the xiRT performance."""

import glob
import logging
import os

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    r2_score,
)
from statannotations.Annotator import Annotator

sns.set(context="notebook", style="white", palette="deep", font_scale=1)

# constants
hSAX_color = "#9b59b6"
SCX_color = "#3498db"
RP_color = "#e74c3c"
colors = [hSAX_color, SCX_color, RP_color, "C1", "C2", "C7"]
colormaps = [plt.cm.Purples, plt.cm.Blues, plt.cm.Reds, plt.cm.Oranges, plt.cm.Greens, plt.cm.Greys]
targetdecoys_cm = sns.xkcd_palette(["faded green", "orange", "dark orange"])
TOTAL_color = palettable.cartocolors.qualitative.Bold_6.mpl_colors[1:3][1]
new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)
logger = logging.getLogger(__name__)


def prediction_metrics(observed, predicted):
    """Return regression errors and correlations, with undefined scores as None.

    Values must be aligned, finite and in the same units. Predictive R² is
    distinct from Pearson r squared; a constant prediction has undefined r.
    """
    actual, estimate = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if (
        actual.ndim != 1
        or actual.shape != estimate.shape
        or not actual.size
        or not np.isfinite(actual).all()
        or not np.isfinite(estimate).all()
    ):
        raise ValueError("Metrics require aligned, nonempty finite one-dimensional arrays.")
    residual = estimate - actual
    total = float(np.square(actual - actual.mean()).sum())
    varied = len(actual) > 1 and np.ptp(actual) > 0
    correlated = varied and np.ptp(estimate) > 0
    return {
        "n": len(actual),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1 - float(np.square(residual).sum()) / total if varied else None,
        "pearson_r": float(pearsonr(actual, estimate).statistic) if correlated else None,
        "spearman_rho": float(spearmanr(actual, estimate).statistic) if correlated else None,
    }


def plot_predictions(observed, predicted, *, name="Retention time", unit="", ax=None, color=None):
    """Plot observed/predicted values with identity line and annotated correlations."""
    metrics = prediction_metrics(observed, predicted)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
    values = np.concatenate((np.asarray(observed), np.asarray(predicted)))
    low, high = float(values.min()), float(values.max())
    margin = max((high - low) * 0.06, abs(high) * 0.01, 0.1)
    limits = (low - margin, high + margin)
    ax.scatter(
        observed,
        predicted,
        s=20,
        alpha=0.6,
        linewidths=0,
        color=color or "#287b8e",
        rasterized=True,
    )
    ax.plot(limits, limits, "--", color="#54616a", linewidth=1, zorder=0)
    suffix = f" ({unit})" if unit else ""
    ax.set(
        title=name,
        xlabel=f"Observed{suffix}",
        ylabel=f"Predicted{suffix}",
        xlim=limits,
        ylim=limits,
    )
    labels = [("pearson_r", "Pearson r"), ("spearman_rho", "Spearman rho"), ("r2", "R²")]
    annotation = "\n".join(
        f"{label} = {metrics[key]:.3f}" if metrics[key] is not None else f"{label} undefined"
        for key, label in labels
    )
    annotation += f"\nRMSE = {metrics['rmse']:.2f} {unit}\nn = {metrics['n']}"
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def prediction_qc(observed, predicted, *, name="Retention time", unit=""):
    """Create the same annotated prediction/residual QC figure for RT and FAIMS."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    plot_predictions(observed, predicted, name=name, unit=unit, ax=axes[0])
    axes[1].scatter(
        observed,
        np.asarray(predicted) - np.asarray(observed),
        s=20,
        alpha=0.6,
        color="#287b8e",
        linewidths=0,
        rasterized=True,
    )
    axes[1].axhline(0, linestyle="--", color="#54616a", linewidth=1)
    suffix = f" ({unit})" if unit else ""
    axes[1].set(
        title="Residuals", xlabel=f"Observed{suffix}", ylabel=f"Predicted − observed{suffix}"
    )
    axes[1].spines[["top", "right"]].set_visible(False)
    return figure


def plot_learning_curve(history, *, scale=1.0, unit=""):
    """Plot training and validation RMSE from scalar standardized MSE histories."""
    data = pd.DataFrame(history)
    if not {"loss", "val_loss"} <= set(data) or data.empty:
        raise ValueError("Learning curves require nonempty loss and val_loss histories.")
    figure, axis = plt.subplots(figsize=(6, 3.5), layout="constrained")
    for column, label, color in (
        ("loss", "Training", "#287b8e"),
        ("val_loss", "Validation", "#d67b30"),
    ):
        axis.plot(
            np.arange(1, len(data) + 1), np.sqrt(data[column]) * scale, label=label, color=color
        )
    axis.set(xlabel="Epoch", ylabel=f"RMSE ({unit})" if unit else "RMSE")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    return figure


def plot_residue_attributions(explanation, *, sample=0):
    """Plot signed residue and precursor contributions in the prediction's units.

    Positive contributions raise the prediction relative to the explanation
    baseline; they are model attributions, not causal chemical effects.
    """
    residues = explanation.residues.loc[explanation.residues["sample"] == sample]
    numeric = explanation.numeric.loc[explanation.numeric["sample"] == sample]
    if residues.empty:
        raise ValueError(f"No explained residues for sample {sample}.")
    branches = sorted(residues["peptide"].unique())
    figure, axes = plt.subplots(
        len(branches) + bool(len(numeric)),
        1,
        figsize=(
            max(8, residues.groupby("peptide").size().max() * 0.26),
            2.5 * (len(branches) + bool(len(numeric))),
        ),
        squeeze=False,
        layout="constrained",
    )
    for axis, branch in zip(axes[:, 0], branches):
        part = residues.loc[(residues["peptide"] == branch) & (residues["kind"] == "residue")]
        values = part["attribution"]
        axis.bar(
            part["position"],
            values,
            color=np.where(values >= 0, "#c95a4a", "#327f9e"),
            width=0.8,
        )
        axis.axhline(0, color="0.6", linewidth=0.7)
        axis.set(
            title=f"Peptide {branch}",
            ylabel=f"Contribution ({explanation.unit})",
            xticks=part["position"],
            xticklabels=part["token"],
        )
        for tick, marked in zip(axis.get_xticklabels(), part["crosslinked"]):
            if marked:
                tick.set_fontweight("bold")
        axis.spines[["top", "right"]].set_visible(False)
    if len(numeric):
        axis = axes[-1, 0]
        axis.bar(
            numeric["feature"],
            numeric["attribution"],
            color=np.where(numeric["attribution"] >= 0, "#c95a4a", "#327f9e"),
        )
        axis.axhline(0, color="0.6", linewidth=0.7)
        axis.set(title="Precursor context", ylabel=f"Contribution ({explanation.unit})")
        axis.spines[["top", "right"]].set_visible(False)
    summary = explanation.summary.iloc[sample]
    termini = residues.loc[residues["kind"] == "terminus", "attribution"].sum()
    figure.suptitle(
        f"{explanation.task}: prediction {summary['prediction']:.2f} {explanation.unit} | "
        f"baseline {summary['baseline_prediction']:.2f}\n"
        f"Termini {termini:+.2f} | completeness error {summary['completeness_error']:+.3g}",
        fontsize=10,
    )
    return figure


def encode_pval(pvalue):
    """
    Given a p-value returns a significant annotation of the float in * decoding.

    According to: * <= 0.05, ** <= 0.01, *** <= 0.001, **** <= 0.0001

    Args:
        pvalue: float, p-value

    Returns:
        str, annotation
    """
    reference_pval = np.array([0.0001, 0.001, 0.01, 0.05, 1])
    reference_str = ["****", "****", "***", "**", "*", "ns"]
    return reference_str[np.where(reference_pval > pvalue)[0][0]]


def statistical_annotation(x1, x2, yvalues, txt, ax):  # pragma: no cover
    """
    Add annotation to axes plot.

    Args:
        x1:
        x2:
        yvalues:
        txt: str, text to add
        ax: axes object, matplotlib axes

    Returns:
        None
    """
    # statistical annotation
    y, h, col = yvalues.max() + 2, 2, "k"
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * 0.5, y + h, txt, ha="center", va="bottom", color=col)


def add_heatmap(y, yhat, task, ax, colormap, dims):  # pragma: no cover
    """Generate a heatmap to visualize classification results and return plot to given axes.

    Args:
        y: ar-like, observed
        yhat: ar-like, predicted
        task: ar-like, task names
        ax: axes, matplotlib axes
        colormap: colormap, plt.cm colormap instance
        dims: ar-like, unique dimensions for the input data

    Returns:
        axes, matplotlib axes object
    """
    # prepare confusion matrix
    # note that yhat and y are change in the order here! This is needed because sklearn
    # arranges the confusion matrix very confusing ... this way bottom left, to top right
    # will equal a correaltion / straight line = expected behavior
    cm_scx = pd.DataFrame(np.flip(confusion_matrix(yhat, y), axis=0))
    cm_scx.columns = cm_scx.columns
    cm_scx.index = np.flip(cm_scx.columns)
    # mask for not writing zeroes into the plot
    mask = cm_scx <= 0
    cm_scx[mask] = np.nan
    # annotation
    metric_str = """r2: {:.2f} f1: {:.2f} acc: {:.2f} racc: {:.2f}""".format(
        custom_r2(y, yhat),
        f1_score(y, yhat, average="macro"),
        accuracy_score(y, yhat),
        relaxed_accuracy(y, yhat),
    )
    metric_str = """r2: {:.2f} f1: {:.2f} acc: {:.2f} racc: {:.2f}""".format(
        custom_r2(y, yhat),
        f1_score(y, yhat, average="macro"),
        accuracy_score(y, yhat),
        relaxed_accuracy(y, yhat),
    )
    logger.info(f"QC: {task}")
    logger.info(f"Metrics: {metric_str}")
    ax = sns.heatmap(
        cm_scx,
        cmap=colormap,
        annot=True,
        annot_kws={"size": 12},
        fmt=".0f",
        cbar=True,
        mask=mask,
        ax=ax,
    )
    ax.axhline(y=dims[-1], color="k")
    ax.axvline(x=0, color="k")
    ax.set(
        ylim=(cm_scx.shape[0], 0),
        xlabel=f"Observed {task}\n",
        title=f"""{task}\n{metric_str}""",
        ylabel=f"Predicted {task}",
    )
    sns.despine()
    return ax


def add_scatter(y, yhat, task, ax, color):  # pragma: no cover
    """Generate a scatter plot to visualize prediction results and return plot to given axes.

    Args:
        y: ar-like, observed
        yhat: ar-like, predicted
        task: ar-like, task names
        ax: axes, matplotlib axes
        color: color, Either named color RGB for usage in matplotlib / seaborn.

    Returns:
        axes, matplotlib axes object
    """
    ax = plot_predictions(y, yhat, name=task.upper(), ax=ax, color=color)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    sns.despine()
    return ax


def save_fig(fig, path, outname):  # pragma: no cover
    """Save a figure in png/svg/pdf format and 600 dpi.

    Args:
        fig: figure object, matplotlib figure object
        path: str, path to store the result

    Returns:
        None
    """
    # fig.savefig(os.path.join(path, outname + ".png"), dpi=600)
    fig.savefig(os.path.join(path, outname + ".pdf"), dpi=600)
    fig.savefig(os.path.join(path, outname + ".svg"), dpi=600)


def custom_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Compute predictive R-squared, not the squared Pearson correlation.

    Args:
        y: Observed values.
        yhat: Predictions in the same order and units.

    Returns:
        Coefficient of determination; NaN with an explicit warning for constant
        observations or fewer than two observations. Negative values are valid.

    Example:
        >>> custom_r2(np.array([1, 2, 3]), np.array([1, 2, 3]))
        1.0
    """
    if len(y) < 2 or np.var(y) == 0:
        logger.warning(
            "Predictive R-squared is undefined for fewer than two or constant observations."
        )
        return float("nan")
    return float(np.round(r2_score(y, yhat), 2))


def relaxed_accuracy(y, yhat):
    """Compute the relaxed accuracy (within an error margin of 1).

    Args:
        y:  ar-like, observed values
        yhat:  ar-like, predicted values

    Returns:
        Float, relaxed accuracy (error +- 1
    """
    return np.round(sum(np.abs(y - yhat) <= 1) / len(yhat), 2)


def plot_epoch_cv(callback_path, tasks, xirt_params, outpath, show=False):  # pragma: no cover
    """Do a QC plot of the metrics and losses for all tasks across the epochs (over all cvs).

    Args:
        callback_path: str, path to the callback logs
        tasks: ar-like, list of tasks that were used as identifier in xirt
        xirt_params: dict, parsed yaml file for xirt config.
        outpath: str, location to store the plots
        show: bool, if True plot figure
    Returns:
        None
    """
    # %%    # read epoch log from callbacks
    epochlogs = glob.glob(os.path.join(callback_path, "*epochlog*"))
    df = pd.concat(pd.read_csv(i) for i in epochlogs)

    if len(tasks) == 1 and f"{tasks[0]}_loss" not in df:
        # write task as prefix for columns
        df.columns = [tasks[0] + "_" + i if i not in ["epoch", "lr"] else i for i in df.columns]

    # transform and melt dataframe for easier plotting, can probably be optimized
    # RT = retention time
    dfs = []
    for rt_task in tasks:
        df_temp = df.filter(regex=f"{rt_task}|epoch").melt(id_vars="epoch")
        df_temp["RT"] = rt_task
        # normalize to max 1
        df_temp["variable_norm"] = (
            df_temp[["variable", "value"]]
            .groupby("variable")
            .transform(lambda x: x / x.max())["value"]
        )
        dfs.append(df_temp)

    dfs = pd.concat(dfs)
    dfs["Split"] = ["Training" if "val" not in i else "Validation" for i in dfs["variable"]]
    dfs["Metric"] = ["Loss" if "loss" in i.lower() else "Metric" for i in dfs["variable"]]

    # split loss and metric data
    nfracs = len(xirt_params["predictions"]["fractions"])
    ncont = len(xirt_params["predictions"]["continues"])
    for filter in ["Metric", "Loss"]:
        if filter == "Metric":
            cname = "metrics"
        elif filter == "Loss":
            cname = "loss"

        # filter for metric/loss
        df_temp = dfs[dfs["Metric"] == filter]

        # split data by fractionation / continuous
        df_temp_frac = df_temp[df_temp.RT.isin(xirt_params["predictions"]["fractions"])]
        df_temp_cont = df_temp[df_temp.RT.isin(xirt_params["predictions"]["continues"])]
        df_temp_cont = df_temp_cont[~df_temp_cont["variable"].str.contains("r_square")]

        # get the metrics and stuff for training, validation (thus the, 2 in tile)
        frac_metrics = sorted(df_temp_frac["variable"].drop_duplicates())
        task_colors = dict(zip(tasks, colors))
        frac_hues = {
            variable: task_colors[
                df_temp_frac.loc[df_temp_frac["variable"] == variable, "RT"].iloc[0]
            ]
            for variable in frac_metrics
        }

        cont_metrics = sorted(df_temp_cont["variable"].drop_duplicates())
        cont_hues = {
            variable: task_colors[
                df_temp_cont.loc[df_temp_cont["variable"] == variable, "RT"].iloc[0]
            ]
            for variable in cont_metrics
        }

        # metric
        f, ax = plt.subplots()
        # make custom legend because lineplot will duplicate the legends with the same color
        if (len(frac_hues) > 0) & (len(cont_hues) > 0):
            handles_frac = [
                Line2D([0], [0], label=li, color=frac_hues[li])
                for li, ci in zip(frac_metrics[:nfracs], frac_hues)
            ]
            handles_cont = [
                Line2D([0], [0], label=li, color=cont_hues[li])
                for li, ci in zip(cont_metrics[:ncont], cont_hues)
            ]

            ax = sns.lineplot(
                x="epoch",
                y="value",
                hue="variable",
                style="Split",
                data=df_temp_frac,
                ax=ax,
                palette=frac_hues,
                legend=False,
            )

            ax.set(
                ylabel=xirt_params["output"][
                    xirt_params["predictions"]["fractions"][0] + "-" + cname
                ]
            )
            ax2 = ax.twinx()
            ax2 = sns.lineplot(
                x="epoch",
                y="value",
                hue="variable",
                style="Split",
                data=df_temp_cont,
                ax=ax2,
                legend=False,
                palette=cont_hues,
            )
            ax2.set(
                ylabel=xirt_params["output"][
                    xirt_params["predictions"]["continues"][0] + "-" + cname
                ]
            )
            sns.despine(right=False)
            ax.legend(
                handles=list(
                    np.hstack(
                        [
                            handles_frac,
                            handles_cont,
                            Line2D([0], [0], color="k", ls="-", label="training"),
                            Line2D([0], [0], color="k", ls="--", label="validation"),
                        ]
                    )
                ),
                loc=2,
                ncol=3,
                bbox_to_anchor=(0.01, 1.2),
                borderaxespad=0.0,
            )

            for axt in [ax, ax2]:
                axt.yaxis.set_major_locator(ticker.MaxNLocator(5))

        else:
            # deal with single plot case ...
            hues = cont_hues if len(cont_hues) > 0 else frac_hues
            param_col = "continues" if len(cont_hues) > 0 else "fractions"
            ax = sns.lineplot(
                x="epoch",
                y="value",
                hue="variable",
                style="Split",
                data=df_temp_cont if len(cont_hues) else df_temp_frac,
                ax=ax,
                palette=hues,
                legend="full",
            )
            ax.set(
                ylabel=xirt_params["output"][xirt_params["predictions"][param_col][0] + "-" + cname]
            )
            sns.despine(right=False)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.tight_layout()
        if show:
            plt.show()
        save_fig(f, outpath, outname=f"cv_epochs_{cname}")
        plt.close(f)


def plot_summary_strip(summary_df, tasks, xirt_params, outpath):  # pragma: no cover
    """Plot stripplot to summarize learning results.

    Args:
        summary_df: df, summary dataframe from xirt
        tasks: ar-like, list of tasks
        xirt_params: dict, xirt parameters
        outpath: str, location for storing plots
    Returns:
        None
    """
    # %%
    summary_df = summary_df.copy()
    if len(tasks) == 1 and f"{tasks[0]}_loss" not in summary_df:
        colstr = f"{tasks[0]}_ordinal-accuracy"
        summary_df.columns = [
            tasks[0] + "_" + i if i not in [colstr, "CV", "Split"] else i
            for i in summary_df.columns
        ]

    for col in tasks:
        if "ordinal" in xirt_params["output"][col + "-column"]:
            # store at right place
            summary_df[col + "_accuracy"] = summary_df[col + "_ordinal-accuracy"].astype(float)

    metrics = [
        "accuracy"
        if "ordinal" in xirt_params["output"][i + "-column"]
        else xirt_params["output"][i + "-metrics"]
        for i in tasks
    ]
    for eval_method in [metrics, "losses"]:
        f, axes = plt.subplots(1, len(tasks), figsize=(5, 4))
        if len(tasks) == 1:
            axes = [axes]
        if type(axes) is matplotlib.axes._axes.Axes:
            axes = [axes]

        for ii, m, t in zip(range(len(tasks)), eval_method, tasks):
            if eval_method == "losses":
                m = "loss"
                store_str = "loss"
            else:
                store_str = "metric"
            # points
            axes[ii] = sns.stripplot(
                x="Split", y=f"{t}_{m}", data=summary_df, ax=axes[ii], color=colors[ii]
            )
            # errors bars
            axes[ii] = sns.pointplot(
                x="Split",
                y=f"{t}_{m}",
                data=summary_df,
                color="grey",
                ax=axes[ii],
                markersize=4,
                linewidth=1,
            )
            if m == "accuracy":
                ylim_ax = 1
            else:
                ylim_ax = axes[ii].get_ylim()[1] + 0.2 * axes[ii].get_ylim()[1]

            axes[ii].set(title=t, ylabel=m, xlabel="", ylim=(0, ylim_ax))
            sns.despine(ax=axes[ii])

        # set axes
        for ax in axes:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.tick_params(axis="x", labelrotation=90)
        plt.tight_layout()
        save_fig(f, outpath, outname=f"cv_summary_strip_{store_str}")
        plt.close(f)
        # %%


def plot_cv_predictions(
    df_predictions, input_psms, xirt_params, outpath, show=False
):  # pragma: no cover
    """Plot observed vs. predicted figure for all tasks.

    Args:
        df_predictions: df, dataframe with predictions and cv annotation
        input_psms: df, input dataframe after xiRT is completed
        xirt_params: dict, xirt parameters
        outpath: str, location to store the figures
        show: bool, if true plots the figure, else just store them

    Returns:
        None
    df_predictions, input_psms = training_data.prediction_df, training_data.psms
                               xirt_params, outpath=xirt_params, outpath
    """
    # %%
    # iterate over the cv folds
    for i in df_predictions["cv"].drop_duplicates():
        # filter data to CV iteration
        temp_cv_df = df_predictions[df_predictions["cv"] == i]
        temp_in_df = input_psms.loc[temp_cv_df.index.values]

        # organize plotting stuff
        fracs = sorted(xirt_params["predictions"]["fractions"])
        cont = sorted(xirt_params["predictions"]["continues"])
        ntasks = len(fracs) + len(cont)

        f, axes = plt.subplots(1, ntasks, figsize=(4 * ntasks, 4))
        axes = np.atleast_1d(axes)

        idx = 0
        # plot heatmap variant for variant for fraction prediction
        for frac_task in fracs:
            dims = np.arange(1, xirt_params["output"][frac_task + "-dimension"] + 1)
            axes[idx] = add_heatmap(
                temp_in_df[frac_task + "_0based"],
                temp_cv_df[frac_task + "-prediction"],
                task=frac_task,
                ax=axes[idx],
                colormap=colormaps[idx],
                dims=dims,
            )
            idx += 1
        # plot scatter variant for continues predictions
        for cont_task in cont:
            axes[idx] = add_scatter(
                temp_in_df[cont_task].values,
                temp_cv_df[cont_task + "-prediction"].values,
                task=cont_task,
                ax=axes[idx],
                color=colors[idx],
            )
            idx += 1
        axes[int(ntasks / 2)].set(xlabel=axes[int(ntasks / 2)].get_xlabel() + f"\nCV: {i}")
        plt.tight_layout()
        if show:
            plt.show()
        save_fig(f, outpath, f"qc_cv{str(i).zfill(2)}_obs_pred")
        plt.close(f)
    # %%


def plot_classification_report(
    y, yhat, title, path, colormap=plt.cm.Blues, name=""
):  # pragma: no cover
    """Plot a classification report from sklearns API.

    Args:
        y: ar-like, observed
        yhat: ar-like, predicted
        title: str, title
        path: str, location for storing
        colormap: colormap, matplotlib colormap (default: plt.cm.Blues)
        name: str, name for the plot (for storing)

    Returns:
        None
    """
    cr = pd.DataFrame(classification_report(y, yhat, output_dict=True)).transpose()
    f, ax = plt.subplots()
    # annotate the number of observations per row
    cr.index = cr.index + " (" + cr.support.astype(int).astype(str).values + ")"
    # drop this column now for better plotting
    cr = cr.drop("support", axis=1)
    ax = sns.heatmap(cr, cmap=colormap, annot=True, ax=ax)
    ax.set(xlabel="Metrics", ylabel="Classes", title=f"{title} Classification Report")
    plt.tight_layout()
    save_fig(f, path, "classification_report_" + name)
    plt.close(f)


def plot_error_characteristics(
    df_errors,
    input_psms,
    tasks,
    xirt_params,
    outpath,
    plt_func=sns.boxenplot,
    min_fdr=0.0,
    max_fdr=0.01,
):  # pragma: no cover
    """Generate error characteristic boxplot.

    *in development*

    Args:
        df_errors:
        input_psms:
        tasks:
        xirt_params:
        outpath:
        plt_func:
        min_fdr:
        max_fdr:

    Returns:
        None
    """
    # %%
    # fdr filter
    df_all_info = input_psms[(input_psms["fdr"] <= max_fdr) & (input_psms["fdr"] >= min_fdr)]
    # remove duplicates
    df_all_info = df_all_info[~df_all_info["Duplicate"]]
    # merge dataf rames
    df_all_info = df_all_info.join(df_errors)
    # unify decoy annotation
    if {"isTD", "isDD"} <= set(df_all_info.columns):
        ref = np.array(["TT", "TD", "DD"])
        df_all_info["Type"] = ref[np.argmax(df_all_info[["isTT", "isTD", "isDD"]].values, axis=1)]
    else:
        logger.warning("isTD/isDD unavailable: QC reports combined Decoy rather than TD/DD.")
        ref = np.array(["TT", "Decoy"])
        df_all_info["Type"] = np.where(df_all_info["isTT"], "TT", "Decoy")
    hue_order = [label for label in ref if label in set(df_all_info["Type"])]
    # convenient ass to columns
    tasks_errors = [i + "-error" for i in tasks]
    # melt data
    df_ec_melt = df_all_info[np.concatenate([tasks_errors, ["Type"]])].melt(
        id_vars=["Type"],
    )
    df_ec_melt = df_ec_melt.rename({"Type": "PSM type"}, axis=1)
    df_ec_melt["variable"] = df_ec_melt["variable"].str.replace("-error", "")
    # counts = dict(df_all_info["isTT"].value_counts())

    # organize plotting stuff
    fracs = xirt_params["predictions"]["fractions"]
    cont = xirt_params["predictions"]["continues"]
    ntasks = len(fracs) + len(cont)

    # %%
    if (len(fracs) > 0) & (len(cont) > 0):
        f, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [len(fracs), len(cont)]})
    else:
        f, axes = plt.subplots(1, ntasks)
    if type(axes) is matplotlib.axes._axes.Axes:
        axes = [axes]

    idx = 0
    if len(fracs) > 0:
        df_ec_melt_frac = df_ec_melt[df_ec_melt["variable"].str.contains("|".join(fracs))]
        axes[idx] = plt_func(
            x="variable",
            y="value",
            hue="PSM type",
            data=df_ec_melt_frac,
            ax=axes[idx],
            hue_order=hue_order,
            palette=dict(zip(ref, targetdecoys_cm)),
        )
        pairs = (
            [((i, "TT"), (i, ref[1])) for i in fracs]
            if ("TT" in hue_order and ref[1] in hue_order)
            else []
        )
        if pairs:
            Annotator(
                axes[idx],
                pairs,
                data=df_ec_melt_frac,
                x="variable",
                y="value",
                hue="PSM type",
                hue_order=hue_order,
            ).configure(test="t-test_ind", text_format="star", loc="outside").apply_and_annotate()
        idx += 1

    if len(cont) > 0:
        df_ec_melt_cont = df_ec_melt[df_ec_melt["variable"].str.contains("|".join(cont))]
        axes[idx] = plt_func(
            x="variable",
            y="value",
            hue="PSM type",
            data=df_ec_melt_cont,
            ax=axes[idx],
            hue_order=hue_order,
            palette=dict(zip(ref, targetdecoys_cm)),
        )
        pairs = (
            [((i, "TT"), (i, ref[1])) for i in cont]
            if ("TT" in hue_order and ref[1] in hue_order)
            else []
        )
        if pairs:
            Annotator(
                axes[idx],
                pairs,
                data=df_ec_melt_cont,
                x="variable",
                y="value",
                hue="PSM type",
                hue_order=hue_order,
            ).configure(test="t-test_ind", text_format="star", loc="outside").apply_and_annotate()
        idx += 1

    for idx, ax in enumerate(axes):
        ax.axhline(0, lw=1, zorder=0, c="k")
        ax.set(xlabel="", ylabel="Observed - Predicted")
        if idx > 0:
            ax.set(ylabel="")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        sns.despine(ax=ax)
    save_fig(f, outpath, "error_characteristics")
    plt.close(f)
