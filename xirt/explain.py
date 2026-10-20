"""Residue-level integrated gradients for native recurrent xiRT models.

References:
    Sundararajan, Taly and Yan (2017), Axiomatic Attribution for Deep Networks.
    https://proceedings.mlr.press/v70/sundararajan17a.html
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Signed contributions and their completeness check in the target's units.

    ``residues`` includes separate terminus rows when the tokenizer uses them.
    ``numeric`` records values, training-mean baselines and contributions.
    ``summary`` preserves sample order; its completeness error is prediction
    minus baseline minus the sum of *all* token and numeric contributions.

    The embedding baseline is zero at unchanged sequence lengths. It is not
    an empty peptide, a physical counterfactual or evidence of causality.
    """

    residues: pd.DataFrame
    numeric: pd.DataFrame
    summary: pd.DataFrame
    task: str
    unit: str


def integrated_gradients(
    model: Any,
    inputs: tuple[Tensor, ...],
    *,
    task: str | None = None,
    index: Any = None,
    tokens: list[list[list[str]]] | None = None,
    steps: int = 64,
) -> Explanation:
    """Integrate a scalar head along an embedding/precursor path using public PyTorch APIs.

    Gauss-Legendre integration holds token IDs, peptide lengths and padding
    masks fixed. No derivatives of integer token IDs are taken. Numeric
    features move jointly from their fitted training means to their values.
    Model mode, hooks and existing parameter gradients are preserved.
    """
    if model.route != "recurrent" or not isinstance(model.embedding, nn.Embedding):
        raise ValueError(
            "Residue integrated gradients requires a native recurrent embedding model."
        )
    if not isinstance(steps, int) or isinstance(steps, bool) or not 2 <= steps <= 512:
        raise ValueError("steps must be an integer between 2 and 512.")
    if task is None:
        if len(model.tasks) != 1:
            raise ValueError("Select a task when explaining a multi-task model.")
        task = model.tasks[0]
    if task not in model.tasks:
        raise ValueError(f"Unknown prediction task: {task}")
    if model.params["output"][f"{task}-dimension"] != 1 or model.params["output"][
        f"{task}-activation"
    ] not in {"linear", "relu"}:
        raise ValueError("Residue explanations currently require a scalar continuous task.")
    branches = 2 if model.siamese else 1
    size = len(inputs[0])
    if not size:
        raise ValueError("Cannot explain an empty batch.")
    if index is not None and len(index) != size:
        raise ValueError("Explanation index must match the number of samples.")
    if tokens is not None and (
        len(tokens) != branches
        or any(len(part) != size for part in tokens)
        or any(
            len(tokens[branch][row]) != int((inputs[branch][row] != 0).sum())
            for branch in range(branches)
            for row in range(size)
        )
    ):
        raise ValueError("Token labels must match the nonpadding tokens in every peptide.")
    scale, offset, unit = model.target_transform(task)
    if not np.isfinite([scale, offset]).all() or scale <= 0:
        raise ValueError("Target transforms must have a finite positive scale and finite offset.")
    mode = model.training
    modes = {module: module.training for module in model.modules()}
    activity = model.activity
    handle = None
    totals: list[Tensor] = []
    actual_embeddings: list[Tensor] = []
    point: list[float] = [1.0]
    captured: list[Tensor] = []

    def embedding_path(module: nn.Module, arguments: Any, output: Tensor) -> Tensor:
        value = (output.detach() * point[0]).requires_grad_(True)
        captured.append(value)
        return value

    try:
        model.eval()
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.backends.cudnn.flags(enabled=False),
        ):
            values = tuple(value.to(model.device).clone() for value in inputs)
            numeric = values[-1] if model.numeric_features else None
            if numeric is not None and not bool(model.numeric_fitted):
                raise RuntimeError("Numeric normalization is not fitted; train or load a model.")
            reference: Tensor = (
                model.numeric_mean.detach()
                if numeric is not None
                else torch.empty(0, device=model.device)
            )
            handle = model.embedding.register_forward_hook(embedding_path)

            def forward(alpha: float) -> tuple[Tensor, list[Tensor]]:
                point[0] = alpha
                captured.clear()
                context = (
                    (reference + alpha * (numeric - reference)).detach().requires_grad_(True)
                    if numeric is not None
                    else None
                )
                result = model(values[0], values[1] if model.siamese else None, numeric=context)[
                    task
                ].reshape(-1)
                leaves = list(captured) + ([context] if context is not None else [])
                if len(captured) != branches:
                    raise RuntimeError(
                        "Unexpected shared-embedding calls while explaining the model."
                    )
                return result, leaves

            prediction, leaves = forward(1.0)
            actual_embeddings = [value.detach() for value in leaves[:branches]]
            totals = [torch.zeros_like(value) for value in leaves]
            baseline, _ = forward(0.0)
            nodes, weights = np.polynomial.legendre.leggauss(steps)
            for alpha, weight in zip((nodes + 1) / 2, weights / 2, strict=True):
                output, leaves = forward(float(alpha))
                gradients = torch.autograd.grad(output.sum(), leaves, allow_unused=True)
                for total, gradient in zip(totals, gradients, strict=True):
                    if gradient is not None:
                        total.add_(gradient.detach(), alpha=float(weight))
            token_contributions = [
                (gradient * embedding).sum(-1).detach().cpu().numpy() * scale
                for gradient, embedding in zip(totals[:branches], actual_embeddings, strict=True)
            ]
            numeric_contributions = (
                (totals[-1] * (numeric - reference)).detach().cpu().numpy() * scale
                if numeric is not None
                else np.empty((size, 0))
            )
            predicted = prediction.detach().cpu().numpy() * scale + offset
            baseline_values = baseline.detach().cpu().numpy() * scale + offset
    finally:
        if handle is not None:
            handle.remove()
        model.train(mode)
        for module, training in modes.items():
            module.training = training
        model.activity = activity
    alphabet = model.params.get("_alphabet", [])
    vocabulary = {i + 1: token for i, token in enumerate(alphabet)}
    tutorial = model.params.get("_preprocessing", {})
    if tutorial.get("schema") == "xirt-api-tutorial-v1":
        vocabulary = {int(value): token for token, value in tutorial["vocabulary"].items()}
    residue_rows = []
    for branch in range(branches):
        for row in range(size):
            ids = inputs[branch][row].cpu().numpy()
            ids = ids[ids != 0]
            labels = (
                tokens[branch][row]
                if tokens is not None
                else [vocabulary.get(int(token), f"token:{token}") for token in ids]
            )
            position = 0
            for token_index, (identifier, label) in enumerate(zip(ids, labels, strict=True)):
                terminus = label in {"H-", "-OH"}
                if not terminus:
                    position += 1
                residue_rows.append(
                    {
                        "sample": row,
                        "peptide": branch + 1,
                        "position": position + (label == "-OH"),
                        "token": label,
                        "model_token": vocabulary.get(int(identifier), f"token:{identifier}"),
                        "kind": "terminus" if terminus else "residue",
                        "crosslinked": label.startswith("cl") or label.endswith("*"),
                        "attribution": float(token_contributions[branch][row, token_index]),
                    }
                )
    numeric_rows = []
    for row in range(size):
        for column, name in enumerate(model.numeric_features):
            numeric_rows.append(
                {
                    "sample": row,
                    "feature": name,
                    "value": float(inputs[-1][row, column]),
                    "baseline": float(model.numeric_mean[column]),
                    "attribution": float(numeric_contributions[row, column]),
                }
            )
    attribution_sum = np.array(
        [
            sum(values[row].sum() for values in token_contributions)
            + numeric_contributions[row].sum()
            for row in range(size)
        ]
    )
    error = predicted - baseline_values - attribution_sum
    if not np.isfinite(error).all():
        raise FloatingPointError("Nonfinite residue attribution or completeness error.")
    if (np.abs(error) > np.maximum(0.01, np.abs(predicted - baseline_values) * 0.05)).any():
        logger.warning(
            "Integrated-gradient completeness error is large; increase steps and inspect convergence."
        )
    return Explanation(
        residues=pd.DataFrame(residue_rows),
        numeric=pd.DataFrame(
            numeric_rows, columns=["sample", "feature", "value", "baseline", "attribution"]
        ),
        summary=pd.DataFrame(
            {
                "prediction": predicted,
                "baseline_prediction": baseline_values,
                "attribution_sum": attribution_sum,
                "completeness_error": error,
            },
            index=index,
        ),
        task=task,
        unit=unit,
    )
