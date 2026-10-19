"""Numerically stable task losses and explicit optimizer parameter routing."""

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def task_loss(
    logits: Tensor, target: Tensor, name: str, gamma: float = 2.0, alpha: float | None = None
) -> Tensor:
    """Compute a scalar task loss from raw logits and shape-matched targets.

    ``binary_focal`` supports ordinal multi-label targets; categorical focal
    supports one-hot targets. Gamma zero reproduces ordinary cross entropy.
    Alpha, when provided, is a binary positive-class weight only.

    Example:
        >>> task_loss(torch.zeros(2, 1), torch.ones(2, 1), "mse").item()
        1.0
    """
    if target.shape != logits.shape:
        raise ValueError(f"Target shape {target.shape} differs from output {logits.shape}.")
    if gamma < 0 or (alpha is not None and not 0 <= alpha <= 1):
        raise ValueError("Focal gamma must be nonnegative and alpha in [0, 1].")
    if name == "mse":
        return F.mse_loss(logits, target)
    if name == "mae":
        return F.l1_loss(logits, target)
    if name == "huber":
        return F.huber_loss(logits, target)
    if name in {"binary_crossentropy", "binary_focal"}:
        if torch.any((target < 0) | (target > 1)):
            raise ValueError("Binary targets must be in [0, 1].")
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if name == "binary_focal":
            p = torch.sigmoid(logits)
            pt = p * target + (1 - p) * (1 - target)
            loss = loss * (1 - pt).pow(gamma)
            if alpha is not None:
                loss = loss * (alpha * target + (1 - alpha) * (1 - target))
        return loss.mean()
    if name in {"categorical_crossentropy", "categorical_focal", "ordered"}:
        if torch.any(target < 0) or not torch.allclose(
            target.sum(-1), torch.ones_like(target[:, 0]), atol=1e-5
        ):
            raise ValueError("Categorical targets must be probability distributions.")
        loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1)
        if name == "categorical_focal":
            pt = (target * F.softmax(logits, dim=-1)).sum(-1)
            loss = loss * (1 - pt).pow(gamma)
        if name == "ordered":
            distance = (target.argmax(-1) - logits.argmax(-1)).abs()
            loss = loss * (1 + distance / (logits.shape[-1] - 1))
        return loss.mean()
    raise ValueError(f"Unsupported loss: {name}")


def make_optimizers(model: nn.Module, options: dict[str, Any]) -> list[torch.optim.Optimizer]:
    """Build optimizers, assigning each trainable parameter exactly once.

    Muon receives hidden two-dimensional weights only; embeddings, prediction
    layers and vector parameters use AdamW. Lion is an optional dependency.

    Example:
        >>> opts = make_optimizers(nn.Linear(4, 1), {"optimizer": "adamw",
        ...                                       "learningrate": 0.001})
        >>> type(opts[0]).__name__
        'AdamW'
    """
    name = options["optimizer"].lower()
    lr = options["learningrate"]
    wd = options.get("weight_decay", 0.0)
    named = [(key, p) for key, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    if not params:
        raise ValueError("No trainable parameters.")
    ratio = options.get("adapter_learning_rate_ratio", 1.0)
    if not 0 < ratio < float("inf"):
        raise ValueError("adapter_learning_rate_ratio must be finite and positive.")
    adapters = [p for key, p in named if "lora_" in key or "lokr_" in key]
    adapter_ids = {id(p) for p in adapters}

    def groups(parameters: list[nn.Parameter]) -> list[dict[str, Any]]:
        ordinary = [p for p in parameters if id(p) not in adapter_ids]
        adapted = [p for p in parameters if id(p) in adapter_ids]
        return ([{"params": ordinary, "lr": lr}] if ordinary else []) + (
            [{"params": adapted, "lr": lr * ratio}] if adapted else []
        )

    if name == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError as exc:
            raise ImportError(
                "Lion requires the experiments extra: uv sync --extra experiments."
            ) from exc
        return [Lion(groups(params), lr=lr, weight_decay=wd)]
    if name == "muon":
        matrix: list[nn.Parameter] = []
        other: list[nn.Parameter] = []
        for key, p in model.named_parameters():
            if not p.requires_grad:
                continue
            hidden = (
                p.ndim == 2
                and not isinstance(model, nn.Linear)
                and id(p) not in adapter_ids
                and not any(part in key for part in ("embedding", "embed.", "output", "regressor"))
            )
            (matrix if hidden else other).append(p)
        opts: list[torch.optim.Optimizer] = []
        if matrix:
            opts.append(
                torch.optim.Muon(matrix, lr=lr, weight_decay=wd, adjust_lr_fn="match_rms_adamw")
            )
        if other:
            opts.append(torch.optim.AdamW(groups(other), lr=lr, weight_decay=wd))
        return opts
    classes = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "nadam": torch.optim.NAdam,
        "rmsprop": torch.optim.RMSprop,
        "rmsprob": torch.optim.RMSprop,
    }
    if name not in classes:
        raise ValueError(f"Unsupported optimizer: {name}")
    return [classes[name](groups(params), lr=lr, weight_decay=wd)]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch; optionally require deterministic kernels.

    Example:
        >>> seed_everything(42)
    """
    import os
    import random

    import numpy as np

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
