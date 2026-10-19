"""LoRA/LoKr adapters for linear layers and GRU gate matrices.

This deliberately small variant of LyCORIS v4.0.0 LoKr uses a full first factor and
a low-rank second factor: ``delta = alpha/rank * kron(W1, W2a @ W2b)``.
It omits convolution, dropout, DoRA and LyCORIS' full-matrix heuristics.
Reconstruct checkpoints by applying the same rank/factor/alpha and target names
to the original architecture before loading its ordinary PyTorch state_dict.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


def _factor_dimension(size: int, factor: int) -> tuple[int, int]:
    """Split a positive dimension into divisors, balancing when factor is -1."""
    limit = math.isqrt(size) if factor == -1 else min(factor, math.isqrt(size))
    first = max(divisor for divisor in range(1, limit + 1) if size % divisor == 0)
    return first, size // first


class _LoKrUpdate(nn.Module):
    """Share the exact zero-initialized update without changing checkpoint keys."""

    def __init__(self, weight: torch.Tensor, rank: int, factor: int, alpha: float) -> None:
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if isinstance(factor, bool) or not isinstance(factor, int) or (factor != -1 and factor < 1):
            raise ValueError("factor must be positive or -1 (balanced factorization)")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if weight.ndim != 2:
            raise ValueError("LoKr requires a two-dimensional weight")
        self.rank, self.factor, self.alpha = rank, factor, float(alpha)
        self.out_features, self.in_features = weight.shape
        out_first, out_second = _factor_dimension(self.out_features, factor)
        in_first, in_second = _factor_dimension(self.in_features, factor)
        self.lokr_w1 = nn.Parameter(weight.new_empty((out_first, in_first)))
        self.lokr_w2_a = nn.Parameter(weight.new_empty((out_second, rank)))
        self.lokr_w2_b = nn.Parameter(weight.new_zeros((rank, in_second)))
        nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        """Return the exact dense update; gradients reach all adapter factors.

        >>> LoKrLinear(nn.Linear(4, 4)).delta_weight().count_nonzero().item()
        0
        """
        return torch.kron(self.lokr_w1, self.lokr_w2_a @ self.lokr_w2_b) * (self.alpha / self.rank)


class LoKrLinear(_LoKrUpdate):
    """Frozen Linear plus a trainable, initially zero Kronecker-product update.

    >>> layer = LoKrLinear(nn.Linear(12, 8), rank=2, factor=2, alpha=2)
    >>> layer(torch.ones(3, 12)).shape
    torch.Size([3, 8])
    """

    def __init__(self, base: nn.Linear, rank: int = 4, factor: int = 4, alpha: float = 4.0) -> None:
        """Freeze the base and preserve the established LoKr state_dict layout."""
        super().__init__(base.weight, rank, factor, alpha)
        self.base = base.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        """Return the differentiable base-plus-adapter effective weight."""
        return self.base.weight + self.delta_weight()

    @property
    def bias(self) -> torch.Tensor | None:
        """Return the original frozen bias, or None for a bias-free base."""
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the frozen linear mapping plus the trainable Kronecker update."""
        return self.base(inputs) + F.linear(inputs, self.delta_weight())


class LoRALinear(nn.Module):
    """Frozen Linear plus ``alpha / rank * B @ A``, initially exactly zero.

    >>> layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)
    >>> layer(torch.ones(5, 4)).shape
    torch.Size([5, 3])
    """

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 4.0) -> None:
        """Freeze the original layer and initialize only the low-rank branch."""
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_a = nn.Parameter(base.weight.new_empty((rank, base.in_features)))
        self.lora_b = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        """Return the differentiable low-rank update."""
        return (self.lora_b @ self.lora_a) * (self.alpha / self.rank)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the differentiable effective weight for normal Linear consumers."""
        return self.base.weight + self.delta_weight()

    @property
    def bias(self) -> torch.Tensor | None:
        """Return the unchanged frozen bias."""
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the base mapping and low-rank update."""
        return self.base(inputs) + F.linear(F.linear(inputs, self.lora_a), self.lora_b) * (
            self.alpha / self.rank
        )


class LoKrWeight(_LoKrUpdate):
    """Parametrize a frozen GRU gate matrix without replacing its recurrent math.

    >>> weight = torch.ones(12, 4)
    >>> torch.equal(LoKrWeight(weight, rank=2)(weight), weight)
    True
    """

    def __init__(
        self, weight: torch.Tensor, rank: int = 4, factor: int = 4, alpha: float = 4.0
    ) -> None:
        """Initialize the Kronecker branch using the matrix's dtype and device."""
        super().__init__(weight, rank, factor, alpha)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        """Return the original gate matrix plus its trainable update."""
        return original + self.delta_weight()


class LoRAWeight(nn.Module):
    """Add a zero-initialized low-rank update to a frozen recurrent gate matrix."""

    def __init__(self, weight: torch.Tensor, rank: int = 4, alpha: float = 4.0) -> None:
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if weight.ndim != 2:
            raise ValueError("LoRA requires a two-dimensional weight")
        self.rank, self.alpha = rank, float(alpha)
        self.lora_a = nn.Parameter(weight.new_empty((rank, weight.shape[1])))
        self.lora_b = nn.Parameter(weight.new_zeros((weight.shape[0], rank)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        """Return the trainable update in the original matrix layout."""
        return self.lora_b @ self.lora_a * (self.alpha / self.rank)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        """Preserve the original matrix and recurrent computation."""
        return original + self.delta_weight()


def apply_lokr(
    model: nn.Module,
    target_names: Sequence[str] | None = None,
    *,
    rank: int = 4,
    factor: int = 4,
    alpha: float = 4.0,
) -> tuple[str, ...]:
    """Replace selected Linear descendants in-place, returning their exact paths.

    ``None`` selects all Linear descendants. Explicit targets must be exact
    named_modules paths. Only replaced base layers are frozen; callers can freeze
    the whole backbone first. Repeated calls skip already wrapped descendants.

    >>> heads = nn.Sequential(nn.Linear(12, 8), nn.ReLU())
    >>> apply_lokr(heads, rank=2, factor=2, alpha=2)
    ('0',)
    """
    modules = dict(model.named_modules())
    wrapped = {
        name for name, module in modules.items() if isinstance(module, (LoKrLinear, LoRALinear))
    }
    candidates = {
        name: module
        for name, module in modules.items()
        if name
        and isinstance(module, nn.Linear)
        and not any(name == path or name.startswith(path + ".") for path in wrapped)
    }
    if isinstance(target_names, str):
        raise TypeError("target_names must be a sequence of module paths, not a string")
    targets = list(candidates) if target_names is None else list(dict.fromkeys(target_names))
    invalid = set(targets) - candidates.keys() - wrapped
    if invalid:
        raise ValueError(f"LoKr targets are not Linear descendants: {sorted(invalid)}")
    if any(isinstance(modules[name], LoRALinear) for name in targets if name in wrapped):
        raise ValueError("Cannot reinterpret an existing LoRA layer as LoKr")
    replacements = {
        name: LoKrLinear(candidates[name], rank, factor, alpha)
        for name in targets
        if name not in wrapped
    }
    for name, replacement in replacements.items():
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, replacement)
    return tuple(targets)


def apply_lora(
    model: nn.Module,
    target_names: Sequence[str] | None = None,
    *,
    rank: int = 4,
    alpha: float = 4.0,
) -> tuple[str, ...]:
    """Inject LoRA into exact Linear paths; freeze only the replaced base layers.

    >>> model = nn.Sequential(nn.Linear(4, 3))
    >>> apply_lora(model, rank=2)
    ('0',)
    """
    if isinstance(target_names, str):
        raise TypeError("target_names must be a sequence of module paths, not a string")
    modules = dict(model.named_modules())
    wrapped = {
        name for name, module in modules.items() if isinstance(module, (LoKrLinear, LoRALinear))
    }
    candidates = {
        name: module
        for name, module in modules.items()
        if name
        and isinstance(module, nn.Linear)
        and not any(name == path or name.startswith(path + ".") for path in wrapped)
    }
    targets = list(candidates) if target_names is None else list(dict.fromkeys(target_names))
    invalid = set(targets) - candidates.keys() - wrapped
    if invalid:
        raise ValueError(f"LoRA targets are not Linear descendants: {sorted(invalid)}")
    if any(isinstance(modules[name], LoKrLinear) for name in targets if name in wrapped):
        raise ValueError("Cannot reinterpret an existing LoKr layer as LoRA")
    replacements = {
        name: LoRALinear(candidates[name], rank, alpha) for name in targets if name not in wrapped
    }
    for name, replacement in replacements.items():
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, replacement)
    return tuple(targets)


def apply_lokr_gru(
    model: nn.Module, *, rank: int = 4, factor: int = 4, alpha: float = 4.0
) -> tuple[str, ...]:
    """Adapt input/hidden GRU gate matrices, preserving layers and directions.

    Biases and other modules are not frozen here: freeze a pretrained model first
    when performing parameter-efficient fine-tuning. Reconstruct the same
    parametrizations before loading an adapted checkpoint.

    >>> gru = nn.GRU(4, 8, batch_first=True)
    >>> apply_lokr_gru(gru, rank=2)
    ('weight_ih_l0', 'weight_hh_l0')
    """
    return _apply_gru_adapters(model, "lokr", rank=rank, factor=factor, alpha=alpha)


def apply_lora_gru(model: nn.Module, *, rank: int = 4, alpha: float = 4.0) -> tuple[str, ...]:
    """Parametrize every GRU input/hidden gate matrix with a zero-initialized LoRA update."""
    return _apply_gru_adapters(model, "lora", rank=rank, factor=4, alpha=alpha)


def _apply_gru_adapters(
    model: nn.Module, kind: str, *, rank: int, factor: int, alpha: float
) -> tuple[str, ...]:
    targets: list[tuple[str, nn.GRU, str, nn.Module]] = []
    existing: list[str] = []
    for path, module in model.named_modules():
        if not isinstance(module, nn.GRU):
            continue
        for name in module._flat_weights_names:
            if not name.startswith(("weight_ih_", "weight_hh_")):
                continue
            full_name = f"{path}.{name}" if path else name
            if parametrize.is_parametrized(module, name):
                registered = module.get_submodule(f"parametrizations.{name}")
                expected = LoKrWeight if kind == "lokr" else LoRAWeight
                if not isinstance(next(registered.children()), expected):
                    raise ValueError(f"GRU matrix already has another parametrization: {full_name}")
                existing.append(full_name)
            else:
                weight = getattr(module, name)
                update: nn.Module = (
                    LoKrWeight(weight, rank, factor, alpha)
                    if kind == "lokr"
                    else LoRAWeight(weight, rank, alpha)
                )
                targets.append((full_name, module, name, update))
    if not targets and not existing:
        raise ValueError("No GRU gate weight matrices found")
    for _, module, name, update in targets:
        getattr(module, name).requires_grad_(False)
        parametrize.register_parametrization(module, name, update)
    return tuple(existing + [name for name, _, _, _ in targets])
