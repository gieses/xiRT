"""Bundled, trained prediction checkpoints."""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

MODEL_NAMES = ("multitask", "rp", "scx", "hsax", "faims")


@contextmanager
def checkpoint_path(model: str = "multitask") -> Iterator[Path]:
    """Resolve a bundled checkpoint for the lifetime of a load, including ZIP imports."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown bundled model {model!r}; choose from {', '.join(MODEL_NAMES)}.")
    with as_file(files(__package__).joinpath(f"{model}.pt")) as path:
        yield path
