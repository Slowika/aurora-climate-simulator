"""Save/load an `AuroraCS` as a weights file plus a JSON config, so loading never unpickles code."""

import json
from pathlib import Path

import torch

from aurora_cs.model import AuroraCS

__all__ = ["save_checkpoint", "load_checkpoint"]

WEIGHTS_FILE = "weights.pt"
CONFIG_FILE = "config.json"


def save_checkpoint(model: AuroraCS, config: dict, directory: str | Path) -> None:
    """Write `model`'s weights (as CPU tensors) and the JSON-serialisable `AuroraCS(**config)`
    kwargs it was built from."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONFIG_FILE).write_text(json.dumps(config))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, directory / WEIGHTS_FILE)


def load_checkpoint(directory: str | Path) -> AuroraCS:
    """Rebuild an `AuroraCS` from a directory written by `save_checkpoint` (on CPU)."""
    directory = Path(directory)
    raw = json.loads((directory / CONFIG_FILE).read_text())
    config = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    model = AuroraCS(**config)
    model.load_state_dict(torch.load(directory / WEIGHTS_FILE, map_location="cpu", weights_only=True))
    return model
