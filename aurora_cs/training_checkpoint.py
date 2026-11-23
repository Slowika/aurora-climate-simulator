"""Resumable training state for `train_aurora_cs`.

Each checkpoint is `ckpt-<step>.pt` plus a small `ckpt-<step>.done` marker written last, so a
checkpoint interrupted mid-write is never picked up. Files are never overwritten. Only the
trainable weights are stored: the wrapped Aurora is frozen and reloaded from its pretrained
checkpoint.

Checkpoints live on any fsspec filesystem: a local directory, or Azure blob storage reached as
the compute's managed identity (see `open_checkpoint_store`). Writing to blob storage directly,
rather than through a registered job output, means nothing at the end of the job depends on the
submitter's permissions.
"""

import logging
import os
import re
import tempfile
from pathlib import Path

import fsspec
import torch

logger = logging.getLogger(__name__)

KEEP = 2
_DONE = re.compile(r"ckpt-(\d+)\.done")


def _stem(step: int) -> str:
    return f"ckpt-{step:09d}"


def _is_local(fs) -> bool:
    return isinstance(fs, fsspec.implementations.local.LocalFileSystem)


def open_checkpoint_store(
    directory: str | None = None,
    *,
    prefix: str | None = None,
    account: str | None = None,
    container: str | None = None,
):
    """`(filesystem, path)` for the checkpoints, or `(None, None)` if checkpointing is off.

    Give either a local `directory`, or blob storage as `prefix` + `account` + `container`. On
    Azure ML the blob filesystem authenticates as the compute's user-assigned identity
    (`DEFAULT_IDENTITY_CLIENT_ID`), as the fetch jobs do; elsewhere it uses the default Azure
    credential chain.
    """
    if directory and prefix:
        raise ValueError("Give either a local checkpoint directory or a blob prefix, not both.")
    if directory:
        return fsspec.filesystem("file"), str(directory)
    if not prefix:
        return None, None
    for name, value in (("account", account), ("container", container), ("prefix", prefix)):
        if not value:
            raise ValueError(f"Blob checkpoints need a storage {name}.")
        if "<" in value or ">" in value:
            raise ValueError(f"Checkpoint {name} '{value}' is an unfilled placeholder.")

    import adlfs
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

    if os.environ.get("AZUREML_RUN_ID"):
        credential = ManagedIdentityCredential(
            client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID") or None
        )
    else:
        credential = DefaultAzureCredential()
    fs = adlfs.AzureBlobFileSystem(account_name=account, credential=credential)
    return fs, f"{container}/{prefix.strip('/')}"


def check_writable(fs, directory: str) -> None:
    """Fail now, not hours in, if checkpoints cannot be written and removed."""
    probe = f"{directory}/.write-check"
    if _is_local(fs):
        fs.makedirs(directory, exist_ok=True)
    fs.pipe_file(probe, b"ok")
    fs.rm(probe)


def save_training_state(
    directory,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    position: int,
    global_step: int,
    epoch_loss: float,
    fingerprint: dict,
    fs=None,
) -> None:
    """Save the state after `position` examples of `epoch` (`position` may equal the epoch's
    length, meaning the epoch is complete but not yet wrapped up)."""
    fs = fs or fsspec.filesystem("file")
    directory = str(directory)
    if _is_local(fs):
        fs.makedirs(directory, exist_ok=True)
    state = {
        "model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "position": position,
        "global_step": global_step,
        "epoch_loss": epoch_loss,
        "fingerprint": fingerprint,
    }
    stem = _stem(global_step)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "state.pt"
        torch.save(state, local)
        fs.put_file(str(local), f"{directory}/{stem}.pt")
    fs.pipe_file(f"{directory}/{stem}.done", b"done")
    for old in sorted(_complete_steps(fs, directory))[:-KEEP]:
        for suffix in (".done", ".pt"):
            try:
                fs.rm(f"{directory}/{_stem(old)}{suffix}")
            except Exception as e:  # pruning is housekeeping; never stop training for it
                logger.warning(f"Could not remove old checkpoint {_stem(old)}{suffix}: {e}")


def _complete_steps(fs, directory: str) -> list[int]:
    try:
        names = {entry.rsplit("/", 1)[-1] for entry in fs.ls(directory, detail=False)}
    except FileNotFoundError:
        return []
    return [
        int(m.group(1))
        for name in names
        if (m := _DONE.fullmatch(name)) and f"{_stem(int(m.group(1)))}.pt" in names
    ]


def load_latest_training_state(directory, device: torch.device, fs=None) -> dict | None:
    """The newest complete checkpoint in `directory`, or `None` if there is none."""
    fs = fs or fsspec.filesystem("file")
    directory = str(directory)
    steps = _complete_steps(fs, directory)
    if not steps:
        return None
    path = f"{directory}/{_stem(max(steps))}.pt"
    logger.info(f"Loading training state from {path}")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "state.pt"
        fs.get_file(path, str(local))
        return torch.load(local, map_location=device, weights_only=True)


def restore_training_state(
    state: dict, model: torch.nn.Module, optimizer: torch.optim.Optimizer, fingerprint: dict
) -> None:
    """Load `state` into `model` and `optimizer`, refusing a checkpoint from a different setup."""
    differing = sorted(k for k in fingerprint.keys() | state["fingerprint"].keys()
                       if fingerprint.get(k) != state["fingerprint"].get(k))
    if differing:
        raise RuntimeError(
            f"Checkpoint was made with a different setup ({', '.join(differing)}). "
            "Use a different checkpoint path to start fresh."
        )
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state["model"].keys():
        raise RuntimeError("Checkpoint trainable parameters do not match the model.")
    with torch.no_grad():
        for name, value in state["model"].items():
            params[name].copy_(value)
    optimizer.load_state_dict(state["optimizer"])
