"""Checkpoints work on any fsspec filesystem, not only local directories.

An in-memory filesystem stands in for blob storage here: no real directories, flat names.
"""

import fsspec
import pytest
import torch

from aurora_cs import training_checkpoint as ckpt
from components.train_aurora_cs import train_aurora_cs as training
from test_training_checkpoint import EXAMPLES, FINGERPRINT, assert_same, build_model, get_example


@pytest.fixture(autouse=True)
def quiet_mlflow(monkeypatch):
    monkeypatch.setattr(training.mlflow, "log_metric", lambda *a, **k: None)


def run(epochs, fs, directory="container/checkpoints/run", every=2, fail_after_calls=None):
    model = build_model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    calls = 0

    def counting(cur_t):
        nonlocal calls
        calls += 1
        if fail_after_calls is not None and calls > fail_after_calls:
            raise RuntimeError("simulated crash")
        return get_example(cur_t)

    training.train_loop(
        model, optimizer, EXAMPLES, counting, epochs, checkpoint_dir=directory,
        checkpoint_fs=fs, checkpoint_every=every, fingerprint=FINGERPRINT,
    )
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("crash_after", [3, 5, 8])
def test_resume_from_blob_style_storage_matches_uninterrupted_training(remote, crash_after):
    expected = run(epochs=3, fs=remote, directory="container/uninterrupted")
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(epochs=3, fs=remote, fail_after_calls=crash_after)
    assert any(name.endswith(".done") for name in remote.ls("container/checkpoints/run", detail=False))
    assert_same(run(epochs=3, fs=remote), expected)


def test_only_the_newest_checkpoints_are_kept_remotely(remote):
    run(epochs=3, fs=remote, every=1)
    names = sorted(n.rsplit("/", 1)[-1] for n in remote.ls("container/checkpoints/run", detail=False))
    assert len([n for n in names if n.endswith(".done")]) == ckpt.KEEP
    assert len([n for n in names if n.endswith(".pt")]) == ckpt.KEEP


def test_a_checkpoint_without_its_marker_is_ignored_remotely(remote):
    run(epochs=1, fs=remote, every=1)
    directory = "container/checkpoints/run"
    newest = max(n for n in remote.ls(directory, detail=False) if n.endswith(".done"))
    remote.rm(newest)
    state = ckpt.load_latest_training_state(directory, torch.device("cpu"), remote)
    assert state["global_step"] == int(max(n for n in remote.ls(directory, detail=False) if n.endswith(".done"))[-14:-5])


def test_a_marker_without_its_data_is_ignored_remotely(remote):
    run(epochs=1, fs=remote, every=1)
    directory = "container/checkpoints/run"
    newest = max(n for n in remote.ls(directory, detail=False) if n.endswith(".pt"))
    remote.rm(newest)
    state = ckpt.load_latest_training_state(directory, torch.device("cpu"), remote)
    assert state["global_step"] < int(newest[-12:-3])


def test_loading_from_a_location_with_no_checkpoints_returns_none(remote):
    assert ckpt.load_latest_training_state("container/absent", torch.device("cpu"), remote) is None


def test_local_directories_still_work_without_a_filesystem(tmp_path):
    model = build_model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    ckpt.save_training_state(
        tmp_path / "new" / "dir", model, optimizer, epoch=0, position=1, global_step=1,
        epoch_loss=0.0, fingerprint={},
    )
    state = ckpt.load_latest_training_state(tmp_path / "new" / "dir", torch.device("cpu"))
    assert state["global_step"] == 1


def test_store_is_off_without_a_location():
    assert ckpt.open_checkpoint_store() == (None, None)


def test_local_store(tmp_path):
    fs, path = ckpt.open_checkpoint_store(str(tmp_path))
    assert path == str(tmp_path)
    assert isinstance(fs, fsspec.implementations.local.LocalFileSystem)
