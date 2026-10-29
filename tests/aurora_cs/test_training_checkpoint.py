import fsspec
import torch

from aurora_cs import training_checkpoint as ckpt


def test_loading_from_a_location_with_no_checkpoints_returns_none(remote):
    assert ckpt.load_latest_training_state("container/absent", torch.device("cpu"), remote) is None


def test_store_is_off_without_a_location():
    assert ckpt.open_checkpoint_store() == (None, None)


def test_local_store(tmp_path):
    fs, path = ckpt.open_checkpoint_store(str(tmp_path))
    assert path == str(tmp_path)
    assert isinstance(fs, fsspec.implementations.local.LocalFileSystem)
