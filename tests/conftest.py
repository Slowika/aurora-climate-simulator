import fsspec
import pytest


@pytest.fixture
def remote():
    """An in-memory filesystem standing in for blob storage: no real directories, flat names."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs[:] = [""]
    yield fs
    fs.store.clear()
