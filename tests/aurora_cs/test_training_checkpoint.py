from unittest.mock import Mock

import fsspec
import pytest
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


def test_no_blob_location_means_no_store():
    assert ckpt.open_checkpoint_store(None, prefix=None, account="a", container="c") == (None, None)


def test_blob_store_uses_the_compute_identity_on_azure_ml(monkeypatch):
    monkeypatch.setenv("AZUREML_RUN_ID", "job")
    monkeypatch.setenv("DEFAULT_IDENTITY_CLIENT_ID", "compute-identity")
    credential, default, blob = Mock(name="mi"), Mock(name="default"), Mock(name="fs")
    monkeypatch.setattr("azure.identity.ManagedIdentityCredential", Mock(return_value=credential))
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", default)
    monkeypatch.setattr("adlfs.AzureBlobFileSystem", Mock(return_value=blob))
    fs, path = ckpt.open_checkpoint_store(prefix="/checkpoints/run/", account="acct", container="cont")
    import azure.identity
    import adlfs
    azure.identity.ManagedIdentityCredential.assert_called_once_with(client_id="compute-identity")
    default.assert_not_called()
    adlfs.AzureBlobFileSystem.assert_called_once_with(account_name="acct", credential=credential)
    assert fs is blob
    assert path == "cont/checkpoints/run"


def test_blob_store_uses_default_credentials_off_azure_ml(monkeypatch):
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    managed, default = Mock(), Mock(return_value="default-credential")
    monkeypatch.setattr("azure.identity.ManagedIdentityCredential", managed)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", default)
    monkeypatch.setattr("adlfs.AzureBlobFileSystem", Mock())
    ckpt.open_checkpoint_store(prefix="p", account="a", container="c")
    managed.assert_not_called()
    import adlfs
    adlfs.AzureBlobFileSystem.assert_called_once_with(account_name="a", credential="default-credential")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"prefix": "p", "container": "c"}, "storage account"),
        ({"prefix": "p", "account": "a"}, "storage container"),
        ({"prefix": "checkpoints/<MODEL_NAME>", "account": "a", "container": "c"}, "unfilled placeholder"),
        ({"prefix": "p", "account": "<STORAGE_ACCOUNT>", "container": "c"}, "unfilled placeholder"),
        ({"prefix": "p", "account": "a", "container": "<BLOB_CONTAINER>"}, "unfilled placeholder"),
    ],
)
def test_blob_store_rejects_incomplete_or_placeholder_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ckpt.open_checkpoint_store(**kwargs)


def test_cannot_combine_a_local_directory_with_blob_checkpoints(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        ckpt.open_checkpoint_store(str(tmp_path), prefix="p", account="a", container="c")


def test_write_check_passes_and_leaves_nothing_behind(remote):
    ckpt.check_writable(remote, "container/checkpoints/run")
    assert remote.find("container/checkpoints/run") == []


def test_write_check_fails_loudly_when_storage_refuses_writes():
    class ReadOnly(fsspec.implementations.memory.MemoryFileSystem):
        def pipe_file(self, *args, **kwargs):
            raise PermissionError("AuthorizationPermissionMismatch")

    with pytest.raises(PermissionError, match="AuthorizationPermissionMismatch"):
        ckpt.check_writable(ReadOnly(), "container/checkpoints/run")
