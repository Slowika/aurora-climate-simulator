import sys
from unittest.mock import Mock

import fsspec
import pytest

from aurora_cs import training_checkpoint as ckpt
from components.train_aurora_cs import train_aurora_cs as training


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


def invoke_main(monkeypatch, *extra, write_check=None):
    monkeypatch.setattr(
        sys, "argv",
        ["train_aurora_cs.py", "--era5-data", "era5", "--co2-data", "co2.csv", "--model-name", "m", *extra],
    )
    monkeypatch.setattr(training, "check_writable", write_check or Mock())
    order = []
    monkeypatch.setattr(training, "configure_mlflow_auth", lambda: order.append("auth"))
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: order.append("device") or False)
    monkeypatch.setattr(
        training.pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("data loading started"))
    )
    with pytest.raises(RuntimeError, match="data loading started"):
        training.main()
    return order


def test_main_checks_blob_access_before_any_data_is_loaded(monkeypatch):
    fs = Mock()
    monkeypatch.setattr(training, "open_checkpoint_store", lambda *a, **k: (fs, "cont/prefix"))
    check = Mock()
    invoke_main(monkeypatch, "--checkpoint-prefix", "prefix", write_check=check)
    check.assert_called_once_with(fs, "cont/prefix")


def test_main_stops_at_startup_when_blob_is_not_writable(monkeypatch):
    monkeypatch.setattr(training, "open_checkpoint_store", lambda *a, **k: (Mock(), "cont/prefix"))
    monkeypatch.setattr(sys, "argv", ["t", "--era5-data", "e", "--co2-data", "c", "--model-name", "m",
                                      "--checkpoint-prefix", "p"])
    monkeypatch.setattr(training, "configure_mlflow_auth", lambda: None)
    monkeypatch.setattr(training, "check_writable", Mock(side_effect=PermissionError("denied")))
    monkeypatch.setattr(training.pd, "read_csv", Mock(side_effect=AssertionError("data was loaded")))
    with pytest.raises(PermissionError, match="denied"):
        training.main()


@pytest.mark.parametrize(
    "args,message",
    [
        (["--checkpoint-prefix", "p"], "storage account"),
        (["--checkpoint-prefix", "p", "--checkpoint-account", "a"], "storage container"),
        (["--checkpoint-prefix", "checkpoints/<MODEL_NAME>", "--checkpoint-account", "a",
          "--checkpoint-container", "c"], "unfilled placeholder"),
        (["--checkpoint-dir", "/tmp/x", "--checkpoint-prefix", "p", "--checkpoint-account", "a",
          "--checkpoint-container", "c"], "not both"),
    ],
)
def test_main_reports_bad_checkpoint_settings_as_usage_errors(monkeypatch, capsys, args, message):
    monkeypatch.setattr(
        sys, "argv", ["t", "--era5-data", "e", "--co2-data", "c", "--model-name", "m", *args]
    )
    with pytest.raises(SystemExit) as exc:
        training.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_main_without_checkpointing_skips_the_write_check(monkeypatch):
    check = Mock()
    invoke_main(monkeypatch, write_check=check)
    check.assert_not_called()
