import os

import pytest

from aurora_cs import configure_mlflow_auth


def test_azure_job_uses_compute_identity(monkeypatch, caplog):
    monkeypatch.setenv("AZUREML_RUN_ID", "training-job")
    monkeypatch.setenv("MLFLOW_RUN_ID", "tracking-run")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "azureml://test-workspace")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "submitter-token")
    monkeypatch.setenv("AZURE_CLIENT_ID", "other-identity")
    monkeypatch.setenv("DEFAULT_IDENTITY_CLIENT_ID", "compute-identity")

    with caplog.at_level("INFO"):
        configure_mlflow_auth()
        configure_mlflow_auth()

    assert os.environ["AZURE_CLIENT_ID"] == "compute-identity"
    assert "MLFLOW_TRACKING_TOKEN" not in os.environ
    assert os.environ["MLFLOW_RUN_ID"] == "tracking-run"
    assert os.environ["MLFLOW_TRACKING_URI"] == "azureml://test-workspace"
    assert "compute-identity" in caplog.text
    assert "submitter-token" not in caplog.text


def test_local_authentication_is_unchanged(monkeypatch):
    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    monkeypatch.delenv("DEFAULT_IDENTITY_CLIENT_ID", raising=False)
    monkeypatch.setenv("MLFLOW_RUN_ID", "local-run")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "local-token")
    monkeypatch.setenv("AZURE_CLIENT_ID", "local-identity")

    configure_mlflow_auth()

    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "local-token"
    assert os.environ["AZURE_CLIENT_ID"] == "local-identity"


@pytest.mark.parametrize("client_id", [None, ""])
def test_azure_job_without_compute_identity_fails_explicitly(monkeypatch, client_id):
    monkeypatch.setenv("AZUREML_RUN_ID", "training-job")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "submitter-token")
    if client_id is None:
        monkeypatch.delenv("DEFAULT_IDENTITY_CLIENT_ID", raising=False)
    else:
        monkeypatch.setenv("DEFAULT_IDENTITY_CLIENT_ID", client_id)

    with pytest.raises(RuntimeError, match="DEFAULT_IDENTITY_CLIENT_ID"):
        configure_mlflow_auth()

    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "submitter-token"
