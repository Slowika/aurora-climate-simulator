"""The rollout must authenticate MLflow as the compute identity, like training does.

A rollout of thousands of steps logs metrics throughout, so a submitter-scoped token that
expires partway (e.g. a time-limited PIM role) would fail it with a 403 mid-run.
"""

import os
import sys

import pytest

import aurora_cs
from components.rollout_aurora_cs import rollout_aurora_cs as rollout


class Stop(Exception):
    pass


def run_until_first_mlflow_request(monkeypatch, on_request):
    def first_request(**kwargs):
        on_request()
        raise Stop

    monkeypatch.setattr(sys, "argv", [
        "r", "--initial-condition", "x", "--forcing-trajectory", "y", "--model-id", "m/1",
        "--num-steps", "1", "--prefix", "p", "--asset-name", "a", "--subscription-id", "s",
        "--resource-group", "g", "--workspace-name", "w",
    ])
    monkeypatch.setattr(rollout.mlflow.artifacts, "download_artifacts", first_request)
    with pytest.raises(Stop):
        rollout.main()


def test_rollout_and_training_share_one_implementation():
    from components.train_aurora_cs import train_aurora_cs as training

    assert rollout.configure_mlflow_auth is aurora_cs.configure_mlflow_auth
    assert training.configure_mlflow_auth is aurora_cs.configure_mlflow_auth


def test_auth_is_configured_before_the_first_mlflow_request(monkeypatch):
    calls = []
    monkeypatch.setattr(rollout, "configure_mlflow_auth", lambda: calls.append("auth"))
    run_until_first_mlflow_request(monkeypatch, lambda: calls.append("mlflow"))
    assert calls == ["auth", "mlflow"]


def test_the_submitter_token_is_gone_by_the_first_mlflow_request_on_azure_ml(monkeypatch):
    monkeypatch.setenv("AZUREML_RUN_ID", "job")
    monkeypatch.setenv("DEFAULT_IDENTITY_CLIENT_ID", "compute-identity")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "submitter-token")
    seen = {}
    run_until_first_mlflow_request(
        monkeypatch,
        lambda: seen.update(token=os.environ.get("MLFLOW_TRACKING_TOKEN"), client=os.environ.get("AZURE_CLIENT_ID")),
    )
    assert seen == {"token": None, "client": "compute-identity"}
