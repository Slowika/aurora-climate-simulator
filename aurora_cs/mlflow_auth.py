"""MLflow authentication for Azure ML jobs."""

import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["configure_mlflow_auth"]


def configure_mlflow_auth() -> None:
    """Make MLflow in an Azure ML job authenticate as the compute's user-assigned identity.

    The token Azure ML injects carries the submitter's permissions, which can expire mid-job
    (e.g. a time-limited PIM role) and then fail every later metric call with a 403. Without it,
    azureml-mlflow uses DefaultAzureCredential and refreshes managed-identity tokens as needed.
    Does nothing outside Azure ML. Call it before the first MLflow request.
    """
    if not os.environ.get("AZUREML_RUN_ID"):
        return
    client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
    if not client_id:
        raise RuntimeError(
            "Azure ML jobs require a user-assigned compute identity; "
            "DEFAULT_IDENTITY_CLIENT_ID is not set."
        )
    os.environ["AZURE_CLIENT_ID"] = client_id
    os.environ.pop("MLFLOW_TRACKING_TOKEN", None)
    logger.info(f"MLflow authentication: compute managed identity {client_id}")
