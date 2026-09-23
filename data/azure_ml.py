"""Shared Azure ML plumbing reused by every data-producer script."""

from typing import Optional

import adlfs
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential


def get_ml_client(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    managed_identity_client_id: Optional[str] = None,
) -> MLClient:
    """Build an `MLClient` for the given workspace using the ambient Azure credentials.

    `managed_identity_client_id`, if given, targets a specific user-assigned identity - needed
    when running on a compute whose only attached identity is user-assigned, since
    `DefaultAzureCredential` otherwise only resolves a compute's system-assigned identity.
    """
    return MLClient(
        DefaultAzureCredential(managed_identity_client_id=managed_identity_client_id),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


def get_blob_filesystem(
    ml_client: MLClient,
    datastore_name: str,
    managed_identity_client_id: Optional[str] = None,
) -> tuple[adlfs.AzureBlobFileSystem, str]:
    """Filesystem + container name for direct read/write access to the given blob datastore.

    For identity-based (credential-less) datastores, `datastore.credentials` carries no
    account key or SAS token, and `adlfs` would otherwise fall back to its own unscoped
    `DefaultAzureCredential()` - losing `managed_identity_client_id` and breaking on computes
    with only a user-assigned identity. Passing our own credential explicitly avoids that.
    """
    datastore = ml_client.datastores.get(datastore_name, include_secrets=True)
    credential = datastore.credentials
    account_key = getattr(credential, "account_key", None)
    sas_token = getattr(credential, "sas_token", None)
    fs = adlfs.AzureBlobFileSystem(
        account_name=datastore.account_name,
        account_key=account_key,
        sas_token=sas_token,
        credential=(
            None
            if (account_key or sas_token)
            else DefaultAzureCredential(managed_identity_client_id=managed_identity_client_id)
        ),
    )
    return fs, datastore.container_name


def next_version(ml_client: MLClient, name: str) -> str:
    """Next version string for a Data asset, `"1"` if `name` hasn't been registered yet."""
    versions = [int(asset.version) for asset in ml_client.data.list(name=name)]
    return str(max(versions) + 1) if versions else "1"


def register_data_asset(
    ml_client: MLClient,
    *,
    name: str,
    path: str,
    asset_type: str,
    description: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[dict] = None,
) -> None:
    """Register (or version-bump) a Data asset pointing at `path`."""
    ml_client.data.create_or_update(
        Data(
            name=name,
            version=version,
            path=path,
            type=asset_type,
            description=description,
            tags=tags,
        )
    )
