"""A registered AuroraCS must round-trip through the model registry via weights + config.

The rollout component rebuilds the model with `aurora_cs.load_checkpoint` from the downloaded
registry artifact; nothing is unpickled, so no training-time code needs to travel with it.
"""

import mlflow
import pytest
import torch

from aurora_cs import AuroraCS, load_checkpoint
from train_aurora_cs import register_model

TINY_CONFIG = dict(
    encoder_depths=(2, 2),
    encoder_num_heads=(2, 2),
    decoder_depths=(2, 2),
    decoder_num_heads=(2, 2),
    embed_dim=32,
    num_heads=2,
    patch_size=4,
    max_history_size=1,
    use_lora=False,
)


@pytest.mark.parametrize(
    "train_device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))],
)
def test_registered_model_round_trips(tmp_path, train_device):
    """A model registered in MLflow loads back on the CPU with identical weights."""
    model = AuroraCS(**TINY_CONFIG).to(train_device)
    # Make the zero-initialised adapter heads non-trivial so the comparison is meaningful.
    for head in model.block_heads:
        torch.nn.init.normal_(head.weight)

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment("t", artifact_location=str(tmp_path / "artifacts"))
    with mlflow.start_run(experiment_id=experiment_id):
        register_model(model, TINY_CONFIG, "aurora-cs-test")

    model_dir = mlflow.artifacts.download_artifacts(artifact_uri="models:/aurora-cs-test/1")
    loaded = load_checkpoint(model_dir)

    assert next(model.parameters()).device.type == train_device  # registering must not move it
    assert next(loaded.parameters()).device.type == "cpu"
    expected = model.state_dict()
    actual = loaded.state_dict()
    assert expected.keys() == actual.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value.cpu())
