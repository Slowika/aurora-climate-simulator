"""Zero-init identity check for `AuroraCS`.

At construction, every per-block adapter head is zero-initialised, so the adapters
`AuroraCS.forward` builds are all-zero `Swin3DBlockAdapter`s. Per the upstream fork's own
tests, all-zero adapters are bit-for-bit equivalent to omitting `backbone_adapters` entirely.
This test checks that invariant holds through the `AuroraCS` wrapper, regardless of the
forcing value.
"""

from datetime import datetime

import pytest
import torch
from aurora import Batch, Metadata

from aurora_cs.model import AuroraCS


@pytest.fixture
def tiny_batch() -> Batch:
    torch.manual_seed(0)
    h = w = 16
    b, t, c = 1, 1, 3
    lat = torch.linspace(90, -90, h)
    lon = torch.linspace(0, 360, w + 1)[:-1]
    return Batch(
        surf_vars={name: torch.randn(b, t, h, w) for name in ("2t", "10u", "10v", "msl")},
        static_vars={name: torch.randn(h, w) for name in ("lsm", "z", "slt")},
        atmos_vars={name: torch.randn(b, t, c, h, w) for name in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(datetime(2020, 1, 1),),
            atmos_levels=(1000, 500, 100),
        ),
    )


@pytest.fixture
def tiny_model() -> AuroraCS:
    torch.manual_seed(0)
    model = AuroraCS(
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
    # Aurora zero-inits each block's AdaLN modulation, which silences every block until trained;
    # randomise it so the tiny backbone behaves like a pretrained one and adapters can matter.
    with torch.no_grad():
        for name, param in model.aurora.named_parameters():
            if "ln_modulation" in name and param.dim() == 2:
                torch.nn.init.normal_(param, std=0.1)
    model.eval()
    return model


def _surf_atmos(pred: Batch) -> list[torch.Tensor]:
    return [*pred.surf_vars.values(), *pred.atmos_vars.values()]


def _max_abs_diff(a: Batch, b: Batch) -> float:
    return max((x - y).abs().max().item() for x, y in zip(_surf_atmos(a), _surf_atmos(b)))


def test_zero_init_adapters_are_a_no_op(tiny_model: AuroraCS, tiny_batch: Batch):
    """Verifies that an AuroraCS model with a no-op conditioning network behaves identically
    to the baseline Aurora."""

    with torch.no_grad():
        conditioned = tiny_model(tiny_batch, forcing=417.0)
        baseline = tiny_model.aurora(tiny_batch)

    for name in conditioned.surf_vars:
        assert torch.equal(conditioned.surf_vars[name], baseline.surf_vars[name])
    for name in conditioned.atmos_vars:
        assert torch.equal(conditioned.atmos_vars[name], baseline.atmos_vars[name])


def test_nonzero_adapters_change_outputs_and_depend_on_forcing(
    tiny_model: AuroraCS, tiny_batch: Batch
):
    """Non-zero adapters change the output, deterministically and as a function of the forcing."""
    for head in tiny_model.block_heads:
        torch.nn.init.normal_(head.weight, std=0.5)
        torch.nn.init.normal_(head.bias, std=0.5)

    with torch.no_grad():
        baseline = tiny_model.aurora(tiny_batch)
        low = tiny_model(tiny_batch, forcing=280.0)
        high = tiny_model(tiny_batch, forcing=560.0)
        high_again = tiny_model(tiny_batch, forcing=560.0)

    assert _max_abs_diff(low, baseline) > 1e-4, "conditioning had no effect on the output"
    assert _max_abs_diff(low, high) > 1e-4, "output does not depend on the forcing value"
    assert _max_abs_diff(high, high_again) == 0.0, "same forcing must give the same output"


def test_one_training_step_moves_only_the_conditioning_pathway(
    tiny_model: AuroraCS, tiny_batch: Batch
):
    """One optimiser step updates the conditioning pathway and leaves the frozen Aurora alone."""
    aurora_before = {k: v.clone() for k, v in tiny_model.aurora.state_dict().items()}
    with torch.no_grad():
        baseline = tiny_model.aurora(tiny_batch)

    trainable = [p for p in tiny_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-2)
    target = {k: v + 1.0 for k, v in baseline.surf_vars.items()}
    pred = tiny_model(tiny_batch, forcing=417.0)
    loss = sum(torch.nn.functional.mse_loss(pred.surf_vars[k], target[k]) for k in target)
    loss.backward()

    assert all(
        p.grad is None for p in tiny_model.aurora.parameters()
    ), "the base Aurora must stay frozen"
    assert any(head.weight.grad.abs().max() > 0 for head in tiny_model.block_heads)
    optimizer.step()

    for k, v in tiny_model.aurora.state_dict().items():
        assert torch.equal(v, aurora_before[k]), f"base Aurora weight {k} changed"
    with torch.no_grad():
        after = tiny_model(tiny_batch, forcing=417.0)
    assert _max_abs_diff(after, baseline) > 1e-6, "training did not change the model's output"
