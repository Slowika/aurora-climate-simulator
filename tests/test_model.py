"""Zero-init identity check for `AuroraCS`.

At construction, every per-block adapter head is zero-initialised, so the adapters
`AuroraCS.forward` builds are all-zero `Swin3DBlockAdapter`s. Per the upstream fork's own
tests, all-zero adapters are bit-for-bit equivalent to omitting `backbone_adapters` entirely.
This test checks that invariant holds through the `AuroraCS` wrapper, regardless of the
forcing value.
"""

from datetime import datetime

import torch
from aurora import Batch, Metadata

from aurora_cs.model import AuroraCS


def _tiny_batch() -> Batch:
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


def test_zero_init_adapters_are_a_no_op():
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
    model.eval()
    batch = _tiny_batch()

    with torch.no_grad():
        conditioned = model(batch, forcing=417.0)
        baseline = model.aurora(batch)

    for name in conditioned.surf_vars:
        assert torch.equal(conditioned.surf_vars[name], baseline.surf_vars[name])
    for name in conditioned.atmos_vars:
        assert torch.equal(conditioned.atmos_vars[name], baseline.atmos_vars[name])
