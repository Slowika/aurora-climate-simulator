"""Variable maps, model config and other constants for `train_aurora_cs`."""

import pandas as pd

# Must match the variables `data/fetch_era5.py` writes into each monthly shard / static.zarr.
SURF_VAR_MAP = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
ATMOS_VAR_MAP = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}
STATIC_VAR_MAP = {
    "lsm": "land_sea_mask",
    "slt": "soil_type",
    "z": "geopotential_at_surface",
}

CHECKPOINT = "aurora-0.25-small-pretrained.ckpt"
# "AuroraSmallPretrained" architecture; stored with the registered model to rebuild it.
MODEL_CONFIG = dict(
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
    use_lora=False,
    # CO2 in ppm is ~350-430 for the historical period; raw values make the first Adam step
    # blow the loss up by orders of magnitude. Stored in the checkpoint, so rollout matches.
    forcing_mean=350.0,
    forcing_scale=100.0,
)
PATCH_SIZE = 4  # AuroraCS's default `patch_size`; history windows must be pre-cropped to it.
SIX_HOURS = pd.Timedelta(hours=6)
