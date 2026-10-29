"""Raw CO2 values (~400 ppm) must not make one optimiser step wreck the pretrained Aurora."""

import torch
from test_model import _max_abs_diff, tiny_batch  # noqa: F401,F811  (pytest fixture)

from aurora_cs import AuroraCS, load_checkpoint, save_checkpoint
from aurora_cs.model import AuroraCS as Model
from components.train_aurora_cs.train_aurora_cs import MODEL_CONFIG

TINY = dict(
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


def _drift_after_one_adam_step(model, batch, forcing):
    with torch.no_grad():
        baseline = model.aurora(batch)
    target = {k: v + 0.1 for k, v in baseline.surf_vars.items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    pred = model(batch, forcing=forcing)
    sum(torch.nn.functional.mse_loss(pred.surf_vars[k], target[k]) for k in target).backward()
    optimizer.step()
    with torch.no_grad():
        return _max_abs_diff(model(batch, forcing=forcing), baseline)


def test_standardising_forcing_tames_the_first_optimiser_step(tiny_batch):
    def build(**kwargs):
        torch.manual_seed(0)
        model = AuroraCS(**TINY, **kwargs)
        with torch.no_grad():
            for name, param in model.aurora.named_parameters():
                if "ln_modulation" in name and param.dim() == 2:
                    torch.nn.init.normal_(param, std=0.1)
        return model.eval()

    raw = _drift_after_one_adam_step(build(), tiny_batch, 400.0)
    scaled = _drift_after_one_adam_step(
        build(forcing_mean=350.0, forcing_scale=100.0), tiny_batch, 400.0
    )
    assert scaled < raw / 10


def test_training_config_standardises_co2_to_unit_scale():
    for ppm in (350.0, 430.0, 1000.0):
        z = (ppm - MODEL_CONFIG["forcing_mean"]) / MODEL_CONFIG["forcing_scale"]
        assert abs(z) < 10
    assert MODEL_CONFIG["forcing_scale"] > 1


def test_forcing_standardisation_survives_checkpoint_round_trip(tmp_path, tiny_batch):
    config = dict(TINY, forcing_mean=350.0, forcing_scale=100.0)
    model = Model(**config)
    for head in model.block_heads:
        torch.nn.init.normal_(head.weight, std=0.5)
    save_checkpoint(model, config, tmp_path)
    loaded = load_checkpoint(tmp_path)
    assert (loaded.forcing_mean, loaded.forcing_scale) == (350.0, 100.0)
    with torch.no_grad():
        assert _max_abs_diff(model.eval()(tiny_batch, forcing=420.0), loaded(tiny_batch, forcing=420.0)) == 0.0


def test_defaults_leave_existing_checkpoints_unchanged():
    config = dict(TINY)
    model = Model(**config)
    assert (model.forcing_mean, model.forcing_scale) == (0.0, 1.0)
