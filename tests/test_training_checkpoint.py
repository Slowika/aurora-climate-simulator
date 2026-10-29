from datetime import datetime

import pandas as pd
import pytest
import torch
from aurora import Batch, Metadata

from aurora_cs import training_checkpoint as ckpt
from components.train_aurora_cs import train_aurora_cs as training
from aurora_cs import AuroraCS

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
    forcing_mean=350.0,
    forcing_scale=100.0,
)
EXAMPLES = list(pd.date_range("2000-01-01 06:00", periods=5, freq="6h"))
FINGERPRINT = {"lr": 1e-3, "num_examples": len(EXAMPLES), "model_config": TINY}


def build_model():
    torch.manual_seed(0)
    model = AuroraCS(**TINY)
    with torch.no_grad():
        for name, param in model.aurora.named_parameters():
            if "ln_modulation" in name and param.dim() == 2:
                torch.nn.init.normal_(param, std=0.1)
    return model


def get_example(cur_t):
    generator = torch.Generator().manual_seed(int(cur_t.timestamp()))
    h = w = 16
    batch = Batch(
        surf_vars={n: torch.randn(1, 1, h, w, generator=generator) for n in ("2t", "10u", "10v", "msl")},
        static_vars={n: torch.randn(h, w, generator=generator) for n in ("lsm", "z", "slt")},
        atmos_vars={n: torch.randn(1, 1, 3, h, w, generator=generator) for n in ("z", "u", "v", "t", "q")},
        metadata=Metadata(lat=torch.linspace(90, -90, h), lon=torch.linspace(0, 360, w + 1)[:-1],
                          time=(datetime(2000, 1, 1),), atmos_levels=(1000, 500, 100)),
    )
    target = {
        "surf_vars": {k: torch.randn(v.shape, generator=generator) for k, v in batch.surf_vars.items()},
        "atmos_vars": {k: torch.randn(v.shape, generator=generator) for k, v in batch.atmos_vars.items()},
    }
    return batch, 400.0 + cur_t.hour, target


@pytest.fixture(autouse=True)
def quiet_mlflow(monkeypatch):
    monkeypatch.setattr(training.mlflow, "log_metric", lambda *a, **k: None)


def run(epochs, checkpoint_dir=None, every=2, fail_after_calls=None):
    model = build_model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    calls = 0

    def counting(cur_t):
        nonlocal calls
        calls += 1
        if fail_after_calls is not None and calls > fail_after_calls:
            raise RuntimeError("simulated crash")
        return get_example(cur_t)

    training.train_loop(
        model, optimizer, EXAMPLES, counting, epochs,
        checkpoint_dir=checkpoint_dir, checkpoint_every=every, fingerprint=FINGERPRINT,
    )
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def assert_same(a, b):
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


@pytest.mark.parametrize("crash_after", [3, 5, 7, 9])
def test_resume_after_crash_matches_uninterrupted_training(tmp_path, crash_after):
    expected = run(epochs=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(epochs=3, checkpoint_dir=tmp_path, fail_after_calls=crash_after)
    assert any(tmp_path.glob("ckpt-*.done"))
    assert_same(run(epochs=3, checkpoint_dir=tmp_path), expected)


def test_checkpointing_does_not_change_training(tmp_path):
    assert_same(run(epochs=2, checkpoint_dir=tmp_path), run(epochs=2))


def test_finished_run_can_be_extended(tmp_path):
    run(epochs=2, checkpoint_dir=tmp_path)
    assert_same(run(epochs=3, checkpoint_dir=tmp_path), run(epochs=3))


def test_resuming_a_finished_run_does_no_extra_training(tmp_path):
    first = run(epochs=2, checkpoint_dir=tmp_path)
    assert_same(run(epochs=2, checkpoint_dir=tmp_path, fail_after_calls=0), first)


def test_only_latest_complete_checkpoints_are_kept(tmp_path):
    run(epochs=3, checkpoint_dir=tmp_path, every=1)
    done = sorted(p.name for p in tmp_path.glob("ckpt-*.done"))
    assert len(done) == ckpt.KEEP
    assert sorted(p.stem for p in tmp_path.glob("ckpt-*.pt")) == [d[:-5] for d in done]


def test_incomplete_checkpoint_is_ignored(tmp_path):
    run(epochs=1, checkpoint_dir=tmp_path, every=1)
    newest = max(tmp_path.glob("ckpt-*.done"))
    newest.unlink()
    state = ckpt.load_latest_training_state(tmp_path, torch.device("cpu"))
    assert state["global_step"] == int(max(tmp_path.glob("ckpt-*.done")).stem[5:])
    assert ckpt.load_latest_training_state(tmp_path / "missing", torch.device("cpu")) is None


@pytest.mark.parametrize("changed", [{"lr": 1e-4}, {"num_examples": 99}, {"model_config": {}}])
def test_mismatched_checkpoint_is_refused(tmp_path, changed):
    run(epochs=1, checkpoint_dir=tmp_path)
    state = ckpt.load_latest_training_state(tmp_path, torch.device("cpu"))
    model = build_model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    with pytest.raises(RuntimeError, match="different setup"):
        ckpt.restore_training_state(state, model, optimizer, {**FINGERPRINT, **changed})
