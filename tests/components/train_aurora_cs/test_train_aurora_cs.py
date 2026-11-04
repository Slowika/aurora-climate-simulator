import argparse
import collections
import random
import sys
from datetime import date, datetime
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
import xarray as xr
from aurora import Batch, Metadata

from aurora_cs import AuroraCS
from aurora_cs import training_checkpoint as ckpt
from components.train_aurora_cs import train_aurora_cs as training


@pytest.fixture
def prediction():
    return Batch(
        surf_vars={"2t": torch.ones(1, 1, 2, 2, requires_grad=True)},
        static_vars={},
        atmos_vars={"t": torch.ones(1, 1, 1, 2, 2, requires_grad=True)},
        metadata=Metadata(
            lat=torch.tensor([90.0, -90.0]),
            lon=torch.tensor([0.0, 180.0]),
            time=(datetime(1990, 1, 1, 12),),
            atmos_levels=(1000,),
        ),
    )


@pytest.fixture
def target(prediction):
    return {
        "surf_vars": {name: torch.zeros_like(v) for name, v in prediction.surf_vars.items()},
        "atmos_vars": {name: torch.zeros_like(v) for name, v in prediction.atmos_vars.items()},
    }


def test_finite_loss_and_gradients_unchanged(prediction, target):
    loss = training.step_loss(prediction, target)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    for value in (*prediction.surf_vars.values(), *prediction.atmos_vars.values()):
        torch.testing.assert_close(value.grad, torch.full_like(value, 0.25))


@pytest.mark.parametrize("group,name", [("surf_vars", "2t"), ("atmos_vars", "t")])
@pytest.mark.parametrize("source", ["target", "prediction"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_prediction_or_target_fails(prediction, target, group, name, source, bad_value):
    values = target[group] if source == "target" else getattr(prediction, group)
    with torch.no_grad():
        values[name].fill_(bad_value)
    with pytest.raises(FloatingPointError, match=rf"{source}.{group}.{name} at 1990-01-01"):
        training.step_loss(prediction, target)


def test_loss_overflow_is_rejected(prediction, target):
    with torch.no_grad():
        prediction.surf_vars["2t"].fill_(1e30)
    with pytest.raises(FloatingPointError, match="loss.surf_vars.2t"):
        training.step_loss(prediction, target)


@pytest.mark.parametrize("bad_source", ["input", "forcing"])
def test_load_example_rejects_nonfinite_data(monkeypatch, prediction, target, tmp_path, bad_source):
    if bad_source == "input":
        with torch.no_grad():
            prediction.atmos_vars["t"].fill_(float("nan"))
    monkeypatch.setattr(training, "build_batch", lambda *args: prediction)
    monkeypatch.setattr(training, "load_target", lambda *args: target)
    monkeypatch.setattr(
        training, "co2_ppm_for", lambda *args: float("nan") if bad_source == "forcing" else 350.0
    )
    with pytest.raises(FloatingPointError, match="input.atmos_vars.t|CO2 forcing"):
        training.load_example(
            tmp_path, {}, xr.Dataset(), pd.DataFrame(), pd.Timestamp("1990-01-01 06:00"),
            torch.device("cpu"),
        )


@pytest.mark.parametrize("bad_stage", ["static", "gradient"])
def test_main_fails_before_unsafe_work(monkeypatch, prediction, target, bad_stage):
    monkeypatch.setattr(
        sys, "argv",
        ["train_aurora_cs.py", "--era5-data", "era5", "--co2-data", "co2.csv", "--model-name", "test"],
    )
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training.pd, "read_csv", lambda *args, **kwargs: pd.DataFrame())
    static = xr.Dataset({
        name: float("nan") if bad_stage == "static" else 0.0
        for name in training.STATIC_VAR_MAP.values()
    })
    monkeypatch.setattr(training.xr, "open_zarr", lambda *args, **kwargs: static)
    monkeypatch.setattr(
        training, "discover_examples", lambda *args: [pd.Timestamp("1990-01-01 06:00")]
    )
    monkeypatch.setattr(training, "find_incomplete_shards", lambda *args: {})
    monkeypatch.setattr(training, "load_example", lambda *args: (prediction, 350.0, target))
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor(1.0)))
    model.aurora = Mock()
    model_factory = Mock(return_value=model)
    monkeypatch.setattr(training, "AuroraCS", model_factory)
    monkeypatch.setattr(model, "forward", lambda *args, **kwargs: prediction)
    monkeypatch.setattr(training, "step_loss", lambda *args: model.weight * 1.0)
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    optimizer_step = Mock()
    monkeypatch.setattr(training.torch.optim.Adam, "step", optimizer_step)
    log_metric = Mock()
    monkeypatch.setattr(training.mlflow, "log_metric", log_metric)

    with pytest.raises(FloatingPointError, match="ERA5 static|gradient.weight"):
        training.main()

    optimizer_step.assert_not_called()
    log_metric.assert_not_called()
    if bad_stage == "static":
        model_factory.assert_not_called()
    else:
        assert model.weight.item() == 1.0


def test_training_accepts_complete_shards(shard):
    path, times = shard
    assert training.find_incomplete_shards(path.parent, [times[1], times[2]]) == {}


def test_training_reports_every_damaged_variable(shard, remove_chunks):
    path, times = shard
    remove_chunks(path, "2m_temperature")
    remove_chunks(path, "temperature", 1)
    problems = training.find_incomplete_shards(path.parent, [times[1]])
    assert problems == {"1995-08.zarr": ["2m_temperature: 0/4 chunks", "temperature: 3/4 chunks"]}


def test_training_only_checks_shards_it_will_read(shard, write_shard, remove_chunks):
    path, times = shard
    remove_chunks(path, "2m_temperature")  # August is damaged, but no example touches it
    write_shard(path.parent / "1995-09.zarr")
    assert training.find_incomplete_shards(path.parent, [pd.Timestamp("1995-09-01 06:00")]) == {}


def test_training_flags_a_shard_with_no_metadata(tmp_path):
    problems = training.find_incomplete_shards(tmp_path, [pd.Timestamp("1995-08-01 06:00")])
    assert set(problems) == {"1995-08.zarr"}
    assert all(issue.endswith("missing") for issue in problems["1995-08.zarr"])


@pytest.fixture
def era5_times(tmp_path):
    times = pd.date_range("1989-12-31", "2016-01-01 18:00", freq="6h")
    cache = {}
    for month in times.to_period("M").unique():
        (tmp_path / f"{month}.zarr").mkdir()
        cache[(month.year, month.month)] = xr.Dataset(
            coords={"time": times[times.to_period("M") == month]}
        )
    return tmp_path, cache, times


@pytest.mark.parametrize(
    "start_date,end_date",
    [
        (None, None),
        (date(1990, 1, 1), None),
        (None, date(2015, 12, 31)),
        (date(1990, 1, 1), date(2015, 12, 31)),
        (date(2000, 2, 29), date(2000, 2, 29)),
        (date(2020, 1, 1), date(2020, 12, 31)),
    ],
)
def test_discover_examples_respects_inclusive_dates(era5_times, start_date, end_date):
    path, cache, times = era5_times
    examples = training.discover_examples(path, cache, start_date, end_date)
    selected = times
    if start_date is not None:
        selected = selected[selected >= pd.Timestamp(start_date)]
    if end_date is not None:
        selected = selected[selected < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    assert examples == list(selected[1:-1])
    if start_date == date(1990, 1, 1) and end_date == date(2015, 12, 31):
        assert examples[0] == pd.Timestamp("1990-01-01 06:00")
        assert examples[-1] == pd.Timestamp("2015-12-31 12:00")
        assert len(examples) == 37982


def test_date_filter_preserves_gap_detection(tmp_path, monkeypatch):
    times = pd.date_range("1995-01-01", periods=8, freq="6h").delete(3)
    monkeypatch.setattr(training, "discover_times", lambda *_: list(times))
    assert training.discover_examples(
        tmp_path, {}, date(1995, 1, 1), date(1995, 1, 2)
    ) == [times[1], times[4], times[5]]


@pytest.mark.parametrize("value", ["unbounded", "1990-01-01"])
def test_parse_date(value):
    expected = None if value == "unbounded" else date(1990, 1, 1)
    assert training.parse_date(value) == expected


@pytest.mark.parametrize(
    "date_args,error",
    [
        (["--start-date", "1990-02-30"], "invalid parse_date value"),
        (["--end-date", "2015-12-31T18:00:00"], "invalid parse_date value"),
        (
            ["--start-date", "2016-01-01", "--end-date", "1990-01-01"],
            "--start-date must be on or before --end-date",
        ),
    ],
)
def test_invalid_date_arguments_fail_before_training(monkeypatch, capsys, date_args, error):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_aurora_cs.py",
            "--era5-data", "era5",
            "--co2-data", "co2.csv",
            "--model-name", "test-model",
            *date_args,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        training.main()
    assert exc.value.code == 2
    assert error in capsys.readouterr().err


def test_main_passes_bounds_and_rejects_empty_selection(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_aurora_cs.py",
            "--era5-data", "era5",
            "--co2-data", "co2.csv",
            "--model-name", "test-model",
            "--start-date", "1990-01-01",
            "--end-date", "2015-12-31",
        ],
    )
    monkeypatch.setattr(training.pd, "read_csv", lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(
        training.xr,
        "open_zarr",
        lambda *_, **__: xr.Dataset({name: 0.0 for name in training.STATIC_VAR_MAP.values()}),
    )

    def discover_examples(path, cache, start_date, end_date):
        assert start_date == date(1990, 1, 1)
        assert end_date == date(2015, 12, 31)
        return []

    monkeypatch.setattr(training, "discover_examples", discover_examples)
    with pytest.raises(RuntimeError, match="No adjacent-triple training examples"):
        training.main()


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


def run_local(epochs, checkpoint_dir=None, every=2, fail_after_calls=None):
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
    expected = run_local(epochs=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_local(epochs=3, checkpoint_dir=tmp_path, fail_after_calls=crash_after)
    assert any(tmp_path.glob("ckpt-*.done"))
    assert_same(run_local(epochs=3, checkpoint_dir=tmp_path), expected)


def test_checkpointing_does_not_change_training(tmp_path):
    assert_same(run_local(epochs=2, checkpoint_dir=tmp_path), run_local(epochs=2))


def test_finished_run_can_be_extended(tmp_path):
    run_local(epochs=2, checkpoint_dir=tmp_path)
    assert_same(run_local(epochs=3, checkpoint_dir=tmp_path), run_local(epochs=3))


def test_resuming_a_finished_run_does_no_extra_training(tmp_path):
    first = run_local(epochs=2, checkpoint_dir=tmp_path)
    assert_same(run_local(epochs=2, checkpoint_dir=tmp_path, fail_after_calls=0), first)


def test_only_latest_complete_checkpoints_are_kept(tmp_path):
    run_local(epochs=3, checkpoint_dir=tmp_path, every=1)
    done = sorted(p.name for p in tmp_path.glob("ckpt-*.done"))
    assert len(done) == ckpt.KEEP
    assert sorted(p.stem for p in tmp_path.glob("ckpt-*.pt")) == [d[:-5] for d in done]


def test_incomplete_checkpoint_is_ignored(tmp_path):
    run_local(epochs=1, checkpoint_dir=tmp_path, every=1)
    newest = max(tmp_path.glob("ckpt-*.done"))
    newest.unlink()
    state = ckpt.load_latest_training_state(tmp_path, torch.device("cpu"))
    assert state["global_step"] == int(max(tmp_path.glob("ckpt-*.done")).stem[5:])
    assert ckpt.load_latest_training_state(tmp_path / "missing", torch.device("cpu")) is None


@pytest.mark.parametrize("changed", [{"lr": 1e-4}, {"num_examples": 99}, {"model_config": {}}])
def test_mismatched_checkpoint_is_refused(tmp_path, changed):
    run_local(epochs=1, checkpoint_dir=tmp_path)
    state = ckpt.load_latest_training_state(tmp_path, torch.device("cpu"))
    model = build_model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    with pytest.raises(RuntimeError, match="different setup"):
        ckpt.restore_training_state(state, model, optimizer, {**FINGERPRINT, **changed})


def run_remote(epochs, fs, directory="container/checkpoints/run", every=2, fail_after_calls=None):
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
        model, optimizer, EXAMPLES, counting, epochs, checkpoint_dir=directory,
        checkpoint_fs=fs, checkpoint_every=every, fingerprint=FINGERPRINT,
    )
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("crash_after", [3, 5, 8])
def test_resume_from_blob_style_storage_matches_uninterrupted_training(remote, crash_after):
    expected = run_remote(epochs=3, fs=remote, directory="container/uninterrupted")
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_remote(epochs=3, fs=remote, fail_after_calls=crash_after)
    assert any(name.endswith(".done") for name in remote.ls("container/checkpoints/run", detail=False))
    assert_same(run_remote(epochs=3, fs=remote), expected)


def test_only_the_newest_checkpoints_are_kept_remotely(remote):
    run_remote(epochs=3, fs=remote, every=1)
    names = sorted(n.rsplit("/", 1)[-1] for n in remote.ls("container/checkpoints/run", detail=False))
    assert len([n for n in names if n.endswith(".done")]) == ckpt.KEEP
    assert len([n for n in names if n.endswith(".pt")]) == ckpt.KEEP


def test_a_checkpoint_without_its_marker_is_ignored_remotely(remote):
    run_remote(epochs=1, fs=remote, every=1)
    directory = "container/checkpoints/run"
    newest = max(n for n in remote.ls(directory, detail=False) if n.endswith(".done"))
    remote.rm(newest)
    state = ckpt.load_latest_training_state(directory, torch.device("cpu"), remote)
    assert state["global_step"] == int(max(n for n in remote.ls(directory, detail=False) if n.endswith(".done"))[-14:-5])


def test_a_marker_without_its_data_is_ignored_remotely(remote):
    run_remote(epochs=1, fs=remote, every=1)
    directory = "container/checkpoints/run"
    newest = max(n for n in remote.ls(directory, detail=False) if n.endswith(".pt"))
    remote.rm(newest)
    state = ckpt.load_latest_training_state(directory, torch.device("cpu"), remote)
    assert state["global_step"] < int(newest[-12:-3])


def test_local_directories_still_work_without_a_filesystem(tmp_path):
    model = build_model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    ckpt.save_training_state(
        tmp_path / "new" / "dir", model, optimizer, epoch=0, position=1, global_step=1,
        epoch_loss=0.0, fingerprint={},
    )
    state = ckpt.load_latest_training_state(tmp_path / "new" / "dir", torch.device("cpu"))
    assert state["global_step"] == 1


def invoke_main_until_data_load(monkeypatch, *extra, write_check=None):
    monkeypatch.setattr(
        sys, "argv",
        ["train_aurora_cs.py", "--era5-data", "era5", "--co2-data", "co2.csv", "--model-name", "m", *extra],
    )
    monkeypatch.setattr(training, "check_writable", write_check or Mock())
    order = []
    monkeypatch.setattr(training, "configure_mlflow_auth", lambda: order.append("auth"))
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: order.append("device") or False)
    monkeypatch.setattr(
        training.pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("data loading started"))
    )
    with pytest.raises(RuntimeError, match="data loading started"):
        training.main()
    return order


def test_main_checks_blob_access_before_any_data_is_loaded(monkeypatch):
    fs = Mock()
    monkeypatch.setattr(training, "open_checkpoint_store", lambda *a, **k: (fs, "cont/prefix"))
    check = Mock()
    invoke_main_until_data_load(monkeypatch, "--checkpoint-prefix", "prefix", write_check=check)
    check.assert_called_once_with(fs, "cont/prefix")


def test_main_stops_at_startup_when_blob_is_not_writable(monkeypatch):
    monkeypatch.setattr(training, "open_checkpoint_store", lambda *a, **k: (Mock(), "cont/prefix"))
    monkeypatch.setattr(sys, "argv", ["t", "--era5-data", "e", "--co2-data", "c", "--model-name", "m",
                                      "--checkpoint-prefix", "p"])
    monkeypatch.setattr(training, "configure_mlflow_auth", lambda: None)
    monkeypatch.setattr(training, "check_writable", Mock(side_effect=PermissionError("denied")))
    monkeypatch.setattr(training.pd, "read_csv", Mock(side_effect=AssertionError("data was loaded")))
    with pytest.raises(PermissionError, match="denied"):
        training.main()


@pytest.mark.parametrize(
    "args,message",
    [
        (["--checkpoint-prefix", "p"], "storage account"),
        (["--checkpoint-prefix", "p", "--checkpoint-account", "a"], "storage container"),
        (["--checkpoint-prefix", "checkpoints/<MODEL_NAME>", "--checkpoint-account", "a",
          "--checkpoint-container", "c"], "unfilled placeholder"),
        (["--checkpoint-dir", "/tmp/x", "--checkpoint-prefix", "p", "--checkpoint-account", "a",
          "--checkpoint-container", "c"], "not both"),
    ],
)
def test_main_reports_bad_checkpoint_settings_as_usage_errors(monkeypatch, capsys, args, message):
    monkeypatch.setattr(
        sys, "argv", ["t", "--era5-data", "e", "--co2-data", "c", "--model-name", "m", *args]
    )
    with pytest.raises(SystemExit) as exc:
        training.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_main_without_checkpointing_skips_the_write_check(monkeypatch):
    check = Mock()
    invoke_main_until_data_load(monkeypatch, write_check=check)
    check.assert_not_called()


POOL = list(pd.date_range("2000-01-01 06:00", periods=20, freq="6h"))


def run_sampled(epochs, k, checkpoint_dir=None, every=2, fail_after_calls=None, visited=None):
    model = build_model()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    calls = 0

    def counting(cur_t):
        nonlocal calls
        calls += 1
        if fail_after_calls is not None and calls > fail_after_calls:
            raise RuntimeError("simulated crash")
        if visited is not None:
            visited.append(cur_t)
        return get_example(cur_t)

    training.train_loop(
        model, optimizer, POOL, counting, epochs, samples_per_epoch=k,
        checkpoint_dir=checkpoint_dir, checkpoint_every=every,
        fingerprint={**FINGERPRINT, "samples_per_epoch": k},
    )
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def test_an_epoch_draws_exactly_k_distinct_examples():
    for epoch in range(20):
        order = training.epoch_order(POOL, epoch, 7)
        assert len(order) == 7 == len(set(order))
        assert set(order) <= set(POOL)


def test_selection_is_uniform_over_the_examples():
    counts = collections.Counter(t for epoch in range(4000) for t in training.epoch_order(POOL, epoch, 5))
    expected = 4000 * 5 / len(POOL)  # 1000, standard deviation about 27
    assert set(counts) == set(POOL)
    assert all(abs(c - expected) < 150 for c in counts.values()), counts


def test_every_epoch_draws_afresh_and_deterministically():
    draws = [training.epoch_order(POOL, epoch, 5) for epoch in range(10)]
    assert len({tuple(d) for d in draws}) > 1
    assert draws == [training.epoch_order(POOL, epoch, 5) for epoch in range(10)]


def test_without_samples_per_epoch_the_full_shuffle_is_unchanged():
    """Keeps old checkpoints valid: the same order the loop used before this option existed."""
    expected = list(POOL)
    random.Random(3).shuffle(expected)
    assert training.epoch_order(POOL, 3) == expected
    assert sorted(training.epoch_order(POOL, 3)) == POOL


def test_taking_every_example_is_a_permutation():
    assert sorted(training.epoch_order(POOL, 0, len(POOL))) == POOL


@pytest.mark.parametrize("k", [len(POOL) + 1, 10**6])
def test_asking_for_more_than_exists_is_rejected_before_training(k):
    visited = []
    with pytest.raises(ValueError, match="exceeds the 20 available"):
        run_sampled(epochs=2, k=k, visited=visited)
    assert visited == []


def test_each_epoch_trains_on_exactly_its_k_examples():
    visited = []
    run_sampled(epochs=4, k=3, visited=visited)
    assert len(visited) == 12
    for epoch in range(4):
        assert visited[epoch * 3:(epoch + 1) * 3] == training.epoch_order(POOL, epoch, 3)


def test_steps_and_epoch_losses_are_counted_per_sample(monkeypatch):
    steps, epochs = [], []
    monkeypatch.setattr(
        training.mlflow, "log_metric",
        lambda name, value, step: (steps if name == "loss" else epochs).append(step),
    )
    run_sampled(epochs=3, k=4)
    assert steps == list(range(12))
    assert epochs == [0, 1, 2]


@pytest.mark.parametrize("crash_after", [2, 3, 4, 5, 8])
def test_resume_after_crash_matches_uninterrupted_sampled_training(tmp_path, crash_after):
    expected = run_sampled(epochs=4, k=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_sampled(epochs=4, k=3, checkpoint_dir=tmp_path, fail_after_calls=crash_after)
    assert any(tmp_path.glob("ckpt-*.done"))
    resumed_visits = []
    assert_same(run_sampled(epochs=4, k=3, checkpoint_dir=tmp_path, visited=resumed_visits), expected)
    full = [t for epoch in range(4) for t in training.epoch_order(POOL, epoch, 3)]
    assert 0 < len(resumed_visits) < len(full)  # resumed, rather than starting over
    assert resumed_visits == full[len(full) - len(resumed_visits):]


def test_a_finished_run_can_be_extended_with_more_epochs(tmp_path):
    run_sampled(epochs=2, k=3, checkpoint_dir=tmp_path)
    assert_same(run_sampled(epochs=4, k=3, checkpoint_dir=tmp_path), run_sampled(epochs=4, k=3))


def test_checkpoint_from_a_different_sample_size_is_refused(tmp_path):
    run_sampled(epochs=1, k=3, checkpoint_dir=tmp_path)
    with pytest.raises(RuntimeError, match="samples_per_epoch"):
        run_sampled(epochs=1, k=4, checkpoint_dir=tmp_path)


@pytest.mark.parametrize("value,expected", [("all", None), ("100", 100), ("1", 1)])
def test_parse_samples_per_epoch(value, expected):
    assert training.parse_samples_per_epoch(value) == expected


@pytest.mark.parametrize("value", ["0", "-5", "abc", "", "1.5", "ALL"])
def test_parse_rejects_anything_else(value):
    with pytest.raises(argparse.ArgumentTypeError):
        training.parse_samples_per_epoch(value)


def invoke_main_capturing_train_loop(monkeypatch, *extra):
    monkeypatch.setattr(
        sys, "argv",
        ["train_aurora_cs.py", "--era5-data", "era5", "--co2-data", "co2.csv",
         "--model-name", "m", *extra],
    )
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training.pd, "read_csv", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        training.xr, "open_zarr",
        lambda *a, **k: xr.Dataset({n: 0.0 for n in training.STATIC_VAR_MAP.values()}),
    )
    monkeypatch.setattr(training, "discover_examples", lambda *a: POOL)
    monkeypatch.setattr(training, "find_incomplete_shards", lambda *a: {})
    model = torch.nn.Linear(1, 1)
    model.aurora = Mock()
    monkeypatch.setattr(training, "AuroraCS", Mock(return_value=model))
    monkeypatch.setattr(training, "register_model", Mock())
    train_loop = Mock()
    monkeypatch.setattr(training, "train_loop", train_loop)
    training.main()
    return train_loop.call_args


def test_main_passes_samples_per_epoch_and_records_it_in_the_fingerprint(monkeypatch):
    call = invoke_main_capturing_train_loop(monkeypatch, "--samples-per-epoch", "100")
    assert call.kwargs["samples_per_epoch"] == 100
    assert call.kwargs["fingerprint"]["samples_per_epoch"] == 100


@pytest.mark.parametrize("extra", [(), ("--samples-per-epoch", "all")])
def test_main_default_keeps_the_old_fingerprint(monkeypatch, extra):
    call = invoke_main_capturing_train_loop(monkeypatch, *extra)
    assert call.kwargs["samples_per_epoch"] is None
    assert "samples_per_epoch" not in call.kwargs["fingerprint"]


def test_main_rejects_a_non_positive_sample_size(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        invoke_main_capturing_train_loop(monkeypatch, "--samples-per-epoch", "0")
    assert "positive integer or 'all'" in capsys.readouterr().err
