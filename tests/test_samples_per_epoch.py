import argparse
import collections
import random
import sys
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
import xarray as xr

from components.train_aurora_cs import train_aurora_cs as training
from test_training_checkpoint import FINGERPRINT, assert_same, build_model, get_example

POOL = list(pd.date_range("2000-01-01 06:00", periods=20, freq="6h"))


@pytest.fixture(autouse=True)
def quiet_mlflow(monkeypatch):
    monkeypatch.setattr(training.mlflow, "log_metric", lambda *a, **k: None)


def run(epochs, k, checkpoint_dir=None, every=2, fail_after_calls=None, visited=None):
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
        run(epochs=2, k=k, visited=visited)
    assert visited == []


def test_each_epoch_trains_on_exactly_its_k_examples():
    visited = []
    run(epochs=4, k=3, visited=visited)
    assert len(visited) == 12
    for epoch in range(4):
        assert visited[epoch * 3:(epoch + 1) * 3] == training.epoch_order(POOL, epoch, 3)


def test_steps_and_epoch_losses_are_counted_per_sample(monkeypatch):
    steps, epochs = [], []
    monkeypatch.setattr(
        training.mlflow, "log_metric",
        lambda name, value, step: (steps if name == "loss" else epochs).append(step),
    )
    run(epochs=3, k=4)
    assert steps == list(range(12))
    assert epochs == [0, 1, 2]


@pytest.mark.parametrize("crash_after", [2, 3, 4, 5, 8])
def test_resume_after_crash_matches_uninterrupted_sampled_training(tmp_path, crash_after):
    expected = run(epochs=4, k=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(epochs=4, k=3, checkpoint_dir=tmp_path, fail_after_calls=crash_after)
    assert any(tmp_path.glob("ckpt-*.done"))
    resumed_visits = []
    assert_same(run(epochs=4, k=3, checkpoint_dir=tmp_path, visited=resumed_visits), expected)
    full = [t for epoch in range(4) for t in training.epoch_order(POOL, epoch, 3)]
    assert 0 < len(resumed_visits) < len(full)  # resumed, rather than starting over
    assert resumed_visits == full[len(full) - len(resumed_visits):]


def test_a_finished_run_can_be_extended_with_more_epochs(tmp_path):
    run(epochs=2, k=3, checkpoint_dir=tmp_path)
    assert_same(run(epochs=4, k=3, checkpoint_dir=tmp_path), run(epochs=4, k=3))


def test_checkpoint_from_a_different_sample_size_is_refused(tmp_path):
    run(epochs=1, k=3, checkpoint_dir=tmp_path)
    with pytest.raises(RuntimeError, match="samples_per_epoch"):
        run(epochs=1, k=4, checkpoint_dir=tmp_path)


@pytest.mark.parametrize("value,expected", [("all", None), ("100", 100), ("1", 1)])
def test_parse_samples_per_epoch(value, expected):
    assert training.parse_samples_per_epoch(value) == expected


@pytest.mark.parametrize("value", ["0", "-5", "abc", "", "1.5", "ALL"])
def test_parse_rejects_anything_else(value):
    with pytest.raises(argparse.ArgumentTypeError):
        training.parse_samples_per_epoch(value)


def invoke_main(monkeypatch, *extra):
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
    call = invoke_main(monkeypatch, "--samples-per-epoch", "100")
    assert call.kwargs["samples_per_epoch"] == 100
    assert call.kwargs["fingerprint"]["samples_per_epoch"] == 100


@pytest.mark.parametrize("extra", [(), ("--samples-per-epoch", "all")])
def test_main_default_keeps_the_old_fingerprint(monkeypatch, extra):
    call = invoke_main(monkeypatch, *extra)
    assert call.kwargs["samples_per_epoch"] is None
    assert "samples_per_epoch" not in call.kwargs["fingerprint"]


def test_main_rejects_a_non_positive_sample_size(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        invoke_main(monkeypatch, "--samples-per-epoch", "0")
    assert "positive integer or 'all'" in capsys.readouterr().err
