import numpy as np
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from game_adapters import ChefshatAdapter
from train import (
    MAX_OPPONENT_POOL_SIZE,
    DEFAULT_ELO,
    _resolve_seat_action,
    build_training_opponent_pool,
    WandbMetricsCallback,
    WinRateCallback,
    _log_current_player_elo_to_wandb,
    _build_arg_parser,
    build_initial_opponent_pool,
    LATEST_MODEL_PATH,
)
from single_agent_wrapper import RANDOM_OPPONENT_TOKEN


class DummyLogger:
    def __init__(self):
        self.records = []

    def record(self, key, value, *args, **kwargs):
        self.records.append((key, value))


def test_win_rate_callback_logs_to_sb3():
    callback = WinRateCallback(adapter=ChefshatAdapter(), learning_seat=0, wandb_run=MagicMock())
    dummy_logger = DummyLogger()

    callback.locals = {
        "dones": [True, True, False],
        "infos": [
            {"Match_Score": [3, 0, 0, 0]},
            {"Match_Score": [0, 3, 0, 0]},
            {},
        ],
    }
    callback.num_timesteps = 128

    with patch.object(type(callback), "logger", new_callable=PropertyMock, return_value=dummy_logger):
        assert callback._on_step() is True
        callback._on_rollout_end()

    assert dummy_logger.records == [("rollout/win_rate", pytest.approx(0.5))]



def test_wandb_metrics_callback_logs_scalar_sb3_metrics():
    wandb_run = MagicMock()
    callback = WandbMetricsCallback(wandb_run=wandb_run)
    callback.num_timesteps = 512

    logger = MagicMock()
    logger.name_to_value = {
        "rollout/win_rate": np.float32(0.75),
        "time/fps": 1234,
        "train/loss": 0.45,
        "non_scalar": [1, 2, 3],
    }

    model = MagicMock()
    model.logger = logger
    callback.model = model

    callback._on_rollout_end()

    wandb_run.log.assert_called_once_with(
        {
            "rollout/win_rate": pytest.approx(0.75),
            "time/fps": 1234,
            "train/loss": pytest.approx(0.45),
            "timesteps": 512,
        },
        step=512,
    )


def test_log_current_player_elo_to_wandb_logs_rollout_elo_metric():
    wandb_run = MagicMock()

    _log_current_player_elo_to_wandb(
        wandb_run,
        timesteps=10_000,
        elo_rating=DEFAULT_ELO + 42,
    )

    wandb_run.log.assert_called_once_with(
        {
            "rollout/elo": pytest.approx(DEFAULT_ELO + 42),
            "timesteps": 10_000,
        },
        step=10_000,
    )


def test_log_current_player_elo_to_wandb_noop_without_wandb():
    _log_current_player_elo_to_wandb(
        None,
        timesteps=10_000,
        elo_rating=DEFAULT_ELO,
    )

def test_arg_parser_defaults():
    args = _build_arg_parser().parse_args([])

    assert args.game == "chefshat"
    assert args.env_id is None
    assert args.wandb is False
    assert args.wandb_project == "chefhats-rl"
    assert args.wandb_run_name is None


# ---------------------------------------------------------------------------
# Tests for build_initial_opponent_pool
# ---------------------------------------------------------------------------

def test_build_initial_opponent_pool_empty_dir(tmp_path):
    """With no models present, pool contains only RANDOM_OPPONENT_TOKEN."""
    pool = build_initial_opponent_pool(tmp_path)

    assert pool == [RANDOM_OPPONENT_TOKEN]


def test_build_initial_opponent_pool_with_latest(tmp_path):
    """Pool includes latest.zip when it exists, followed by RANDOM token."""
    (tmp_path / LATEST_MODEL_PATH.name).touch()

    pool = build_initial_opponent_pool(tmp_path)

    assert pool[0] == str(tmp_path / LATEST_MODEL_PATH.name)
    assert pool[-1] == RANDOM_OPPONENT_TOKEN


def test_build_initial_opponent_pool_snapshot_limit(tmp_path):
    """Pool includes at most 4 snapshots even when more exist."""
    for i in range(6):
        (tmp_path / f"snapshot_{i}.zip").touch()

    pool = build_initial_opponent_pool(tmp_path)

    snapshot_entries = [p for p in pool if "snapshot_" in p]
    assert len(snapshot_entries) == 4


def test_build_initial_opponent_pool_snapshot_ordering(tmp_path):
    """Snapshots are ordered newest-first by iteration number."""
    for i in [1, 2, 3, 4, 5]:
        (tmp_path / f"snapshot_{i}.zip").touch()

    pool = build_initial_opponent_pool(tmp_path)

    snapshot_entries = [p for p in pool if "snapshot_" in p]
    # Highest iteration numbers should appear first
    assert str(tmp_path / "snapshot_5.zip") == snapshot_entries[0]
    assert str(tmp_path / "snapshot_4.zip") == snapshot_entries[1]
    assert str(tmp_path / "snapshot_3.zip") == snapshot_entries[2]
    assert str(tmp_path / "snapshot_2.zip") == snapshot_entries[3]


def test_build_initial_opponent_pool_deduplication(tmp_path):
    """latest.zip is not duplicated even if it matches the snapshot glob somehow."""
    latest = tmp_path / LATEST_MODEL_PATH.name
    latest.touch()
    # Pool should not contain the latest path twice
    pool = build_initial_opponent_pool(tmp_path)

    assert pool.count(str(latest)) == 1


def test_build_initial_opponent_pool_always_ends_with_random_token(tmp_path):
    """RANDOM_OPPONENT_TOKEN is always the last element of the pool."""
    (tmp_path / LATEST_MODEL_PATH.name).touch()
    for i in range(3):
        (tmp_path / f"snapshot_{i}.zip").touch()

    pool = build_initial_opponent_pool(tmp_path)

    assert pool[-1] == RANDOM_OPPONENT_TOKEN


def test_build_initial_opponent_pool_uses_latest_model_path_name(tmp_path):
    """Pool uses LATEST_MODEL_PATH.name, not a hardcoded 'latest.zip' string."""
    # If LATEST_MODEL_PATH.name ever changes, this test ensures the function
    # picks up the correct filename rather than a stale hardcoded value.
    expected_name = LATEST_MODEL_PATH.name
    (tmp_path / expected_name).touch()

    pool = build_initial_opponent_pool(tmp_path)

    assert str(tmp_path / expected_name) in pool


def test_build_training_opponent_pool_includes_latest_and_random(tmp_path):
    latest = tmp_path / "latest.zip"
    latest.touch()
    top = tmp_path / "snapshot_10.zip"
    top.touch()

    pool = build_training_opponent_pool(
        ratings={str(top): 1200.0},
        latest_model=str(latest),
    )

    assert pool[0] == str(latest)
    assert str(top) in pool
    assert pool[-1] == RANDOM_OPPONENT_TOKEN


def test_build_training_opponent_pool_respects_max_size(tmp_path):
    latest = tmp_path / "latest.zip"
    latest.touch()
    ratings = {}
    for i in range(MAX_OPPONENT_POOL_SIZE + 5):
        path = tmp_path / f"snapshot_{i}.zip"
        path.touch()
        ratings[str(path)] = 1000.0 + i

    pool = build_training_opponent_pool(ratings=ratings, latest_model=str(latest))

    assert len(pool) <= MAX_OPPONENT_POOL_SIZE
    assert pool[0] == str(latest)
    assert pool[-1] == RANDOM_OPPONENT_TOKEN


def test_build_training_opponent_pool_keeps_latest_first_without_duplication(tmp_path):
    latest = tmp_path / "latest.zip"
    latest.touch()
    other = tmp_path / "snapshot_1.zip"
    other.touch()
    ratings = {
        str(latest): 2000.0,
        str(other): 1900.0,
    }

    pool = build_training_opponent_pool(ratings=ratings, latest_model=str(latest))

    assert pool[0] == str(latest)
    assert pool.count(str(latest)) == 1


def test_resolve_seat_action_raises_on_short_observation():
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="Expected action mask of length"):
        _resolve_seat_action(
            seat_agent=RANDOM_OPPONENT_TOKEN,
            obs=np.zeros(50, dtype=np.float32),
            action_space_n=200,
            adapter=ChefshatAdapter(),
            policy_cache={},
            rng=rng,
        )
