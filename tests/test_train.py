import numpy as np
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from train import WandbMetricsCallback, WinRateCallback, _build_arg_parser


class DummyLogger:
    def __init__(self):
        self.records = []

    def record(self, key, value, *args, **kwargs):
        self.records.append((key, value))


def test_win_rate_callback_logs_to_sb3():
    callback = WinRateCallback(learning_seat=0, wandb_run=MagicMock())
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

def test_arg_parser_defaults():
    args = _build_arg_parser().parse_args([])

    assert args.wandb is False
    assert args.wandb_project == "chefhats-rl"
    assert args.wandb_run_name is None
