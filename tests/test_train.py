from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from train import WinRateCallback, _build_arg_parser


class DummyLogger:
    def __init__(self):
        self.records = []

    def record(self, key, value, *args, **kwargs):
        self.records.append((key, value))


def test_win_rate_callback_logs_to_sb3_and_wandb():
    wandb_run = MagicMock()
    callback = WinRateCallback(learning_seat=0, wandb_run=wandb_run)
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
    wandb_run.log.assert_called_once_with(
        {"rollout/win_rate": pytest.approx(0.5), "timesteps": 128},
        step=128,
    )


def test_arg_parser_defaults():
    args = _build_arg_parser().parse_args([])

    assert args.wandb is False
    assert args.wandb_project == "chefhats-rl"
    assert args.wandb_run_name is None
