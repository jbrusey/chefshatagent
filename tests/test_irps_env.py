import numpy as np
import pytest

from irps_env import IrpsEnv, make_single_agent_env


def test_reset_initializes_buckets_to_zero():
    env = IrpsEnv(seed=7)
    obs, info = env.reset(seed=11)

    np.testing.assert_array_equal(obs, np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(info["agent_wins_by_move"], np.zeros(3, dtype=np.int32))
    np.testing.assert_array_equal(info["opponent_wins_by_move"], np.zeros(3, dtype=np.int32))
    assert info["scores"] == [0, 0]
    assert info["winner"] is None


def test_step_updates_winning_bucket_for_agent_win():
    env = IrpsEnv(seed=0, opponent_policy=lambda _obs, _rng: 2)  # scissors
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)  # rock beats scissors

    np.testing.assert_array_equal(info["agent_wins_by_move"], np.array([1, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(info["opponent_wins_by_move"], np.array([0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(obs, np.array([1, 0, 0, 0, 0, 0], dtype=np.float32))
    assert reward == pytest.approx(1.0)
    assert terminated is False
    assert truncated is False


def test_step_updates_winning_bucket_for_opponent_win():
    env = IrpsEnv(seed=0, opponent_policy=lambda _obs, _rng: 1)  # paper
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)  # rock loses to paper

    np.testing.assert_array_equal(info["agent_wins_by_move"], np.array([0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(info["opponent_wins_by_move"], np.array([0, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(obs, np.array([0, 0, 0, 0, 1, 0], dtype=np.float32))
    assert reward == pytest.approx(-1.0)
    assert terminated is False
    assert truncated is False


def test_episode_ends_when_either_player_reaches_three_wins():
    env = IrpsEnv(seed=0, opponent_policy=lambda _obs, _rng: 2)
    env.reset()

    out = None
    for _ in range(3):
        out = env.step(0)

    obs, reward, terminated, truncated, info = out
    np.testing.assert_array_equal(obs, np.array([2, 0, 0, 0, 0, 0], dtype=np.float32))
    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False
    assert info["winner"] == 0
    assert info["scores"] == [3, 0]
    np.testing.assert_array_equal(info["agent_wins_by_move"], np.array([3, 0, 0], dtype=np.int32))


def test_observation_is_bounded_to_two_per_bucket():
    env = IrpsEnv(seed=0, opponent_policy=lambda _obs, _rng: 2)
    env.reset()
    for _ in range(3):
        obs, *_ = env.step(0)

    assert obs.max() <= 2.0
    np.testing.assert_array_equal(obs[:3], np.array([2, 0, 0], dtype=np.float32))


def test_invalid_action_raises_value_error():
    env = IrpsEnv(seed=0)
    env.reset()
    with pytest.raises(ValueError, match="must be in"):
        env.step(5)


def test_make_single_agent_env_routes_irps_without_single_agent_wrapper(monkeypatch):
    def crash(*_args, **_kwargs):
        raise AssertionError("SingleAgentWrapper should not be used for irps")

    monkeypatch.setattr("single_agent_wrapper.SingleAgentWrapper", crash)
    env = make_single_agent_env(game="irps", env_id="irps-v1", seed=5)
    assert isinstance(env, IrpsEnv)


def test_make_single_agent_env_routes_other_games_through_single_agent_wrapper(monkeypatch):
    sentinel = object()

    def fake_wrapper(**kwargs):
        assert kwargs["env_id"] == "chefshat-v1"
        return sentinel

    monkeypatch.setattr("single_agent_wrapper.SingleAgentWrapper", fake_wrapper)
    env = make_single_agent_env(game="chefshat", env_id="chefshat-v1", seed=5)
    assert env is sentinel
