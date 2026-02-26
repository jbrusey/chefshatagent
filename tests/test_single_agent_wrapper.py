import gymnasium as gym
import numpy as np
import pytest

from single_agent_wrapper import SingleAgentWrapper


class DummyChefsHatEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)
        self.current_player = 0
        self.turn = 0

    def reset(self, *, seed=None, options=None):
        self.current_player = 1
        self.turn = 0
        return np.array([0], dtype=np.int32), {
            "current_player": self.current_player,
            "action_mask": [True, True, False, False],
        }

    def step(self, action):
        if self.current_player == 0:
            reward = 1.0 if action == 1 else -1.0
        else:
            reward = -0.1

        self.turn += 1
        self.current_player = (self.current_player + 1) % 4
        terminated = self.turn >= 4
        info = {
            "current_player": self.current_player,
            "action_mask": [False, True, True, False] if self.current_player == 0 else [True, False, True, False],
            "winner": 0 if terminated else None,
        }
        return np.array([self.turn], dtype=np.int32), reward, terminated, False, info


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: DummyChefsHatEnv())
    return SingleAgentWrapper(seed=123)


def test_reset_advances_to_learning_seat(env):
    obs, info = env.reset(seed=123)

    assert info["current_player"] == 0
    np.testing.assert_array_equal(obs, np.array([3], dtype=np.int32))


def test_step_accumulates_opponent_rewards(env):
    env.reset(seed=123)

    obs, reward, terminated, truncated, info = env.step(1)

    # Learning turn gives +1.0; three opponent turns give -0.3 total.
    assert reward == pytest.approx(0.7)
    assert terminated is True
    assert truncated is False
    assert info.get("winner") == 0
    np.testing.assert_array_equal(obs, np.array([4], dtype=np.int32))


def test_illegal_action_rejected(env):
    env.reset(seed=123)

    with pytest.raises(ValueError):
        env.step(0)
