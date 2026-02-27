import gym
import numpy as np
import pytest

from single_agent_wrapper import SingleAgentWrapper


class DummyChefsHatEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(228,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(200)
        self.current_player = 0
        self.turn = 0
        self._last_action = None

    @staticmethod
    def _make_obs(turn: int, valid_actions: list[int]) -> np.ndarray:
        obs = np.zeros(228, dtype=np.float32)
        obs[0] = turn
        obs[28 + np.asarray(valid_actions, dtype=int)] = 1.0
        return obs

    def reset(self, *, seed=None, options=None):
        self.current_player = 1
        self.turn = 0
        self._last_action = None
        return self._make_obs(self.turn, [0, 1]), {
            "current_player": self.current_player,
        }

    def step(self, action):
        self._last_action = action
        action_idx = int(np.argmax(action))
        if self.current_player == 0:
            reward = 1.0 if action_idx == 1 else -1.0
        else:
            reward = -0.1

        self.turn += 1
        self.current_player = (self.current_player + 1) % 4
        terminated = self.turn >= 4
        legal = [1, 2] if self.current_player == 0 else [0, 2]
        info = {
            "current_player": self.current_player,
            "winner": 0 if terminated else None,
        }
        return self._make_obs(self.turn, legal), reward, terminated, False, info


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: DummyChefsHatEnv())
    return SingleAgentWrapper(seed=123)


def test_reset_advances_to_learning_seat(env):
    obs, info = env.reset(seed=123)

    assert info["current_player"] == 0
    assert obs.shape == (228,)
    assert obs[0] == 3
    np.testing.assert_array_equal(np.flatnonzero(obs[28:]), np.array([1, 2]))


def test_step_accumulates_opponent_rewards(env):
    env.reset(seed=123)

    obs, reward, terminated, truncated, info = env.step(1)

    # Learning turn gives +1.0; three opponent turns give -0.3 total.
    assert reward == pytest.approx(0.7)
    assert terminated is True
    assert truncated is False
    assert info.get("winner") == 0
    assert obs[0] == 4


def test_action_mask_read_from_observation(env):
    env.reset(seed=123)
    mask = env.action_masks()

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(np.flatnonzero(mask), np.array([1, 2]))


def test_step_converts_discrete_action_to_one_hot(env):
    env.reset(seed=123)
    env.step(1)

    action_vec = env.base_env._last_action
    assert action_vec.shape == (200,)
    assert action_vec.sum() == pytest.approx(1.0)
    assert action_vec[1] == pytest.approx(1.0)


def test_illegal_action_rejected(env):
    env.reset(seed=123)

    with pytest.raises(ValueError):
        env.step(0)


def test_init_without_start_experiment(monkeypatch):
    class MinimalEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(30,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros(30, dtype=np.float32), {"current_player": 0}

        def step(self, action):
            return np.zeros(30, dtype=np.float32), 0.0, True, False, {"current_player": 0}

        def close(self):
            return None

    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: MinimalEnv())

    wrapper = SingleAgentWrapper(seed=123)

    assert wrapper.base_env.__class__.__name__ == "MinimalEnv"
