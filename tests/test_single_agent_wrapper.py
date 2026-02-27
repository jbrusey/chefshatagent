import gym
import numpy as np
import pytest

from evaluate import _resolve_model_path
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

    phi_prev = -0.0
    phi_next = -0.0
    expected_reward = 1.0 - 0.3 + env._win_reward(info) + (0.99 * phi_next - phi_prev) + env._step_penalty()
    assert reward == pytest.approx(expected_reward)
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


def test_step_applies_shaping_when_terminal_on_agent_step(monkeypatch):
    class TerminalOnAgentStepEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(228,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(3)
            self.current_player = 0

        @staticmethod
        def _obs(cards_in_hand: int, valid_actions: list[int]) -> np.ndarray:
            obs = np.zeros(228, dtype=np.float32)
            obs[11:11 + cards_in_hand] = 1.0
            obs[28 + np.asarray(valid_actions)] = 1.0
            return obs

        def reset(self, *, seed=None, options=None):
            self.current_player = 0
            return self._obs(cards_in_hand=2, valid_actions=[1]), {"current_player": 0}

        def step(self, action):
            self.current_player = 1
            info = {"current_player": 1, "Match_Score": [3, 0, 0, 0]}
            return self._obs(cards_in_hand=1, valid_actions=[0]), 2.0, True, False, info

    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: TerminalOnAgentStepEnv())
    wrapper = SingleAgentWrapper(seed=123)
    wrapper.reset(seed=123)

    phi_prev = wrapper._phi(wrapper._last_obs)
    obs, reward, terminated, truncated, info = wrapper.step(1)
    phi_next = wrapper._phi(obs)

    expected = 2.0 + wrapper._win_reward(info) + (0.99 * phi_next - phi_prev) + wrapper._step_penalty()
    assert reward == pytest.approx(expected)
    assert terminated is True
    assert truncated is False


def test_step_applies_shaping_when_terminal_during_opponent_advance(monkeypatch):
    class TerminalDuringOpponentAdvanceEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(228,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(3)
            self.current_player = 0

        @staticmethod
        def _obs(cards_in_hand: int, valid_actions: list[int]) -> np.ndarray:
            obs = np.zeros(228, dtype=np.float32)
            obs[11:11 + cards_in_hand] = 1.0
            obs[28 + np.asarray(valid_actions)] = 1.0
            return obs

        def reset(self, *, seed=None, options=None):
            self.current_player = 0
            return self._obs(cards_in_hand=3, valid_actions=[1]), {"current_player": 0}

        def step(self, action):
            if self.current_player == 0:
                self.current_player = 1
                return self._obs(cards_in_hand=2, valid_actions=[0]), 0.5, False, False, {"current_player": 1}

            self.current_player = 2
            info = {"current_player": 2, "Match_Score": [3, 0, 0, 0]}
            return self._obs(cards_in_hand=1, valid_actions=[0]), -0.2, True, False, info

    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: TerminalDuringOpponentAdvanceEnv())
    wrapper = SingleAgentWrapper(seed=123)
    wrapper.reset(seed=123)

    phi_prev = wrapper._phi(wrapper._last_obs)
    obs, reward, terminated, truncated, info = wrapper.step(1)
    phi_next = wrapper._phi(obs)

    expected = 0.5 - 0.2 + wrapper._win_reward(info) + (0.99 * phi_next - phi_prev) + wrapper._step_penalty()
    assert reward == pytest.approx(expected)
    assert terminated is True
    assert truncated is False


# ---------------------------------------------------------------------------
# Tests for evaluate._resolve_model_path
# ---------------------------------------------------------------------------

def test_resolve_model_path_final_model(tmp_path):
    """Returns the base path when the .zip final model exists."""
    model_path = tmp_path / "mymodel"
    (tmp_path / "mymodel.zip").touch()
    assert _resolve_model_path(model_path) == model_path


def test_resolve_model_path_latest_checkpoint_numeric(tmp_path):
    """Picks the checkpoint with the highest step count (numeric, not lexicographic)."""
    model_path = tmp_path / "mymodel"
    # Create checkpoints with step counts that would sort incorrectly alphabetically
    # (e.g. 20000 > 100000 lexicographically because '2' > '1').
    for steps in [10000, 20000, 100000, 200000]:
        (tmp_path / f"mymodel_{steps}_steps.zip").touch()

    resolved = _resolve_model_path(model_path)
    assert resolved == tmp_path / "mymodel_200000_steps"


def test_resolve_model_path_no_model_raises(tmp_path):
    """Raises FileNotFoundError when no model or checkpoint exists."""
    model_path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        _resolve_model_path(model_path)
