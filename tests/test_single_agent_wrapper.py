from unittest.mock import MagicMock

import gym
import numpy as np
import pytest

from evaluate import _resolve_model_path
from single_agent_wrapper import FrozenPolicyOpponent, SingleAgentWrapper


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


class DummyOpponent:
    """Fake opponent that always picks the first valid action."""

    def act(self, obs, mask):
        valid = np.flatnonzero(mask)
        return int(valid[0])


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: DummyChefsHatEnv())
    monkeypatch.setattr("single_agent_wrapper.FrozenPolicyOpponent", lambda path: DummyOpponent())
    return SingleAgentWrapper(seed=123, opponent_pool=["model_a", "model_b"])


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
    expected_reward = 1.0 + env._win_reward(info) + (0.99 * phi_next - phi_prev) + env._step_penalty()
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
            self.action_space = gym.spaces.Discrete(200)
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
    monkeypatch.setattr("single_agent_wrapper.FrozenPolicyOpponent", lambda path: DummyOpponent())
    wrapper = SingleAgentWrapper(seed=123, opponent_pool=["dummy"])
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
            self.action_space = gym.spaces.Discrete(200)
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
    monkeypatch.setattr("single_agent_wrapper.FrozenPolicyOpponent", lambda path: DummyOpponent())
    wrapper = SingleAgentWrapper(seed=123, opponent_pool=["dummy"])
    wrapper.reset(seed=123)

    phi_prev = wrapper._phi(wrapper._last_obs)
    obs, reward, terminated, truncated, info = wrapper.step(1)
    phi_next = wrapper._phi(obs)

    expected = 0.5 - 0.2 + wrapper._win_reward(info) + (0.99 * phi_next - phi_prev) + wrapper._step_penalty()
    assert reward == pytest.approx(expected)
    assert terminated is True
    assert truncated is False


# ---------------------------------------------------------------------------
# Tests for FrozenPolicyOpponent
# ---------------------------------------------------------------------------

def test_frozen_policy_opponent_loads_model(monkeypatch):
    """FrozenPolicyOpponent calls MaskablePPO.load with the given path."""
    mock_model = MagicMock()
    monkeypatch.setattr("single_agent_wrapper.MaskablePPO.load", lambda path: mock_model)
    opp = FrozenPolicyOpponent("my_model_path")
    assert opp.model is mock_model


def test_frozen_policy_opponent_act_returns_int(monkeypatch):
    """FrozenPolicyOpponent.act() returns an int action from model.predict."""
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array(3), None)
    monkeypatch.setattr("single_agent_wrapper.MaskablePPO.load", lambda path: mock_model)
    opp = FrozenPolicyOpponent("path")
    obs = np.zeros(228, dtype=np.float32)
    mask = np.ones(200, dtype=bool)
    action = opp.act(obs, mask)
    assert action == 3
    assert isinstance(action, int)
    mock_model.predict.assert_called_once_with(obs, action_masks=mask, deterministic=True)


# ---------------------------------------------------------------------------
# Tests for opponent_pool sampling in reset()
# ---------------------------------------------------------------------------

def test_reset_raises_when_opponent_pool_empty(monkeypatch):
    """reset() raises RuntimeError when no opponent_pool is provided."""
    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: DummyChefsHatEnv())
    wrapper = SingleAgentWrapper(seed=0)
    with pytest.raises(RuntimeError, match="opponent_pool"):
        wrapper.reset()


def test_reset_samples_three_opponents_from_pool(monkeypatch):
    """reset() creates exactly 3 FrozenPolicyOpponent instances from the pool."""
    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: DummyChefsHatEnv())
    created_paths = []

    def mock_frozen(path):
        created_paths.append(path)
        return DummyOpponent()

    monkeypatch.setattr("single_agent_wrapper.FrozenPolicyOpponent", mock_frozen)
    wrapper = SingleAgentWrapper(seed=0, opponent_pool=["p_a", "p_b", "p_c"])
    wrapper.reset()

    assert len(created_paths) == 3
    assert all(p in ["p_a", "p_b", "p_c"] for p in created_paths)


def test_reset_maps_opponents_to_non_learning_seats(monkeypatch):
    """After reset(), each non-learning seat has an assigned opponent."""
    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: DummyChefsHatEnv())
    monkeypatch.setattr("single_agent_wrapper.FrozenPolicyOpponent", lambda path: DummyOpponent())
    wrapper = SingleAgentWrapper(seed=0, opponent_pool=["p"])
    wrapper.reset()

    assert set(wrapper._opponent_by_seat.keys()) == {1, 2, 3}
    for opp in wrapper._opponent_by_seat.values():
        assert isinstance(opp, DummyOpponent)


# ---------------------------------------------------------------------------
# Tests for _advance_until_learning_turn() opponent interaction
# ---------------------------------------------------------------------------

def test_advance_calls_opponent_for_each_non_learning_turn(monkeypatch):
    """_advance_until_learning_turn uses the opponent for every non-learning seat."""
    call_log = []

    class LoggingOpponent:
        def __init__(self, seat):
            self.seat = seat

        def act(self, obs, mask):
            call_log.append(self.seat)
            return int(np.flatnonzero(mask)[0])

    seat_counter = [1]

    def make_logging_opponent(path):
        opp = LoggingOpponent(seat_counter[0])
        seat_counter[0] += 1
        return opp

    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: DummyChefsHatEnv())
    monkeypatch.setattr("single_agent_wrapper.FrozenPolicyOpponent", make_logging_opponent)
    wrapper = SingleAgentWrapper(seed=0, opponent_pool=["p"])
    wrapper.reset()

    # DummyChefsHatEnv starts at seat 1; seats 1, 2, 3 each take one turn before seat 0
    assert len(call_log) == 3


def test_advance_raises_when_frozen_opponent_picks_illegal_action(monkeypatch):
    """_advance_until_learning_turn raises RuntimeError if opponent returns illegal action."""
    class BadOpponent:
        def act(self, obs, mask):
            return 999  # always illegal

    monkeypatch.setattr("single_agent_wrapper.gym.make", lambda _env_id: DummyChefsHatEnv())
    monkeypatch.setattr("single_agent_wrapper.FrozenPolicyOpponent", lambda path: BadOpponent())
    wrapper = SingleAgentWrapper(seed=0, opponent_pool=["p"])
    with pytest.raises(RuntimeError, match="illegal action"):
        wrapper.reset()


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
