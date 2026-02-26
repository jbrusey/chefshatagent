import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

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


class SingleAgentWrapperTests(unittest.TestCase):
    @patch("single_agent_wrapper.gym.make", return_value=DummyChefsHatEnv())
    def test_reset_advances_to_learning_seat(self, _mock_make):
        env = SingleAgentWrapper(seed=123)
        obs, info = env.reset(seed=123)

        self.assertEqual(info["current_player"], 0)
        np.testing.assert_array_equal(obs, np.array([3], dtype=np.int32))

    @patch("single_agent_wrapper.gym.make", return_value=DummyChefsHatEnv())
    def test_step_accumulates_opponent_rewards(self, _mock_make):
        env = SingleAgentWrapper(seed=123)
        env.reset(seed=123)

        obs, reward, terminated, truncated, info = env.step(1)

        # Learning turn gives +1.0; three opponent turns give -0.3 total.
        self.assertAlmostEqual(reward, 0.7, places=6)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info.get("winner"), 0)
        np.testing.assert_array_equal(obs, np.array([4], dtype=np.int32))

    @patch("single_agent_wrapper.gym.make", return_value=DummyChefsHatEnv())
    def test_illegal_action_rejected(self, _mock_make):
        env = SingleAgentWrapper(seed=123)
        env.reset(seed=123)

        with self.assertRaises(ValueError):
            env.step(0)


if __name__ == "__main__":
    unittest.main()
