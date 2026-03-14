from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gym
import numpy as np
from gym.envs.registration import register, registry


ActionPolicy = Callable[[np.ndarray, np.random.Generator], int]


class IrpsEnv(gym.Env):
    """Iterated Rock-Paper-Scissors environment.

    Observation is a 6-vector: [agent_r, agent_p, agent_s, opp_r, opp_p, opp_s].
    Each bucket is clipped to [0, 2] for policy input stability.
    The underlying win counters continue growing until a player reaches 3 total wins.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42, opponent_policy: ActionPolicy | None = None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(6,),
            dtype=np.float32,
        )
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._opponent_policy = opponent_policy
        self._agent_wins = np.zeros(3, dtype=np.int32)
        self._opponent_wins = np.zeros(3, dtype=np.int32)

    def _sample_opponent_action(self) -> int:
        if self._opponent_policy is None:
            return int(self._rng.integers(0, 3))
        action = int(self._opponent_policy(self._get_observation(), self._rng))
        if action not in (0, 1, 2):
            raise RuntimeError(f"Opponent policy produced invalid action {action}; expected one of (0,1,2).")
        return action

    def _winner(self, agent_action: int, opponent_action: int) -> int | None:
        if agent_action == opponent_action:
            return None
        if (agent_action - opponent_action) % 3 == 1:
            return 0
        return 1

    def _get_observation(self) -> np.ndarray:
        obs = np.concatenate([self._agent_wins, self._opponent_wins]).astype(np.float32)
        return np.clip(obs, 0, 2)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

        self._agent_wins = np.zeros(3, dtype=np.int32)
        self._opponent_wins = np.zeros(3, dtype=np.int32)

        info = {
            "agent_wins_by_move": self._agent_wins.copy(),
            "opponent_wins_by_move": self._opponent_wins.copy(),
            "scores": [0, 0],
            "winner": None,
        }
        return self._get_observation(), info

    def step(self, action: int):
        agent_action = int(action)
        if agent_action not in (0, 1, 2):
            raise ValueError(f"IRPS action must be in {{0,1,2}}, got {agent_action}")

        opponent_action = self._sample_opponent_action()
        winner = self._winner(agent_action, opponent_action)

        reward = 0.0
        if winner == 0:
            self._agent_wins[agent_action] += 1
            reward = 1.0
        elif winner == 1:
            self._opponent_wins[opponent_action] += 1
            reward = -1.0

        agent_total = int(self._agent_wins.sum())
        opponent_total = int(self._opponent_wins.sum())
        terminated = agent_total >= 3 or opponent_total >= 3
        truncated = False

        terminal_winner = None
        if terminated:
            terminal_winner = 0 if agent_total >= 3 else 1

        info = {
            "agent_action": agent_action,
            "opponent_action": opponent_action,
            "round_winner": winner,
            "winner": terminal_winner,
            "agent_wins_by_move": self._agent_wins.copy(),
            "opponent_wins_by_move": self._opponent_wins.copy(),
            "scores": [agent_total, opponent_total],
        }

        return self._get_observation(), float(reward), terminated, truncated, info


def make_single_agent_env(
    *,
    game: str,
    env_id: str,
    learning_seat: int = 0,
    seed: int = 42,
    opponent_pool: list[str] | None = None,
):
    if game == "irps":
        return IrpsEnv(seed=seed)

    from single_agent_wrapper import SingleAgentWrapper

    return SingleAgentWrapper(
        env_id=env_id,
        learning_seat=learning_seat,
        seed=seed,
        opponent_pool=opponent_pool,
    )


if "irps-v1" not in registry:
    register(id="irps-v1", entry_point="irps_env:IrpsEnv")
