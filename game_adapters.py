from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gym
import numpy as np


@dataclass(frozen=True)
class GameAdapterConfig:
    game: str
    env_id: str


class GameAdapter(ABC):
    """Interface for game-specific environment and action handling."""

    def __init__(self, config: GameAdapterConfig):
        self.config = config

    def make_env(self) -> gym.Env:
        return gym.make(self.config.env_id)

    def reset_env(self, env: gym.Env, *, seed: int, options: dict[str, Any] | None = None):
        return env.reset(seed=seed, options=options)

    def step_env(self, env: gym.Env, action: Any):
        return env.step(action)

    def num_seats(self, env: gym.Env) -> int:
        """Return number of player seats used by the environment."""
        for attr in ("n_players", "num_players", "nPlayers"):
            value = getattr(env, attr, None)
            if value is not None:
                return int(value)

        players = getattr(env, "players", None)
        if players is not None:
            return int(len(players))

        raise RuntimeError("Could not determine number of seats from environment")

    @abstractmethod
    def is_win(self, info: dict[str, Any], learning_seat: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_valid_action_mask(self, obs: Any, action_space_n: int) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def format_env_action(self, policy_action: int, action_space_n: int) -> Any:
        raise NotImplementedError

    def shape_reward(self, reward: float, _obs: Any, _next_obs: Any, _info: dict[str, Any]) -> float:
        """Optional reward shaping hook; default keeps environment reward unchanged."""
        return float(reward)


class ChefshatAdapter(GameAdapter):
    def __init__(self, env_id: str = "chefshat-v1"):
        super().__init__(GameAdapterConfig(game="chefshat", env_id=env_id))

    def make_env(self) -> gym.Env:
        __import__("ChefsHatGym.env")
        return super().make_env()

    def num_seats(self, env: gym.Env) -> int:
        return int(getattr(env, "n_players", 4))

    def is_win(self, info: dict[str, Any], learning_seat: int) -> bool:
        scores = info.get("Match_Score", [])
        return bool(scores) and int(scores[learning_seat]) == 3

    def extract_valid_action_mask(self, obs: Any, action_space_n: int) -> np.ndarray:
        obs_array = np.asarray(obs).ravel()
        start = 28
        end = start + action_space_n
        mask = obs_array[start:end].astype(bool)
        if mask.size != action_space_n:
            raise RuntimeError(
                f"Expected action mask of length {action_space_n} starting at index {start}, "
                f"but got slice of length {mask.size} from observation of length {obs_array.size}."
            )
        return mask

    def format_env_action(self, policy_action: int, action_space_n: int) -> np.ndarray:
        action_vec = np.zeros(action_space_n, dtype=np.float32)
        action_vec[int(policy_action)] = 1.0
        return action_vec


class IrpsAdapter(GameAdapter):
    def __init__(self, env_id: str = "irps-v1"):
        super().__init__(GameAdapterConfig(game="irps", env_id=env_id))

    def num_seats(self, env: gym.Env) -> int:
        for attr in ("n_players", "num_players", "nPlayers"):
            value = getattr(env, attr, None)
            if value is not None:
                return int(value)
        return 2

    def is_win(self, info: dict[str, Any], learning_seat: int) -> bool:
        winner = info.get("winner")
        if winner is not None:
            return int(winner) == int(learning_seat)

        scores = info.get("scores")
        if isinstance(scores, (list, tuple)) and len(scores) > learning_seat:
            learning_score = scores[learning_seat]
            return learning_score == max(scores)

        return bool(info.get("win", False))

    def extract_valid_action_mask(self, obs: Any, action_space_n: int) -> np.ndarray:
        return np.ones(action_space_n, dtype=bool)

    def format_env_action(self, policy_action: int, action_space_n: int) -> int:
        return int(policy_action)


def get_game_adapter(game: str, env_id: str | None = None) -> GameAdapter:
    normalized = game.lower().strip()
    if normalized == "chefshat":
        return ChefshatAdapter(env_id=env_id or "chefshat-v1")
    if normalized == "irps":
        return IrpsAdapter(env_id=env_id or "irps-v1")
    raise ValueError(f"Unsupported game '{game}'. Expected one of: chefshat, irps")
