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

    def adapt_observation(self, obs: Any, info: dict[str, Any] | None = None) -> Any:
        """Return the policy-facing observation representation."""
        del info
        return obs

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

    def get_action_mask(self, obs: Any, env: gym.Env) -> np.ndarray | None:
        """Resolve a valid-action mask from an observation and environment."""
        action_space_n = int(getattr(env.action_space, "n"))
        return self.extract_valid_action_mask(obs, action_space_n)

    @abstractmethod
    def format_env_action(self, policy_action: int, action_space_n: int) -> Any:
        raise NotImplementedError

    def shape_reward(self, reward: float, _obs: Any, _next_obs: Any, _info: dict[str, Any]) -> float:
        """Optional reward shaping hook; default keeps environment reward unchanged."""
        return float(reward)

    def get_state_features_for_shaping(self, obs: Any, info: dict[str, Any] | None = None) -> np.ndarray | None:
        """Optional compact state feature vector used by wrapper shaping."""
        del obs, info
        return None


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

    def get_state_features_for_shaping(self, obs: Any, info: dict[str, Any] | None = None) -> np.ndarray:
        del info
        obs_array = np.asarray(obs).ravel()
        hand = obs_array[11:28]
        if hand.size != 17:
            raise RuntimeError(
                f"Expected 17 hand-card features at observation indices [11:28), got {hand.size}."
            )
        return hand.astype(np.float32)


class IrpsAdapter(GameAdapter):
    def __init__(self, env_id: str = "irps-v1"):
        super().__init__(GameAdapterConfig(game="irps", env_id=env_id))

    def make_env(self) -> gym.Env:
        __import__("irps_env")
        return super().make_env()

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

    def adapt_observation(self, obs: Any, info: dict[str, Any] | None = None) -> np.ndarray:
        """Compact IRPS observation: [p_rock,p_paper,p_scissors,o_rock,o_paper,o_scissors]."""

        def _first_non_none(*values: np.ndarray | None) -> np.ndarray | None:
            for value in values:
                if value is not None:
                    return value
            return None

        def _wins_from(source: Any) -> np.ndarray | None:
            if source is None:
                return None
            if isinstance(source, dict):
                values = [source.get("rock", 0), source.get("paper", 0), source.get("scissors", 0)]
                return np.asarray(values, dtype=np.int32)
            arr = np.asarray(source).ravel()
            if arr.size >= 3:
                return arr[:3].astype(np.int32)
            return None

        info = info or {}
        player_wins = _first_non_none(
            _wins_from(info.get("player_wins_by_move")),
            _wins_from(info.get("agent_wins_by_move")),
            _wins_from(info.get("wins_by_move")),
        )
        opponent_wins = _first_non_none(
            _wins_from(info.get("opponent_wins_by_move")),
            _wins_from(info.get("enemy_wins_by_move")),
        )

        if player_wins is None or opponent_wins is None:
            obs_array = np.asarray(obs).ravel()
            if obs_array.size >= 6:
                if player_wins is None:
                    player_wins = obs_array[:3].astype(np.int32)
                if opponent_wins is None:
                    opponent_wins = obs_array[3:6].astype(np.int32)

        if player_wins is None:
            player_wins = np.zeros(3, dtype=np.int32)
        if opponent_wins is None:
            opponent_wins = np.zeros(3, dtype=np.int32)

        compact = np.concatenate([player_wins, opponent_wins]).astype(np.float32)
        return np.clip(compact, 0, 2)

    def extract_valid_action_mask(self, obs: Any, action_space_n: int) -> np.ndarray:
        if action_space_n != 3:
            raise RuntimeError(f"IRPS expects exactly 3 actions (rock,paper,scissors), got {action_space_n}")
        return np.ones(3, dtype=bool)

    def format_env_action(self, policy_action: int, action_space_n: int) -> int:
        return int(policy_action)


def get_game_adapter(game: str, env_id: str | None = None) -> GameAdapter:
    normalized = game.lower().strip()
    if normalized == "chefshat":
        return ChefshatAdapter(env_id=env_id or "chefshat-v1")
    if normalized == "irps":
        return IrpsAdapter(env_id=env_id or "irps-v1")
    raise ValueError(f"Unsupported game '{game}'. Expected one of: chefshat, irps")
