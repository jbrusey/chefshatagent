from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import gymnasium as gym
import numpy as np


@dataclass
class TurnStep:
    obs: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class SingleAgentWrapper(gym.Env):
    """
    Collapses a 4-player Chef's Hat game into a single-agent MDP for seat 0.

    The learning agent controls seat 0. Seats 1-3 are played by random legal moves.
    During both reset() and step(), this wrapper auto-advances turns until it is
    seat 0's turn (or the episode ends), while accumulating intermediate rewards.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_id: str = "ChefsHat-v0", learning_seat: int = 0, seed: int = 42):
        super().__init__()
        self.learning_seat = learning_seat
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.base_env = gym.make(env_id)
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

        self._last_obs: Any | None = None
        self._last_info: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        reset_out = self.base_env.reset(seed=self._seed, options=options)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}

        self._last_obs = obs
        self._last_info = info

        obs, _, terminated, truncated, info = self._advance_until_learning_turn(obs, info)
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def step(self, action: int):
        if not self._is_action_valid(action):
            raise ValueError(f"Illegal action {action} for current action mask")

        step = self._step_base(action)
        total_reward = step.reward

        if step.terminated or step.truncated:
            self._last_obs = step.obs
            self._last_info = step.info
            return step.obs, total_reward, step.terminated, step.truncated, step.info

        obs, reward_add, terminated, truncated, info = self._advance_until_learning_turn(step.obs, step.info)
        total_reward += reward_add

        self._last_obs = obs
        self._last_info = info
        return obs, float(total_reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean valid-action mask compatible with sb3-contrib MaskablePPO."""
        return self._compute_action_mask(self._last_info)

    # Alias requested in the prompt text.
    def get_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def render(self):
        if hasattr(self.base_env, "render"):
            return self.base_env.render()
        return None

    def close(self):
        self.base_env.close()

    def _advance_until_learning_turn(self, obs: Any, info: dict[str, Any]):
        total_reward = 0.0
        terminated = False
        truncated = False
        current_obs = obs
        current_info = info

        while not (terminated or truncated) and self._current_player(current_info) != self.learning_seat:
            mask = self._compute_action_mask(current_info)
            valid_actions = np.flatnonzero(mask)
            if valid_actions.size == 0:
                raise RuntimeError("No valid actions available for non-learning player")

            random_action = int(self._rng.choice(valid_actions))
            step = self._step_base(random_action)
            total_reward += step.reward
            current_obs, current_info = step.obs, step.info
            terminated, truncated = step.terminated, step.truncated

        return current_obs, float(total_reward), terminated, truncated, current_info

    def _step_base(self, action: int) -> TurnStep:
        out = self.base_env.step(action)

        # Gymnasium API: obs, reward, terminated, truncated, info
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return TurnStep(obs, float(reward), bool(terminated), bool(truncated), dict(info))

        # Legacy API fallback: obs, reward, done, info
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return TurnStep(obs, float(reward), bool(done), False, dict(info))

        raise RuntimeError("Unsupported environment step() return format")

    def _is_action_valid(self, action: int) -> bool:
        mask = self._compute_action_mask(self._last_info)
        return 0 <= int(action) < mask.size and bool(mask[int(action)])

    def _compute_action_mask(self, info: dict[str, Any] | None = None) -> np.ndarray:
        info = info or {}

        candidate_masks = [
            info.get("action_mask"),
            info.get("action_masks"),
            getattr(self.base_env, "action_mask", None),
            getattr(self.base_env, "action_masks", None),
        ]
        for mask in candidate_masks:
            if mask is None:
                continue
            mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
            if mask_arr.size == self.action_space.n:
                return mask_arr

        valid_list = self._extract_valid_actions(info)
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[valid_list] = True
        return mask

    def _extract_valid_actions(self, info: dict[str, Any]) -> np.ndarray:
        candidates: list[Iterable[int] | None] = [
            info.get("valid_actions"),
            info.get("legal_actions"),
            getattr(self.base_env, "valid_actions", None),
            getattr(self.base_env, "legal_actions", None),
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            arr = np.asarray(list(candidate), dtype=int).reshape(-1)
            arr = arr[(arr >= 0) & (arr < self.action_space.n)]
            if arr.size > 0:
                return np.unique(arr)

        raise RuntimeError(
            "Could not determine valid actions. Base env must expose either an action mask "
            "or a valid/legal action list."
        )

    def _current_player(self, info: dict[str, Any]) -> int:
        candidates = [
            info.get("current_player"),
            info.get("currentPlayer"),
            getattr(self.base_env, "current_player", None),
            getattr(self.base_env, "currentPlayer", None),
            getattr(getattr(self.base_env, "unwrapped", None), "current_player", None),
            getattr(getattr(self.base_env, "unwrapped", None), "currentPlayer", None),
        ]
        for value in candidates:
            if value is not None:
                return int(value)

        raise RuntimeError("Could not determine current player from env info/attributes")
