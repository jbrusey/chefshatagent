from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass
from typing import Any

import gym
import numpy as np
import ChefsHatGym.env  # Registers gym.make("chefshat-v1") environment.
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning
from ChefsHatGym.rewards.reward import Reward

GAMMA=0.99


@contextlib.contextmanager
def _silence():
    """Suppress all stdout, stderr, and logging output from the ChefsHat library."""
    null = io.StringIO()
    root_logger = logging.getLogger()
    original_level = root_logger.level
    # Raise root logger level so the library's loggers don't emit anything
    root_logger.setLevel(logging.CRITICAL + 1)
    with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
        try:
            yield
        finally:
            root_logger.setLevel(original_level)


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

    def __init__(self, env_id: str = "chefshat-v1", learning_seat: int = 0, seed: int = 42,
                 reward_fn: Reward | None = None):
        super().__init__()
        self.learning_seat = learning_seat
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # Use RewardOnlyWinning by default; pass any ChefsHatGym Reward subclass to override.
        self._reward_fn: Reward = reward_fn if reward_fn is not None else RewardOnlyWinning()

        self.base_env = gym.make(env_id)

        # Start experiment once when the backend environment supports it.
        if hasattr(self.base_env, "startExperiment"):
            with _silence():
                self.base_env.startExperiment(
                    playerNames=["RL", "Random1", "Random2", "Random3"],
                    logDirectory="log",
                    verbose=False
                )
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

        self._last_obs: Any | None = None
        self._last_info: dict[str, Any] = {}

    def _phi(self, obs):
        hand = obs[11:28]
        cards_left = np.count_nonzero(hand)
        return -(cards_left / 17.0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        with _silence():
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

    def _win_reward(self, info: dict[str, Any]) -> float:
        """Terminal bonus from the ChefsHatGym reward function.

        Match_Score = 3 - finishing_position, so finishing_position = 3 - Match_Score.
        RewardOnlyWinning:  position=0 (1st) → +1.0, else → 0.0  (matchFinished=True)
        """
        scores = info.get("Match_Score", [])
        if not scores:
            return 0.0
        match_score = int(scores[self.learning_seat])       # 0–3
        finishing_position = 3 - match_score                # 0=1st … 3=last
        return float(self._reward_fn.getReward(finishing_position, True))

    def _step_penalty(self) -> float:
        """Per-step penalty as defined by the reward function (non-terminal call)."""
        return float(self._reward_fn.getReward(0, False))

    def step(self, action: int):
        if not self._is_action_valid(action):
            raise ValueError(f"Illegal action {action} for current action mask")

        # Capture phi BEFORE the agent's action and before any opponent turns
        # overwrite self._last_obs — this is the timing bug fix.
        phi_prev = self._phi(self._last_obs)

        step = self._step_base(action)
        total_reward = step.reward

        if step.terminated or step.truncated:
            terminal_reward = self._win_reward(step.info)
            self._last_obs = step.obs
            self._last_info = step.info
            return step.obs, total_reward + terminal_reward, step.terminated, step.truncated, step.info

        obs, reward_add, terminated, truncated, info = self._advance_until_learning_turn(step.obs, step.info)
        total_reward += reward_add

        # potential-based reward shaping: F(s,s') = γ·φ(s') − φ(s)
        phi_next = self._phi(obs)
        shaped_reward = total_reward + GAMMA * phi_next - phi_prev + self._step_penalty()

        # Add terminal win bonus if the episode ended during opponent turns
        if terminated or truncated:
            shaped_reward += self._win_reward(info)

        self._last_obs = obs
        self._last_info = info

        return obs, shaped_reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean valid-action mask compatible with sb3-contrib MaskablePPO."""
        if self._last_obs is None:
            raise RuntimeError("No observation available yet. Call reset() before action_masks().")
        mask = np.asarray(self._last_obs)[28:]
        if mask.size != self.action_space.n:
            raise RuntimeError(
                f"Observation mask slice has size {mask.size}, expected {self.action_space.n}"
            )
        return mask.astype(bool)

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
            self._last_obs = current_obs
            mask = self.action_masks()
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
        action_vec = np.zeros(self.action_space.n, dtype=np.float32)
        action_vec[int(action)] = 1.0
        out = self.base_env.step(action_vec)

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
        mask = self.action_masks()
        return 0 <= int(action) < mask.size and bool(mask[int(action)])

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
