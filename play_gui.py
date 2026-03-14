from __future__ import annotations

import argparse
from pathlib import Path

import gym
import numpy as np
import ChefsHatGym.env  # noqa: F401  # Registers gym.make("chefshat-v1") environment.
from sb3_contrib import MaskablePPO

from game_adapters import GameAdapter, get_game_adapter
from utils import resolve_model_path


DEFAULT_MODEL_PATH = Path("models/ppo_chefhats_masked")


def _positive_int(value: str, arg_name: str) -> int:
    try:
        v = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{arg_name} must be a positive integer, got: {value!r}")
    if v < 1:
        raise argparse.ArgumentTypeError(f"{arg_name} must be >= 1, got {v}")
    return v


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local Chef's Hat game with a trained MaskablePPO model and env GUI/logging"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--game", choices=["chefshat", "irps"], default="chefshat")
    parser.add_argument("--env-id", default="chefshat-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=lambda v: _positive_int(v, "--episodes"), default=1)
    parser.add_argument("--learning-seat", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions for the learning seat.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ChefsHat startExperiment verbosity (use --no-verbose to reduce output).",
    )
    parser.add_argument("--log-directory", default="log")
    return parser


def _action_mask_from_obs(obs: np.ndarray, env: gym.Env, adapter: GameAdapter) -> np.ndarray:
    mask = adapter.get_action_mask(obs, env)
    if mask is None:
        return np.ones(env.action_space.n, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _to_one_hot(action: int, action_count: int) -> np.ndarray:
    if not (0 <= action < action_count):
        raise ValueError(f"Action {action} out of bounds for size {action_count}")
    one_hot = np.zeros(action_count, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot


def _current_player(info: dict, env: gym.Env) -> int:
    candidates = [
        info.get("current_player"),
        info.get("currentPlayer"),
        getattr(env, "current_player", None),
        getattr(env, "currentPlayer", None),
        getattr(getattr(env, "unwrapped", None), "current_player", None),
        getattr(getattr(env, "unwrapped", None), "currentPlayer", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    raise RuntimeError("Could not determine current player from env info/attributes")


def _play_episode(
    env: gym.Env,
    model: MaskablePPO,
    rng: np.random.Generator,
    *,
    learning_seat: int,
    adapter: GameAdapter,
    deterministic: bool,
    seed: int,
) -> tuple[bool, dict]:
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        current_player = _current_player(info, env)
        mask = _action_mask_from_obs(obs, env, adapter)
        valid_actions = np.flatnonzero(mask)
        if valid_actions.size == 0:
            raise RuntimeError("No valid actions available")

        if current_player == learning_seat:
            policy_obs = adapter.adapt_observation(obs, info)
            action, _ = model.predict(policy_obs, action_masks=mask, deterministic=deterministic)
            action = int(action)
            if action not in valid_actions:
                raise RuntimeError(
                    f"Model predicted illegal action {action}; valid actions are {valid_actions.tolist()}"
                )
        else:
            action = int(rng.choice(valid_actions))

        obs, _, terminated, truncated, info = env.step(adapter.format_env_action(action, env.action_space.n))

    scores = info.get("Match_Score", [])
    won = bool(scores) and int(scores[learning_seat]) == 3
    return won, info


def main() -> None:
    args = _build_arg_parser().parse_args()

    resolved_model = resolve_model_path(args.model_path)
    adapter = get_game_adapter(args.game, env_id=args.env_id)
    model = MaskablePPO.load(str(resolved_model))
    print(f"Loaded model: {resolved_model}")

    env = gym.make(args.env_id)
    if hasattr(env, "startExperiment"):
        env.startExperiment(
            playerNames=["Seat0", "Seat1", "Seat2", "Seat3"],
            logDirectory=args.log_directory,
            verbose=args.verbose,
        )

    rng = np.random.default_rng(args.seed)
    wins = 0

    try:
        for episode in range(args.episodes):
            won, info = _play_episode(
                env,
                model,
                rng,
                learning_seat=args.learning_seat,
                adapter=adapter,
                deterministic=args.deterministic,
                seed=args.seed + episode,
            )
            wins += int(won)
            scores = info.get("Match_Score", [])
            print(f"Episode {episode + 1}/{args.episodes} - win={won} scores={scores}")
    finally:
        env.close()

    print(f"Wins: {wins}/{args.episodes} ({(100.0 * wins / args.episodes):.1f}%)")


if __name__ == "__main__":
    main()
