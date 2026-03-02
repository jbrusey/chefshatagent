from __future__ import annotations

import argparse
from pathlib import Path

import gym
import numpy as np
import ChefsHatGym.env  # noqa: F401  # Registers gym.make("chefshat-v1") environment.
from sb3_contrib import MaskablePPO

from evaluate import _resolve_model_path


DEFAULT_MODEL_PATH = Path("models/ppo_chefhats_masked")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local Chef's Hat game with a trained MaskablePPO model and env GUI/logging"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--env-id", default="chefshat-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
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


def _action_mask_from_obs(obs: np.ndarray, action_count: int) -> np.ndarray:
    mask = np.asarray(obs)[28:28 + action_count]
    if mask.size != action_count:
        raise RuntimeError(
            f"Observation mask slice has size {mask.size}, expected {action_count}."
        )
    return mask.astype(bool)


def _to_one_hot(action: int, action_count: int) -> np.ndarray:
    if not (0 <= action < action_count):
        raise ValueError(f"Action {action} out of bounds for size {action_count}")
    one_hot = np.zeros(action_count, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot


def _current_player(info: dict, fallback_seat: int) -> int:
    return int(info.get("current_player", fallback_seat))


def _play_episode(
    env: gym.Env,
    model: MaskablePPO,
    rng: np.random.Generator,
    *,
    learning_seat: int,
    deterministic: bool,
    seed: int,
) -> tuple[bool, dict]:
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        current_player = _current_player(info, learning_seat)
        mask = _action_mask_from_obs(obs, env.action_space.n)
        valid_actions = np.flatnonzero(mask)
        if valid_actions.size == 0:
            raise RuntimeError("No valid actions available")

        if current_player == learning_seat:
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            action = int(action)
            if action not in valid_actions:
                raise RuntimeError(
                    f"Model predicted illegal action {action}; valid actions are {valid_actions.tolist()}"
                )
        else:
            action = int(rng.choice(valid_actions))

        obs, _, terminated, truncated, info = env.step(_to_one_hot(action, env.action_space.n))

    scores = info.get("Match_Score", [])
    won = bool(scores) and int(scores[learning_seat]) == 3
    return won, info


def main() -> None:
    args = _build_arg_parser().parse_args()

    resolved_model = _resolve_model_path(args.model_path)
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
