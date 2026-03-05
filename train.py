from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from single_agent_wrapper import SingleAgentWrapper

SEED = 42
GAMMA = 0.99
TOTAL_TIMESTEPS = 200_000
SELF_PLAY_ITERATIONS = 5
MODEL_PATH = Path("models/ppo_chefhats_masked")
LATEST_MODEL_PATH = Path("models/latest.zip")
RANDOM_OPPONENT_TOKEN = "__RANDOM__"


def _snapshot_sort_key(path: Path) -> tuple[int, float, str]:
    match = re.search(r"snapshot_(\d+)\.zip$", path.name)
    iteration = int(match.group(1)) if match else -1
    return (iteration, path.stat().st_mtime, path.name)


def build_initial_opponent_pool(models_dir: Path) -> list[str]:
    """Build deterministic initial opponent pool from available frozen models."""
    latest = models_dir / "latest.zip"
    pool: list[str] = []
    if latest.exists():
        pool.append(str(latest))

    snapshots = sorted(models_dir.glob("snapshot_*.zip"), key=_snapshot_sort_key, reverse=True)
    for snapshot_path in snapshots[:4]:
        snapshot_str = str(snapshot_path)
        if snapshot_str not in pool:
            pool.append(snapshot_str)

    pool.append(RANDOM_OPPONENT_TOKEN)
    return pool


class WinRateCallback(BaseCallback):
    """Tracks win rate (1st place finishes) and logs it to the SB3 console output.

    A 'win' is defined as Match_Score[learning_seat] == 3 (1st place / Chef).
    The rate is computed over all episodes completed since the last rollout log.
    """

    def __init__(
        self,
        learning_seat: int = 0,
        verbose: int = 0,
        wandb_run: Any | None = None,
    ):
        super().__init__(verbose)
        self.learning_seat = learning_seat
        self.wandb_run = wandb_run
        self._episode_outcomes: list[int] = []  # 1 = win, 0 = loss

    def _on_step(self) -> bool:
        # SB3 stores per-step info in self.locals["infos"] (list, one per env)
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                scores = info.get("Match_Score", [])
                if scores:
                    won = int(int(scores[self.learning_seat]) == 3)
                    self._episode_outcomes.append(won)
        return True

    def _on_rollout_end(self) -> None:
        if self._episode_outcomes:
            win_rate = float(np.mean(self._episode_outcomes))
            self.logger.record("rollout/win_rate", win_rate)
            self._episode_outcomes.clear()


class WandbMetricsCallback(BaseCallback):
    """Mirrors SB3 scalar logger outputs to Weights & Biases.

    This captures the standard PPO diagnostics (e.g. fps, policy/value losses,
    entropy, explained variance, KL) whenever they are present in the SB3 logger,
    and logs them with `num_timesteps` as the global step.
    """

    def __init__(self, wandb_run: Any, verbose: int = 0):
        super().__init__(verbose)
        self.wandb_run = wandb_run

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        logger_values = getattr(self.model.logger, "name_to_value", {})
        metrics: dict[str, float | int] = {}

        for key, value in logger_values.items():
            if isinstance(value, np.generic):
                metrics[key] = value.item()
            elif isinstance(value, (float, int)):
                metrics[key] = value

        if metrics:
            metrics["timesteps"] = self.num_timesteps
            self.wandb_run.log(metrics, step=self.num_timesteps)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MaskablePPO for Chef's Hat self-play")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires `uv pip install wandb`).",
    )
    parser.add_argument(
        "--wandb-project",
        default="chefhats-rl",
        help="wandb project name when --wandb is enabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional wandb run name when --wandb is enabled.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "--wandb was set but the `wandb` package is not installed. "
                "Install it with `uv pip install wandb`."
            ) from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "seed": SEED,
                "gamma": GAMMA,
                "total_timesteps": TOTAL_TIMESTEPS,
                "self_play_iterations": SELF_PLAY_ITERATIONS,
            },
        )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Bootstrap: if no latest model exists yet, create and save a fresh one.
    if not LATEST_MODEL_PATH.exists():
        _env = SingleAgentWrapper(env_id="chefshat-v1", learning_seat=0, seed=SEED)
        try:
            MaskablePPO("MlpPolicy", _env, gamma=GAMMA, seed=SEED).save(str(LATEST_MODEL_PATH))
        finally:
            _env.close()

    opponent_pool = build_initial_opponent_pool(MODEL_PATH.parent)
    print(f"Initial opponent pool: {opponent_pool}")
    env = SingleAgentWrapper(
        env_id="chefshat-v1",
        learning_seat=0,
        seed=SEED,
        opponent_pool=opponent_pool,
    )

    model = MaskablePPO.load(
        str(LATEST_MODEL_PATH),
        env=env,
    )
    model.gamma = GAMMA
    model.seed = SEED

    try:
        for iteration in range(1, SELF_PLAY_ITERATIONS + 1):
            if iteration > 1:
                env.close()
                env = SingleAgentWrapper(
                    env_id="chefshat-v1",
                    learning_seat=0,
                    seed=SEED,
                    opponent_pool=opponent_pool,
                )
                model.set_env(env)

            checkpoint_callback = CheckpointCallback(
                save_freq=10_000,
                save_path=str(MODEL_PATH.parent),
                name_prefix=f"{MODEL_PATH.name}_iter_{iteration}",
            )

            callbacks: list[BaseCallback] = [
                WinRateCallback(learning_seat=0, wandb_run=wandb_run),
                checkpoint_callback,
            ]
            if wandb_run is not None:
                callbacks.append(WandbMetricsCallback(wandb_run=wandb_run))

            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=CallbackList(callbacks),
                reset_num_timesteps=False,
            )

            snapshot_path = MODEL_PATH.parent / f"snapshot_{iteration}.zip"
            model.save(str(snapshot_path))
            model.save(str(LATEST_MODEL_PATH))
            opponent_pool.append(str(snapshot_path))
            print(f"Iteration {iteration}: saved snapshot to {snapshot_path}")
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
