from __future__ import annotations

import argparse
import contextlib
import io
import logging
import re
from pathlib import Path
from typing import Any

import gym
import numpy as np
import ChefsHatGym.env  # noqa: F401  # Registers gym.make("chefshat-v1") environment.
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from population import DEFAULT_ELO, load_ratings, save_ratings, top_rated_players, update_ratings_from_match
from single_agent_wrapper import RANDOM_OPPONENT_TOKEN, SingleAgentWrapper

SEED = 42
GAMMA = 0.99
TOTAL_TIMESTEPS = 200_000
SELF_PLAY_ITERATIONS = 5
MODEL_PATH = Path("models/ppo_chefhats_masked")
LATEST_MODEL_PATH = Path("models/latest.zip")
ELO_RATINGS_PATH = Path("models/elo_ratings.json")
ELO_EVAL_GAMES = 24
ELO_K_FACTOR = 24.0
MAX_OPPONENT_POOL_SIZE = 12


def _snapshot_sort_key(path: Path) -> tuple[int, float, str]:
    match = re.search(r"snapshot_(\d+)\.zip$", path.name)
    iteration = int(match.group(1)) if match else -1
    return (iteration, path.stat().st_mtime, path.name)


def build_initial_opponent_pool(models_dir: Path) -> list[str]:
    """Build deterministic initial opponent pool from available frozen models."""
    latest = models_dir / LATEST_MODEL_PATH.name
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


@contextlib.contextmanager
def _silence():
    """Suppress noisy Chef's Hat stdout/stderr/logging during evaluation matches."""
    null = io.StringIO()
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.CRITICAL + 1)
    with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
        try:
            yield
        finally:
            root_logger.setLevel(original_level)


def _current_player(base_env: gym.Env, info: dict[str, Any]) -> int:
    candidates = [
        info.get("current_player"),
        info.get("currentPlayer"),
        getattr(base_env, "current_player", None),
        getattr(base_env, "currentPlayer", None),
        getattr(getattr(base_env, "unwrapped", None), "current_player", None),
        getattr(getattr(base_env, "unwrapped", None), "currentPlayer", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    raise RuntimeError("Could not determine current player from env info/attributes")


def _resolve_seat_action(
    seat_agent: str,
    obs: np.ndarray,
    action_space_n: int,
    policy_cache: dict[str, MaskablePPO],
    rng: np.random.Generator,
) -> int:
    mask = np.asarray(obs)[28:28 + action_space_n].astype(bool)
    valid_actions = np.flatnonzero(mask)
    if valid_actions.size == 0:
        raise RuntimeError("No valid actions available")

    if seat_agent == RANDOM_OPPONENT_TOKEN:
        return int(rng.choice(valid_actions))

    policy = policy_cache.get(seat_agent)
    if policy is None:
        policy = MaskablePPO.load(seat_agent)
        policy_cache[seat_agent] = policy

    action, _ = policy.predict(obs, deterministic=True, action_masks=mask)
    action = int(action)
    if action not in valid_actions:
        raise RuntimeError(f"Agent {seat_agent} produced illegal action {action}")
    return action


def play_match_and_get_scores(
    seat_agents: list[str],
    seed: int,
    policy_cache: dict[str, MaskablePPO],
) -> list[int]:
    rng = np.random.default_rng(seed)
    base_env = gym.make("chefshat-v1")
    try:
        if hasattr(base_env, "startExperiment"):
            with _silence():
                base_env.startExperiment(
                    playerNames=["Elo0", "Elo1", "Elo2", "Elo3"],
                    logDirectory="log",
                    verbose=False,
                )

        with _silence():
            reset_out = base_env.reset(seed=seed)

        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}

        terminated = False
        truncated = False
        while not (terminated or truncated):
            current_player = _current_player(base_env, info)
            action = _resolve_seat_action(
                seat_agent=seat_agents[current_player],
                obs=obs,
                action_space_n=base_env.action_space.n,
                policy_cache=policy_cache,
                rng=rng,
            )

            action_vec = np.zeros(base_env.action_space.n, dtype=np.float32)
            action_vec[action] = 1.0
            with _silence():
                step_out = base_env.step(action_vec)

            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, _reward, terminated, truncated, info = step_out
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                obs, _reward, done, info = step_out
                terminated, truncated = bool(done), False
            else:
                raise RuntimeError("Unsupported environment step() return format")

        scores = info.get("Match_Score", [])
        if not scores:
            raise RuntimeError("Match_Score missing from terminal info")
        return [int(v) for v in scores]
    finally:
        base_env.close()


def evaluate_snapshot_elo(
    snapshot_path: str,
    ratings: dict[str, float],
    evaluation_pool: list[str],
    games: int,
    seed: int,
) -> None:
    policy_cache: dict[str, MaskablePPO] = {}
    rng = np.random.default_rng(seed)

    for _ in range(games):
        if len(evaluation_pool) >= 3:
            chosen = list(rng.choice(evaluation_pool, size=3, replace=False))
        else:
            chosen = list(rng.choice(evaluation_pool, size=3, replace=True))

        seat_agents = [snapshot_path, *chosen]
        match_scores = play_match_and_get_scores(
            seat_agents=seat_agents,
            seed=int(rng.integers(0, 2**31 - 1)),
            policy_cache=policy_cache,
        )
        update_ratings_from_match(
            ratings=ratings,
            players=seat_agents,
            match_scores=match_scores,
            k_factor=ELO_K_FACTOR,
        )


def build_training_opponent_pool(ratings: dict[str, float], latest_model: str) -> list[str]:
    rated_models = [path for path in top_rated_players(ratings, MAX_OPPONENT_POOL_SIZE) if Path(path).exists()]
    if latest_model not in rated_models:
        rated_models.insert(0, latest_model)
    rated_models.append(RANDOM_OPPONENT_TOKEN)
    return rated_models


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
    ratings = load_ratings(ELO_RATINGS_PATH)
    ratings.setdefault(str(LATEST_MODEL_PATH), DEFAULT_ELO)
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
            snapshot_key = str(snapshot_path)
            ratings.setdefault(snapshot_key, ratings.get(str(LATEST_MODEL_PATH), DEFAULT_ELO))

            eval_candidates = [path for path in opponent_pool if path != RANDOM_OPPONENT_TOKEN]
            if not eval_candidates:
                eval_candidates = [str(LATEST_MODEL_PATH)]
            evaluate_snapshot_elo(
                snapshot_path=snapshot_key,
                ratings=ratings,
                evaluation_pool=eval_candidates,
                games=ELO_EVAL_GAMES,
                seed=SEED + iteration,
            )

            save_ratings(ELO_RATINGS_PATH, ratings)
            opponent_pool = build_training_opponent_pool(ratings, latest_model=str(LATEST_MODEL_PATH))
            print(f"Iteration {iteration}: saved snapshot to {snapshot_path}")
            print(f"Iteration {iteration}: saved Elo ratings to {ELO_RATINGS_PATH}")
            print(f"Iteration {iteration}: top pool candidates {opponent_pool[:5]}")
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
