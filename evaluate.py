from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from single_agent_wrapper import SingleAgentWrapper

SEED = 42
N_EPISODES = 100
MODEL_PATH = Path("models/ppo_chefhats_masked")
LEARNING_SEAT = 0


def _resolve_model_path(model_path: Path) -> Path:
    """Return the final model if it exists, otherwise the latest checkpoint."""
    if Path(str(model_path) + ".zip").exists():
        return model_path

    def _step_count(p: Path) -> int:
        m = re.search(r"_(\d+)_steps", p.stem)
        return int(m.group(1)) if m else 0

    checkpoints = sorted(
        (p for p in model_path.parent.glob(f"{model_path.name}_*_steps.zip")
         if re.search(r"_(\d+)_steps", p.stem)),
        key=_step_count,
    )
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Final model not found — loading latest checkpoint: {latest.name}")
        return latest.with_suffix("")
    raise FileNotFoundError(
        f"No model found at {model_path}.zip and no checkpoints in {model_path.parent}. "
        "Run train.py first."
    )


def _is_win(info: dict, learning_seat: int) -> bool:
    """Return True if the learning agent finished 1st (Match_Score == 3)."""
    scores = info.get("Match_Score", [])
    return bool(scores) and int(scores[learning_seat]) == 3


def main() -> None:
    env = SingleAgentWrapper(env_id="chefshat-v1", learning_seat=LEARNING_SEAT, seed=SEED)
    resolved = _resolve_model_path(MODEL_PATH)
    model = MaskablePPO.load(str(resolved), env=env)
    print(f"Loaded model: {resolved}")

    episode_rewards = []
    wins = 0

    for episode in range(N_EPISODES):
        obs, info = env.reset(seed=SEED + episode)
        done = False
        truncated = False
        total_reward = 0.0
        final_info = info

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            final_info = info

        if _is_win(final_info, LEARNING_SEAT):
            wins += 1

        episode_rewards.append(total_reward)

    win_rate = 100.0 * wins / N_EPISODES
    avg_reward = float(np.mean(episode_rewards))

    print(f"\nEpisodes:   {N_EPISODES}")
    print(f"Wins:       {wins}")
    print(f"Win rate:   {win_rate:.1f}%")
    print(f"Avg reward: {avg_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()
