from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from single_agent_wrapper import SingleAgentWrapper
from utils import resolve_model_path

SEED = 42
N_EPISODES = 100
MODEL_PATH = Path("models/ppo_chefhats_masked")
LEARNING_SEAT = 0

# Backward-compatible alias so existing callers using _resolve_model_path continue to work.
_resolve_model_path = resolve_model_path


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
