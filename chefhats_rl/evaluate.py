from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from single_agent_wrapper import SingleAgentWrapper

SEED = 42
N_EPISODES = 100
MODEL_PATH = Path("models/ppo_chefhats_masked")


def _extract_winner(info: dict) -> int | None:
    for key in ("winner", "winning_player", "winningPlayer"):
        if key in info:
            return int(info[key])
    return None


def main() -> None:
    env = SingleAgentWrapper(env_id="ChefsHat-v0", learning_seat=0, seed=SEED)
    model = MaskablePPO.load(str(MODEL_PATH), env=env)

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

        winner = _extract_winner(final_info)
        if winner is not None:
            wins += int(winner == 0)
        else:
            # Fallback heuristic: terminal positive reward counts as a win.
            wins += int(total_reward > 0)

        episode_rewards.append(total_reward)

    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    win_rate = 100.0 * wins / N_EPISODES

    print(f"Episodes: {N_EPISODES}")
    print(f"Wins (seat 0): {wins}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average reward: {avg_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()
