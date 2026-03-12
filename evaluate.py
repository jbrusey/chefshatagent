from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from game_adapters import get_game_adapter
from single_agent_wrapper import SingleAgentWrapper
from utils import resolve_model_path

SEED = 42
N_EPISODES = 100
MODEL_PATH = Path("models/ppo_chefhats_masked")
LEARNING_SEAT = 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO model")
    parser.add_argument("--game", choices=["chefshat", "irps"], default="chefshat")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    adapter = get_game_adapter(args.game, env_id=args.env_id)

    env = SingleAgentWrapper(env_id=adapter.config.env_id, learning_seat=LEARNING_SEAT, seed=SEED)
    resolved = resolve_model_path(MODEL_PATH)
    model = MaskablePPO.load(str(resolved), env=env)
    print(f"Loaded model: {resolved}")

    episode_rewards = []
    wins = 0

    for episode in range(args.episodes):
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

        if adapter.is_win(final_info, LEARNING_SEAT):
            wins += 1

        episode_rewards.append(total_reward)

    win_rate = 100.0 * wins / args.episodes
    avg_reward = float(np.mean(episode_rewards))

    print(f"\nEpisodes:   {args.episodes}")
    print(f"Wins:       {wins}")
    print(f"Win rate:   {win_rate:.1f}%")
    print(f"Avg reward: {avg_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()
