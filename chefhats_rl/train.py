from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from sb3_contrib import MaskablePPO

from single_agent_wrapper import SingleAgentWrapper

SEED = 42
TOTAL_TIMESTEPS = 1_000_000
MODEL_PATH = Path("models/ppo_chefhats_masked")


def main() -> None:
    env = SingleAgentWrapper(env_id="ChefsHat-v0", learning_seat=0, seed=SEED)

    # Dict observations require a multi-input policy; flat observations use MLP.
    is_dict_obs = isinstance(env.observation_space, gym.spaces.Dict)
    policy = "MaskableMultiInputPolicy" if is_dict_obs else "MaskableMlpPolicy"

    model = MaskablePPO(
        policy=policy,
        env=env,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        seed=SEED,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save(str(MODEL_PATH))

    env.close()


if __name__ == "__main__":
    main()
