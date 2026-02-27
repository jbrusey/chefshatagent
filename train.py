from __future__ import annotations

from pathlib import Path

import gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from single_agent_wrapper import SingleAgentWrapper

SEED = 42
GAMMA = 0.99
TOTAL_TIMESTEPS = 200_000
MODEL_PATH = Path("models/ppo_chefhats_masked")


class WinRateCallback(BaseCallback):
    """Tracks win rate (1st place finishes) and logs it to the SB3 console output.

    A 'win' is defined as Match_Score[learning_seat] == 3 (1st place / Chef).
    The rate is computed over all episodes completed since the last rollout log.
    """

    def __init__(self, learning_seat: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.learning_seat = learning_seat
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


def main() -> None:
    env = SingleAgentWrapper(env_id="chefshat-v1", learning_seat=0, seed=SEED)

    # Dict observations require a multi-input policy; flat observations use MLP.
    is_dict_obs = isinstance(env.observation_space, gym.spaces.Dict)
    policy = "MultiInputPolicy" if is_dict_obs else "MlpPolicy"

    model = MaskablePPO(
        policy=policy,
        env=env,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        seed=SEED,
        gamma=GAMMA,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_PATH.parent),
        name_prefix=MODEL_PATH.name,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([WinRateCallback(learning_seat=0), checkpoint_callback]),
    )
    model.save(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}.zip")

    env.close()


if __name__ == "__main__":
    main()
