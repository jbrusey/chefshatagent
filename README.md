# Chef's Hat RL with MaskablePPO

This project trains a PPO agent to play Chef's Hat from **seat 0** against three random opponents using action masking.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
uv venv
uv pip install -e .
```

## Train

```bash
uv run train.py
```

To stream the rollout win rate (`rollout/win_rate`) to [Weights & Biases](https://wandb.ai):

```bash
uv pip install wandb
uv run train.py --wandb --wandb-project chefhats-rl --wandb-run-name self-play-seed42
```

Training runs for `200_000` timesteps and saves the model to:

- `models/ppo_chefhats_masked`

During self-play, snapshots are scored with pairwise Elo (derived from multiplayer match
rankings), and ratings are persisted to:

- `models/elo_ratings.json`

## Evaluate

```bash
uv run evaluate.py
```

Evaluation runs 100 games and prints:

- wins
- win rate
- average reward

## Run tests

```bash
uv run pytest -v
```

## Play a trained model with the Chef's Hat GUI/logging

After you have a trained model (`models/ppo_chefhats_masked.zip`), run:

```bash
uv run play_gui.py --episodes 1 --verbose
```

Useful flags:

- `--model-path models/ppo_chefhats_masked` to choose a different model/checkpoint prefix
- `--learning-seat 0|1|2|3` to control which seat uses the trained model
- `--no-deterministic` to sample policy actions instead of greedy actions
- `--log-directory log` to choose where ChefsHat logs are written

This script runs the real `chefshat-v1` environment (not the single-agent training wrapper),
uses action masks for the trained seat, and plays random valid actions for the other seats.
