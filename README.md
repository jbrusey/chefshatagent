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

Training runs for `200_000` timesteps and saves the model to:

- `models/ppo_chefhats_masked`

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
