from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

DEFAULT_ELO = 1000.0
DEFAULT_K = 24.0


def load_ratings(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {str(key): float(value) for key, value in data.items()}


def save_ratings(path: Path, ratings: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(dict(sorted(ratings.items())), fp, indent=2)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _actual_score(score_a: int, score_b: int) -> float:
    if score_a > score_b:
        return 1.0
    if score_a < score_b:
        return 0.0
    return 0.5


def update_ratings_from_match(
    ratings: dict[str, float],
    players: list[str],
    match_scores: list[int],
    k_factor: float = DEFAULT_K,
) -> dict[str, float]:
    """Apply pairwise Elo updates for a multiplayer match.

    `match_scores` should be aligned to `players`, where higher score means better finish.
    In Chef's Hat this maps to Match_Score (3 best -> 0 last).
    """
    if len(players) != len(match_scores):
        raise ValueError("players and match_scores must have same length")

    for player in players:
        ratings.setdefault(player, DEFAULT_ELO)

    deltas = {player: 0.0 for player in players}

    for i, j in combinations(range(len(players)), 2):
        player_a = players[i]
        player_b = players[j]
        rating_a = ratings[player_a]
        rating_b = ratings[player_b]

        expected_a = expected_score(rating_a, rating_b)
        actual_a = _actual_score(match_scores[i], match_scores[j])
        delta = k_factor * (actual_a - expected_a)
        deltas[player_a] += delta
        deltas[player_b] -= delta

    for player, delta in deltas.items():
        ratings[player] += delta

    return ratings


def top_rated_players(ratings: dict[str, float], max_players: int) -> list[str]:
    ordered = sorted(ratings.items(), key=lambda item: item[1], reverse=True)
    return [path for path, _ in ordered[:max_players]]
