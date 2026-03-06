from pathlib import Path

import pytest

from population import (
    DEFAULT_ELO,
    expected_score,
    load_ratings,
    save_ratings,
    top_rated_players,
    update_ratings_from_match,
)


def test_expected_score_is_symmetric():
    a = expected_score(1200, 1000)
    b = expected_score(1000, 1200)
    assert a + b == pytest.approx(1.0)


def test_update_ratings_from_multiplayer_match_changes_all_players():
    ratings = {"a": 1000.0, "b": 1000.0, "c": 1000.0, "d": 1000.0}

    # a first, b second, c third, d fourth
    update_ratings_from_match(
        ratings=ratings,
        players=["a", "b", "c", "d"],
        match_scores=[3, 2, 1, 0],
        k_factor=24.0,
    )

    assert ratings["a"] > 1000.0
    assert ratings["d"] < 1000.0
    assert ratings["a"] > ratings["b"] > ratings["c"] > ratings["d"]


def test_update_ratings_from_match_supports_ties():
    ratings = {"a": 1000.0, "b": 1000.0}

    update_ratings_from_match(
        ratings=ratings,
        players=["a", "b"],
        match_scores=[1, 1],
        k_factor=24.0,
    )

    assert ratings["a"] == pytest.approx(1000.0)
    assert ratings["b"] == pytest.approx(1000.0)


def test_ratings_roundtrip_json(tmp_path: Path):
    ratings_path = tmp_path / "elo_ratings.json"
    ratings = {"models/a.zip": 1012.5, "models/b.zip": 987.75}

    save_ratings(ratings_path, ratings)
    loaded = load_ratings(ratings_path)

    assert loaded == ratings


def test_top_rated_players_returns_descending_subset():
    ratings = {"a": DEFAULT_ELO, "b": DEFAULT_ELO + 50, "c": DEFAULT_ELO - 20}
    assert top_rated_players(ratings, 2) == ["b", "a"]
