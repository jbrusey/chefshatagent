import numpy as np
import pytest

from game_adapters import ChefshatAdapter, IrpsAdapter, get_game_adapter


def test_get_game_adapter_returns_expected_types():
    assert isinstance(get_game_adapter("chefshat"), ChefshatAdapter)
    assert isinstance(get_game_adapter("irps"), IrpsAdapter)


def test_get_game_adapter_rejects_unknown_game():
    with pytest.raises(ValueError, match="Unsupported game"):
        get_game_adapter("unknown")


def test_chefshat_adapter_formats_one_hot_action():
    adapter = ChefshatAdapter()
    action = adapter.format_env_action(policy_action=2, action_space_n=5)
    assert np.array_equal(action, np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_irps_adapter_uses_index_actions_and_unmasked_space():
    adapter = IrpsAdapter()
    assert adapter.format_env_action(policy_action=1, action_space_n=3) == 1
    assert np.array_equal(adapter.extract_valid_action_mask(obs=None, action_space_n=3), np.ones(3, dtype=bool))


def test_irps_adapter_requires_three_actions_for_mask():
    adapter = IrpsAdapter()
    with pytest.raises(RuntimeError, match="exactly 3 actions"):
        adapter.extract_valid_action_mask(obs=None, action_space_n=5)


def test_irps_adapter_adapts_to_compact_observation_from_info():
    adapter = IrpsAdapter()
    compact = adapter.adapt_observation(
        obs=np.zeros(10, dtype=np.float32),
        info={
            "player_wins_by_move": {"rock": 2, "paper": 1, "scissors": 0},
            "opponent_wins_by_move": {"rock": 0, "paper": 2, "scissors": 1},
        },
    )
    np.testing.assert_array_equal(compact, np.array([2, 1, 0, 0, 2, 1], dtype=np.float32))


def test_chefshat_adapter_provides_shaping_features_from_hand_slice():
    adapter = ChefshatAdapter()
    obs = np.zeros(228, dtype=np.float32)
    obs[11:13] = 1.0
    features = adapter.get_state_features_for_shaping(obs)
    assert features.shape == (17,)
    assert int(np.count_nonzero(features)) == 2


def test_irps_adapter_win_detection():
    adapter = IrpsAdapter()
    assert adapter.is_win({"winner": 0}, learning_seat=0)
    assert not adapter.is_win({"winner": 1}, learning_seat=0)


def test_irps_adapter_num_seats_defaults_to_two_without_env_metadata():
    adapter = IrpsAdapter()

    class Env:
        pass

    assert adapter.num_seats(Env()) == 2


def test_game_adapter_num_seats_uses_env_attribute_when_present():
    adapter = ChefshatAdapter()

    class Env:
        n_players = 6

    assert adapter.num_seats(Env()) == 6
