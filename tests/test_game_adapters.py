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


def test_irps_adapter_win_detection():
    adapter = IrpsAdapter()
    assert adapter.is_win({"winner": 0}, learning_seat=0)
    assert not adapter.is_win({"winner": 1}, learning_seat=0)
