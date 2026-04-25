import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))


from realm.eval_defaults import resolve_rollout_horizon


def test_gr00t_default_horizon_matches_official_droid_example():
    assert resolve_rollout_horizon("gr00t_n17", None) == 15
    assert resolve_rollout_horizon("GR00T", None) == 15


def test_non_gr00t_default_horizon_stays_legacy_value():
    assert resolve_rollout_horizon("openpi", None) == 8
    assert resolve_rollout_horizon("debug", None) == 8
    assert resolve_rollout_horizon("GR00T_N16", None) == 8


def test_explicit_horizon_is_preserved():
    assert resolve_rollout_horizon("gr00t_n17", 8) == 8
    assert resolve_rollout_horizon("openpi", 12) == 12
    assert resolve_rollout_horizon("GR00T_N16", 12) == 12