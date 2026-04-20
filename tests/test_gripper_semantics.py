import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))


from realm.inference.utils import discretize_gripper_action, extract_from_obs, normalize_gripper_position


class _FakeController:
    def __init__(self):
        lower = np.zeros(11, dtype=np.float32)
        upper = np.zeros(11, dtype=np.float32)
        upper[7] = 0.04
        upper[8] = 0.04
        upper[9] = 0.57
        upper[10] = -0.57

        self.dof_idx = np.array([7, 8, 9, 10], dtype=np.int64)
        self._open_qpos = None
        self._closed_qpos = None
        self._control_limits = {"position": (lower, upper)}


class _FakeRobot:
    default_arm = 0

    def __init__(self):
        self._controllers = {"gripper_0": _FakeController()}


def test_normalize_gripper_position_uses_controller_limits():
    value = normalize_gripper_position(
        np.array([0.02, 0.02], dtype=np.float32),
        open_qpos=np.array([0.04, 0.04], dtype=np.float32),
        closed_qpos=np.array([0.0, 0.0], dtype=np.float32),
    )

    assert np.isclose(value, 0.5)


def test_extract_from_obs_prefers_robot_controller_limits():
    obs = {
        "external": {
            "external_sensor0": {
                "rgb": torch.zeros((128, 128, 4), dtype=torch.uint8),
            },
        },
        "DROID": {
            "proprio": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02], dtype=torch.float32),
        },
    }

    _, _, _, _, _, robot_state, gripper_state = extract_from_obs(
        obs,
        robot_name="DROID",
        robot=_FakeRobot(),
    )

    assert robot_state.shape == (7,)
    assert np.isclose(gripper_state, 0.5)


def test_discretize_gripper_action_matches_realm_controller_semantics():
    assert discretize_gripper_action(0.75, open_if_above_threshold=True) == 1.0
    assert discretize_gripper_action(0.25, open_if_above_threshold=True) == -1.0
    assert discretize_gripper_action(0.25, open_if_above_threshold=False) == 1.0
    assert discretize_gripper_action(0.75, open_if_above_threshold=False) == -1.0