import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))


from realm.inference.gr00t_n16 import (
    Gr00tN16Client,
    _clear_gr00t_modules,
    _load_gr00t_n16_policy_client_class,
)


class _FakePolicyClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_observation = None

    def ping(self):
        return True

    def get_modality_config(self):
        return {
            "video": {
                "delta_indices": [0],
                "modality_keys": ["exterior_image_1_left", "wrist_image_left"],
            },
            "state": {
                "delta_indices": [0],
                "modality_keys": ["joint_position", "gripper_position"],
            },
            "action": {
                "delta_indices": list(range(32)),
                "modality_keys": ["joint_position", "gripper_position"],
            },
            "language": {
                "delta_indices": [0],
                "modality_keys": ["annotation.language.language_instruction"],
            },
        }

    def get_action(self, observation, options=None):
        self.last_observation = observation
        horizon = 32
        joint = np.tile(np.arange(7, dtype=np.float32), (1, horizon, 1))
        gripper = np.ones((1, horizon, 1), dtype=np.float32) * 0.25
        return {
            "action.joint_position": joint,
            "action.gripper_position": gripper,
        }, {}

    def reset(self, options=None):
        return {}


class _HistoryPolicyClient(_FakePolicyClient):
    def get_modality_config(self):
        return {
            "video": {
                "delta_indices": [-1, 0],
                "modality_keys": [
                    "wrist_image_left",
                    "exterior_image_2_left",
                    "exterior_image_1_left",
                ],
            },
            "state": {
                "delta_indices": [-1, 0],
                "modality_keys": ["joint_position", "gripper_position"],
            },
            "action": {
                "delta_indices": list(range(32)),
                "modality_keys": ["joint_position", "gripper_position"],
            },
            "language": {
                "delta_indices": [0],
                "modality_keys": ["annotation.language.language_instruction"],
            },
        }


class _N17LikePolicyClient(_FakePolicyClient):
    def get_modality_config(self):
        return {
            "video": {
                "delta_indices": [-15, 0],
                "modality_keys": ["exterior_image_1_left", "wrist_image_left"],
            },
            "state": {
                "delta_indices": [0],
                "modality_keys": ["eef_9d", "gripper_position", "joint_position"],
            },
            "action": {
                "delta_indices": list(range(40)),
                "modality_keys": ["eef_9d", "gripper_position", "joint_position"],
            },
            "language": {
                "delta_indices": [0],
                "modality_keys": ["annotation.language.language_instruction"],
            },
        }


def _constant_rgb(fill_value, dtype=np.uint8):
    return np.full((2, 2, 3), fill_value, dtype=dtype)


def test_gr00t_n16_build_observation_uses_expected_flat_keys():
    client = Gr00tN16Client(
        host="localhost",
        port=5555,
        policy_client_cls=_FakePolicyClient,
        image_size=(2, 2),
        print_observation_stats=False,
    )
    client.get_modality_config(refresh=True)

    observation = client.build_observation(
        instruction="put the banana in the box",
        base_im=_constant_rgb(10),
        base_im_second=_constant_rgb(20),
        wrist_im=_constant_rgb(30),
        robot_state=np.arange(7, dtype=np.float32),
        gripper_state=0.5,
        use_base_im_second=True,
    )

    assert sorted(observation) == [
        "annotation.language.language_instruction",
        "state.gripper_position",
        "state.joint_position",
        "video.exterior_image_1_left",
        "video.wrist_image_left",
    ]
    assert observation["video.exterior_image_1_left"].shape == (1, 1, 2, 2, 3)
    assert observation["video.wrist_image_left"].shape == (1, 1, 2, 2, 3)
    assert np.all(observation["video.exterior_image_1_left"][0, 0] == 20)
    assert np.all(observation["video.wrist_image_left"][0, 0] == 30)
    assert observation["state.joint_position"].shape == (1, 1, 7)
    assert observation["state.gripper_position"].shape == (1, 1, 1)
    assert observation["annotation.language.language_instruction"] == [
        "put the banana in the box"
    ]


def test_gr00t_n16_prepare_image_scales_float_inputs_to_uint8():
    client = Gr00tN16Client(
        host="localhost",
        port=5555,
        policy_client_cls=_FakePolicyClient,
        image_size=(2, 2),
        print_observation_stats=False,
    )

    prepared = client._prepare_image(_constant_rgb(0.5, dtype=np.float32))

    assert prepared.dtype == np.uint8
    assert prepared.shape == (2, 2, 3)
    assert np.all(prepared == 127)


def test_gr00t_n16_uses_modality_history_and_camera_mapping():
    client = Gr00tN16Client(
        host="localhost",
        port=5555,
        policy_client_cls=_HistoryPolicyClient,
        image_size=(2, 2),
        print_observation_stats=False,
    )
    client.get_modality_config(refresh=True)

    client.observe(
        base_im=_constant_rgb(10),
        base_im_second=_constant_rgb(20),
        wrist_im=_constant_rgb(30),
        robot_state=np.arange(7, dtype=np.float32),
        gripper_state=0.25,
    )
    client.observe(
        base_im=_constant_rgb(11),
        base_im_second=_constant_rgb(21),
        wrist_im=_constant_rgb(31),
        robot_state=np.arange(7, dtype=np.float32) + 1,
        gripper_state=0.75,
    )

    observation = client.build_observation(
        instruction="open the drawer",
        base_im=_constant_rgb(99),
        base_im_second=_constant_rgb(98),
        wrist_im=_constant_rgb(97),
        robot_state=np.zeros(7, dtype=np.float32),
        gripper_state=0.0,
        update_frame_buffer=False,
    )

    assert observation["video.exterior_image_1_left"].shape == (1, 2, 2, 2, 3)
    assert observation["video.exterior_image_2_left"].shape == (1, 2, 2, 2, 3)
    assert observation["video.wrist_image_left"].shape == (1, 2, 2, 2, 3)
    assert np.all(observation["video.exterior_image_1_left"][0, 0] == 10)
    assert np.all(observation["video.exterior_image_1_left"][0, 1] == 11)
    assert np.all(observation["video.exterior_image_2_left"][0, 0] == 20)
    assert np.all(observation["video.exterior_image_2_left"][0, 1] == 21)
    assert np.all(observation["video.wrist_image_left"][0, 0] == 30)
    assert np.all(observation["video.wrist_image_left"][0, 1] == 31)
    assert observation["state.joint_position"].shape == (1, 2, 7)
    assert observation["state.gripper_position"].shape == (1, 2, 1)
    assert np.allclose(observation["state.joint_position"][0, 0], np.arange(7, dtype=np.float32))
    assert np.allclose(observation["state.joint_position"][0, 1], np.arange(7, dtype=np.float32) + 1)
    assert np.allclose(observation["state.gripper_position"][0, :, 0], [0.25, 0.75])


def test_gr00t_n16_rejects_n17_eef_modality_with_clear_error():
    client = Gr00tN16Client(
        host="localhost",
        port=5555,
        policy_client_cls=_N17LikePolicyClient,
        image_size=(2, 2),
        print_observation_stats=False,
    )

    with pytest.raises(ValueError, match="OXE_DROID"):
        client.get_modality_config(refresh=True)


def test_gr00t_n16_infer_action_chunk_returns_joint_and_gripper_chunk():
    client = Gr00tN16Client(
        host="localhost",
        port=5555,
        policy_client_cls=_FakePolicyClient,
        image_size=(2, 2),
        print_observation_stats=False,
    )
    client.connect(fetch_modality_config=True)

    action_chunk = client.infer_action_chunk(
        {
            "instruction": "rotate the marker",
            "base_im": _constant_rgb(10),
            "wrist_im": _constant_rgb(30),
            "robot_state": np.arange(7, dtype=np.float32),
            "gripper_state": 0.5,
        }
    )

    assert action_chunk.shape == (32, 8)
    assert np.allclose(action_chunk[0, :7], np.arange(7, dtype=np.float32))
    assert np.allclose(action_chunk[:, -1], 0.25)


def test_gr00t_n16_loader_skips_policy_package_init(monkeypatch, tmp_path):
    policy_root = tmp_path / "gr00t" / "policy"
    data_root = tmp_path / "gr00t" / "data"
    policy_root.mkdir(parents=True)
    data_root.mkdir(parents=True)

    (tmp_path / "gr00t" / "__init__.py").write_text("", encoding="utf-8")
    (policy_root / "__init__.py").write_text(
        "raise RuntimeError('gr00t.policy.__init__ should not be imported')\n",
        encoding="utf-8",
    )
    (policy_root / "policy.py").write_text(
        "class BasePolicy:\n"
        "    def __init__(self, *, strict=False):\n"
        "        self.strict = strict\n",
        encoding="utf-8",
    )
    (policy_root / "server_client.py").write_text(
        "from .policy import BasePolicy\n\n"
        "class PolicyClient(BasePolicy):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (data_root / "__init__.py").write_text("", encoding="utf-8")
    (data_root / "types.py").write_text(
        "class ModalityConfig:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n",
        encoding="utf-8",
    )
    (data_root / "utils.py").write_text(
        "def to_json_serializable(value):\n"
        "    return value\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("GR00T_N16_ROOT", str(tmp_path))
    monkeypatch.delenv("ISAAC_GR00T_N16_ROOT", raising=False)
    _clear_gr00t_modules()

    try:
        policy_client_cls = _load_gr00t_n16_policy_client_class()
    finally:
        _clear_gr00t_modules()

    assert policy_client_cls.__name__ == "PolicyClient"