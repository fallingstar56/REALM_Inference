import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))


from realm.inference.gr00t_n17 import Gr00tN17Client
from realm.inference.utils import extract_from_obs


class _FakePolicyClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def ping(self):
        return True

    def get_modality_config(self):
        return {
            "video": {
                "delta_indices": [0],
                "modality_keys": [
                    "exterior_image_1_left",
                    "exterior_image_2_left",
                    "wrist_image_left",
                ],
            },
            "state": {
                "delta_indices": [0],
                "modality_keys": ["joint_position", "gripper_position", "eef_9d"],
            },
            "action": {
                "delta_indices": [0],
                "modality_keys": ["joint_position", "gripper_position"],
            },
            "language": {
                "delta_indices": [0],
                "modality_keys": ["annotation.language.language_instruction"],
            },
        }


class _ReorderedVideoPolicyClient(_FakePolicyClient):
    def get_modality_config(self):
        config = super().get_modality_config()
        config["video"]["modality_keys"] = [
            "exterior_image_2_left",
            "wrist_image_left",
            "exterior_image_1_left",
        ]
        return config


class _StateHistoryPolicyClient(_FakePolicyClient):
    def get_modality_config(self):
        config = super().get_modality_config()
        config["state"]["delta_indices"] = [-1, 0]
        return config


def _constant_rgb(fill_value):
    return np.full((2, 2, 3), fill_value, dtype=np.uint8)


def test_extract_from_obs_uses_sorted_external_sensor_order_and_wrist_aliases():
    obs = {
        "external": {
            "external_sensor10": {
                "rgb": torch.full((2, 2, 4), 200, dtype=torch.uint8),
            },
            "external_sensor2": {
                "rgb": torch.full((2, 2, 4), 50, dtype=torch.uint8),
            },
        },
        "DROID": {
            "DROID:wrist_camera:Camera:0": {
                "rgb": torch.full((2, 2, 4), 125, dtype=torch.uint8),
            },
            "proprio": torch.tensor(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.02, 0.02],
                dtype=torch.float32,
            ),
        },
    }

    base_im, _, base_im_second, _, wrist_im, robot_state, _ = extract_from_obs(
        obs,
        robot_name="DROID",
    )

    assert np.all(base_im == 50)
    assert np.all(base_im_second == 200)
    assert np.all(wrist_im == 125)
    assert robot_state.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_gr00t_n17_observe_preserves_exterior_camera_identity():
    client = Gr00tN17Client(
        host="localhost",
        port=5555,
        policy_client_cls=_FakePolicyClient,
        image_size=(2, 2),
        print_observation_stats=False,
    )
    client.get_modality_config(refresh=True)

    client.observe(
        base_im=_constant_rgb(10),
        base_im_second=_constant_rgb(20),
        wrist_im=_constant_rgb(30),
        use_base_im_second=False,
    )

    latest = client._frame_buffer[-1]

    assert np.all(latest["exterior_image_1_left"] == 10)
    assert np.all(latest["exterior_image_2_left"] == 20)
    assert np.all(latest["wrist_image_left"] == 30)


def test_gr00t_n17_observe_keeps_selected_primary_camera_in_sync():
    client = Gr00tN17Client(
        host="localhost",
        port=5555,
        policy_client_cls=_FakePolicyClient,
        image_size=(2, 2),
        print_observation_stats=False,
    )
    client.get_modality_config(refresh=True)

    client.observe(
        base_im=_constant_rgb(10),
        base_im_second=_constant_rgb(20),
        wrist_im=_constant_rgb(30),
        use_base_im_second=True,
    )

    latest = client._frame_buffer[-1]

    assert np.all(latest["exterior_image_1_left"] == 20)
    assert np.all(latest["exterior_image_2_left"] == 10)
    assert np.all(latest["wrist_image_left"] == 30)


def test_gr00t_n17_observe_maps_exterior_cameras_by_key_not_config_order():
    client = Gr00tN17Client(
        host="localhost",
        port=5555,
        policy_client_cls=_ReorderedVideoPolicyClient,
        image_size=(2, 2),
        print_observation_stats=False,
    )
    client.get_modality_config(refresh=True)

    client.observe(
        base_im=_constant_rgb(10),
        base_im_second=_constant_rgb(20),
        wrist_im=_constant_rgb(30),
        use_base_im_second=False,
    )

    latest = client._frame_buffer[-1]

    assert np.all(latest["exterior_image_1_left"] == 10)
    assert np.all(latest["exterior_image_2_left"] == 20)
    assert np.all(latest["wrist_image_left"] == 30)


def test_gr00t_n17_prepare_image_matches_resize_with_pad_behavior():
    client = Gr00tN17Client(
        host="localhost",
        port=5555,
        policy_client_cls=_FakePolicyClient,
        image_size=(4, 4),
        print_observation_stats=False,
    )

    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
            [[130, 140, 150], [160, 170, 180], [190, 200, 210], [220, 230, 240]],
        ],
        dtype=np.uint8,
    )

    prepared = client._prepare_image(image)

    assert prepared.shape == (4, 4, 3)
    assert prepared.dtype == np.uint8
    assert np.all(prepared[0] == 0)
    assert np.all(prepared[-1] == 0)
    assert np.array_equal(prepared[1:3], image)


def test_gr00t_n17_build_observation_uses_state_history_buffer():
    client = Gr00tN17Client(
        host="localhost",
        port=5555,
        policy_client_cls=_StateHistoryPolicyClient,
        image_size=(2, 2),
        print_observation_stats=False,
    )
    client.get_modality_config(refresh=True)

    first_joint_state = np.arange(7, dtype=np.float32)
    second_joint_state = first_joint_state + 10.0

    client.observe(
        base_im=_constant_rgb(10),
        base_im_second=_constant_rgb(20),
        wrist_im=_constant_rgb(30),
        robot_state=first_joint_state,
        gripper_state=0.1,
        cartesian_position=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    client.observe(
        base_im=_constant_rgb(11),
        base_im_second=_constant_rgb(21),
        wrist_im=_constant_rgb(31),
        robot_state=second_joint_state,
        gripper_state=0.9,
        cartesian_position=np.array([0.4, 0.5, 0.6, 0.1, 0.2, 0.3], dtype=np.float32),
    )

    observation = client.build_observation(
        instruction="stack blocks",
        base_im=_constant_rgb(11),
        base_im_second=_constant_rgb(21),
        wrist_im=_constant_rgb(31),
        robot_state=second_joint_state,
        gripper_state=0.9,
        cartesian_position=np.array([0.4, 0.5, 0.6, 0.1, 0.2, 0.3], dtype=np.float32),
        update_frame_buffer=False,
    )

    assert observation["state"]["joint_position"].shape == (1, 2, 7)
    assert np.allclose(observation["state"]["joint_position"][0, 0], first_joint_state)
    assert np.allclose(observation["state"]["joint_position"][0, 1], second_joint_state)
    assert np.allclose(observation["state"]["gripper_position"][0, :, 0], [0.1, 0.9])