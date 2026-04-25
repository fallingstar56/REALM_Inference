import time
import numpy as np
from PIL import Image
import omnigibson as og
from openpi_client import websocket_client_policy, image_tools

from realm.helpers import axisangle_to_rpy
from realm.inference.utils import scene_gripper_position_to_model_position
#from realm.inference.base import ExternalRobotInferenceClient
#from realm.inference.hamster import HamsterClient
#from realm.inference.dreamzero import DreamZeroClient


class InferenceClient:
    def __init__(self, model_type, port, host="127.0.0.1", timeout=150.0):
        self.model_type = model_type
        self.host = host
        self.port = port
        self.gr00t_n16_modality_config = None
        self.gr00t_n17_modality_config = None
        # if model_type == "hamster":
        #     self.client = HamsterClient(host=self.host, port=self.port)
        # elif model_type == "dreamzero":
        #     self.client = DreamZeroClient(host=self.host, port=self.port)
        if model_type == "openpi":
            og.log.info("Connecting to server...")
            self.client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port
            )
        elif model_type in {"GR00T_N16", "gr00t_n16"}:
            self.client = self._init_gr00t_n16_client(host, port, timeout)
        elif model_type == "gr00t_n17":
            self.client = self._init_gr00t_n17_client(host, port, timeout)
        elif model_type == "GR00T":
            og.log.warning(
                "Model type 'GR00T' is kept as a compatibility alias. "
                "Use 'gr00t_n17' for explicit GR00T N1.7 rollout logic."
            )
            self.client = self._init_gr00t_n17_client(host, port, timeout)
        elif model_type == "debug":
            self.client = None
        else:
            raise NotImplementedError()

    def _init_gr00t_n16_client(self, host, port, timeout):
        from realm.inference.gr00t_n16 import Gr00tN16Client

        og.log.info(f"Connecting to GR00T N1.6 server at {host}:{port}...")
        client = Gr00tN16Client(
            host=host,
            port=port,
            timeout_ms=int(timeout * 1000),
        )
        self.gr00t_n16_modality_config = client.connect(fetch_modality_config=True)
        og.log.info(
            "Connected to GR00T N1.6 server. "
            f"Modality config keys: {list(self.gr00t_n16_modality_config.keys())}"
        )
        return client

    def _init_gr00t_n17_client(self, host, port, timeout):
        from realm.inference.gr00t_n17 import Gr00tN17Client

        og.log.info(f"Connecting to GR00T N1.7 server at {host}:{port}...")
        client = Gr00tN17Client(
            host=host,
            port=port,
            timeout_ms=int(timeout * 1000),
        )
        self.gr00t_n17_modality_config = client.connect(fetch_modality_config=True)
        og.log.info(
            "Connected to GR00T N1.7 server. "
            f"Modality config keys: {list(self.gr00t_n17_modality_config.keys())}"
        )
        return client

    def _uses_gr00t_n17_adapter(self):
        return self.model_type in {"gr00t_n17", "GR00T"}

    def _uses_gr00t_n16_adapter(self):
        return self.model_type in {"GR00T_N16", "gr00t_n16"}

    def _prepare_gripper_state_for_model(self, gripper_state):
        """Forward the normalized scene gripper scalar to the GR00T DROID client.

        The official GR00T DROID example forwards robot_state['gripper_position']
        to the policy server without an additional semantic inversion, so REALM
        should do the same.
        """
        if self._uses_gr00t_n17_adapter():
            return scene_gripper_position_to_model_position(gripper_state)
        return gripper_state

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state, use_base_im_second=False, ee_control=False, cartesian_position=None):
        if self.model_type == "debug":
            if ee_control:
                pred_action_chunk = np.array([0.41402626, -0.13211727, 0.57253086, -3.09742367, 0.2580259, -0.24700592, -1])
            else:
                pred_action_chunk = np.atleast_1d(np.zeros(8))

            return pred_action_chunk

        # TODO: all DROID EE control poses need to have flip_pose_pointing_down() applied before being passed to the step
        if self.model_type in {"GR00T_N16", "gr00t_n16"}:
            pred_action_chunk = self.client.infer_action_chunk(
                {
                    "instruction": instruction,
                    "base_im": base_im,
                    "base_im_second": base_im_second,
                    "wrist_im": wrist_im,
                    "robot_state": robot_state,
                    "gripper_state": gripper_state,
                    "use_base_im_second": use_base_im_second,
                    "update_frame_buffer": False,
                }
            )
            return pred_action_chunk

        elif self._uses_gr00t_n17_adapter():
            model_gripper_state = self._prepare_gripper_state_for_model(gripper_state)
            pred_action_chunk = self.client.infer_action_chunk(
                {
                    "instruction": instruction,
                    "base_im": base_im,
                    "base_im_second": base_im_second,
                    "wrist_im": wrist_im,
                    "robot_state": robot_state,
                    "gripper_state": model_gripper_state,
                    "cartesian_position": cartesian_position,
                    "use_base_im_second": use_base_im_second,
                    "update_frame_buffer": False,
                }
            )
            return pred_action_chunk

        elif self.model_type == "molmoact":
            img_to_use = base_im_second if use_base_im_second else base_im
            obs_dict = {
                "images": [img_to_use, wrist_im],
                "instruction": instruction,
            }
            _t0 = time.perf_counter()
            pred = self.client.infer(obs_dict)
            og.log.info(f"[molmoact] inference time: {time.perf_counter() - _t0:.3f}s")
            pred_action_chunk = pred["action"]

            if ee_control:
                pred_action_chunk = axisangle_to_rpy(pred_action_chunk)

            return pred_action_chunk

        elif self.model_type == "hamster":
            img_to_use = base_im_second if use_base_im_second else base_im
            # Hamster expects BGR for cv2.imencode
            import cv2
            img_bgr = cv2.cvtColor(img_to_use, cv2.COLOR_RGB2BGR)
            _t0 = time.perf_counter()
            trajectory = self.client.infer(img_bgr, instruction)
            og.log.info(f"[hamster] inference time: {time.perf_counter() - _t0:.3f}s")
            return np.array(trajectory)

        elif self.model_type == "dreamzero":
            assert base_im_second is not None, "DreamZero requires --multi-view (second external camera)"
            assert cartesian_position is not None, "DreamZero requires cartesian_position (robot-relative EE pose)"

            # DreamZero expects 180x320 RGB and strictly numpy arrays
            # H=180, W=320. Initial frames MUST be strictly 3D (H, W, 3) np.ndarray
            base_im_resized = np.array(Image.fromarray(base_im).resize((320, 180)), dtype=np.uint8)
            base_im_second_resized = np.array(Image.fromarray(base_im_second).resize((320, 180)), dtype=np.uint8)
            wrist_im_resized = np.array(Image.fromarray(wrist_im).resize((320, 180)), dtype=np.uint8)

            obs_dict = {
                "observation/exterior_image_0_left": base_im_resized,
                "observation/exterior_image_1_left": base_im_second_resized,
                "observation/wrist_image_left": wrist_im_resized,
                "observation/joint_position": np.array(robot_state, dtype=np.float32),
                "observation/cartesian_position": np.array(cartesian_position, dtype=np.float32),
                "observation/gripper_position": np.array(np.atleast_1d(gripper_state), dtype=np.float32),
                "prompt": instruction
            }

            pred_action_chunk = self.client.infer(obs_dict)
            return pred_action_chunk

        else:
            img_to_use = base_im_second if use_base_im_second else base_im

            obs_dict = {
                "prompt": instruction,
                "observation/joint_position": robot_state,
                "observation/gripper_position": np.atleast_1d(np.array(gripper_state)),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(img_to_use, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(wrist_im, 224, 224)
            }
            pred = self.client.infer(obs_dict)
            pred_action_chunk = pred["actions"]
            return pred_action_chunk

    def reset(self):
        if hasattr(self.client, "reset"):
            self.client.reset()

    def observe(
        self,
        base_im,
        base_im_second,
        wrist_im,
        use_base_im_second=False,
        robot_state=None,
        gripper_state=None,
        cartesian_position=None,
    ):
        if (self._uses_gr00t_n16_adapter() or self._uses_gr00t_n17_adapter()) and hasattr(self.client, "observe"):
            self.client.observe(
                base_im=base_im,
                base_im_second=base_im_second,
                wrist_im=wrist_im,
                use_base_im_second=use_base_im_second,
                robot_state=robot_state,
                gripper_state=self._prepare_gripper_state_for_model(gripper_state)
                if gripper_state is not None
                else None,
                cartesian_position=cartesian_position,
            )
