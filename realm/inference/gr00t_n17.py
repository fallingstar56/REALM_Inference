from __future__ import annotations

from collections import deque
import importlib
import os
import sys
from pathlib import Path
from typing import Any


def _candidate_gr00t_n17_roots() -> list[Path]:
    env_roots = [
        os.environ.get("GR00T_ROOT"),
        os.environ.get("ISAAC_GR00T_ROOT"),
        os.environ.get("POLICY_RUN_DIR"),
    ]
    workspace_root = Path(__file__).resolve().parents[2]
    candidates = [
        workspace_root / "Isaac-GR00T",
        workspace_root.parent / "Isaac-GR00T",
    ]

    resolved: list[Path] = []
    for root in env_roots:
        if root:
            resolved.append(Path(root).expanduser().resolve())
    for root in candidates:
        resolved.append(root.resolve())
    return resolved


def _ensure_gr00t_n17_on_path() -> Path | None:
    for candidate in _candidate_gr00t_n17_roots():
        if not candidate.exists():
            continue
        if not (candidate / "gr00t" / "policy" / "server_client.py").is_file():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return candidate
    return None


def _load_gr00t_n17_policy_client_class():
    try:
        module = importlib.import_module("gr00t.policy.server_client")
        return module.PolicyClient
    except ModuleNotFoundError as exc:
        if exc.name not in {"gr00t", "gr00t.policy", "gr00t.policy.server_client"}:
            raise ImportError(
                "Isaac-GR00T dependencies are missing while importing the official GR00T N1.7 PolicyClient."
            ) from exc

    root = _ensure_gr00t_n17_on_path()
    if root is None:
        raise ImportError(
            "Could not locate Isaac-GR00T. Set GR00T_ROOT (or ISAAC_GR00T_ROOT) to the "
            "Isaac-GR00T repository root so REALM can import gr00t.policy.server_client.PolicyClient."
        )

    try:
        module = importlib.import_module("gr00t.policy.server_client")
    except Exception as exc:
        raise ImportError(
            f"Failed to import Isaac-GR00T GR00T N1.7 PolicyClient from {root}. "
            "Ensure the Isaac-GR00T environment dependencies are installed."
        ) from exc
    return module.PolicyClient


GR00T_N17_DEFAULT_IMAGE_SIZE = (180, 320)
GR00T_N17_DROID_EEF_ROTATION_CORRECT = (
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


def _cfg_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def compute_gr00t_n17_eef_9d(cartesian_position: Any) -> Any:
    import numpy as np
    from scipy.spatial.transform import Rotation

    cartesian = np.asarray(cartesian_position, dtype=np.float64).reshape(6)
    xyz = cartesian[:3]
    euler = cartesian[3:6]
    rot_robot = Rotation.from_euler("XYZ", euler).as_matrix()
    rot_mat = rot_robot @ np.asarray(GR00T_N17_DROID_EEF_ROTATION_CORRECT, dtype=np.float64)
    rot6d = rot_mat[:2, :].reshape(6)
    return np.concatenate([xyz, rot6d]).astype(np.float32)


def _gr00t_n17_action_key_candidates(action_key: str) -> tuple[str, ...]:
    if action_key.startswith("action."):
        return (action_key, action_key.removeprefix("action."))
    return (action_key, f"action.{action_key}")


class Gr00tN17Client:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
        strict: bool = False,
        image_size: tuple[int, int] = GR00T_N17_DEFAULT_IMAGE_SIZE,
        print_observation_stats: bool = True,
        policy_client_cls: type[Any] | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.strict = strict
        self.image_size = image_size
        self.print_observation_stats = print_observation_stats
        self._modality_config: dict[str, Any] | None = None
        self.video_keys: list[str] = []
        self.state_keys: list[str] = []
        self.action_keys: list[str] = []
        self.language_key: str | None = None
        self.video_delta_indices: list[int] = []
        self.state_delta_indices: list[int] = []
        self.action_delta_indices: list[int] = []
        self.video_horizon = 0
        self.state_horizon = 0
        self.action_horizon = 0
        self.video_history_len = 1
        self._frame_buffer: deque[dict[str, Any]] = deque(maxlen=1)
        self._observation_stats_printed = False

        client_cls = policy_client_cls or _load_gr00t_n17_policy_client_class()
        self._client = client_cls(
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            api_token=api_token,
            strict=strict,
        )

    @property
    def client(self) -> Any:
        return self._client

    def connect(self, fetch_modality_config: bool = True) -> dict[str, Any] | None:
        if not self.ping():
            raise ConnectionError(
                f"Failed to reach GR00T N1.7 server at tcp://{self.host}:{self.port}"
            )
        if fetch_modality_config:
            return self.get_modality_config(refresh=True)
        return None

    def ping(self) -> bool:
        return self._client.ping()

    def _cache_modality_metadata(self, modality_config: dict[str, Any]) -> None:
        video_config = modality_config["video"]
        state_config = modality_config["state"]
        action_config = modality_config["action"]
        language_config = modality_config["language"]

        self.video_keys = list(_cfg_value(video_config, "modality_keys"))
        self.state_keys = list(_cfg_value(state_config, "modality_keys"))
        self.action_keys = list(_cfg_value(action_config, "modality_keys"))

        self.video_delta_indices = list(_cfg_value(video_config, "delta_indices"))
        self.state_delta_indices = list(_cfg_value(state_config, "delta_indices"))
        self.action_delta_indices = list(_cfg_value(action_config, "delta_indices"))

        self.video_horizon = len(self.video_delta_indices)
        self.state_horizon = len(self.state_delta_indices)
        self.action_horizon = len(self.action_delta_indices)

        language_keys = list(_cfg_value(language_config, "modality_keys"))
        if not language_keys:
            raise ValueError("GR00T N1.7 modality config is missing a language key")
        self.language_key = language_keys[0]

        history_window = (
            max(-min(self.video_delta_indices), 0) + 1 if self.video_delta_indices else 1
        )
        self.video_history_len = max(history_window, self.video_horizon, 1)
        self._frame_buffer = deque(maxlen=self.video_history_len)

    def get_modality_config(self, refresh: bool = False) -> dict[str, Any]:
        if refresh or self._modality_config is None:
            self._modality_config = self._client.get_modality_config()
            self._cache_modality_metadata(self._modality_config)
        return self._modality_config

    def _prepare_image(self, image: Any) -> Any:
        import numpy as np
        from PIL import Image

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(f"Expected an RGB image with shape (H, W, C), got {array.shape}")
        array = np.clip(array[..., :3], 0, 255).astype(np.uint8)
        image_height, image_width = self.image_size
        resized = Image.fromarray(array).resize((image_width, image_height))
        return np.asarray(resized, dtype=np.uint8)

    def observe(
        self,
        *,
        base_im: Any,
        wrist_im: Any,
        base_im_second: Any | None = None,
        use_base_im_second: bool = False,
    ) -> None:
        if not self.video_keys:
            self.get_modality_config()

        exterior_image = base_im_second if use_base_im_second and base_im_second is not None else base_im
        prepared_frames: dict[str, Any] = {}
        for video_key in self.video_keys:
            if "wrist" in video_key:
                prepared_frames[video_key] = self._prepare_image(wrist_im)
            else:
                prepared_frames[video_key] = self._prepare_image(exterior_image)
        self._frame_buffer.append(prepared_frames)

    def _build_video_observation(self) -> dict[str, Any]:
        import numpy as np

        if not self._frame_buffer:
            raise ValueError(
                "GR00T N1.7 frame buffer is empty. Call observe() before requesting an action."
            )

        last_idx = len(self._frame_buffer) - 1
        video_observation: dict[str, Any] = {}
        for video_key in self.video_keys:
            frames = []
            for delta_idx in self.video_delta_indices:
                frame_idx = min(max(last_idx + delta_idx, 0), last_idx)
                frames.append(self._frame_buffer[frame_idx][video_key])
            video_observation[video_key] = np.stack(frames, axis=0)[None, ...].astype(np.uint8)
        return video_observation

    def _build_state_observation(
        self,
        *,
        robot_state: Any,
        gripper_state: Any,
        cartesian_position: Any,
    ) -> dict[str, Any]:
        import numpy as np

        state_sources = {
            "joint_position": np.asarray(robot_state, dtype=np.float32).reshape(-1),
            "gripper_position": np.asarray(gripper_state, dtype=np.float32).reshape(-1),
            "eef_9d": compute_gr00t_n17_eef_9d(cartesian_position),
        }

        state_observation: dict[str, Any] = {}
        for state_key in self.state_keys:
            if state_key not in state_sources:
                raise KeyError(f"Unsupported GR00T N1.7 state key: {state_key}")
            value = state_sources[state_key].reshape(1, 1, -1).astype(np.float32)
            if self.state_horizon > 1:
                value = np.repeat(value, self.state_horizon, axis=1)
            state_observation[state_key] = value
        return state_observation

    def _build_language_observation(self, instruction: str) -> dict[str, list[list[str]]]:
        if self.language_key is None:
            self.get_modality_config()
        return {self.language_key: [[instruction]]}

    def _print_observation_stats(self, observation: dict[str, Any]) -> None:
        if not self.print_observation_stats or self._observation_stats_printed:
            return

        for modality_name in ("video", "state"):
            for key, value in observation[modality_name].items():
                print(
                    f"[GR00T N1.7 observation] {modality_name}.{key}: "
                    f"shape={value.shape} dtype={value.dtype} min={value.min()} max={value.max()}",
                    flush=True,
                )

        for key, value in observation["language"].items():
            rows = len(value)
            cols = len(value[0]) if value else 0
            flat = [item for batch in value for item in batch]
            print(
                f"[GR00T N1.7 observation] language.{key}: "
                f"shape=({rows}, {cols}) dtype=str min={min(flat)!r} max={max(flat)!r}",
                flush=True,
            )

        self._observation_stats_printed = True

    def build_observation(
        self,
        *,
        instruction: str,
        base_im: Any,
        wrist_im: Any,
        robot_state: Any,
        gripper_state: Any,
        cartesian_position: Any,
        base_im_second: Any | None = None,
        use_base_im_second: bool = False,
        update_frame_buffer: bool = True,
    ) -> dict[str, Any]:
        if self._modality_config is None:
            self.get_modality_config()

        if update_frame_buffer or not self._frame_buffer:
            self.observe(
                base_im=base_im,
                wrist_im=wrist_im,
                base_im_second=base_im_second,
                use_base_im_second=use_base_im_second,
            )

        observation = {
            "video": self._build_video_observation(),
            "state": self._build_state_observation(
                robot_state=robot_state,
                gripper_state=gripper_state,
                cartesian_position=cartesian_position,
            ),
            "language": self._build_language_observation(str(instruction)),
        }
        self._print_observation_stats(observation)
        return observation

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._client.get_action(observation, options)

    def extract_action_chunk(self, action: dict[str, Any]) -> Any:
        import numpy as np

        joint_key = next(
            (
                candidate
                for candidate in _gr00t_n17_action_key_candidates("joint_position")
                if candidate in action
            ),
            None,
        )
        gripper_key = next(
            (
                candidate
                for candidate in _gr00t_n17_action_key_candidates("gripper_position")
                if candidate in action
            ),
            None,
        )

        if joint_key is None or gripper_key is None:
            raise KeyError(
                "GR00T N1.7 action dict must contain joint_position and gripper_position entries"
            )

        joint_chunk = np.asarray(action[joint_key], dtype=np.float32)
        gripper_chunk = np.asarray(action[gripper_key], dtype=np.float32)

        if joint_chunk.ndim != 3:
            raise ValueError(
                f"Expected joint_position action to have shape (B, T, 7), got {joint_chunk.shape}"
            )
        if gripper_chunk.ndim != 3:
            raise ValueError(
                f"Expected gripper_position action to have shape (B, T, 1), got {gripper_chunk.shape}"
            )

        if joint_chunk.shape[0] != 1 or gripper_chunk.shape[0] != 1:
            raise ValueError(
                "REALM GR00T N1.7 adapter currently supports batch size 1 for action chunk extraction"
            )

        if joint_chunk.shape[1] != gripper_chunk.shape[1]:
            raise ValueError(
                f"Joint/gripper horizons must match, got {joint_chunk.shape[1]} and {gripper_chunk.shape[1]}"
            )

        if joint_chunk.shape[2] != 7:
            raise ValueError(
                f"Expected joint_position action dimension 7, got {joint_chunk.shape[2]}"
            )
        if gripper_chunk.shape[2] != 1:
            raise ValueError(
                f"Expected gripper_position action dimension 1, got {gripper_chunk.shape[2]}"
            )

        pred_action_chunk = np.concatenate([joint_chunk[0], gripper_chunk[0]], axis=-1)
        if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[1] != 8:
            raise ValueError(
                f"Expected parsed action chunk to have shape (T, 8), got {pred_action_chunk.shape}"
            )
        return pred_action_chunk.astype(np.float32)

    def infer_action_chunk(
        self, payload: dict[str, Any], options: dict[str, Any] | None = None
    ) -> Any:
        action = self.infer(payload, options=options)
        return self.extract_action_chunk(action)

    def infer(
        self, payload: dict[str, Any], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if all(key in payload for key in ("video", "state", "language")):
            observation = payload
            request_options = options
        elif "observation" in payload and isinstance(payload["observation"], dict):
            observation = payload["observation"]
            request_options = payload.get("options", options)
        elif all(
            key in payload
            for key in (
                "instruction",
                "base_im",
                "wrist_im",
                "robot_state",
                "gripper_state",
                "cartesian_position",
            )
        ):
            observation = self.build_observation(
                instruction=payload["instruction"],
                base_im=payload["base_im"],
                base_im_second=payload.get("base_im_second"),
                wrist_im=payload["wrist_im"],
                robot_state=payload["robot_state"],
                gripper_state=payload["gripper_state"],
                cartesian_position=payload["cartesian_position"],
                use_base_im_second=payload.get("use_base_im_second", False),
                update_frame_buffer=payload.get("update_frame_buffer", True),
            )
            request_options = payload.get("options", options)
        else:
            observation = payload
            request_options = options
        action, _ = self.get_action(observation, request_options)
        return action

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._frame_buffer.clear()
        self._observation_stats_printed = False
        return self._client.reset(options=options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)