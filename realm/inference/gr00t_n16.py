from __future__ import annotations

import importlib
import os
import re
import sys
import types
from collections import deque
from pathlib import Path
from typing import Any

from realm.inference.gr00t_n17 import _load_resize_with_pad


def _candidate_gr00t_n16_roots() -> list[Path]:
    env_roots = [
        os.environ.get("GR00T_N16_ROOT"),
        os.environ.get("ISAAC_GR00T_N16_ROOT"),
    ]
    workspace_root = Path(__file__).resolve().parents[2]
    candidates = [
        workspace_root / "Isaac-GR00T-n1.6-release",
        workspace_root.parent / "Isaac-GR00T-n1.6-release",
    ]

    resolved: list[Path] = []
    for root in env_roots:
        if root:
            resolved.append(Path(root).expanduser().resolve())
    for root in candidates:
        resolved.append(root.resolve())
    return resolved


def _ensure_gr00t_n16_on_path() -> Path | None:
    for candidate in _candidate_gr00t_n16_roots():
        if not candidate.exists():
            continue
        if not (candidate / "gr00t" / "policy" / "server_client.py").is_file():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return candidate
    return None


def _clear_gr00t_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "gr00t" or module_name.startswith("gr00t."):
            sys.modules.pop(module_name, None)


def _module_has_search_path(module: Any, expected_path: Path) -> bool:
    module_paths = getattr(module, "__path__", None)
    if not module_paths:
        return False
    try:
        return any(Path(path).resolve() == expected_path for path in module_paths)
    except OSError:
        return False


def _module_loaded_from_root(module: Any, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        return root in Path(module_file).resolve().parents
    except OSError:
        return False


def _install_gr00t_n16_package_stubs(root: Path) -> None:
    package_roots = {
        "gr00t": root / "gr00t",
        "gr00t.policy": root / "gr00t" / "policy",
    }
    for module_name, package_root in package_roots.items():
        existing_module = sys.modules.get(module_name)
        if existing_module is not None and _module_has_search_path(existing_module, package_root):
            continue

        package_stub = types.ModuleType(module_name)
        package_stub.__file__ = str(package_root / "__init__.py")
        package_stub.__package__ = module_name
        package_stub.__path__ = [str(package_root)]
        sys.modules[module_name] = package_stub


def _cfg_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)


def _gr00t_n16_action_key_candidates(action_key: str) -> tuple[str, ...]:
    if action_key.startswith("action."):
        return (action_key, action_key.removeprefix("action."))
    return (f"action.{action_key}", action_key)


def _is_wrist_video_key(video_key: str) -> bool:
    lowered = video_key.lower()
    return any(token in lowered for token in ("wrist", "hand", "eef", "ee", "gripper"))


def _video_stream_index(video_key: str) -> int | None:
    match = re.search(r"(?:exterior|image)[^0-9]*([0-9]+)", video_key.lower())
    if match is None:
        return None
    return int(match.group(1))


def _load_gr00t_n16_policy_client_class():
    root = _ensure_gr00t_n16_on_path()
    if root is None:
        raise ImportError(
            "Could not locate Isaac-GR00T-n1.6-release. Set GR00T_N16_ROOT "
            "(or ISAAC_GR00T_N16_ROOT) to the Isaac-GR00T N1.6 repository root so "
            "REALM can import gr00t.policy.server_client.PolicyClient."
        )

    existing_module = sys.modules.get("gr00t.policy.server_client")
    if existing_module is not None and _module_loaded_from_root(existing_module, root):
        return existing_module.PolicyClient
    if existing_module is not None:
        _clear_gr00t_modules()

    _install_gr00t_n16_package_stubs(root)

    try:
        module = importlib.import_module("gr00t.policy.server_client")
    except Exception as exc:
        _clear_gr00t_modules()
        raise ImportError(
            f"Failed to import Isaac-GR00T N1.6 PolicyClient from {root}. "
            "Ensure the Isaac-GR00T N1.6 client runtime dependencies are installed."
        ) from exc

    if not _module_loaded_from_root(module, root):
        raise ImportError(
            f"Imported gr00t.policy.server_client from an unexpected location while "
            f"loading Isaac-GR00T N1.6: {getattr(module, '__file__', '<unknown>')}"
        )
    return module.PolicyClient


GR00T_N16_DEFAULT_IMAGE_SIZE = (180, 320)
GR00T_N16_SUPPORTED_STATE_KEYS = {"joint_position", "gripper_position"}
GR00T_N16_SUPPORTED_ACTION_KEYS = {"joint_position", "gripper_position"}


class Gr00tN16Client:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
        strict: bool = False,
        image_size: tuple[int, int] = GR00T_N16_DEFAULT_IMAGE_SIZE,
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
        self.state_history_len = 1
        self._frame_buffer: deque[dict[str, Any]] = deque(maxlen=1)
        self._state_buffer: deque[dict[str, Any]] = deque(maxlen=1)
        self.exterior_video_key: str | None = None
        self.wrist_video_key: str | None = None
        self._observation_stats_printed = False

        client_cls = policy_client_cls or _load_gr00t_n16_policy_client_class()
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
                f"Failed to reach GR00T N1.6 server at tcp://{self.host}:{self.port}"
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
            raise ValueError("GR00T N1.6 modality config is missing a language key")
        self.language_key = language_keys[0]

        self.wrist_video_key = next(
            (video_key for video_key in self.video_keys if _is_wrist_video_key(video_key)),
            None,
        )
        self.exterior_video_key = next(
            (video_key for video_key in self.video_keys if video_key != self.wrist_video_key),
            None,
        )

        self._validate_supported_modality()

        history_window = (
            max(-min(self.video_delta_indices), 0) + 1 if self.video_delta_indices else 1
        )
        self.video_history_len = max(history_window, self.video_horizon, 1)
        state_history_window = (
            max(-min(self.state_delta_indices), 0) + 1 if self.state_delta_indices else 1
        )
        self.state_history_len = max(state_history_window, self.state_horizon, 1)
        self._frame_buffer = deque(maxlen=self.video_history_len)
        self._state_buffer = deque(maxlen=self.state_history_len)

    def _validate_supported_modality(self) -> None:
        state_keys = set(self.state_keys)
        action_keys = set(self.action_keys)
        unsupported_state_keys = sorted(state_keys - GR00T_N16_SUPPORTED_STATE_KEYS)
        unsupported_action_keys = sorted(action_keys - GR00T_N16_SUPPORTED_ACTION_KEYS)
        missing_state_keys = sorted(GR00T_N16_SUPPORTED_STATE_KEYS - state_keys)
        missing_action_keys = sorted(GR00T_N16_SUPPORTED_ACTION_KEYS - action_keys)

        if (
            unsupported_state_keys
            or unsupported_action_keys
            or missing_state_keys
            or missing_action_keys
        ):
            raise ValueError(
                "GR00T N1.6 REALM adapter supports the OXE_DROID joint/gripper modality only. "
                f"Received state_keys={self.state_keys}, action_keys={self.action_keys}. "
                f"Unsupported state keys={unsupported_state_keys}, unsupported action keys={unsupported_action_keys}, "
                f"missing state keys={missing_state_keys}, missing action keys={missing_action_keys}. "
                "If you intended to run GR00T N1.7, use model_type='gr00t_n17'. "
                "If you intended to run GR00T N1.6, restart the N1.6 server with "
                "--embodiment-tag OXE_DROID --use_sim_policy_wrapper."
            )

    def get_modality_config(self, refresh: bool = False) -> dict[str, Any]:
        if refresh or self._modality_config is None:
            self._modality_config = self._client.get_modality_config()
            self._cache_modality_metadata(self._modality_config)
        return self._modality_config

    def _prepare_image(self, image: Any) -> Any:
        import numpy as np

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(f"Expected an RGB image with shape (H, W, C), got {array.shape}")
        array = array[..., :3]

        if np.issubdtype(array.dtype, np.floating):
            finite_max = float(np.nanmax(array)) if array.size > 0 else 0.0
            if finite_max <= 1.0 + 1e-6:
                array = array * 255.0

        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
        image_height, image_width = self.image_size
        resize_with_pad = _load_resize_with_pad()
        return np.asarray(
            resize_with_pad(array, image_height, image_width),
            dtype=np.uint8,
        )

    def _resolve_video_frame_sources(
        self,
        *,
        base_im: Any,
        wrist_im: Any,
        base_im_second: Any | None = None,
        use_base_im_second: bool = False,
    ) -> dict[str, Any]:
        selected_exterior = base_im_second if use_base_im_second and base_im_second is not None else base_im
        alternate_exteriors: list[Any] = []
        if use_base_im_second:
            if base_im is not None:
                alternate_exteriors.append(base_im)
        elif base_im_second is not None:
            alternate_exteriors.append(base_im_second)

        prepared_wrist = self._prepare_image(wrist_im)
        prepared_selected_exterior = self._prepare_image(selected_exterior)
        prepared_alternate_exteriors = [
            self._prepare_image(image)
            for image in alternate_exteriors
            if image is not None
        ]
        prepared_exteriors = [prepared_selected_exterior, *prepared_alternate_exteriors]

        resolved_frames: dict[str, Any] = {}
        for video_key in self.video_keys:
            if _is_wrist_video_key(video_key):
                resolved_frames[video_key] = prepared_wrist
                continue

            stream_index = _video_stream_index(video_key)
            if stream_index is None or stream_index < 1:
                resolved_frames[video_key] = prepared_selected_exterior
                continue

            resolved_idx = min(stream_index - 1, len(prepared_exteriors) - 1)
            resolved_frames[video_key] = prepared_exteriors[resolved_idx]
        return resolved_frames

    def _build_state_snapshot(self, *, robot_state: Any, gripper_state: Any) -> dict[str, Any]:
        import numpy as np

        state_sources = {
            "joint_position": np.asarray(robot_state, dtype=np.float32).reshape(-1),
            "gripper_position": np.asarray(gripper_state, dtype=np.float32).reshape(-1),
        }

        state_snapshot: dict[str, Any] = {}
        for state_key in self.state_keys:
            if state_key not in state_sources:
                raise KeyError(f"Unsupported GR00T N1.6 state key: {state_key}")
            state_snapshot[state_key] = state_sources[state_key].astype(np.float32, copy=False)
        return state_snapshot

    def observe(
        self,
        *,
        base_im: Any,
        wrist_im: Any,
        base_im_second: Any | None = None,
        use_base_im_second: bool = False,
        robot_state: Any | None = None,
        gripper_state: Any | None = None,
        **_: Any,
    ) -> None:
        if not self.video_keys:
            self.get_modality_config()

        prepared_frames = self._resolve_video_frame_sources(
            base_im=base_im,
            wrist_im=wrist_im,
            base_im_second=base_im_second,
            use_base_im_second=use_base_im_second,
        )
        self._frame_buffer.append(prepared_frames)

        if robot_state is not None and gripper_state is not None:
            self._state_buffer.append(
                self._build_state_snapshot(
                    robot_state=robot_state,
                    gripper_state=gripper_state,
                )
            )

    def _build_video_observation(self) -> dict[str, Any]:
        import numpy as np

        if not self._frame_buffer:
            raise ValueError(
                "GR00T N1.6 frame buffer is empty. Call observe() before requesting an action."
            )

        last_idx = len(self._frame_buffer) - 1
        video_observation: dict[str, Any] = {}
        for video_key in self.video_keys:
            frames = []
            for delta_idx in self.video_delta_indices:
                frame_idx = min(max(last_idx + delta_idx, 0), last_idx)
                frames.append(self._frame_buffer[frame_idx][video_key])
            video_observation[f"video.{video_key}"] = np.stack(frames, axis=0)[None, ...].astype(np.uint8)
        return video_observation

    def _build_state_observation(
        self,
        *,
        robot_state: Any | None = None,
        gripper_state: Any | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        if not self._state_buffer:
            if robot_state is None or gripper_state is None:
                raise ValueError(
                    "GR00T N1.6 state buffer is empty. Provide robot_state and gripper_state before requesting an action."
                )
            self._state_buffer.append(
                self._build_state_snapshot(
                    robot_state=robot_state,
                    gripper_state=gripper_state,
                )
            )

        last_idx = len(self._state_buffer) - 1
        state_observation: dict[str, Any] = {}
        for state_key in self.state_keys:
            states = []
            for delta_idx in self.state_delta_indices:
                state_idx = min(max(last_idx + delta_idx, 0), last_idx)
                states.append(self._state_buffer[state_idx][state_key])
            state_observation[f"state.{state_key}"] = np.stack(states, axis=0)[None, ...].astype(np.float32)
        return state_observation

    def _print_observation_stats(self, observation: dict[str, Any]) -> None:
        if not self.print_observation_stats or self._observation_stats_printed:
            return

        for key, value in observation.items():
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                print(
                    f"[GR00T N1.6 observation] {key}: "
                    f"shape={value.shape} dtype={value.dtype} min={value.min()} max={value.max()}",
                    flush=True,
                )
            else:
                print(
                    f"[GR00T N1.6 observation] {key}: "
                    f"shape=({len(value)},) dtype={type(value[0]).__name__}",
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
        base_im_second: Any | None = None,
        use_base_im_second: bool = False,
        update_frame_buffer: bool = True,
    ) -> dict[str, Any]:
        if self._modality_config is None:
            self.get_modality_config()

        if self.exterior_video_key is None or self.wrist_video_key is None:
            raise ValueError(
                "GR00T N1.6 modality config must include one exterior video key and one wrist video key"
            )

        if update_frame_buffer or not self._frame_buffer:
            self.observe(
                base_im=base_im,
                wrist_im=wrist_im,
                base_im_second=base_im_second,
                use_base_im_second=use_base_im_second,
                robot_state=robot_state,
                gripper_state=gripper_state,
            )

        observation = {}
        observation.update(self._build_video_observation())
        observation.update(
            self._build_state_observation(
                robot_state=robot_state,
                gripper_state=gripper_state,
            )
        )
        observation[str(self.language_key)] = [str(instruction)]
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
                for candidate in _gr00t_n16_action_key_candidates("joint_position")
                if candidate in action
            ),
            None,
        )
        gripper_key = next(
            (
                candidate
                for candidate in _gr00t_n16_action_key_candidates("gripper_position")
                if candidate in action
            ),
            None,
        )

        if joint_key is None or gripper_key is None:
            raise KeyError(
                "GR00T N1.6 action dict must contain joint_position and gripper_position entries"
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
                "REALM GR00T N1.6 adapter currently supports batch size 1 for action chunk extraction"
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
        if all(
            key in payload
            for key in (
                "instruction",
                "base_im",
                "wrist_im",
                "robot_state",
                "gripper_state",
            )
        ):
            observation = self.build_observation(
                instruction=payload["instruction"],
                base_im=payload["base_im"],
                base_im_second=payload.get("base_im_second"),
                wrist_im=payload["wrist_im"],
                robot_state=payload["robot_state"],
                gripper_state=payload["gripper_state"],
                use_base_im_second=payload.get("use_base_im_second", False),
                update_frame_buffer=payload.get("update_frame_buffer", True),
            )
            request_options = payload.get("options", options)
        elif "observation" in payload and isinstance(payload["observation"], dict):
            observation = payload["observation"]
            request_options = payload.get("options", options)
        else:
            observation = payload
            request_options = options

        action, _ = self.get_action(observation, request_options)
        return action

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._observation_stats_printed = False
        self._frame_buffer.clear()
        self._state_buffer.clear()
        return self._client.reset(options=options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)