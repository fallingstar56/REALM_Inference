import numpy as np


def _to_numpy(value):
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _extract_rgb(sensor_payload):
    if sensor_payload is None or "rgb" not in sensor_payload:
        return None
    rgb = _to_numpy(sensor_payload["rgb"])
    if rgb.ndim != 3:
        return None
    return rgb[..., :3]


def _extract_depth(sensor_payload):
    if sensor_payload is None or "depth_linear" not in sensor_payload:
        return None
    depth = _to_numpy(sensor_payload["depth_linear"])
    return depth


def _extract_wrist_key(robot_obs: dict, robot_name: str):
    preferred = f"{robot_name}:gripper_link_camera:Camera:0"
    if preferred in robot_obs:
        return preferred

    for key in robot_obs:
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if "camera" in lowered and "gripper" in lowered:
            return key
    return None


def _extract_gripper_limits_from_robot(robot):
    if robot is None:
        return None, None

    default_arm = getattr(robot, "default_arm", 0)
    controller_name = f"gripper_{default_arm}"
    controllers = getattr(robot, "_controllers", {})
    controller = controllers.get(controller_name)
    if controller is None:
        return None, None

    dof_idx = np.asarray(getattr(controller, "dof_idx", []), dtype=np.int64)
    if dof_idx.size == 0:
        return None, None

    finger_dof_idx = dof_idx[:2]
    open_qpos = getattr(controller, "_open_qpos", None)
    closed_qpos = getattr(controller, "_closed_qpos", None)

    if open_qpos is not None:
        open_qpos = _to_numpy(open_qpos).astype(np.float32).reshape(-1)[: finger_dof_idx.size]
    if closed_qpos is not None:
        closed_qpos = _to_numpy(closed_qpos).astype(np.float32).reshape(-1)[: finger_dof_idx.size]

    control_limits = getattr(controller, "_control_limits", {})
    if open_qpos is None and "position" in control_limits:
        _, upper = control_limits["position"]
        upper = _to_numpy(upper).astype(np.float32)
        open_qpos = upper[finger_dof_idx]

    if closed_qpos is None and "position" in control_limits:
        lower, _ = control_limits["position"]
        lower = _to_numpy(lower).astype(np.float32)
        closed_qpos = lower[finger_dof_idx]

    return open_qpos, closed_qpos


def normalize_gripper_position(finger_qpos, open_qpos=None, closed_qpos=None, eps=1e-6):
    finger = np.asarray(finger_qpos, dtype=np.float32).reshape(-1)
    if finger.size == 0:
        return 0.0

    # Some environments expose gripper as normalized [0, 1] already.
    if open_qpos is None and closed_qpos is None:
        finite_max = float(np.nanmax(finger)) if finger.size > 0 else 0.0
        finite_min = float(np.nanmin(finger)) if finger.size > 0 else 0.0
        if finite_min >= -eps and finite_max <= 1.0 + eps and finite_max > 0.2:
            return float(np.clip(np.mean(finger), 0.0, 1.0))

    if open_qpos is None:
        open_qpos = np.full_like(finger, 0.05, dtype=np.float32)
    else:
        open_qpos = np.asarray(open_qpos, dtype=np.float32).reshape(-1)

    if closed_qpos is None:
        closed_qpos = np.zeros_like(finger, dtype=np.float32)
    else:
        closed_qpos = np.asarray(closed_qpos, dtype=np.float32).reshape(-1)

    if open_qpos.size != finger.size:
        open_qpos = np.full_like(finger, open_qpos[0], dtype=np.float32)
    if closed_qpos.size != finger.size:
        closed_qpos = np.full_like(finger, closed_qpos[0], dtype=np.float32)

    denom = open_qpos - closed_qpos
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + (denom == 0) * eps, denom)
    normalized = (finger - closed_qpos) / denom
    return float(np.clip(np.mean(normalized), 0.0, 1.0))


def discretize_gripper_action(value, threshold=0.5, open_if_above_threshold=False):
    scalar = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
    should_open = scalar >= threshold if open_if_above_threshold else scalar < threshold
    return 1.0 if should_open else -1.0


def extract_from_obs(obs: dict, robot_name="DROID", enable_depth=False, robot=None):
    external_obs = obs.get("external", {})

    sensor0 = external_obs.get("external_sensor0")
    base_im = _extract_rgb(sensor0)
    if base_im is None:
        base_im = np.zeros((128, 128, 3), dtype=np.uint8)
    base_depth = _extract_depth(sensor0) if enable_depth else None
    if enable_depth and base_depth is None:
        base_depth = np.zeros(base_im.shape[:2], dtype=np.float32)

    sensor1 = external_obs.get("external_sensor1")
    base_im_second = _extract_rgb(sensor1)
    base_depth_second = _extract_depth(sensor1) if enable_depth else None

    robot_obs = obs.get(robot_name, {})
    wrist_im = None
    wrist_key = _extract_wrist_key(robot_obs, robot_name)
    if wrist_key is not None:
        wrist_im = _extract_rgb(robot_obs.get(wrist_key))
    if wrist_im is None:
        wrist_im = np.zeros_like(base_im)

    proprio = _to_numpy(robot_obs["proprio"]).astype(np.float32).reshape(-1)
    robot_state = proprio[:7]

    if proprio.size >= 9:
        finger_qpos = proprio[7:9]
    elif proprio.size >= 8:
        finger_qpos = proprio[7:8]
    else:
        finger_qpos = proprio[-1:]

    open_qpos, closed_qpos = _extract_gripper_limits_from_robot(robot)
    gripper_state = normalize_gripper_position(
        finger_qpos,
        open_qpos=open_qpos,
        closed_qpos=closed_qpos,
    )

    return base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state
