import numpy as np


def extract_from_obs(obs: dict, robot_name='DROID', enable_depth=False):
    # Fallback to zeros if external sensors are missing (e.g. during no_render)
    if 'external' in obs and 'external_sensor0' in obs['external']:
        base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
        base_depth = obs['external']['external_sensor0']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        # Dummy 128x128 image
        base_im = np.zeros((128, 128, 3), dtype=np.uint8)
        base_depth = np.zeros((128, 128), dtype=np.float32) if enable_depth else None

    if 'external' in obs and 'external_sensor1' in obs['external']:
        base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
        base_depth_second = obs['external']['external_sensor1']['depth_linear'].cpu().numpy() if enable_depth else None
    else:
        base_im_second = None
        base_depth_second = None

    # Handle wrist camera robustly across robot naming / mounting variants.
    wrist_im = None
    if robot_name in obs:
        robot_obs = obs[robot_name]
        wrist_cam_key = f'{robot_name}:gripper_link_camera:Camera:0'
        if wrist_cam_key in robot_obs:
            wrist_im = robot_obs[wrist_cam_key]['rgb'].cpu().numpy()[..., :3]
        else:
            for key, value in robot_obs.items():
                if 'gripper_link_camera' in key and isinstance(value, dict) and 'rgb' in value:
                    wrist_im = value['rgb'].cpu().numpy()[..., :3]
                    break
    if wrist_im is None:
        wrist_im = np.zeros((128, 128, 3), dtype=np.uint8)

    # Proprio is always present in DROID and other robots
    proprio = obs[robot_name]['proprio'].cpu().numpy()
    robot_state = proprio[:7]

    # DROID exposes multiple finger joints; collapse them to a single normalized scalar.
    gripper_qpos = proprio[7:9] if proprio.shape[0] >= 9 else proprio[7:8]
    gripper_state = float(np.mean(gripper_qpos)) if gripper_qpos.size > 0 else 0.0
    if -1e-4 <= gripper_state <= 0.06:
        gripper_state = gripper_state / 0.05
    gripper_state = float(np.clip(gripper_state, 0.0, 1.0))

    return base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state
