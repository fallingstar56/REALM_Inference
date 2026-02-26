import argparse
import sys
import os
import numpy as np
import torch
from datetime import datetime

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import set_sim_config
from realm.utils import replay_traj, plot_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    parser.add_argument('--max_eps', type=int, required=False, default=5)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/app/logs/replay_trajectory/{run_id}"
    task_cfg_path = f"IMPACT/trajectory_replay/default.yaml"
    traj_path = "/app/data/droid_1.0.1/extracted_eps/chunk-000/"
    rendering_mode = "r"
    #robot = "DROID_polaris_control"
    #robot = "DROID_default_pd_control"
    #robot = "DROID_no_wrist_cam"
    robot = "WidowX"

    ep_names = [d for d in os.listdir(traj_path) if os.path.isdir(os.path.join(traj_path, d))]
    ep_names = ep_names[:args.max_eps]

    set_sim_config(rendering_mode=rendering_mode, robot=robot)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=["Default"],
        rendering_mode=rendering_mode,
        robot=robot,
        no_rendering=True
    )

    for traj_id in range(len(ep_names)):
        traj_qpos_actions = np.load(f"{traj_path}/{ep_names[traj_id]}/action_joint_position.npy")
        traj_qpos_gt = np.load(f"{traj_path}/{ep_names[traj_id]}/observation_state_joint_position.npy")
        traj_ee_gt = np.load(f"{traj_path}/{ep_names[traj_id]}/observation_state_cartesian_position.npy")
        # for i in len(traj_ee_gt):
        #     traj_ee_gt[i, 3:6] = flip_pose_pointing_down(traj_ee_gt[i, 3:6])

        res_dict = replay_traj(env, traj_qpos_actions, traj_qpos_gt, traj_ee_gt)
        plot_err(res_dict, ep_names[traj_id], log_dir=log_dir, plot_title=robot)

    og.shutdown()
    sys.exit(0)
