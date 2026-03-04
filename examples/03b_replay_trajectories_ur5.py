import argparse
import sys
import os
import glob
import h5py
import random
import numpy as np
import torch
from datetime import datetime

import omnigibson as og

from realm.environments.env_dynamic import RealmEnvironmentDynamic
from realm.eval import set_sim_config
from realm.utils import replay_traj, plot_err


def get_joint_data(f, root_key, arm_priority=['arm_left_position_align']):
    if root_key not in list(f.keys()):
        return None
    
    group = f[root_key]
    for arm_prefix in arm_priority:
        return group[arm_prefix]['data'][:]
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    parser.add_argument('--max_eps', type=int, required=False, default=5)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/app/logs/replay_trajectory/{run_id}"
    task_cfg_path = f"IMPACT/trajectory_replay/default.yaml"
    traj_root = "/app/data/RoboMIND2.0-UR5/data/ur/"
    rendering_mode = "r"
    robot = "UR5"

    ep_paths = sorted(glob.glob(os.path.join(traj_root, "**/trajectory.hdf5"), recursive=True))
    random.seed(42)
    random.shuffle(ep_paths)
    ep_paths = ep_paths[:args.max_eps]
    ep_names = [os.path.relpath(os.path.dirname(p), traj_root).replace("/", "_") for p in ep_paths]

    set_sim_config(rendering_mode=rendering_mode, robot=robot)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=["Default"],
        rendering_mode=rendering_mode,
        robot=robot,
        no_rendering=True
    )

    for i, full_path in enumerate(ep_paths):
        print(f"Replaying episode {i+1}/{len(ep_paths)}: {ep_names[i]}")
        
        try:
            with h5py.File(full_path, 'r') as f:
                # 1. Try to get actions (prefer puppet, fallback to master)
                traj_qpos_actions = get_joint_data(f, 'puppet')
                if traj_qpos_actions is None:
                    traj_qpos_actions = get_joint_data(f, 'master')
                
                # 2. Try to get ground truth (prefer master)
                traj_qpos_gt = get_joint_data(f, 'master')
                if traj_qpos_gt is None:
                    traj_qpos_gt = traj_qpos_actions
                
                if traj_qpos_actions is None:
                    print(f"Skipping {ep_names[i]}: Could not find joint data in 'puppet' or 'master'.")
                    continue

                traj_ee_gt = None # Cartesian GT is optional in replay_traj

            res_dict = replay_traj(env, traj_qpos_actions, traj_qpos_gt, traj_ee_gt, dof=6)
            plot_err(res_dict, ep_names[i], log_dir=log_dir, plot_title=robot)
            
        except Exception as e:
            print(f"Error processing {ep_names[i]}: {e}")
            continue

    og.shutdown()
    sys.exit(0)
