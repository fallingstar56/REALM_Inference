from __future__ import annotations

import copy
import numpy as np
from typing import TYPE_CHECKING

import omnigibson as og
from realm.helpers import (
    get_non_colliding_positions_for_objects,
    get_droid_categories_by_theme,
    get_objects_by_names,
    get_default_objects_cfg,
)
from realm.environments.perturbations._helpers import replace_obj

if TYPE_CHECKING:
    from realm.environments.env_dynamic import RealmEnvironmentDynamic


TASK_OBJECT_MIN_XY_DELTA = 0.03
MAX_SCENE_PLACEMENT_ATTEMPTS = 50


def _as_numpy_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _sync_init_poses(env: "RealmEnvironmentDynamic") -> None:
    for scene_obj in env.main_objects + env.target_objects + env.distractors:
        pos, rot = scene_obj.get_position_orientation()
        env.init_poses[scene_obj._relative_prim_path] = {"pos": pos, "rot": rot}


def _task_objects_moved(obj_cfgs: list[dict], original_positions: dict[str, np.ndarray], task_object_names: set[str]) -> bool:
    for obj_cfg in obj_cfgs:
        if obj_cfg["name"] not in task_object_names:
            continue
        new_position = _as_numpy_array(obj_cfg["position"])
        original_position = original_positions[obj_cfg["name"]]
        if np.linalg.norm(new_position[:2] - original_position[:2]) < TASK_OBJECT_MIN_XY_DELTA:
            return False
    return True


def v_sc(env: "RealmEnvironmentDynamic") -> None:
    # --------------- Translation ---------------
    og.sim.stop()

    obj_cfgs = copy.deepcopy(env.cfg["objects"])
    task_objects = env.main_objects + env.target_objects
    task_object_names = {obj.name for obj in task_objects}
    num_mo_to = len(task_objects)
    original_task_positions = {obj.name: _as_numpy_array(obj.get_position_orientation()[0]) for obj in task_objects}

    for scene_obj in task_objects:
        for cfg in obj_cfgs:
            if cfg["name"] == scene_obj.name:
                if "position" not in cfg:
                    cfg["position"] = scene_obj.get_position_orientation()[0].tolist()
                if "bounding_box" not in cfg:
                    cfg["bounding_box"] = scene_obj.aabb_extent.tolist()

    env.cfg["objects"] = None
    num_distractors = len(obj_cfgs) - num_mo_to

    placed_obj_cfgs = None
    for _ in range(MAX_SCENE_PLACEMENT_ATTEMPTS):
        placed_obj_cfgs = get_non_colliding_positions_for_objects(
            xmin=env.spawn_bbox[0],
            xmax=env.spawn_bbox[1],
            ymin=env.spawn_bbox[2],
            ymax=env.spawn_bbox[3],
            z=env.spawn_bbox[4],
            obj_cfg=copy.deepcopy(obj_cfgs[:num_mo_to + num_distractors]),
            objects_to_skip=[],
            main_object_names=[],
            max_attempts_per_object=25000,
            maximum_dim=0.12,
        )
        if _task_objects_moved(placed_obj_cfgs, original_task_positions, task_object_names):
            break
    else:
        og.log.warn(
            f"V-SC could not move every task object by {TASK_OBJECT_MIN_XY_DELTA}m in XY "
            f"after {MAX_SCENE_PLACEMENT_ATTEMPTS} placement attempts."
        )

    env.cfg["objects"] = placed_obj_cfgs

    env.distractors = [env.omnigibson_env.scene.object_registry("name", dist["name"]) for dist in env.cfg["objects"][num_mo_to:]]

    # TODO: check if this works properly in the edge cases where it should trigger
    if num_distractors < len(env.distractors):
        for dist_cfg in env.cfg["objects"][num_mo_to + num_distractors:]:
            obj = env.omnigibson_env.scene.object_registry("name", dist_cfg["name"])
            env.omnigibson_env.scene.remove_object(obj)
        env.cfg["objects"] = env.cfg["objects"][:num_mo_to + num_distractors]

    # --------------- Set Position ---------------
    for obj in env.cfg["objects"]:
        env.omnigibson_env.scene.object_registry("name", obj["name"]).set_position(obj["position"])
    _sync_init_poses(env)

    # --------------- Replace the objects models ---------------
    distractor_obj_cfgs = get_default_objects_cfg(env.omnigibson_env.scene, [obj.name for obj in env.distractors])
    distractor_objs = get_objects_by_names(env.omnigibson_env.scene, list(distractor_obj_cfgs.keys()))
    excluded_categories = [obj.category for obj in env.main_objects + env.target_objects]
    new_distractors = []
    for distractor in distractor_objs:
        cat_dict = get_droid_categories_by_theme()
        t = [k for k, v in cat_dict.items() if any(distractor.category in c for c in v.values())]
        if t:
            cat_dict.pop(t[0])
        l = [o for v in cat_dict.values() for c in v.values() for o in c]
        l = [c for c in l if c not in excluded_categories]
        new_distractor, _ = replace_obj(env, distractor, included_categories=l, maximum_dim=0.12)
        new_distractors.append(new_distractor)
    if new_distractors:
        env.distractors = new_distractors
        _sync_init_poses(env)

    og.sim.play()
    env.reset_joints()
    # fake rest to get to original pose after stopping sim
    for _ in range(30):
        env.omnigibson_env.step(np.concatenate((env.reset_qpos[:7], np.atleast_1d(np.array([-1])))))
    _sync_init_poses(env)
    env.mo_pos_orig, env.mo_rot_orig = env.main_objects[0].get_position_orientation()
