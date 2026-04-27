# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import math
from pathlib import Path

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import open_stage
import isaacsim.core.api.objects as objects
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_update_semantics
from pxr import Usd, UsdGeom


from omni.ext.patosim.occupancy_map import OccupancyMap
from omni.ext.patosim.config import Config
from omni.ext.patosim.utils.global_utils import new_stage, new_world, set_viewport_camera, get_stage
from omni.ext.patosim.scenarios import Scenario, SCENARIOS
from omni.ext.patosim.robots import ROBOTS
from omni.ext.patosim.reader import Reader
from omni.ext.patosim.utils.stage_utils import stage_get_prim


DATASET_OBJECT_ALLOWED_FOLDERS = ("plataforms", "platforms", "statues_temples", "scenario")
DATASET_OBJECT_ASSET_ROOT = Path(__file__).resolve().parents[5] / "assets" / "models"


def _validate_scene_usd_path(scene_path: str) -> str:
    scene_path = str(scene_path or "").strip()
    if scene_path == "":
        raise RuntimeError(
            "build_scenario_from_config: 'scene_usd' is empty. "
            "Please set the USD Path / URL in the UI before clicking Build."
        )

    lower = scene_path.lower()
    supported_suffixes = (".usd", ".usda", ".usdc", ".usdz")
    if not lower.endswith(supported_suffixes):
        raise RuntimeError(
            "build_scenario_from_config: unsupported scene asset path "
            f"'{scene_path}'. Expected a USD asset ({', '.join(supported_suffixes)}), "
            "not a zip/archive or another file type."
        )

    if "://" not in scene_path and not os.path.exists(scene_path):
        raise RuntimeError(
            "build_scenario_from_config: scene asset does not exist at "
            f"'{scene_path}'."
        )

    return scene_path


def _is_underwater_robot_type(robot_type) -> bool:
    return getattr(robot_type, "__name__", "") == "OceanSimROVRobot"


def _make_underwater_placeholder_occupancy_map() -> OccupancyMap:
    size = 256
    resolution = 0.25
    freespace = np.ones((size, size), dtype=bool)
    occupied = np.zeros((size, size), dtype=bool)
    origin = (-(size * resolution) / 2.0, -(size * resolution) / 2.0, 0.0)
    return OccupancyMap.from_masks(
        freespace_mask=freespace,
        occupied_mask=occupied,
        resolution=resolution,
        origin=origin,
    )


async def _make_underwater_occupancy_map_async(
    scene_prim_path: str,
    z_nominal: float = -2.0,
    slice_half: float = 0.5,
    cell_size: float = 0.25,
) -> OccupancyMap:
    try:
        from omni.ext.patosim.utils.stage_utils import occupancy_map_generate_from_prim_async as _gen
        omap = await _gen(
            scene_prim_path,
            cell_size=cell_size,
            z_min=z_nominal - slice_half,
            z_max=z_nominal + slice_half,
        )
        if getattr(omap, "data", None) is not None and omap.data.size > 0:
            return omap
    except Exception as exc:
        import warnings
        warnings.warn(
            f"_make_underwater_occupancy_map_async: failed to generate scene occupancy map "
            f"(z={z_nominal:.1f}m) — falling back to empty map. Reason: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return _make_underwater_placeholder_occupancy_map()


def _enable_scene_collisions(scene_root_path: str) -> int:
    stage = get_stage()
    root_prim = stage_get_prim(stage, scene_root_path)
    if root_prim is None or not root_prim.IsValid():
        return 0

    enabled = 0

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid():
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        # Nunca habilite colisao no prim raiz do mundo referenciado:
        # isso pode gerar um collider englobando toda a cena e prender o ROV
        # "dentro" do mundo logo no spawn.
        prim_path = prim.GetPath().pathString
        if prim_path == scene_root_path:
            continue
        try:
            geom = SingleGeometryPrim(prim_path=prim_path, collision=True)
            try:
                geom.set_collision_approximation("convexDecomposition")
            except Exception:
                pass
            enabled += 1
        except Exception:
            continue
    return enabled


def _apply_sonar_reflectivity_to_scene(scene_root_path: str, default_reflectivity: float = 1.0) -> int:
    stage = get_stage()
    root_prim = stage_get_prim(stage, scene_root_path)
    if root_prim is None or not root_prim.IsValid():
        return 0

    applied = 0
    reflectivity_label = str(float(default_reflectivity))
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid():
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        try:
            add_update_semantics(
                prim=prim,
                type_label="reflectivity",
                semantic_label=reflectivity_label,
            )
            applied += 1
        except Exception:
            continue
    return applied


def _compute_world_bbox(scene_root_path: str):
    stage = get_stage()
    root_prim = stage_get_prim(stage, scene_root_path)
    if root_prim is None or not root_prim.IsValid():
        return None
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        bbox = bbox_cache.ComputeWorldBound(root_prim).ComputeAlignedBox()
        if bbox.IsEmpty():
            return None
        return bbox
    except Exception:
        return None


def _compute_safe_underwater_spawn(scene_root_path: str, desired_position: np.ndarray) -> np.ndarray:
    desired_position = np.asarray(desired_position, dtype=np.float32).reshape(3)
    bbox = _compute_world_bbox(scene_root_path)
    if bbox is None:
        return desired_position

    min_v = np.asarray(bbox.GetMin(), dtype=np.float32)
    max_v = np.asarray(bbox.GetMax(), dtype=np.float32)
    safe_position = desired_position.copy()
    inside = bool(np.all(desired_position >= min_v) and np.all(desired_position <= max_v))
    below_world = bool(desired_position[2] <= (min_v[2] + 0.25))

    if not inside and not below_world:
        return desired_position

    # If the requested spawn is below the world or intersects the scene bounds,
    # lift the robot above the scene to avoid starting inside geometry/colliders.
    safe_position[2] = float(max_v[2] + 1.0)

    if inside:
        xy_inside = bool(
            (min_v[0] <= desired_position[0] <= max_v[0])
            and (min_v[1] <= desired_position[1] <= max_v[1])
        )
        if xy_inside:
            safe_position[0] = float((min_v[0] + max_v[0]) * 0.5)
            safe_position[1] = float((min_v[1] + max_v[1]) * 0.5)

    return safe_position


def list_dataset_object_assets(asset_root: str | Path | None = None) -> list[dict]:
    root = Path(asset_root) if asset_root is not None else DATASET_OBJECT_ASSET_ROOT
    entries: list[dict] = []
    if not root.exists():
        return entries

    exts = {".usd", ".usda", ".usdc", ".usdz"}
    for folder_name in DATASET_OBJECT_ALLOWED_FOLDERS:
        folder = root / folder_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            rel_path = path.relative_to(root)
            entries.append(
                {
                    "label": str(rel_path).replace(os.sep, " / "),
                    "path": str(path.resolve()),
                    "relative_path": str(rel_path).replace(os.sep, "/"),
                    "category": folder_name,
                    "class_name": path.stem,
                }
            )
    return entries


def _dataset_object_entry_from_path(asset_path: str) -> dict | None:
    asset_path = str(asset_path or "").strip()
    if not asset_path:
        return None
    for entry in list_dataset_object_assets():
        if os.path.normpath(entry["path"]) == os.path.normpath(asset_path):
            return entry
    if not os.path.exists(asset_path):
        return None
    path = Path(asset_path).resolve()
    try:
        rel_path = path.relative_to(DATASET_OBJECT_ASSET_ROOT)
        parts = rel_path.parts
        category = parts[0] if len(parts) > 0 else "custom"
    except Exception:
        rel_path = path.name
        category = "custom"
    return {
        "label": str(rel_path).replace(os.sep, " / "),
        "path": str(path),
        "relative_path": str(rel_path).replace(os.sep, "/"),
        "category": category,
        "class_name": path.stem,
    }


def _set_prim_world_translation(prim_path: str, translation) -> None:
    stage = get_stage()
    prim = stage_get_prim(stage, prim_path)
    if prim is None or not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))


def _enable_collisions_for_subtree(root_prim_path: str) -> int:
    stage = get_stage()
    root_prim = stage_get_prim(stage, root_prim_path)
    if root_prim is None or not root_prim.IsValid():
        return 0

    enabled = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Gprim):
            continue
        try:
            geom = SingleGeometryPrim(prim_path=prim.GetPath().pathString, collision=True)
            try:
                geom.set_collision_approximation("convexDecomposition")
            except Exception:
                pass
            enabled += 1
        except Exception:
            continue
    return enabled


def _mark_dataset_object_for_annotations(root_prim_path: str, asset_entry: dict, reflectivity: float) -> int:
    stage = get_stage()
    root_prim = stage_get_prim(stage, root_prim_path)
    if root_prim is None or not root_prim.IsValid():
        return 0

    class_name = str(asset_entry.get("class_name", "dataset_object"))
    category = str(asset_entry.get("category", "dataset_object"))
    relative_path = str(asset_entry.get("relative_path", class_name))
    reflectivity_label = str(float(reflectivity))

    applied = 0
    try:
        add_update_semantics(prim=root_prim, type_label="class", semantic_label=class_name)
        add_update_semantics(prim=root_prim, type_label="category", semantic_label=category)
        add_update_semantics(prim=root_prim, type_label="asset", semantic_label=relative_path)
        add_update_semantics(prim=root_prim, type_label="dataset_object_root", semantic_label="true")
        applied += 1
    except Exception:
        pass

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid():
            continue
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
            try:
                add_update_semantics(prim=prim, type_label="class", semantic_label=class_name)
                add_update_semantics(prim=prim, type_label="category", semantic_label=category)
                add_update_semantics(prim=prim, type_label="asset", semantic_label=relative_path)
                applied += 1
            except Exception:
                pass
        if prim.IsA(UsdGeom.Mesh):
            try:
                add_update_semantics(
                    prim=prim,
                    type_label="reflectivity",
                    semantic_label=reflectivity_label,
                )
            except Exception:
                pass
    return applied


def _insert_dataset_object_into_scene(
    scene_root_path: str,
    dataset_object_usd: str,
    reflectivity: float,
    config: Config | None = None,
) -> str | None:
    asset_entry = _dataset_object_entry_from_path(dataset_object_usd)
    if asset_entry is None:
        raise RuntimeError(f"Dataset object asset not found: '{dataset_object_usd}'")

    scene_bbox = _compute_world_bbox(scene_root_path)
    if scene_bbox is not None:
        # ComputeAlignedBox() returns a Range3d in this environment, so use
        # its min/max corners instead of relying on a non-portable GetCenter().
        min_v = np.asarray(scene_bbox.GetMin(), dtype=np.float32)
        max_v = np.asarray(scene_bbox.GetMax(), dtype=np.float32)
        center = 0.5 * (min_v + max_v)
    else:
        center = np.zeros(3, dtype=np.float32)

    safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in asset_entry["class_name"]) or "dataset_object"
    prim_path = f"{scene_root_path}/dataset_objects/{safe_name}"
    prim_utils.delete_prim(prim_path)
    add_reference_to_stage(asset_entry["path"], prim_path)
    cfg_pos = getattr(config, "dataset_object_position", None)
    if (
        cfg_pos is not None
        and hasattr(cfg_pos, "__len__")
        and len(cfg_pos) == 3
        and all(math.isfinite(float(v)) for v in cfg_pos)
    ):
        position = np.array([float(v) for v in cfg_pos], dtype=np.float32)
    else:
        position = center

    _set_prim_world_translation(prim_path, position)
    _enable_collisions_for_subtree(prim_path)
    _mark_dataset_object_for_annotations(prim_path, asset_entry, reflectivity=float(reflectivity))
    return prim_path


def load_scenario(path: str) -> Scenario:
    reader = Reader(path)
    config = reader.read_config()
    robot_type = ROBOTS.get(config.robot_type)
    scenario_type = SCENARIOS.get(config.scenario_type)
    occupancy_map = reader.read_occupancy_map()
    scene_usd = str(getattr(config, "scene_usd", "") or "").strip()
    stage_path = os.path.join(path, "stage.usd")

    # Prefer rebuilding replay from the source scene USD instead of reopening
    # the recorded stage. Recorded stages can keep stale runtime graph state
    # from render/sensor pipelines, which later breaks replay.
    rebuilt_from_scene = False
    if scene_usd:
        try:
            scene_usd = _validate_scene_usd_path(scene_usd)
            new_stage()
            add_reference_to_stage(scene_usd, "/World/scene")
            rebuilt_from_scene = True
        except Exception:
            rebuilt_from_scene = False

    if not rebuilt_from_scene:
        open_stage(stage_path)
        prim_utils.delete_prim("/World/robot")

    new_world(physics_dt=robot_type.physics_dt)
    if bool(getattr(config, "apply_sonar_reflectivity_to_world", True)):
        try:
            _apply_sonar_reflectivity_to_scene("/World/scene", default_reflectivity=1.0)
        except Exception:
            pass
    if rebuilt_from_scene and bool(getattr(config, "dataset_object_enabled", False)) and str(
        getattr(config, "dataset_object_usd", "") or ""
    ).strip():
        _insert_dataset_object_into_scene(
            "/World/scene",
            str(getattr(config, "dataset_object_usd", "")),
            float(getattr(config, "dataset_object_reflectivity", 1.5)),
            config=config,
        )
    robot = robot_type.build("/World/robot")
    chase_camera_path = robot.build_chase_camera()
    try:
        robot.chase_camera_path = chase_camera_path
    except Exception:
        pass
    try:
        set_viewport_camera(chase_camera_path)
    except Exception:
        pass

    scenario = scenario_type.from_robot_occupancy_map(robot, occupancy_map)
    return scenario


async def build_scenario_from_config(config: Config):
    from omni.ext.patosim.utils.occupancy_map_utils import occupancy_map_generate_from_prim_async

    robot_type = ROBOTS.get(config.robot_type)
    scenario_type = SCENARIOS.get(config.scenario_type)
    is_underwater_robot = _is_underwater_robot_type(robot_type)
    if is_underwater_robot:
        robot_type.water_profile_path = str(getattr(config, "water_profile_path", "") or "").strip() or None
        robot_type.enable_dvl_debug_lines = bool(getattr(config, "enable_dvl_debug_lines", False))
        robot_type.teleop_linear_speed_gain = float(getattr(config, "rov_linear_speed", 0.75))
        robot_type.teleop_angular_speed_gain = float(getattr(config, "rov_angular_speed", 0.9))
        robot_type.enable_front_camera = bool(getattr(config, "enable_rov_front_camera", True))
        robot_type.enable_stereo_camera = bool(getattr(config, "enable_rov_stereo_camera", False))
        robot_type.enable_lidar = bool(getattr(config, "enable_rov_lidar", False))
        robot_type.enable_sonar = bool(getattr(config, "enable_rov_sonar", True))
        robot_type.enable_dvl = bool(getattr(config, "enable_rov_dvl", True))
        robot_type.enable_barometer = bool(getattr(config, "enable_rov_barometer", True))
    scene_usd = _validate_scene_usd_path(getattr(config, "scene_usd", ""))
    new_stage()
    world = new_world(physics_dt=robot_type.physics_dt)
    await world.initialize_simulation_context_async()
    add_reference_to_stage(scene_usd, "/World/scene")
    _enable_scene_collisions("/World/scene")
    if bool(getattr(config, "apply_sonar_reflectivity_to_world", True)):
        try:
            _apply_sonar_reflectivity_to_scene("/World/scene", default_reflectivity=1.0)
        except Exception:
            pass
    if bool(getattr(config, "dataset_object_enabled", False)) and str(
        getattr(config, "dataset_object_usd", "") or ""
    ).strip():
        _insert_dataset_object_into_scene(
            "/World/scene",
            str(getattr(config, "dataset_object_usd", "")),
            float(getattr(config, "dataset_object_reflectivity", 1.5)),
            config=config,
        )
    if not is_underwater_robot:
        objects.GroundPlane("/World/ground_plane", visible=False)
    robot = robot_type.build("/World/robot")
    if is_underwater_robot:
        safe_spawn = _compute_safe_underwater_spawn(
            "/World/scene",
            np.asarray(getattr(robot_type, "initial_translation", (-2.0, 0.0, -0.8)), dtype=np.float32),
        )
        try:
            robot.spawn_translation = safe_spawn.astype(np.float32)
            robot.initial_translation = tuple(float(v) for v in safe_spawn)
            robot.set_pose_3d(safe_spawn, robot_type._initial_orientation())
        except Exception:
            pass
        z_nominal = float(getattr(config, "rov_operating_depth", -2.0))
        occupancy_map = await _make_underwater_occupancy_map_async(
            "/World/scene",
            z_nominal=z_nominal,
            cell_size=float(getattr(robot, "occupancy_map_cell_size", 0.25)),
        )
    else:
        occupancy_map = await occupancy_map_generate_from_prim_async(
            "/World/scene",
            cell_size=robot.occupancy_map_cell_size,
            z_min=robot.occupancy_map_z_min,
            z_max=robot.occupancy_map_z_max
        )
        if getattr(occupancy_map, "data", None) is None or occupancy_map.data.size == 0:
            raise RuntimeError(
                "build_scenario_from_config: occupancy map generation returned an empty map. "
                f"Scene asset='{scene_usd}'. Check whether the USD loaded successfully and "
                "whether the referenced stage contains visible geometry under /World/scene."
            )
    chase_camera_path = robot.build_chase_camera()
    try:
        robot.chase_camera_path = chase_camera_path
    except Exception:
        pass
    set_viewport_camera(chase_camera_path)
    scenario = scenario_type.from_robot_occupancy_map(robot, occupancy_map)
    if is_underwater_robot:
        waypoint_path = str(getattr(config, "waypoint_path", "") or "").strip()
        if waypoint_path and hasattr(scenario, "_waypoint_path"):
            try:
                from pathlib import Path
                scenario._waypoint_path = Path(waypoint_path)
            except Exception:
                pass
    return scenario
