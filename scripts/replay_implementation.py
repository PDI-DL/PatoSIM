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

"""
Replay and render a recording with Isaac Sim.

This script replays a recording with SimulationApp and re-renders outputs.
"""

import argparse
import asyncio
import glob
import json
import os
from pathlib import Path
import signal
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import tqdm


STOP_REQUESTED = False

if "PATOSIM_DATA" in os.environ:
    DATA_DIR = os.environ["PATOSIM_DATA"]
else:
    DATA_DIR = os.path.expanduser("~/PatoSimData")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=os.path.join(DATA_DIR, "recordings"))
    parser.add_argument("--output_path", type=str, default=os.path.join(DATA_DIR, "replays"))
    parser.add_argument("--pipeline_mode", type=str, default="staged", choices=["staged", "legacy"])
    parser.add_argument("--camera_serial_enabled", type=parse_bool, default=True)
    parser.add_argument("--camera_names", type=str, default="")
    parser.add_argument("--rgb_enabled", type=parse_bool, default=True)
    parser.add_argument("--segmentation_enabled", type=parse_bool, default=True)
    parser.add_argument("--depth_enabled", type=parse_bool, default=True)
    parser.add_argument("--instance_id_segmentation_enabled", type=parse_bool, default=True)
    parser.add_argument("--normals_enabled", type=parse_bool, default=False)
    parser.add_argument("--render_rt_subframes", type=int, default=1)
    parser.add_argument("--render_interval", type=int, default=20)
    parser.add_argument("--pc_enabled", type=parse_bool, default=True)
    parser.add_argument("--pc_format", type=str, default="npy", choices=["npy", "ply", "pcd"])
    parser.add_argument("--annotations_enabled", type=parse_bool, default=True)
    parser.add_argument("--pc_interval", type=int, default=1)
    parser.add_argument("--pc_min_points", type=int, default=32)
    parser.add_argument("--pc_min_extent", type=float, default=0.05)
    parser.add_argument("--pc_require_spread", type=parse_bool, default=True)
    parser.add_argument("--pc_fallback_to_recording", type=parse_bool, default=True)
    parser.add_argument("--no-sonar", action="store_true")
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--verbose", type=parse_bool, default=False)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[replay] Ignoring unknown args: {' '.join(unknown)}", flush=True)
    return args


def normalize_args(args: argparse.Namespace) -> None:
    args.render_interval = max(1, int(getattr(args, "render_interval", 1)))
    args.pc_interval = max(1, int(getattr(args, "pc_interval", 1)))
    args.pc_min_points = max(1, int(getattr(args, "pc_min_points", 1)))
    args.pc_min_extent = max(0.0, float(getattr(args, "pc_min_extent", 0.0)))
    args.camera_names = str(getattr(args, "camera_names", "") or "").strip()
    args.no_sonar = bool(getattr(args, "no_sonar", False))


def log(message: str) -> None:
    print(message, flush=True)


def resolve_output_path(input_path: str, output_path: str) -> str:
    input_name = os.path.basename(os.path.normpath(input_path))
    normalized_output = os.path.normpath(output_path)
    if os.path.basename(normalized_output) == input_name:
        return output_path
    if (
        output_path.endswith("/replays")
        or output_path.endswith(os.path.sep + "replays")
        or (os.path.isdir(output_path) and not os.path.isdir(os.path.join(output_path, "state")))
    ):
        return os.path.join(output_path, input_name)
    return output_path

def install_signal_handlers() -> None:
    def _signal_handler(sig: int, _frame: Any) -> None:
        global STOP_REQUESTED
        log(f"\n[replay] Received signal {sig}; finishing current iteration before stopping.")
        STOP_REQUESTED = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _get_oceansim_sonar_type():
    try:
        from omni.ext.patosim.sensors import OceanSimImagingSonar

        return OceanSimImagingSonar
    except Exception:
        return None


def bootstrap_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    ext_pkg_root = repo_root / "exts" / "omni.ext.patosim"
    ext_omni_ext_root = ext_pkg_root / "omni" / "ext"

    ext_pkg_root_str = str(ext_pkg_root)
    if ext_pkg_root.exists() and ext_pkg_root_str not in sys.path:
        sys.path.insert(0, ext_pkg_root_str)

    try:
        import omni.ext as omni_ext  # type: ignore

        ext_omni_ext_root_str = str(ext_omni_ext_root)
        if ext_omni_ext_root.exists() and ext_omni_ext_root_str not in omni_ext.__path__:
            omni_ext.__path__.append(ext_omni_ext_root_str)
    except Exception:
        # In dry-run outside Isaac Kit this package may not exist yet.
        pass

    return repo_root


def init_simulation_app(headless: bool = True) -> Tuple[Any, Any]:
    from isaacsim import SimulationApp

    log("[replay] Initializing SimulationApp...")
    simulation_app = SimulationApp(launch_config={"headless": headless})
    log("[replay] SimulationApp initialized.")
    log("[replay] Importing omni.replicator.core...")
    import omni.replicator.core as rep
    log("[replay] omni.replicator.core imported.")

    return simulation_app, rep


def load_runtime_modules():
    os.environ["PATOSIM_IMPORT_MODE"] = "lite"
    from omni.ext.patosim.reader import Reader
    from omni.ext.patosim.writer import Writer

    from omni.ext.patosim.build import load_scenario
    from omni.ext.patosim.utils.global_utils import get_world

    return Reader, Writer, load_scenario, get_world


def has_any_data(state_dict: Dict[str, Any]) -> bool:
    return any(value is not None for value in state_dict.values())


def count_non_none(state_dict: Dict[str, Any]) -> int:
    return sum(1 for value in state_dict.values() if value is not None)


def count_written_pointcloud_files(output_path: str) -> int:
    patterns = ("*.npy", "*.ply", "*.pcd")
    total = 0
    for pattern in patterns:
        total += len(glob.glob(os.path.join(output_path, "state", "pointcloud", "*", pattern)))
    return total


def count_written_annotation_files(output_path: str) -> int:
    total = 0
    total += len(glob.glob(os.path.join(output_path, "state", "bboxes2d", "*.json")))
    total += len(glob.glob(os.path.join(output_path, "state", "bboxes3d", "*.json")))
    total += len(glob.glob(os.path.join(output_path, "state", "classes", "*.json")))
    total += len(glob.glob(os.path.join(output_path, "state", "semantic", "*.json")))
    return total


def _parse_requested_names(camera_names: str) -> list[str]:
    if not camera_names:
        return []
    return [chunk.strip() for chunk in str(camera_names).split(",") if chunk.strip()]


def _is_sonar_module(module: Any) -> bool:
    sonar_type = _get_oceansim_sonar_type()
    return bool(sonar_type is not None and module is not None and isinstance(module, sonar_type))


def _is_camera_module(module: Any) -> bool:
    if module is None:
        return False

    # Treat leaf optical cameras as replayable camera modules, including the
    # OceanSim underwater camera wrapper used by the ROV. Parent stereo wrappers
    # are intentionally excluded so the replay pass records left/right views
    # independently.
    try:
        if hasattr(module, "children") and len(module.children()) > 0:
            return False
    except Exception:
        pass

    if _is_sonar_module(module):
        return True

    class_name = module.__class__.__name__
    if class_name in {"Camera", "OceanSimUWCamera"} and hasattr(module, "disable_rendering"):
        return True

    camera_like_buffers = any(
        hasattr(module, attr_name)
        for attr_name in (
            "raw_rgb_image",
            "depth_image",
            "segmentation_image",
            "instance_id_segmentation_image",
            "normals_image",
        )
    )
    camera_like_controls = any(
        hasattr(module, attr_name)
        for attr_name in (
            "enable_rgb_rendering",
            "enable_depth_rendering",
            "enable_segmentation_rendering",
            "enable_instance_id_segmentation_rendering",
            "enable_normals_rendering",
            "disable_rendering",
        )
    )
    return bool(camera_like_buffers and camera_like_controls)


def discover_camera_modules(
    scenario: Any,
    requested_names: str = "",
    include_sonar: bool = True,
) -> "OrderedDict[str, Any]":
    modules = OrderedDict()
    for name, module in scenario.named_modules().items():
        if not name:
            continue
        if _is_camera_module(module):
            if not include_sonar and _is_sonar_module(module):
                continue
            modules[name] = module

    requested = _parse_requested_names(requested_names)
    if not requested:
        return modules

    filtered = OrderedDict()
    missing = []
    for wanted in requested:
        if wanted in modules:
            filtered[wanted] = modules[wanted]
            continue
        suffix_matches = [(name, module) for name, module in modules.items() if name.endswith(wanted)]
        if len(suffix_matches) == 1:
            filtered[suffix_matches[0][0]] = suffix_matches[0][1]
        else:
            missing.append(wanted)
    if missing:
        log(f"[replay] Requested camera names not found or ambiguous: {', '.join(missing)}")
    return filtered


def disable_all_camera_rendering(scenario: Any, include_sonar: bool = True) -> None:
    for _name, module in discover_camera_modules(scenario, include_sonar=include_sonar).items():
        try:
            module.disable_rendering()
        except Exception:
            pass


def enable_camera_modalities(camera_module: Any, args: argparse.Namespace) -> None:
    if _is_sonar_module(camera_module):
        if not getattr(args, "no_sonar", False) and args.rgb_enabled:
            camera_module.enable_rgb_rendering()
        return
    if args.rgb_enabled:
        camera_module.enable_rgb_rendering()
    if args.segmentation_enabled:
        camera_module.enable_segmentation_rendering()
    if args.instance_id_segmentation_enabled:
        camera_module.enable_instance_id_segmentation_rendering()
    if args.depth_enabled:
        camera_module.enable_depth_rendering()
    if args.normals_enabled:
        camera_module.enable_normals_rendering()


def _get_sonar_from_scenario(scenario: Any) -> Any:
    robot = getattr(scenario, "robot", None)
    sonar = getattr(robot, "sonar", None)
    if _is_sonar_module(sonar):
        return sonar
    try:
        for _name, module in scenario.named_modules().items():
            if _is_sonar_module(module):
                return module
    except Exception:
        pass
    return None


def _get_module_name_for_instance(scenario: Any, target: Any) -> str:
    if target is None:
        return ""
    try:
        for name, module in scenario.named_modules().items():
            if module is target:
                return str(name)
    except Exception:
        pass
    return ""


def _filter_state_dict_excluding_module_prefix(state_dict: Dict[str, Any], module_prefix: str) -> Dict[str, Any]:
    prefix = str(module_prefix or "").strip()
    if not prefix:
        return OrderedDict(state_dict.items())
    needle = prefix + "."
    return OrderedDict((name, value) for name, value in state_dict.items() if not name.startswith(needle))


def filter_state_dict_for_module_prefix(state_dict: Dict[str, Any], module_prefix: str) -> Dict[str, Any]:
    prefix = str(module_prefix or "").strip()
    if not prefix:
        return OrderedDict(state_dict.items())
    needle = prefix + "."
    return OrderedDict((name, value) for name, value in state_dict.items() if name.startswith(needle))


def analyze_pointcloud_array(
    value: Any,
    min_points: int = 32,
    min_extent: float = 0.05,
    require_spread: bool = True,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "valid": False,
        "num_points": 0,
        "extent": 0.0,
        "reason": "empty",
    }
    if value is None:
        result["reason"] = "none"
        return result

    try:
        array = np.asarray(value, dtype=np.float32)
    except Exception:
        result["reason"] = "coerce_failed"
        return result

    if array.ndim != 2 or array.shape[1] < 3:
        result["reason"] = "bad_shape"
        return result

    xyz = array[:, :3]
    finite_mask = np.all(np.isfinite(xyz), axis=1)
    if not np.any(finite_mask):
        result["reason"] = "no_finite_points"
        return result

    xyz = xyz[finite_mask]
    result["num_points"] = int(xyz.shape[0])
    if xyz.shape[0] < int(min_points):
        result["reason"] = "too_few_points"
        return result

    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    extent_vec = maxs - mins
    extent = float(np.linalg.norm(extent_vec))
    result["extent"] = extent
    if require_spread and extent < float(min_extent):
        result["reason"] = "too_compact"
        return result

    result["valid"] = True
    result["reason"] = "ok"
    return result


def select_relevant_pointclouds(
    state_pc: Dict[str, Any],
    min_points: int,
    min_extent: float,
    require_spread: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    valid = OrderedDict()
    report = OrderedDict()
    for name, value in (state_pc or {}).items():
        info = analyze_pointcloud_array(
            value=value,
            min_points=min_points,
            min_extent=min_extent,
            require_spread=require_spread,
        )
        report[name] = info
        if info.get("valid"):
            valid[name] = value
    return valid, report


def select_available_pointclouds(state_pc: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    available = OrderedDict()
    report = OrderedDict()
    for name, value in (state_pc or {}).items():
        info = analyze_pointcloud_array(
            value=value,
            min_points=1,
            min_extent=0.0,
            require_spread=False,
        )
        report[name] = info
        if info.get("num_points", 0) > 0:
            available[name] = value
    return available, report


def pointcloud_report_total_points(report: Dict[str, Any]) -> int:
    total = 0
    for info in (report or {}).values():
        try:
            total += int(info.get("num_points", 0))
        except Exception:
            pass
    return total


def lidar_status_snapshot(scenario: Any) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = OrderedDict()
    try:
        modules = scenario.named_modules()
    except Exception:
        return snapshot
    for name, module in modules.items():
        if not hasattr(module, "pointcloud"):
            continue
        status_value = None
        try:
            status_value = module.status.get_value() if hasattr(module, "status") else None
        except Exception:
            status_value = None
        snapshot[name] = {
            "status": status_value,
            "prim_path": getattr(module, "_prim_path", None),
            "sensor_prim_path": getattr(module, "_sensor_prim_path", None),
        }
    return snapshot


def capture_pointcloud_for_step(
    scenario: Any,
    simulation_app: Any,
    rep: Any,
    render_rt_subframes: int,
    min_points: int,
    min_extent: float,
    require_spread: bool,
    extra_attempts: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    pointcloud_delta_time = 1.0 / 60.0
    best_available_pc: Dict[str, Any] = OrderedDict()
    best_available_report: Dict[str, Any] = OrderedDict()
    best_available_points = 0

    for attempt in range(max(1, int(extra_attempts) + 1)):
        progressed = drive_rtx_lidar_render_products(
            scenario=scenario,
            simulation_app=simulation_app,
            rep=rep,
            frames=2 if attempt == 0 else 3,
        )
        if not progressed:
            simulation_app.update()
            rep.orchestrator.step(
                rt_subframes=render_rt_subframes,
                delta_time=pointcloud_delta_time,
                pause_timeline=False,
            )
            scenario.update_state()

        state_pc = scenario.state_dict_pointcloud()
        valid_pc, report = select_relevant_pointclouds(
            state_pc,
            min_points=min_points,
            min_extent=min_extent,
            require_spread=require_spread,
        )
        if valid_pc:
            return valid_pc, report, OrderedDict(), OrderedDict()

        available_pc, available_report = select_available_pointclouds(state_pc)
        available_points = pointcloud_report_total_points(available_report)
        if available_points > best_available_points:
            best_available_points = available_points
            best_available_pc = available_pc
            best_available_report = available_report

    return OrderedDict(), report if 'report' in locals() else OrderedDict(), best_available_pc, best_available_report


def drive_rtx_lidar_render_products(
    scenario: Any,
    simulation_app: Any,
    rep: Any,
    frames: int = 2,
) -> bool:
    try:
        import omni.timeline
    except Exception:
        return False

    lidar_modules = []
    try:
        for module in scenario.named_modules().values():
            rtx = getattr(module, "_rtx", None)
            if rtx is None or not hasattr(rtx, "get_render_product_path"):
                continue
            lidar_modules.append(module)
    except Exception:
        return False

    if not lidar_modules:
        return False

    try:
        omni.timeline.get_timeline_interface().play()
    except Exception:
        pass

    progressed = False
    for module in lidar_modules:
        try:
            rtx = getattr(module, "_rtx", None)
            if rtx is not None and hasattr(rtx, "resume"):
                rtx.resume()
        except Exception:
            pass

    for _ in range(max(1, int(frames))):
        try:
            simulation_app.update()
        except Exception:
            return progressed
        try:
            rep.orchestrator.step(
                rt_subframes=1,
                delta_time=1.0 / 60.0,
                pause_timeline=False,
            )
        except Exception:
            pass
        try:
            scenario.update_state()
            state_pc = scenario.state_dict_pointcloud()
            if any(value is not None for value in (state_pc or {}).values()):
                progressed = True
        except Exception:
            pass
    return progressed


def pointcloud_fields(value: Any) -> Optional[list]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 2:
        return None
    cols = array.shape[1]
    if cols == 3:
        return ["x", "y", "z"]
    if cols == 4:
        return ["x", "y", "z", "intensity"]
    if cols == 6:
        return ["x", "y", "z", "r", "g", "b"]
    if cols == 7:
        return ["x", "y", "z", "r", "g", "b", "intensity"]
    return ["x", "y", "z"]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def _semantic_type_from_attr_prefix(prefix: str) -> str:
    if "_" in prefix:
        return prefix.split("_", 1)[0]
    return prefix


def extract_semantic_labels_from_prim(prim: Any) -> Dict[str, str]:
    semantic_labels: Dict[str, str] = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not name.startswith("semantic:") or not name.endswith(":params:semanticData"):
            continue
        raw_value = attr.Get()
        if raw_value is None:
            continue

        prefix = name[len("semantic:") : -len(":params:semanticData")]
        type_attr = prim.GetAttribute(f"semantic:{prefix}:params:semanticType")
        semantic_type = None
        if type_attr is not None:
            semantic_type = type_attr.Get()
        if semantic_type is None:
            semantic_type = _semantic_type_from_attr_prefix(prefix)

        semantic_labels[str(semantic_type)] = str(raw_value)
    return semantic_labels


def has_dataset_object_root_ancestor(prim: Any) -> bool:
    parent = prim.GetParent()
    while parent is not None and parent.IsValid() and not parent.IsPseudoRoot():
        labels = extract_semantic_labels_from_prim(parent)
        if str(labels.get("dataset_object_root", "")).strip().lower() in {"true", "1", "yes"}:
            return True
        parent = parent.GetParent()
    return False


def extract_semantic_state_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    semantic_state: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        if key.endswith("segmentation_info") or key.endswith("instance_id_segmentation_info"):
            semantic_state[key] = to_jsonable(value)
    return semantic_state


def build_pointcloud_metadata(scenario: Any, state_pc: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    modules = scenario.named_modules()
    for full_name, value in state_pc.items():
        module_name = full_name.rsplit(".", 1)[0] if "." in full_name else full_name
        module = modules.get(module_name)
        if module is None:
            continue

        pos = None
        ori = None
        if hasattr(module, "get_world_pose"):
            pos, ori = module.get_world_pose()
        elif hasattr(module, "_xform_prim"):
            pos, ori = module._xform_prim.get_world_pose()

        fields = pointcloud_fields(value)
        prim_path = getattr(module, "_prim_path", None)
        if pos is None and ori is None and fields is None and prim_path is None:
            continue

        metadata[module_name] = {
            "position": None if pos is None else [float(x) for x in list(pos)],
            "orientation": None if ori is None else [float(x) for x in list(ori)],
            "prim_path": prim_path,
            "fields": fields,
        }
    return metadata


def _camera_projection_contexts(stage: Any, xform_cache: Any, image_w: int = 640, image_h: int = 480):
    from pxr import UsdGeom

    cameras = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Camera" or not prim.IsActive():
            continue
        try:
            cam = UsdGeom.Camera(prim)
            focal = cam.GetFocalLengthAttr().Get()
            h_ap = cam.GetHorizontalApertureAttr().Get()
            v_ap = cam.GetVerticalApertureAttr().Get()
            fx = focal * image_w / h_ap if (h_ap and image_w) else 1.0
            fy = focal * image_h / v_ap if (v_ap and image_h) else fx
            cam_world = xform_cache.GetLocalToWorldTransform(prim)
            cam_mat = cam_world.GetInverse()
        except Exception:
            continue
        cameras.append(
            {
                "prim": prim,
                "name": prim.GetName(),
                "prim_path": prim.GetPath().pathString,
                "image_w": image_w,
                "image_h": image_h,
                "fx": fx,
                "fy": fy,
                "cx": image_w / 2.0,
                "cy": image_h / 2.0,
                "cam_mat": cam_mat,
            }
        )
    return cameras


def gather_annotations(stage: Any) -> Dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom

    annotations = {"bboxes2d": [], "bboxes3d": [], "classes": []}
    if stage is None:
        return annotations

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    camera_contexts = _camera_projection_contexts(stage, xform_cache)

    for prim in stage.Traverse():
        if prim.IsPseudoRoot() or not prim.IsActive() or prim.GetTypeName() == "Camera":
            continue
        if has_dataset_object_root_ancestor(prim):
            continue

        bound = bbox_cache.ComputeWorldBound(prim)
        rng = bound.GetRange()
        if rng is None:
            continue
        if hasattr(rng, "IsEmpty") and rng.IsEmpty():
            continue

        min_pt = rng.GetMin()
        max_pt = rng.GetMax()
        corners = []
        for x in [min_pt[0], max_pt[0]]:
            for y in [min_pt[1], max_pt[1]]:
                for z in [min_pt[2], max_pt[2]]:
                    corners.append([float(x), float(y), float(z)])

        semantic_labels = extract_semantic_labels_from_prim(prim)
        class_name = semantic_labels.get("class", prim.GetName())
        annotations["bboxes3d"].append(
            {
                "prim_path": prim.GetPath().pathString,
                "class": class_name,
                "semantic_labels": semantic_labels,
                "corners": corners,
            }
        )

        for cam_ctx in camera_contexts:
            xs = []
            ys = []
            for corner in corners:
                wc = Gf.Vec4d(corner[0], corner[1], corner[2], 1.0)
                cc = cam_ctx["cam_mat"] * wc
                if cc[2] <= 0.0:
                    continue
                xs.append(float((cam_ctx["fx"] * (cc[0] / cc[2])) + cam_ctx["cx"]))
                ys.append(float((cam_ctx["fy"] * (cc[1] / cc[2])) + cam_ctx["cy"]))

            if xs and ys:
                annotations["bboxes2d"].append(
                    {
                        "prim_path": prim.GetPath().pathString,
                        "camera_name": cam_ctx["name"],
                        "camera_prim_path": cam_ctx["prim_path"],
                        "image_size": [cam_ctx["image_w"], cam_ctx["image_h"]],
                        "class": class_name,
                        "semantic_labels": semantic_labels,
                        "bbox": [
                            max(0.0, min(xs)),
                            max(0.0, min(ys)),
                            min(cam_ctx["image_w"] - 1.0, max(xs)),
                            min(cam_ctx["image_h"] - 1.0, max(ys)),
                        ],
                    }
                )

    annotations["classes"] = sorted(
        {entry.get("class", "") for entry in annotations["bboxes3d"] if entry.get("class", "")}
    )
    return annotations


def read_json_file(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    return None


def build_annotation_payload(
    step: int,
    base_annotations: Optional[Dict[str, Any]] = None,
    semantic_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "step": int(step),
        "bboxes2d": [],
        "bboxes3d": [],
        "classes": [],
        "semantic": semantic_state or {},
    }
    if isinstance(base_annotations, dict):
        payload.update(base_annotations)
        if "semantic" not in payload or payload.get("semantic") is None:
            payload["semantic"] = semantic_state or {}
    payload["step"] = int(step)
    if not isinstance(payload.get("bboxes2d"), list):
        payload["bboxes2d"] = []
    if not isinstance(payload.get("bboxes3d"), list):
        payload["bboxes3d"] = []
    if not isinstance(payload.get("classes"), list):
        payload["classes"] = []
    if not isinstance(payload.get("semantic"), dict):
        payload["semantic"] = {}
    return payload


def read_split_annotations(base_path: str, step: int) -> Optional[Dict[str, Any]]:
    payload = {
        "step": int(step),
        "bboxes2d": read_json_file(os.path.join(base_path, "state", "bboxes2d", f"{step:08d}.json")) or [],
        "bboxes3d": read_json_file(os.path.join(base_path, "state", "bboxes3d", f"{step:08d}.json")) or [],
        "classes": read_json_file(os.path.join(base_path, "state", "classes", f"{step:08d}.json")) or [],
        "semantic": read_json_file(os.path.join(base_path, "state", "semantic", f"{step:08d}.json")) or {},
    }
    if any((payload["bboxes2d"], payload["bboxes3d"], payload["classes"], payload["semantic"])):
        return payload
    legacy = read_json_file(os.path.join(base_path, "state", "annotations", f"{step:08d}.json"))
    if isinstance(legacy, dict):
        return legacy
    return None


def write_summary(
    output_path: str,
    input_path: str,
    pc_written_count: int,
    annotations_written_count: int,
    pointcloud_validation_summary: Optional[Dict[str, Any]],
    verbose: bool,
) -> None:
    actual_pc_written = count_written_pointcloud_files(output_path)
    actual_annotation_written = count_written_annotation_files(output_path)
    log("\n=== Replay summary ===")
    log(f"Pointcloud entries written: {actual_pc_written}")
    log(f"Annotation files written: {actual_annotation_written}")

    os.makedirs(output_path, exist_ok=True)
    summary_path = os.path.join(output_path, "replay_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Replay summary\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"Pointcloud entries written: {actual_pc_written}\n")
        f.write(f"Annotation files written: {actual_annotation_written}\n")
        f.write(f"Pointcloud entries written (run counter): {pc_written_count}\n")
        f.write(f"Annotation files written (run counter): {annotations_written_count}\n")
        if isinstance(pointcloud_validation_summary, dict):
            f.write("Pointcloud validation summary:\n")
            for key in (
                "valid_steps",
                "invalid_steps",
                "fallback_steps",
                "missing_steps",
                "sparse_steps",
            ):
                f.write(f"  {key}: {int(pointcloud_validation_summary.get(key, 0))}\n")
            invalid_reasons = pointcloud_validation_summary.get("invalid_reasons", {})
            if isinstance(invalid_reasons, dict):
                for reason, count in sorted(invalid_reasons.items()):
                    f.write(f"  invalid_reason[{reason}]: {int(count)}\n")

        common_files = glob.glob(os.path.join(output_path, "state", "common", "*.npy"))
        rgb_files = glob.glob(os.path.join(output_path, "state", "rgb", "*", "*.jpg"))
        seg_files = glob.glob(os.path.join(output_path, "state", "segmentation", "*", "*.png"))
        inst_seg_files = glob.glob(
            os.path.join(output_path, "state", "instance_id_segmentation", "*", "*.png")
        )
        depth_files = glob.glob(os.path.join(output_path, "state", "depth", "*", "*.png"))
        normals_files = glob.glob(os.path.join(output_path, "state", "normals", "*", "*.npy"))
        bbox2d_files = glob.glob(os.path.join(output_path, "state", "bboxes2d", "*.json"))
        bbox3d_files = glob.glob(os.path.join(output_path, "state", "bboxes3d", "*.json"))
        class_files = glob.glob(os.path.join(output_path, "state", "classes", "*.json"))
        semantic_files = glob.glob(os.path.join(output_path, "state", "semantic", "*.json"))
        f.write(f"state/common files: {len(common_files)}\n")
        f.write(f"state/rgb images: {len(rgb_files)}\n")
        f.write(f"state/segmentation images: {len(seg_files)}\n")
        f.write(f"state/instance_id_segmentation images: {len(inst_seg_files)}\n")
        f.write(f"state/depth images: {len(depth_files)}\n")
        f.write(f"state/normals arrays: {len(normals_files)}\n")
        f.write(f"state/bboxes2d files: {len(bbox2d_files)}\n")
        f.write(f"state/bboxes3d files: {len(bbox3d_files)}\n")
        f.write(f"state/classes files: {len(class_files)}\n")
        f.write(f"state/semantic files: {len(semantic_files)}\n")

        total_bbox2d = 0
        total_bbox3d = 0
        semantic_payload_files = 0
        class_labels = set()
        annotation_steps = sorted(
            {
                int(Path(p).stem)
                for p in (bbox2d_files + bbox3d_files + class_files + semantic_files)
                if Path(p).stem.isdigit()
            }
        )
        for step in annotation_steps:
            ann_data = read_split_annotations(output_path, step)
            if not isinstance(ann_data, dict):
                continue
            b2d = ann_data.get("bboxes2d")
            b3d = ann_data.get("bboxes3d")
            if isinstance(b2d, list):
                total_bbox2d += len(b2d)
            if isinstance(b3d, list):
                total_bbox3d += len(b3d)
                for entry in b3d:
                    if isinstance(entry, dict) and "class" in entry:
                        class_labels.add(str(entry["class"]))
            if isinstance(ann_data.get("semantic"), dict) and ann_data["semantic"]:
                semantic_payload_files += 1
        f.write(f"Total bbox2d entries: {total_bbox2d}\n")
        f.write(f"Total bbox3d entries: {total_bbox3d}\n")
        f.write(f"Annotation files with semantic payload: {semantic_payload_files}\n")
        f.write(f"Detected class labels: {len(class_labels)}\n")

        pc_root = os.path.join(output_path, "state", "pointcloud")
        if os.path.exists(pc_root):
            for sensor in sorted(os.listdir(pc_root)):
                sensor_folder = os.path.join(pc_root, sensor)
                if os.path.isdir(sensor_folder):
                    file_count = len(
                        [
                            p
                            for p in os.listdir(sensor_folder)
                            if os.path.isfile(os.path.join(sensor_folder, p))
                        ]
                    )
                    f.write(f"  {sensor}: {file_count} files\n")
    if verbose:
        log(f"Wrote replay summary to: {summary_path}")


def _advance_replay_step(
    step: int,
    reader: Any,
    scenario: Any,
    simulation_app: Any,
    rep: Any,
    render_rt_subframes: int,
    run_render_step: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    replay_state = reader.read_state_dict_flat(index=step)
    try:
        scenario.load_state_dict(replay_state)
    except Exception as exc:
        raise RuntimeError(f"step {step}: scenario.load_state_dict failed: {exc}") from exc
    try:
        scenario.write_replay_data()
    except Exception as exc:
        raise RuntimeError(f"step {step}: scenario.write_replay_data failed: {exc}") from exc
    try:
        simulation_app.update()
    except Exception as exc:
        raise RuntimeError(f"step {step}: simulation_app.update failed: {exc}") from exc
    if run_render_step:
        try:
            rep.orchestrator.step(
                rt_subframes=render_rt_subframes,
                delta_time=0.0,
                pause_timeline=False,
            )
        except Exception as exc:
            raise RuntimeError(f"step {step}: rep.orchestrator.step failed: {exc}") from exc
    try:
        scenario.update_state()
    except Exception as exc:
        raise RuntimeError(f"step {step}: scenario.update_state failed: {exc}") from exc
    return replay_state, scenario.state_dict_common()


def run_legacy_replay(
    args: argparse.Namespace,
    reader: Any,
    writer: Any,
    simulation_app: Any,
    rep: Any,
    load_scenario: Any,
    get_world: Any,
) -> Tuple[int, int, Dict[str, Any]]:
    scenario = load_scenario(args.input_path)
    if scenario is None:
        raise RuntimeError(f"Failed to load scenario from recording: {args.input_path}")

    world = get_world()
    if world is None:
        raise RuntimeError("World instance is not available after loading scenario.")
    world.reset()

    log(str(scenario))
    if args.rgb_enabled:
        scenario.enable_rgb_rendering()
    sonar_ref = _get_sonar_from_scenario(scenario)
    if args.no_sonar and sonar_ref is not None:
        try:
            sonar_ref.disable_rendering()
        except Exception:
            pass
    if args.segmentation_enabled:
        scenario.enable_segmentation_rendering()
    if args.depth_enabled:
        scenario.enable_depth_rendering()
    if args.instance_id_segmentation_enabled:
        scenario.enable_instance_id_segmentation_rendering()
    if args.normals_enabled:
        scenario.enable_normals_rendering()

    simulation_app.update()
    rep.orchestrator.step(
        rt_subframes=args.render_rt_subframes,
        delta_time=0.0,
        pause_timeline=False,
    )

    pc_written_count = 0
    annotations_written_count = 0
    validation_summary = {
        "valid_steps": 0,
        "invalid_steps": 0,
        "fallback_steps": 0,
        "missing_steps": 0,
        "sparse_steps": 0,
        "invalid_reasons": OrderedDict(),
    }
    num_steps = len(reader)
    pc_interval = max(1, int(args.pc_interval))
    sonar_ref = None if args.no_sonar else sonar_ref
    sonar_module_name = _get_module_name_for_instance(scenario, sonar_ref)

    for step in tqdm.tqdm(range(0, num_steps, args.render_interval)):
        if STOP_REQUESTED:
            log("[replay] Stop requested; leaving replay loop.")
            break

        _replay_state, state_dict_common = _advance_replay_step(
            step=step,
            reader=reader,
            scenario=scenario,
            simulation_app=simulation_app,
            rep=rep,
            render_rt_subframes=args.render_rt_subframes,
            run_render_step=True,
        )
        writer.write_state_dict_common(state_dict_common, step)

        if args.rgb_enabled:
            rgb_state = scenario.state_dict_rgb()
            if args.no_sonar:
                rgb_state = _filter_state_dict_excluding_module_prefix(rgb_state, sonar_module_name)
            writer.write_state_dict_rgb(rgb_state, step, sonar_ref=sonar_ref)
        if args.segmentation_enabled:
            writer.write_state_dict_segmentation(scenario.state_dict_segmentation(), step)
        if args.instance_id_segmentation_enabled:
            writer.write_state_dict_instance_id_segmentation(
                scenario.state_dict_instance_id_segmentation(), step
            )
        if args.depth_enabled:
            writer.write_state_dict_depth(scenario.state_dict_depth(), step)
        if args.normals_enabled:
            writer.write_state_dict_normals(scenario.state_dict_normals(), step)

        if args.pc_enabled and (step % pc_interval) == 0:
            state_pc = scenario.state_dict_pointcloud()
            used_pc = None
            valid_pc, report = select_relevant_pointclouds(
                state_pc,
                min_points=args.pc_min_points,
                min_extent=args.pc_min_extent,
                require_spread=args.pc_require_spread,
            )
            if valid_pc:
                writer.write_state_dict_pointcloud(valid_pc, step, save_format=args.pc_format)
                pc_written_count += count_non_none(valid_pc)
                used_pc = valid_pc
                validation_summary["valid_steps"] += 1
            elif args.pc_fallback_to_recording and getattr(reader, "pointcloud_names", []):
                fallback_pc = reader.read_state_dict_pointcloud(index=step)
                valid_fallback, fallback_report = select_relevant_pointclouds(
                    fallback_pc,
                    min_points=args.pc_min_points,
                    min_extent=args.pc_min_extent,
                    require_spread=args.pc_require_spread,
                )
                if valid_fallback:
                    log(f"[replay] No scenario pointcloud for step {step}; copying from recording.")
                    writer.write_state_dict_pointcloud(valid_fallback, step, save_format=args.pc_format)
                    pc_written_count += count_non_none(valid_fallback)
                    used_pc = valid_fallback
                    validation_summary["fallback_steps"] += 1
                    report = fallback_report
                else:
                    validation_summary["missing_steps"] += 1
            else:
                validation_summary["missing_steps"] += 1

            if used_pc is None:
                validation_summary["invalid_steps"] += 1
                for info in report.values():
                    reason = str(info.get("reason", "invalid"))
                    validation_summary["invalid_reasons"][reason] = (
                        int(validation_summary["invalid_reasons"].get(reason, 0)) + 1
                    )

            if used_pc is not None:
                metadata = build_pointcloud_metadata(scenario, used_pc)
                if metadata:
                    writer.write_pointcloud_metadata(metadata, step)

        if args.annotations_enabled:
            from omni.ext.patosim.utils.global_utils import get_stage

            annotations = gather_annotations(get_stage())
            semantic_state = extract_semantic_state_from_state_dict(state_dict_common)
            payload = build_annotation_payload(
                step=step,
                base_annotations=annotations,
                semantic_state=semantic_state,
            )
            writer.write_annotations(payload, step)
            annotations_written_count += 1

    return pc_written_count, annotations_written_count, validation_summary


def run_staged_replay(
    args: argparse.Namespace,
    reader: Any,
    writer: Any,
    simulation_app: Any,
    rep: Any,
    load_scenario: Any,
    get_world: Any,
) -> Tuple[int, int, Dict[str, Any]]:
    scenario = load_scenario(args.input_path)
    if scenario is None:
        raise RuntimeError(f"Failed to load scenario from recording: {args.input_path}")

    world = get_world()
    if world is None:
        raise RuntimeError("World instance is not available after loading scenario.")
    world.reset()

    log(str(scenario))
    camera_modules = discover_camera_modules(
        scenario,
        requested_names=args.camera_names,
        include_sonar=not args.no_sonar,
    )
    if args.camera_names and not camera_modules:
        raise RuntimeError("No requested cameras were found for staged replay.")

    disable_all_camera_rendering(scenario, include_sonar=not args.no_sonar)
    try:
        scenario.set_pointcloud_enabled(False)
    except Exception:
        pass

    simulation_app.update()
    rep.orchestrator.step(
        rt_subframes=args.render_rt_subframes,
        delta_time=0.0,
        pause_timeline=False,
    )

    num_steps = len(reader)
    replay_steps = list(range(0, num_steps, args.render_interval))
    written_common_steps: set[int] = set()
    pc_written_count = 0
    annotations_written_count = 0
    validation_summary = {
        "valid_steps": 0,
        "invalid_steps": 0,
        "fallback_steps": 0,
        "missing_steps": 0,
        "sparse_steps": 0,
        "invalid_reasons": OrderedDict(),
    }

    camera_modalities_enabled = any(
        [
            args.rgb_enabled,
            args.segmentation_enabled,
            args.instance_id_segmentation_enabled,
            args.depth_enabled,
            args.normals_enabled,
        ]
    )
    sonar_ref = None if args.no_sonar else _get_sonar_from_scenario(scenario)
    sonar_module_name = _get_module_name_for_instance(scenario, sonar_ref)

    if camera_modalities_enabled:
        if args.camera_serial_enabled:
            if not camera_modules:
                log("[replay] No camera modules discovered; skipping staged camera pass.")
            for module_name, module in camera_modules.items():
                log(f"[replay] Camera pass: {module_name}")
                disable_all_camera_rendering(scenario, include_sonar=not args.no_sonar)
                enable_camera_modalities(module, args)
                for step in tqdm.tqdm(replay_steps, desc=f"camera:{module_name}", leave=False):
                    if STOP_REQUESTED:
                        log("[replay] Stop requested; leaving staged camera pass.")
                        break
                    _replay_state, state_dict_common = _advance_replay_step(
                        step=step,
                        reader=reader,
                        scenario=scenario,
                        simulation_app=simulation_app,
                        rep=rep,
                        render_rt_subframes=args.render_rt_subframes,
                        run_render_step=True,
                    )
                    if step not in written_common_steps:
                        writer.write_state_dict_common(state_dict_common, step)
                        written_common_steps.add(step)
                    if args.rgb_enabled:
                        module_sonar_ref = module if _is_sonar_module(module) else None
                        writer.write_state_dict_rgb(
                            filter_state_dict_for_module_prefix(scenario.state_dict_rgb(), module_name),
                            step,
                            sonar_ref=module_sonar_ref,
                        )
                    if args.segmentation_enabled:
                        writer.write_state_dict_segmentation(
                            filter_state_dict_for_module_prefix(scenario.state_dict_segmentation(), module_name),
                            step,
                        )
                    if args.instance_id_segmentation_enabled:
                        writer.write_state_dict_instance_id_segmentation(
                            filter_state_dict_for_module_prefix(
                                scenario.state_dict_instance_id_segmentation(), module_name
                            ),
                            step,
                        )
                    if args.depth_enabled:
                        writer.write_state_dict_depth(
                            filter_state_dict_for_module_prefix(scenario.state_dict_depth(), module_name),
                            step,
                        )
                    if args.normals_enabled:
                        writer.write_state_dict_normals(
                            filter_state_dict_for_module_prefix(scenario.state_dict_normals(), module_name),
                            step,
                        )
                if STOP_REQUESTED:
                    break
        else:
            log("[replay] Camera pass: all cameras together")
            for module in camera_modules.values():
                enable_camera_modalities(module, args)
            for step in tqdm.tqdm(replay_steps, desc="camera:all", leave=False):
                if STOP_REQUESTED:
                    log("[replay] Stop requested; leaving staged camera pass.")
                    break
                _replay_state, state_dict_common = _advance_replay_step(
                    step=step,
                    reader=reader,
                    scenario=scenario,
                    simulation_app=simulation_app,
                    rep=rep,
                    render_rt_subframes=args.render_rt_subframes,
                    run_render_step=True,
                )
                if step not in written_common_steps:
                    writer.write_state_dict_common(state_dict_common, step)
                    written_common_steps.add(step)
                if args.rgb_enabled:
                    rgb_state = scenario.state_dict_rgb()
                    if args.no_sonar:
                        rgb_state = _filter_state_dict_excluding_module_prefix(rgb_state, sonar_module_name)
                    writer.write_state_dict_rgb(rgb_state, step, sonar_ref=sonar_ref)
                if args.segmentation_enabled:
                    writer.write_state_dict_segmentation(scenario.state_dict_segmentation(), step)
                if args.instance_id_segmentation_enabled:
                    writer.write_state_dict_instance_id_segmentation(
                        scenario.state_dict_instance_id_segmentation(), step
                    )
                if args.depth_enabled:
                    writer.write_state_dict_depth(scenario.state_dict_depth(), step)
                if args.normals_enabled:
                    writer.write_state_dict_normals(scenario.state_dict_normals(), step)

    if args.pc_enabled and not STOP_REQUESTED:
        log("[replay] Pointcloud pass")
        disable_all_camera_rendering(scenario, include_sonar=not args.no_sonar)
        try:
            scenario.set_pointcloud_enabled(True)
        except Exception:
            pass
        try:
            progressed = drive_rtx_lidar_render_products(
                scenario=scenario,
                simulation_app=simulation_app,
                rep=rep,
                frames=3,
            )
            if not progressed:
                simulation_app.update()
                rep.orchestrator.step(
                    rt_subframes=max(1, int(args.render_rt_subframes)),
                    delta_time=1.0 / 60.0,
                    pause_timeline=False,
                )
                scenario.update_state()
        except Exception:
            pass
        pc_steps = [step for step in replay_steps if (step % args.pc_interval) == 0]
        for step in tqdm.tqdm(pc_steps, desc="pointcloud", leave=False):
            if STOP_REQUESTED:
                log("[replay] Stop requested; leaving pointcloud pass.")
                break
            _replay_state, state_dict_common = _advance_replay_step(
                step=step,
                reader=reader,
                scenario=scenario,
                simulation_app=simulation_app,
                rep=rep,
                render_rt_subframes=args.render_rt_subframes,
                run_render_step=True,
            )
            if step not in written_common_steps:
                writer.write_state_dict_common(state_dict_common, step)
                written_common_steps.add(step)

            valid_pc, report, sparse_pc, sparse_report = capture_pointcloud_for_step(
                scenario=scenario,
                simulation_app=simulation_app,
                rep=rep,
                render_rt_subframes=args.render_rt_subframes,
                min_points=args.pc_min_points,
                min_extent=args.pc_min_extent,
                require_spread=args.pc_require_spread,
            )
            used_pc = None
            if valid_pc:
                writer.write_state_dict_pointcloud(valid_pc, step, save_format=args.pc_format)
                pc_written_count += count_non_none(valid_pc)
                used_pc = valid_pc
                validation_summary["valid_steps"] += 1
            elif sparse_pc:
                log(
                    f"[replay] Sparse pointcloud kept for step {step}; "
                    f"LiDAR likely still warming up. Status={to_jsonable(lidar_status_snapshot(scenario))}"
                )
                writer.write_state_dict_pointcloud(sparse_pc, step, save_format=args.pc_format)
                pc_written_count += count_non_none(sparse_pc)
                used_pc = sparse_pc
                report = sparse_report
                validation_summary["sparse_steps"] += 1
            elif args.pc_fallback_to_recording and getattr(reader, "pointcloud_names", []):
                fallback_pc = reader.read_state_dict_pointcloud(index=step)
                valid_fallback, fallback_report = select_relevant_pointclouds(
                    fallback_pc,
                    min_points=args.pc_min_points,
                    min_extent=args.pc_min_extent,
                    require_spread=args.pc_require_spread,
                )
                if valid_fallback:
                    log(f"[replay] No relevant scenario pointcloud for step {step}; copying from recording.")
                    writer.write_state_dict_pointcloud(valid_fallback, step, save_format=args.pc_format)
                    pc_written_count += count_non_none(valid_fallback)
                    used_pc = valid_fallback
                    validation_summary["fallback_steps"] += 1
                    report = fallback_report
                else:
                    log(
                        f"[replay] No pointcloud written for step {step}; "
                        f"scenario report={to_jsonable(report)} "
                        f"fallback report={to_jsonable(fallback_report)} "
                        f"lidar={to_jsonable(lidar_status_snapshot(scenario))}"
                    )
                    validation_summary["missing_steps"] += 1
            else:
                log(
                    f"[replay] No pointcloud written for step {step}; "
                    f"scenario report={to_jsonable(report)} "
                    f"lidar={to_jsonable(lidar_status_snapshot(scenario))}"
                )
                validation_summary["missing_steps"] += 1

            if used_pc is None:
                validation_summary["invalid_steps"] += 1
                for info in report.values():
                    reason = str(info.get("reason", "invalid"))
                    validation_summary["invalid_reasons"][reason] = (
                        int(validation_summary["invalid_reasons"].get(reason, 0)) + 1
                    )
            else:
                metadata = build_pointcloud_metadata(scenario, used_pc)
                if metadata:
                    writer.write_pointcloud_metadata(metadata, step)
        try:
            scenario.set_pointcloud_enabled(False)
        except Exception:
            pass

    if args.annotations_enabled and not STOP_REQUESTED:
        log("[replay] Annotation pass")
        disable_all_camera_rendering(scenario, include_sonar=not args.no_sonar)
        try:
            scenario.set_pointcloud_enabled(False)
        except Exception:
            pass
        from omni.ext.patosim.utils.global_utils import get_stage

        for step in tqdm.tqdm(replay_steps, desc="annotations", leave=False):
            if STOP_REQUESTED:
                log("[replay] Stop requested; leaving annotation pass.")
                break
            _replay_state, state_dict_common = _advance_replay_step(
                step=step,
                reader=reader,
                scenario=scenario,
                simulation_app=simulation_app,
                rep=rep,
                render_rt_subframes=args.render_rt_subframes,
                run_render_step=False,
            )
            if step not in written_common_steps:
                writer.write_state_dict_common(state_dict_common, step)
                written_common_steps.add(step)
            annotations = gather_annotations(get_stage())
            semantic_state = extract_semantic_state_from_state_dict(state_dict_common)
            payload = build_annotation_payload(
                step=step,
                base_annotations=annotations,
                semantic_state=semantic_state,
            )
            writer.write_annotations(payload, step)
            annotations_written_count += 1

    return pc_written_count, annotations_written_count, validation_summary


def run_replay(
    args: argparse.Namespace,
    reader: Any,
    writer: Any,
    simulation_app: Any,
    rep: Any,
    load_scenario: Any,
    get_world: Any,
) -> Tuple[int, int, Dict[str, Any]]:
    if str(getattr(args, "pipeline_mode", "staged")) == "legacy":
        return run_legacy_replay(args, reader, writer, simulation_app, rep, load_scenario, get_world)
    return run_staged_replay(args, reader, writer, simulation_app, rep, load_scenario, get_world)


def main() -> int:
    args = parse_args()
    normalize_args(args)
    args.input_path = os.path.expanduser(args.input_path)
    args.output_path = os.path.expanduser(args.output_path)

    if os.path.isdir(args.input_path) and not os.path.isdir(os.path.join(args.input_path, "state")):
        candidates = sorted(
            [p for p in glob.glob(os.path.join(args.input_path, "*")) if os.path.isdir(p)],
            key=os.path.getmtime,
        )
        if not candidates:
            log(f"[replay] No recording directories found in: {args.input_path}")
            return 1
        args.input_path = candidates[-1]

    args.output_path = resolve_output_path(args.input_path, args.output_path)

    install_signal_handlers()
    bootstrap_repo_paths()

    try:
        simulation_app, rep = init_simulation_app(headless=True)
    except Exception as exc:
        log(f"[replay] Failed to initialize SimulationApp/Replicator: {exc}")
        return 1

    try:
        bootstrap_repo_paths()
        log("[replay] Loading runtime modules...")
        Reader, Writer, load_scenario, get_world = load_runtime_modules()
        log("[replay] Runtime modules loaded.")
        log("[replay] Creating reader...")
        reader = Reader(args.input_path)
        log("[replay] Reader ready.")
        log("[replay] Creating writer...")
        writer = Writer(args.output_path)
        log("[replay] Writer ready.")
        log("[replay] Copying init artifacts...")
        writer.copy_init(args.input_path, overwrite=args.overwrite, verbose=args.verbose)
        log("[replay] Init artifacts copied.")
        log("============== Replaying ==============")
        log(f"\tInput path: {args.input_path}")
        log(f"\tOutput path: {args.output_path}")
        log(f"\tRgb enabled: {args.rgb_enabled}")
        log(f"\tSegmentation enabled: {args.segmentation_enabled}")
        log(f"\tRendering RT subframes: {args.render_rt_subframes}")
        log(f"\tRender interval: {args.render_interval}")
        log(f"\tPipeline mode: {args.pipeline_mode}")
        log(f"\tCamera serial enabled: {args.camera_serial_enabled}")
        pc_count, ann_count, pc_validation_summary = run_replay(
            args=args,
            reader=reader,
            writer=writer,
            simulation_app=simulation_app,
            rep=rep,
            load_scenario=load_scenario,
            get_world=get_world,
        )
    except Exception as exc:
        log(f"[replay] Replay failed: {exc}")
        return 1

    write_summary(
        args.output_path,
        args.input_path,
        pc_count,
        ann_count,
        pc_validation_summary,
        args.verbose,
    )
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
