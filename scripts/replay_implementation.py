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

This script supports:
  - regular replay/render with SimulationApp
  - dry-run mode to copy/write outputs without launching Isaac Sim
"""

import argparse
import glob
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tqdm


STOP_REQUESTED = False


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
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--rgb_enabled", type=parse_bool, default=True)
    parser.add_argument("--segmentation_enabled", type=parse_bool, default=True)
    parser.add_argument("--depth_enabled", type=parse_bool, default=True)
    parser.add_argument("--instance_id_segmentation_enabled", type=parse_bool, default=True)
    parser.add_argument("--normals_enabled", type=parse_bool, default=False)
    parser.add_argument("--render_rt_subframes", type=int, default=1)
    parser.add_argument("--render_interval", type=int, default=1)
    parser.add_argument("--pc_enabled", type=parse_bool, default=True)
    parser.add_argument("--pc_format", type=str, default="npy", choices=["npy", "ply", "pcd"])
    parser.add_argument("--annotations_enabled", type=parse_bool, default=True)
    parser.add_argument("--pc_interval", type=int, default=1)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--verbose", type=parse_bool, default=False)
    parser.add_argument(
        "--dry_run",
        type=parse_bool,
        default=False,
        help="If True, skip SimulationApp and copy/write data from recording to output only.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[replay] Ignoring unknown args: {' '.join(unknown)}")
    return args


def enforce_full_capture(args: argparse.Namespace) -> None:
    required_true = [
        "rgb_enabled",
        "segmentation_enabled",
        "depth_enabled",
        "instance_id_segmentation_enabled",
        "normals_enabled",
        "pc_enabled",
        "annotations_enabled",
    ]
    for key in required_true:
        if not getattr(args, key, False):
            print(f"[replay] Overriding --{key}=False to True to keep full dataset capture.")
        setattr(args, key, True)
    if int(getattr(args, "pc_interval", 1)) != 1:
        print("[replay] Overriding --pc_interval to 1 to keep full per-frame pointcloud capture.")
    args.pc_interval = 1
    if int(getattr(args, "render_interval", 1)) != 1:
        print("[replay] Overriding --render_interval to 1 to keep full per-frame replay coverage.")
    args.render_interval = 1


def install_signal_handlers() -> None:
    def _signal_handler(sig: int, _frame: Any) -> None:
        global STOP_REQUESTED
        print(f"\n[replay] Received signal {sig}; finishing current iteration before stopping.")
        STOP_REQUESTED = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def bootstrap_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    ext_pkg_root = repo_root / "exts" / "omni.ext.mobility_gen"
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

    simulation_app = SimulationApp(launch_config={"headless": headless})
    import omni.replicator.core as rep

    return simulation_app, rep


def load_runtime_modules(dry_run: bool):
    from omni.ext.mobility_gen.reader import Reader
    from omni.ext.mobility_gen.writer import Writer

    if dry_run:
        return Reader, Writer, None, None

    from omni.ext.mobility_gen.build import load_scenario
    from omni.ext.mobility_gen.utils.global_utils import get_world

    return Reader, Writer, load_scenario, get_world


def has_any_data(state_dict: Dict[str, Any]) -> bool:
    return any(value is not None for value in state_dict.values())


def count_non_none(state_dict: Dict[str, Any]) -> int:
    return sum(1 for value in state_dict.values() if value is not None)


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


def gather_annotations(stage: Any) -> Dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom

    annotations = {"bboxes2d": [], "bboxes3d": [], "classes": []}
    if stage is None:
        return annotations

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    image_w, image_h = 640, 480
    camera_prim = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Camera":
            camera_prim = prim
            break

    cam_params = None
    if camera_prim is not None:
        cam = UsdGeom.Camera(camera_prim)
        focal = cam.GetFocalLengthAttr().Get()
        h_ap = cam.GetHorizontalApertureAttr().Get()
        v_ap = cam.GetVerticalApertureAttr().Get()
        fx = focal * image_w / h_ap if (h_ap and image_w) else 1.0
        fy = focal * image_h / v_ap if (v_ap and image_h) else fx
        cam_params = {"fx": fx, "fy": fy, "cx": image_w / 2.0, "cy": image_h / 2.0}

    for prim in stage.Traverse():
        if prim.IsPseudoRoot() or not prim.IsActive():
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

        if cam_params is None or camera_prim is None:
            continue

        cam_world = xform_cache.GetLocalToWorldTransform(camera_prim)
        cam_mat = cam_world.GetInverse()
        xs = []
        ys = []
        for corner in corners:
            wc = Gf.Vec4d(corner[0], corner[1], corner[2], 1.0)
            cc = cam_mat * wc
            if cc[2] <= 0.0:
                continue
            xs.append(float((cam_params["fx"] * (cc[0] / cc[2])) + cam_params["cx"]))
            ys.append(float((cam_params["fy"] * (cc[1] / cc[2])) + cam_params["cy"]))

        if xs and ys:
            annotations["bboxes2d"].append(
                {
                    "prim_path": prim.GetPath().pathString,
                    "class": class_name,
                    "semantic_labels": semantic_labels,
                    "bbox": [
                        max(0.0, min(xs)),
                        max(0.0, min(ys)),
                        min(image_w - 1.0, max(xs)),
                        min(image_h - 1.0, max(ys)),
                    ],
                }
            )

    annotations["classes"] = sorted(
        {entry.get("class", "") for entry in annotations["bboxes3d"] if entry.get("class", "")}
    )
    return annotations


def read_annotation_file(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
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


def write_summary(
    output_path: str,
    input_path: str,
    pc_written_count: int,
    annotations_written_count: int,
    verbose: bool,
    dry_run: bool,
) -> None:
    print("\n=== Replay summary ===")
    print(f"Pointcloud entries written: {pc_written_count}")
    print(f"Annotation files written: {annotations_written_count}")

    os.makedirs(output_path, exist_ok=True)
    summary_path = os.path.join(output_path, "replay_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Replay summary (dry_run)\n" if dry_run else "Replay summary\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"Pointcloud entries written: {pc_written_count}\n")
        f.write(f"Annotation files written: {annotations_written_count}\n")

        common_files = glob.glob(os.path.join(output_path, "state", "common", "*.npy"))
        rgb_files = glob.glob(os.path.join(output_path, "state", "rgb", "*", "*.jpg"))
        seg_files = glob.glob(os.path.join(output_path, "state", "segmentation", "*", "*.png"))
        depth_files = glob.glob(os.path.join(output_path, "state", "depth", "*", "*.png"))
        normals_files = glob.glob(os.path.join(output_path, "state", "normals", "*", "*.npy"))
        ann_files = glob.glob(os.path.join(output_path, "state", "annotations", "*.json"))
        f.write(f"state/common files: {len(common_files)}\n")
        f.write(f"state/rgb images: {len(rgb_files)}\n")
        f.write(f"state/segmentation images: {len(seg_files)}\n")
        f.write(f"state/depth images: {len(depth_files)}\n")
        f.write(f"state/normals arrays: {len(normals_files)}\n")
        f.write(f"state/annotations files: {len(ann_files)}\n")

        total_bbox2d = 0
        total_bbox3d = 0
        semantic_files = 0
        class_labels = set()
        for ann_path in ann_files:
            ann_data = read_annotation_file(ann_path)
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
                semantic_files += 1
        f.write(f"Total bbox2d entries: {total_bbox2d}\n")
        f.write(f"Total bbox3d entries: {total_bbox3d}\n")
        f.write(f"Annotation files with semantic payload: {semantic_files}\n")
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
        print(f"Wrote replay summary to: {summary_path}")


def run_dry_run(args: argparse.Namespace, reader: Any, writer: Any) -> Tuple[int, int]:
    pc_written_count = 0
    annotations_written_count = 0
    num_steps = len(reader)
    pc_interval = max(1, int(args.pc_interval))

    print("[replay] Running in dry_run mode")
    for step in range(0, num_steps, args.render_interval):
        if STOP_REQUESTED:
            print("[replay] Stop requested; leaving dry_run loop.")
            break

        state_dict_common = reader.read_state_dict_common(index=step)
        writer.write_state_dict_common(state_dict_common, step)

        if args.rgb_enabled:
            writer.write_state_dict_rgb(reader.read_state_dict_rgb(index=step), step)
        if args.segmentation_enabled:
            writer.write_state_dict_segmentation(reader.read_state_dict_segmentation(index=step), step)
        if args.depth_enabled:
            writer.write_state_dict_depth(reader.read_state_dict_depth(index=step), step)
        if args.normals_enabled:
            writer.write_state_dict_normals(reader.read_state_dict_normals(index=step), step)

        if args.pc_enabled and (step % pc_interval) == 0:
            pc_from_reader = reader.read_state_dict_pointcloud(index=step)
            if has_any_data(pc_from_reader):
                writer.write_state_dict_pointcloud(pc_from_reader, step, save_format=args.pc_format)
                pc_written_count += count_non_none(pc_from_reader)

        if args.annotations_enabled:
            src_ann = os.path.join(args.input_path, "state", "annotations", f"{step:08d}.json")
            src_payload = read_annotation_file(src_ann)
            semantic_state = extract_semantic_state_from_state_dict(state_dict_common)
            payload = build_annotation_payload(
                step=step,
                base_annotations=src_payload,
                semantic_state=semantic_state,
            )
            writer.write_annotations(payload, step)
            annotations_written_count += 1

    return pc_written_count, annotations_written_count


def run_replay(
    args: argparse.Namespace,
    reader: Any,
    writer: Any,
    simulation_app: Any,
    rep: Any,
    load_scenario: Any,
    get_world: Any,
) -> Tuple[int, int]:
    scenario = load_scenario(args.input_path)
    if scenario is None:
        raise RuntimeError(f"Failed to load scenario from recording: {args.input_path}")

    world = get_world()
    if world is None:
        raise RuntimeError("World instance is not available after loading scenario.")
    world.reset()

    print(scenario)
    if args.rgb_enabled:
        scenario.enable_rgb_rendering()
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
    num_steps = len(reader)
    pc_interval = max(1, int(args.pc_interval))

    for step in tqdm.tqdm(range(0, num_steps, args.render_interval)):
        if STOP_REQUESTED:
            print("[replay] Stop requested; leaving replay loop.")
            break

        replay_state = reader.read_state_dict(index=step)
        scenario.load_state_dict(replay_state)
        scenario.write_replay_data()

        simulation_app.update()
        rep.orchestrator.step(
            rt_subframes=args.render_rt_subframes,
            delta_time=0.0,
            pause_timeline=False,
        )
        scenario.update_state()

        state_dict_common = scenario.state_dict_common()
        writer.write_state_dict_common(state_dict_common, step)

        if args.rgb_enabled:
            writer.write_state_dict_rgb(scenario.state_dict_rgb(), step)
        if args.segmentation_enabled:
            writer.write_state_dict_segmentation(scenario.state_dict_segmentation(), step)
        if args.depth_enabled:
            writer.write_state_dict_depth(scenario.state_dict_depth(), step)
        if args.normals_enabled:
            writer.write_state_dict_normals(scenario.state_dict_normals(), step)

        if args.pc_enabled and (step % pc_interval) == 0:
            state_pc = scenario.state_dict_pointcloud()
            used_pc = None
            if has_any_data(state_pc):
                writer.write_state_dict_pointcloud(state_pc, step, save_format=args.pc_format)
                pc_written_count += count_non_none(state_pc)
                used_pc = state_pc
            elif getattr(reader, "pointcloud_names", []):
                fallback_pc = reader.read_state_dict_pointcloud(index=step)
                if has_any_data(fallback_pc):
                    print(f"[replay] No scenario pointcloud for step {step}; copying from recording.")
                    writer.write_state_dict_pointcloud(fallback_pc, step, save_format=args.pc_format)
                    pc_written_count += count_non_none(fallback_pc)
                    used_pc = fallback_pc

            if used_pc is not None:
                metadata = build_pointcloud_metadata(scenario, used_pc)
                if metadata:
                    writer.write_pointcloud_metadata(metadata, step)

        if args.annotations_enabled:
            from omni.ext.mobility_gen.utils.global_utils import get_stage

            annotations = gather_annotations(get_stage())
            semantic_state = extract_semantic_state_from_state_dict(state_dict_common)
            payload = build_annotation_payload(
                step=step,
                base_annotations=annotations,
                semantic_state=semantic_state,
            )
            writer.write_annotations(payload, step)
            annotations_written_count += 1

    return pc_written_count, annotations_written_count


def main() -> int:
    args = parse_args()
    enforce_full_capture(args)
    args.input_path = os.path.expanduser(args.input_path)
    args.output_path = os.path.expanduser(args.output_path)

    install_signal_handlers()
    bootstrap_repo_paths()

    if args.dry_run:
        try:
            Reader, Writer, _, _ = load_runtime_modules(dry_run=True)
        except Exception as exc:
            print(f"[replay] Failed to import Reader/Writer modules for dry_run: {exc}")
            return 1

        reader = Reader(args.input_path)
        writer = Writer(args.output_path)
        writer.copy_init(args.input_path, overwrite=args.overwrite, verbose=args.verbose)
        print("============== Replaying ==============")
        print(f"\tInput path: {args.input_path}")
        print(f"\tOutput path: {args.output_path}")
        print(f"\tRgb enabled: {args.rgb_enabled}")
        print(f"\tSegmentation enabled: {args.segmentation_enabled}")
        print(f"\tRendering RT subframes: {args.render_rt_subframes}")
        print(f"\tRender interval: {args.render_interval}")
        print(f"\tDry run: {args.dry_run}")
        pc_count, ann_count = run_dry_run(args, reader, writer)
        write_summary(args.output_path, args.input_path, pc_count, ann_count, args.verbose, dry_run=True)
        return 0

    try:
        simulation_app, rep = init_simulation_app(headless=True)
    except Exception as exc:
        print(f"[replay] Failed to initialize SimulationApp/Replicator: {exc}")
        return 1

    try:
        bootstrap_repo_paths()
        Reader, Writer, load_scenario, get_world = load_runtime_modules(dry_run=False)
        reader = Reader(args.input_path)
        writer = Writer(args.output_path)
        writer.copy_init(args.input_path, overwrite=args.overwrite, verbose=args.verbose)
        print("============== Replaying ==============")
        print(f"\tInput path: {args.input_path}")
        print(f"\tOutput path: {args.output_path}")
        print(f"\tRgb enabled: {args.rgb_enabled}")
        print(f"\tSegmentation enabled: {args.segmentation_enabled}")
        print(f"\tRendering RT subframes: {args.render_rt_subframes}")
        print(f"\tRender interval: {args.render_interval}")
        print(f"\tDry run: {args.dry_run}")
        pc_count, ann_count = run_replay(
            args=args,
            reader=reader,
            writer=writer,
            simulation_app=simulation_app,
            rep=rep,
            load_scenario=load_scenario,
            get_world=get_world,
        )
    except Exception as exc:
        print(f"[replay] Replay failed: {exc}")
        return 1

    write_summary(args.output_path, args.input_path, pc_count, ann_count, args.verbose, dry_run=False)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
