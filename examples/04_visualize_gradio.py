# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gradio explorer for PatoSim recordings/replays.

Compatible with the current recording layout:
  - state/common
  - optional rgb/depth/segmentation/instance_id_segmentation/normals
  - optional pointcloud
  - optional split annotations (bboxes2d/bboxes3d/classes/semantic)

Example:
  python examples/04_visualize_gradio.py --input_dir ~/PatoSimData/replays
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def bootstrap_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    ext_pkg_root = repo_root / "exts" / "omni.ext.patosim"
    ext_omni_ext_root = ext_pkg_root / "omni" / "ext"
    repo_root_str = str(repo_root)

    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    ext_pkg_root_str = str(ext_pkg_root)
    if ext_pkg_root.exists() and ext_pkg_root_str not in sys.path:
        sys.path.insert(0, ext_pkg_root_str)

    try:
        import omni.ext as omni_ext  # type: ignore

        ext_omni_ext_root_str = str(ext_omni_ext_root)
        if ext_omni_ext_root.exists() and ext_omni_ext_root_str not in omni_ext.__path__:
            omni_ext.__path__.append(ext_omni_ext_root_str)
    except Exception:
        pass

    return repo_root


REPO_ROOT = bootstrap_repo_paths()

from omni.ext.patosim.reader import Reader  # noqa: E402


gr = None
_OPEN3D_HELPERS: Optional[Tuple[Callable, Callable, Callable]] = None
_INTERACTIVE_POINTCLOUD_PROCESS = None
BOX_EDGE_INDEXES = [
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="~/PatoSimData/replays")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def launch_interactive_pointcloud(recording_path: str, pointcloud_name: str, step: int) -> str:
    global _INTERACTIVE_POINTCLOUD_PROCESS
    if not pointcloud_name:
        return "No pointcloud sensor selected."

    script_path = REPO_ROOT / "scripts" / "view_pointclouds.py"
    if _INTERACTIVE_POINTCLOUD_PROCESS is not None:
        try:
            if _INTERACTIVE_POINTCLOUD_PROCESS.poll() is None:
                return "An interactive pointcloud window is already open. Close it before opening another."
        except Exception:
            pass
        _INTERACTIVE_POINTCLOUD_PROCESS = None

    cmd = [
        sys.executable,
        str(script_path),
        "--path",
        recording_path,
        "--sensor",
        pointcloud_name,
        "--step",
        str(int(step)),
    ]
    try:
        _INTERACTIVE_POINTCLOUD_PROCESS = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
    except Exception as exc:
        return f"Failed to open interactive pointcloud: {exc}"
    return f"Opened interactive Open3D window for `{pointcloud_name}` at step `{int(step)}`."


def discover_recordings(directory: str) -> list[str]:
    candidates = []
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        if not os.path.isdir(path):
            continue
        if os.path.isdir(os.path.join(path, "state", "common")):
            candidates.append(os.path.basename(path))
    return candidates


def blank_image(label: str, shape: Tuple[int, int] = (320, 240)) -> np.ndarray:
    width, height = int(shape[0]), int(shape[1])
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = np.array([45, 45, 45], dtype=np.uint8)
    return image


def get_open3d_helpers() -> Optional[Tuple[Callable, Callable, Callable]]:
    global _OPEN3D_HELPERS
    if _OPEN3D_HELPERS is not None:
        return _OPEN3D_HELPERS
    try:
        from scripts.view_pointclouds import points_to_open3d, reduce_points, require_open3d

        _OPEN3D_HELPERS = (require_open3d, reduce_points, points_to_open3d)
    except Exception:
        _OPEN3D_HELPERS = None
    return _OPEN3D_HELPERS


def normalize_depth_image(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    finite_mask = np.isfinite(arr) & (arr > 0.0)
    if not np.any(finite_mask):
        return blank_image("depth")
    clipped = np.zeros_like(arr, dtype=np.float32)
    clipped[finite_mask] = 1.0 / np.maximum(arr[finite_mask], 1e-6)
    lo = float(np.percentile(clipped[finite_mask], 2))
    hi = float(np.percentile(clipped[finite_mask], 98))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((clipped - lo) / (hi - lo), 0.0, 1.0)
    rgb = (norm[..., None] * 255.0).astype(np.uint8)
    return np.repeat(rgb, 3, axis=2)


def normalize_segmentation_image(segmentation: np.ndarray) -> np.ndarray:
    arr = np.asarray(segmentation)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.asarray(arr[..., :3], dtype=np.uint8)
    if arr.ndim != 2:
        return blank_image("segmentation")
    colored = np.stack(
        [
            (arr * 17) % 255,
            (arr * 67) % 255,
            (arr * 131) % 255,
        ],
        axis=2,
    ).astype(np.uint8)
    return colored


def quat_wxyz_to_rotmat(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(-1)
    if q.size != 4:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q
    n = float(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def transform_world_points_to_sensor(points: np.ndarray, metadata: Optional[dict]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return pts
    if not isinstance(metadata, dict):
        return pts
    position = metadata.get("position")
    orientation = metadata.get("orientation")
    if position is None or orientation is None:
        return pts
    try:
        t = np.asarray(position, dtype=np.float32).reshape(1, 3)
        rot = quat_wxyz_to_rotmat(orientation)
        transformed = (pts[:, :3] - t) @ rot
        if pts.shape[1] > 3:
            transformed = np.hstack([transformed, pts[:, 3:]])
        return transformed
    except Exception:
        return pts


def label_color_rgb01(label: str) -> tuple[float, float, float]:
    cmap = plt.get_cmap("tab20")
    idx = sum(ord(ch) for ch in str(label)) % 20
    color = cmap(idx)
    return float(color[0]), float(color[1]), float(color[2])


def label_color_rgb255(label: str) -> tuple[int, int, int]:
    rgb = label_color_rgb01(label)
    return tuple(int(round(255.0 * channel)) for channel in rgb)


def bbox_line_segments_xy(corners: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (8, 3):
        return []
    segments = []
    for i0, i1 in BOX_EDGE_INDEXES:
        segments.append((corners[i0, :2], corners[i1, :2]))
    return segments


class Context:
    def __init__(self, root_dir: str, recording_name: str):
        self.root_dir = root_dir
        self.recording_name = recording_name
        self.recording_path = os.path.join(root_dir, recording_name)
        self.reader = Reader(self.recording_path)
        self.occupancy_map = self.reader.read_occupancy_map()
        self._path_xy = None

        self.rgb_names = list(self.reader.rgb_names)
        self.depth_names = list(self.reader.depth_names)
        self.segmentation_names = list(self.reader.segmentation_names)
        self.instance_id_segmentation_names = list(self.reader.instance_id_segmentation_names)
        self.pointcloud_names = list(self.reader.pointcloud_names)

        self.camera_groups = self._build_camera_groups()
        self.camera_names = [group["label"] for group in self.camera_groups]
        self.default_camera = self.camera_names[0] if self.camera_names else ""
        self.default_pointcloud = self.pointcloud_names[0] if self.pointcloud_names else ""
        self.pointcloud_steps = self._build_pointcloud_step_index()
        self.step_to_index = {int(step): idx for idx, step in enumerate(self.reader.steps)}

    def _build_pointcloud_step_index(self):
        step_index = {}
        pointcloud_root = Path(self.recording_path) / "state" / "pointcloud"
        for sensor_name in self.pointcloud_names:
            sensor_dir = pointcloud_root / sensor_name
            steps = set()
            if sensor_dir.is_dir():
                for ext in ("*.npy", "*.ply", "*.pcd"):
                    for path in sensor_dir.glob(ext):
                        stem = path.stem
                        if stem.isdigit():
                            steps.add(int(stem))
            step_index[sensor_name] = sorted(steps)
        return step_index

    def index_to_step(self, index: int) -> int:
        if len(self.reader.steps) == 0:
            return int(index)
        clamped = max(0, min(int(index), len(self.reader.steps) - 1))
        return int(self.reader.steps[clamped])

    def resolve_pointcloud_step(self, pointcloud_name: str, requested_step: int) -> int:
        steps = self.pointcloud_steps.get(pointcloud_name, [])
        if not steps:
            return int(requested_step)
        return min(steps, key=lambda step: abs(step - int(requested_step)))

    def _load_step_state(self, index: int):
        return self.reader.read_state_dict(index=int(index))

    @staticmethod
    def _strip_modality_suffix(name: str) -> str:
        for suffix in (
            ".rgb_image",
            ".depth_image",
            ".segmentation_image",
            ".instance_id_segmentation_image",
        ):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @staticmethod
    def _split_sensor_side(name: str) -> Tuple[str, Optional[str]]:
        if name.endswith(".left"):
            return name[:-5], "left"
        if name.endswith(".right"):
            return name[:-6], "right"
        return name, None

    def _display_sensor_name(self, base_name: str) -> str:
        return base_name.split("robot.", 1)[-1]

    def _build_camera_groups(self):
        grouped = {}
        modality_lists = {
            "rgb": self.rgb_names,
            "depth": self.depth_names,
            "segmentation": self.segmentation_names,
        }

        for modality, names in modality_lists.items():
            for full_name in names:
                stripped = self._strip_modality_suffix(full_name)
                base_name, side = self._split_sensor_side(stripped)
                group = grouped.setdefault(
                    base_name,
                    {
                        "base": base_name,
                        "label": self._display_sensor_name(base_name),
                        "sides": {},
                    },
                )
                side_key = side or "mono"
                side_entry = group["sides"].setdefault(side_key, {})
                side_entry[modality] = full_name

        return sorted(grouped.values(), key=lambda group: group["label"])

    def _camera_match_candidates(self, camera_full_name: str) -> set[str]:
        stripped = self._strip_modality_suffix(camera_full_name)
        candidates = {camera_full_name, stripped}
        parts = [part for part in stripped.split(".") if part]
        for start in range(len(parts)):
            suffix_dot = ".".join(parts[start:])
            suffix_slash = "/".join(parts[start:])
            if suffix_dot:
                candidates.add(suffix_dot)
            if suffix_slash:
                candidates.add(suffix_slash)
        if parts:
            candidates.add(parts[-1])
            if len(parts) >= 2:
                candidates.add(".".join(parts[-2:]))
                candidates.add("/".join(parts[-2:]))
        return {candidate for candidate in candidates if candidate}

    def _bbox2d_matches_camera(self, bbox: dict, camera_full_name: str) -> bool:
        if not isinstance(bbox, dict):
            return False
        camera_name = str(bbox.get("camera_name", "") or "")
        prim_path = str(bbox.get("camera_prim_path", "") or "")
        candidates = self._camera_match_candidates(camera_full_name)
        for candidate in candidates:
            if camera_name == candidate or camera_name.endswith(candidate):
                return True
            if prim_path == candidate or prim_path.endswith(candidate):
                return True
            if candidate.startswith("/") and candidate in prim_path:
                return True
            if "/" in candidate and f"/{candidate}" in prim_path:
                return True
        return False

    def _bboxes2d_for_camera(self, index: int, camera_full_name: str) -> list[dict]:
        annotations = self._annotations_for_step(self.index_to_step(index))
        matches = []
        for bbox in annotations.get("bboxes2d", []):
            if self._bbox2d_matches_camera(bbox, camera_full_name):
                matches.append(bbox)
        return matches

    def _pointcloud_metadata_for_step(self, pointcloud_name: str, step: int):
        if not pointcloud_name:
            return None
        sensor_dir = Path(self.recording_path) / "state" / "pointcloud" / pointcloud_name
        meta_path = sensor_dir / f"{int(step):08d}_meta.json"
        if meta_path.exists():
            try:
                import json

                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _annotations_for_step(self, step: int) -> dict:
        actual_step = int(step)
        reader_index = self.step_to_index.get(actual_step)
        if reader_index is not None:
            return self.reader.read_annotations(reader_index) or {}

        import json

        state_root = Path(self.recording_path) / "state"
        payload = {
            "step": actual_step,
            "bboxes2d": [],
            "bboxes3d": [],
            "classes": [],
            "semantic": {},
        }
        found = False
        for key in ("bboxes2d", "bboxes3d", "classes", "semantic"):
            path = state_root / key / f"{actual_step:08d}.json"
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload[key] = json.load(f)
                found = True
            except Exception:
                continue
        return payload if found else {}

    def _pointcloud_state_for_step(self, pointcloud_name: str, step: int):
        sensor_dir = Path(self.recording_path) / "state" / "pointcloud" / pointcloud_name
        if not sensor_dir.is_dir():
            return None
        for ext in (".npy", ".ply", ".pcd"):
            path = sensor_dir / f"{int(step):08d}{ext}"
            if not path.exists():
                continue
            if ext == ".npy":
                try:
                    return np.load(path, allow_pickle=True)
                except Exception:
                    return None
            try:
                if ext in {".ply", ".pcd"}:
                    from scripts.view_pointclouds import load_pointcloud_array

                    return load_pointcloud_array(path)
            except Exception:
                return None
        return None

    def _visible_bbox3d_for_pointcloud(self, step: int, pointcloud_name: str, pts: Optional[np.ndarray]):
        annotations = self._annotations_for_step(step)
        bbox_entries = annotations.get("bboxes3d", [])
        metadata = self._pointcloud_metadata_for_step(pointcloud_name, step)

        xy_limits = None
        if pts is not None:
            arr = np.asarray(pts, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] > 0:
                mins = np.min(arr[:, :2], axis=0)
                maxs = np.max(arr[:, :2], axis=0)
                span = np.maximum(maxs - mins, 1e-3)
                margin = 0.15 * span
                xy_limits = (mins - margin, maxs + margin)

        visible = []
        for entry in bbox_entries:
            corners = np.asarray(entry.get("corners"), dtype=np.float32)
            if corners.shape != (8, 3):
                continue
            local_corners = transform_world_points_to_sensor(corners, metadata)
            if xy_limits is not None:
                mins = np.min(local_corners[:, :2], axis=0)
                maxs = np.max(local_corners[:, :2], axis=0)
                low, high = xy_limits
                overlaps = np.all(maxs >= low) and np.all(mins <= high)
                if not overlaps:
                    continue
            visible.append(
                {
                    "class": str(entry.get("class", "") or entry.get("prim_path", "object")),
                    "prim_path": entry.get("prim_path"),
                    "corners_local": local_corners,
                }
            )
        return visible

    def get_path_xy(self):
        if self._path_xy is not None:
            return self._path_xy
        points_world = []
        for index in range(len(self.reader)):
            state_dict = self.reader.read_state_dict_common(index)
            position = state_dict.get("robot.position")
            if position is None:
                continue
            points_world.append(np.asarray(position, dtype=np.float32)[0:2])
        if not points_world:
            self._path_xy = np.zeros((0, 2), dtype=np.float32)
            return self._path_xy
        points_world = np.asarray(points_world, dtype=np.float32)
        self._path_xy = self.occupancy_map.world_to_pixel_numpy(points_world)
        return self._path_xy

    def get_robot_xy(self, index: int):
        state_dict = self.reader.read_state_dict_common(index)
        pos_world = state_dict.get("robot.position")
        if pos_world is None:
            return np.zeros((1, 2), dtype=np.float32)
        pos_image = self.occupancy_map.world_to_pixel_numpy(np.asarray([pos_world[0:2]], dtype=np.float32))
        return pos_image

    def get_map_plot(self, index: int):
        path_xy = self.get_path_xy()
        robot_xy = self.get_robot_xy(index)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.occupancy_map.ros_image().convert("RGB"))
        if len(path_xy) > 0:
            ax.plot(path_xy[0, 0], path_xy[0, 1], "go", markersize=6)
            ax.plot(path_xy[:, 0], path_xy[:, 1], "--", linewidth=2, color="#00d084")
            ax.plot(path_xy[-1, 0], path_xy[-1, 1], "ro", markersize=6)
        ax.plot(robot_xy[0, 0], robot_xy[0, 1], "y*", markersize=14)
        ax.set_title(f"Trajectory | step={int(index)}")
        ax.axis("off")
        fig.tight_layout()
        return fig

    def get_camera_bundle(self, index: int, camera_name: str):
        state = self._load_step_state(index)
        rgb = state["rgb"].get(camera_name)
        depth = state["depth"].get(camera_name)
        segmentation = state["segmentation"].get(camera_name)

        rgb_image = np.asarray(rgb, dtype=np.uint8) if rgb is not None else blank_image("rgb")
        depth_image = normalize_depth_image(depth) if depth is not None else blank_image("depth")
        seg_image = normalize_segmentation_image(segmentation) if segmentation is not None else blank_image("seg")
        return rgb_image, depth_image, seg_image

    def _get_group_for_label(self, label: str):
        if not self.camera_groups:
            return None
        for group in self.camera_groups:
            if group["label"] == label:
                return group
        return self.camera_groups[0]

    def _group_is_stereo(self, group) -> bool:
        return "left" in group["sides"] and "right" in group["sides"]

    def _get_group_images(self, index: int, group):
        state = self._load_step_state(index)
        images = {
            "rgb_left": blank_image("rgb_left"),
            "rgb_right": blank_image("rgb_right"),
            "depth": blank_image("depth"),
            "segmentation": blank_image("segmentation"),
        }

        if self._group_is_stereo(group):
            left = group["sides"].get("left", {})
            right = group["sides"].get("right", {})
            if left.get("rgb"):
                rgb = state["rgb"].get(left["rgb"])
                images["rgb_left"] = np.asarray(rgb, dtype=np.uint8) if rgb is not None else blank_image("rgb_left")
            if right.get("rgb"):
                rgb = state["rgb"].get(right["rgb"])
                images["rgb_right"] = (
                    np.asarray(rgb, dtype=np.uint8) if rgb is not None else blank_image("rgb_right")
                )
            if left.get("depth"):
                depth = state["depth"].get(left["depth"])
                images["depth"] = normalize_depth_image(depth) if depth is not None else blank_image("depth")
            elif right.get("depth"):
                depth = state["depth"].get(right["depth"])
                images["depth"] = normalize_depth_image(depth) if depth is not None else blank_image("depth")
            if left.get("segmentation"):
                seg = state["segmentation"].get(left["segmentation"])
                images["segmentation"] = (
                    normalize_segmentation_image(seg) if seg is not None else blank_image("segmentation")
                )
            elif right.get("segmentation"):
                seg = state["segmentation"].get(right["segmentation"])
                images["segmentation"] = (
                    normalize_segmentation_image(seg) if seg is not None else blank_image("segmentation")
                )
        else:
            mono = group["sides"].get("mono", {})
            rgb_name = mono.get("rgb")
            depth_name = mono.get("depth")
            seg_name = mono.get("segmentation")
            if rgb_name:
                rgb = state["rgb"].get(rgb_name)
                images["rgb_left"] = np.asarray(rgb, dtype=np.uint8) if rgb is not None else blank_image("rgb_left")
            if depth_name:
                depth = state["depth"].get(depth_name)
                images["depth"] = normalize_depth_image(depth) if depth is not None else blank_image("depth")
            if seg_name:
                seg = state["segmentation"].get(seg_name)
                images["segmentation"] = (
                    normalize_segmentation_image(seg) if seg is not None else blank_image("segmentation")
                )

        return images

    def get_camera_bbox_panel(self, index: int, selected_camera: str):
        group = self._get_group_for_label(selected_camera)
        if group is None:
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#101010")
            ax.set_facecolor("#101010")
            ax.text(0.5, 0.5, "No RGB camera available", ha="center", va="center", color="white")
            ax.axis("off")
            fig.tight_layout()
            return fig

        stereo = self._group_is_stereo(group)
        cols = 2 if stereo else 1
        fig, axes = plt.subplots(1, cols, figsize=(5.2 * cols, 4.8))
        fig.patch.set_facecolor("#101010")
        axes = np.asarray(axes).reshape(1, cols)[0]
        images = self._get_group_images(index, group)

        panel_specs = []
        if stereo:
            panel_specs.append(("rgb_left", group["sides"].get("left", {}).get("rgb"), "RGB + BB2D Left"))
            panel_specs.append(("rgb_right", group["sides"].get("right", {}).get("rgb"), "RGB + BB2D Right"))
        else:
            panel_specs.append(("rgb_left", group["sides"].get("mono", {}).get("rgb"), "RGB + BB2D"))

        for ax, (key, sensor_name, title) in zip(axes, panel_specs):
            ax.set_facecolor("#101010")
            ax.imshow(images[key])
            if sensor_name:
                for bbox in self._bboxes2d_for_camera(index, sensor_name):
                    coords = bbox.get("bbox")
                    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                        continue
                    x0, y0, x1, y1 = [float(v) for v in coords]
                    label = str(bbox.get("class", "") or "object")
                    color = np.asarray(label_color_rgb01(label))
                    ax.add_patch(
                        Rectangle(
                            (x0, y0),
                            max(1.0, x1 - x0),
                            max(1.0, y1 - y0),
                            fill=False,
                            linewidth=1.8,
                            edgecolor=color,
                        )
                    )
                    ax.text(
                        x0,
                        max(2.0, y0 - 4.0),
                        label,
                        fontsize=8,
                        color="white",
                        bbox={"facecolor": color, "alpha": 0.85, "pad": 1.8, "edgecolor": "none"},
                    )
            ax.set_title(title, fontsize=11, fontweight="bold", color="white")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#303030")

        fig.suptitle(
            f"RGB + 2D Bounding Boxes | {group['label']} | step={int(index)}",
            fontsize=13,
            fontweight="bold",
            color="white",
        )
        fig.tight_layout()
        return fig

    def get_camera_rgb_panel(self, index: int, selected_camera: str):
        group = self._get_group_for_label(selected_camera)
        if group is None:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No camera data available", ha="center", va="center")
            ax.axis("off")
            fig.tight_layout()
            return fig

        stereo = self._group_is_stereo(group)
        cols = 2 if stereo else 1
        fig_w = 5.2 * cols
        fig_h = 4.8
        fig, axes = plt.subplots(1, cols, figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#101010")
        axes = np.asarray(axes).reshape(1, cols)[0]
        images = self._get_group_images(index, group)

        panel_specs = [("rgb_left", "RGB")]
        if stereo:
            panel_specs.append(("rgb_right", "RGB Right"))

        for ax, (key, title) in zip(axes, panel_specs):
            ax.set_facecolor("#101010")
            ax.imshow(images[key])
            ax.set_title(title, fontsize=11, fontweight="bold", color="white")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#303030")

        display_name = group["label"]
        if stereo:
            subtitle = f"{display_name} | stereo pair"
        else:
            subtitle = f"{display_name} | mono sensor"
        fig.suptitle(
            f"Camera Panel | {subtitle} | step={int(index)}",
            fontsize=13,
            fontweight="bold",
            color="white",
        )
        fig.tight_layout()
        return fig

    def get_camera_annotation_panel(self, index: int, selected_camera: str):
        group = self._get_group_for_label(selected_camera)
        if group is None:
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#101010")
            ax.set_facecolor("#101010")
            ax.text(0.5, 0.5, "No camera annotations available", ha="center", va="center", color="white")
            ax.axis("off")
            fig.tight_layout()
            return fig

        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
        fig.patch.set_facecolor("#101010")
        axes = np.asarray(axes).reshape(1, 2)[0]
        images = self._get_group_images(index, group)

        for ax, (key, title) in zip(axes, [("depth", "Depth"), ("segmentation", "Segmentation")]):
            ax.set_facecolor("#101010")
            ax.imshow(images[key])
            ax.set_title(title, fontsize=11, fontweight="bold", color="white")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#303030")

        fig.suptitle(
            f"Camera Annotation | {group['label']} | step={int(index)}",
            fontsize=13,
            fontweight="bold",
            color="white",
        )
        fig.tight_layout()
        return fig

    def get_pointcloud_plot(self, index: int, pointcloud_name: str):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_facecolor("#181818")
        fig.patch.set_facecolor("#181818")
        if not pointcloud_name:
            ax.text(0.5, 0.5, "No pointcloud sensor", ha="center", va="center", color="white")
            ax.axis("off")
            fig.tight_layout()
            return fig

        resolved_step = self.resolve_pointcloud_step(pointcloud_name, self.index_to_step(index))
        pts = self._pointcloud_state_for_step(pointcloud_name, resolved_step)
        if pts is None:
            ax.text(0.5, 0.5, "No pointcloud data", ha="center", va="center", color="white")
            ax.axis("off")
            fig.tight_layout()
            return fig

        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
            ax.text(0.5, 0.5, "Invalid pointcloud", ha="center", va="center", color="white")
            ax.axis("off")
            fig.tight_layout()
            return fig

        stride = max(1, int(pts.shape[0] / 15000))
        pts = pts[::stride]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.0, c=pts[:, 2], cmap="viridis")
        for bbox in self._visible_bbox3d_for_pointcloud(resolved_step, pointcloud_name, pts):
            label = bbox["class"]
            color = label_color_rgb01(label)
            for p0, p1 in bbox_line_segments_xy(bbox["corners_local"]):
                ax.plot(
                    [float(p0[0]), float(p1[0])],
                    [float(p0[1]), float(p1[1])],
                    color=color,
                    linewidth=1.0,
                    alpha=0.85,
                )
        ax.set_title(f"{pointcloud_name} | {pts.shape[0]} pts | pc_step={int(resolved_step)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        return fig

    def _make_open3d_bbox_lines(self, o3d, boxes: list[dict]):
        geometries = []
        for bbox in boxes:
            corners = np.asarray(bbox.get("corners_local"), dtype=np.float32)
            if corners.shape != (8, 3):
                continue
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners.astype(np.float64))
            line_set.lines = o3d.utility.Vector2iVector(np.asarray(BOX_EDGE_INDEXES, dtype=np.int32))
            color = np.asarray(label_color_rgb01(bbox.get("class", "object")), dtype=np.float64)
            colors = np.repeat(color.reshape(1, 3), len(BOX_EDGE_INDEXES), axis=0)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)
        return geometries

    def get_pointcloud_open3d_image(self, index: int, pointcloud_name: str) -> np.ndarray:
        helpers = get_open3d_helpers()
        if helpers is None:
            return blank_image("open3d_unavailable", shape=(640, 420))

        require_open3d, reduce_points, points_to_open3d = helpers
        resolved_step = (
            self.resolve_pointcloud_step(pointcloud_name, self.index_to_step(index))
            if pointcloud_name
            else self.index_to_step(index)
        )
        pts = self._pointcloud_state_for_step(pointcloud_name, resolved_step) if pointcloud_name else None
        if pts is None:
            return blank_image("no_pointcloud", shape=(640, 420))

        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
            return blank_image("invalid_pointcloud", shape=(640, 420))

        try:
            o3d = require_open3d()
            reduced = reduce_points(
                pts,
                stride=max(1, int(pts.shape[0] / 30000)),
                max_points=30000,
            )
            cloud = points_to_open3d(reduced)

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name="PatoSim Pointcloud Preview",
                width=640,
                height=420,
                visible=False,
            )
            vis.add_geometry(cloud)
            bbox_geometries = self._make_open3d_bbox_lines(
                o3d,
                self._visible_bbox3d_for_pointcloud(resolved_step, pointcloud_name, reduced),
            )
            for geometry in bbox_geometries:
                vis.add_geometry(geometry)
            render = vis.get_render_option()
            render.background_color = np.asarray([0.05, 0.05, 0.05], dtype=np.float64)
            render.point_size = 2.5

            bbox = cloud.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = np.asarray(bbox.get_extent(), dtype=np.float64)
            radius = max(float(np.linalg.norm(extent)), 1.0)

            ctr = vis.get_view_control()
            ctr.set_lookat(center)
            ctr.set_front([0.9, -0.45, 0.35])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_zoom(max(0.2, min(0.8, 1.8 / radius)))

            vis.poll_events()
            vis.update_renderer()
            image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            vis.destroy_window()
            if image.ndim != 3:
                return blank_image("open3d_capture_failed", shape=(640, 420))
            return np.clip(image * 255.0, 0, 255).astype(np.uint8)
        except Exception:
            return blank_image("open3d_capture_error", shape=(640, 420))

    def get_summary(self, index: int, camera_name: str, pointcloud_name: str) -> str:
        common = self.reader.read_state_dict_common(index)
        annotations = self.reader.read_annotations(index) or {}
        semantic = self.reader.read_semantic_annotations(index) or {}
        pc_meta = None
        resolved_pc_step = self.index_to_step(index)
        if pointcloud_name:
            resolved_pc_step = self.resolve_pointcloud_step(pointcloud_name, self.index_to_step(index))
            pc_meta = self._pointcloud_metadata_for_step(pointcloud_name, resolved_pc_step)
        pointcloud = self._pointcloud_state_for_step(pointcloud_name, resolved_pc_step) if pointcloud_name else None
        num_points = int(np.asarray(pointcloud).shape[0]) if pointcloud is not None else 0

        lines = [
            f"Recording: `{self.recording_name}`",
            f"Step: `{int(index)}` / `{max(0, len(self.reader) - 1)}`",
            f"Camera: `{camera_name or 'none'}`",
            f"Pointcloud sensor: `{pointcloud_name or 'none'}`",
            f"Pointcloud step used: `{resolved_pc_step}`",
            f"Pointcloud points: `{num_points}`",
            f"BBoxes2D: `{len(annotations.get('bboxes2d', []))}`",
            f"BBoxes3D: `{len(annotations.get('bboxes3d', []))}`",
            f"Classes: `{', '.join(annotations.get('classes', [])) if annotations.get('classes') else 'none'}`",
        ]

        robot_position = common.get("robot.position")
        if robot_position is not None:
            rp = np.asarray(robot_position, dtype=np.float32).reshape(-1)
            if rp.size >= 3:
                lines.append(f"Robot position: `({rp[0]:.3f}, {rp[1]:.3f}, {rp[2]:.3f})`")

        robot_orientation = common.get("robot.orientation")
        if robot_orientation is not None:
            ro = np.asarray(robot_orientation, dtype=np.float32).reshape(-1)
            if ro.size >= 4:
                lines.append(f"Robot quat: `({ro[0]:.3f}, {ro[1]:.3f}, {ro[2]:.3f}, {ro[3]:.3f})`")

        lidar_status = common.get("robot.lidar.status")
        if lidar_status is not None:
            lines.append(f"LiDAR status: `{lidar_status}`")

        if pc_meta:
            fields = None
            if isinstance(pc_meta, dict):
                fields = pc_meta.get("fields")
                if fields is None:
                    fields = pc_meta.get(pointcloud_name, {}).get("fields")
            if fields:
                lines.append(f"Pointcloud fields: `{fields}`")

        if semantic:
            semantic_keys = ", ".join(sorted(semantic.keys())[:8])
            lines.append(f"Semantic keys: `{semantic_keys or 'none'}`")

        return "\n\n".join(lines)

    def update(self, index: int, camera_name: str, pointcloud_name: str):
        camera_name = camera_name or self.default_camera
        pointcloud_name = pointcloud_name or self.default_pointcloud
        return [
            self.get_map_plot(index),
            self.get_camera_rgb_panel(index, camera_name),
            self.get_camera_annotation_panel(index, camera_name),
            self.get_camera_bbox_panel(index, camera_name),
            self.get_pointcloud_plot(index, pointcloud_name),
            self.get_pointcloud_open3d_image(index, pointcloud_name),
            self.get_summary(index, camera_name, pointcloud_name),
        ]


args = parse_args()
directory = os.path.expanduser(args.input_dir)
recording_names = discover_recordings(directory)

context: Optional[Context] = None


def update_recording(name: str):
    global context
    if not name:
        raise gr.Error("Select a recording first.")
    context = Context(directory, name)
    main = context.update(
        index=0,
        camera_name=context.default_camera,
        pointcloud_name=context.default_pointcloud,
    )
    slider = gr.Slider(value=0, minimum=0, maximum=max(0, len(context.reader) - 1), step=1)
    camera_dropdown = gr.Dropdown(
        choices=context.camera_names or [""],
        value=context.default_camera,
        label="Camera",
    )
    pointcloud_dropdown = gr.Dropdown(
        choices=context.pointcloud_names or [""],
        value=context.default_pointcloud,
        label="Pointcloud",
    )
    return main + [slider, camera_dropdown, pointcloud_dropdown]


def update_step(index: int, camera_name: str, pointcloud_name: str):
    if context is None:
        raise gr.Error("Select a recording first.")
    return context.update(index=int(index), camera_name=camera_name, pointcloud_name=pointcloud_name)


def open_interactive_pointcloud(index: int, pointcloud_name: str):
    if context is None:
        raise gr.Error("Select a recording first.")
    sensor_name = pointcloud_name or context.default_pointcloud
    resolved_step = context.resolve_pointcloud_step(sensor_name, context.index_to_step(index))
    return launch_interactive_pointcloud(
        recording_path=context.recording_path,
        pointcloud_name=sensor_name,
        step=resolved_step,
    )


def import_gradio():
    try:
        import gradio as _gr
    except Exception as exc:
        raise SystemExit(
            "Gradio is not installed in this environment. "
            "Install it with `python -m pip install gradio` or use the PatoSim environment."
        ) from exc
    return _gr


def build_demo():
    with gr.Blocks(title="PatoSim - Data Explorer") as demo:
        gr.Markdown("# PatoSim - Data Explorer")
        with gr.Row():
            recording_selector = gr.Dropdown(
                label=f"Recording ({os.path.realpath(directory)})",
                choices=recording_names,
                value=recording_names[0] if recording_names else None,
            )
            camera_selector = gr.Dropdown(label="Camera", choices=[], value=None)
            pointcloud_selector = gr.Dropdown(label="Pointcloud", choices=[], value=None)
        slider = gr.Slider(label="Timestep", minimum=0, maximum=0, step=1, value=0)

        with gr.Row():
            map_plot = gr.Plot(label="Occupancy Map / Trajectory")
            pointcloud_plot = gr.Plot(label="Pointcloud Top-Down")

        with gr.Row():
            pointcloud_3d_image = gr.Image(label="Pointcloud 3D (Open3D)", type="numpy")

        with gr.Row():
            open_pointcloud_button = gr.Button("Open Interactive Pointcloud")
        pointcloud_status = gr.Markdown("Interactive pointcloud window is closed.")

        with gr.Row():
            camera_panel = gr.Plot(label="Camera Panel")

        with gr.Row():
            camera_annotation_panel = gr.Plot(label="Camera Annotation")

        with gr.Row():
            camera_bbox_panel = gr.Plot(label="RGB + 2D Bounding Boxes")

        summary_markdown = gr.Markdown()

        update_widgets = [
            map_plot,
            camera_panel,
            camera_annotation_panel,
            camera_bbox_panel,
            pointcloud_plot,
            pointcloud_3d_image,
            summary_markdown,
        ]
        selector_widgets = [slider, camera_selector, pointcloud_selector]

        recording_selector.change(
            update_recording,
            inputs=recording_selector,
            outputs=update_widgets + selector_widgets,
        )
        slider.change(
            update_step,
            inputs=[slider, camera_selector, pointcloud_selector],
            outputs=update_widgets,
            show_progress=False,
        )
        camera_selector.change(
            update_step,
            inputs=[slider, camera_selector, pointcloud_selector],
            outputs=update_widgets,
            show_progress=False,
        )
        pointcloud_selector.change(
            update_step,
            inputs=[slider, camera_selector, pointcloud_selector],
            outputs=update_widgets,
            show_progress=False,
        )
        open_pointcloud_button.click(
            open_interactive_pointcloud,
            inputs=[slider, pointcloud_selector],
            outputs=pointcloud_status,
            show_progress=False,
        )
    return demo


if __name__ == "__main__":
    gr = import_gradio()
    demo = build_demo()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
