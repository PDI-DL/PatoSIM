#!/usr/bin/env python3
"""
Visualize PatoSim pointcloud recordings with Open3D.

Examples:
  python3 scripts/view_pointclouds.py \
    --path ~/PatoSimData/replays/2026-03-04T15:29:33.130987 \
    --list

  python3 scripts/view_pointclouds.py \
    --path ~/PatoSimData/replays/2026-03-04T15:29:33.130987 \
    --sensor robot.lidar.pointcloud \
    --step latest

  python3 scripts/view_pointclouds.py \
    --path ~/PatoSimData/replays/2026-03-04T15:29:33.130987 \
    --sensor robot.lidar.pointcloud \
    --play --fps 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


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
    parser = argparse.ArgumentParser(description="View PatoSim pointclouds with Open3D.")
    parser.add_argument("--path", required=True, help="Recording/replay directory or direct state/pointcloud path.")
    parser.add_argument("--sensor", default="", help="Sensor folder name inside state/pointcloud.")
    parser.add_argument("--step", default="latest", help="Step to visualize: integer, latest, or first.")
    parser.add_argument("--play", action="store_true", help="Play all frames for the selected sensor.")
    parser.add_argument("--fps", type=float, default=8.0, help="Playback FPS when --play is enabled.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth point.")
    parser.add_argument("--max_points", type=int, default=120000, help="Randomly subsample to at most this many points.")
    parser.add_argument("--point_size", type=float, default=2.0, help="Open3D point size.")
    parser.add_argument("--show_frame", action="store_true", help="Show coordinate frame at the origin.")
    parser.add_argument(
        "--hide_bboxes3d",
        action="store_true",
        help="Do not overlay 3D bounding boxes from state/bboxes3d.",
    )
    parser.add_argument("--list", action="store_true", help="List available sensors and steps, then exit.")
    return parser.parse_args()


def require_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Open3D is required for visualization. Install it in the same Python "
            f"environment used to run the script. Original import error: {exc}"
        )


def resolve_pointcloud_root(path_str: str) -> Path:
    raw = Path(os.path.expanduser(path_str)).resolve()
    if raw.name == "pointcloud" and raw.is_dir():
        return raw
    candidate = raw / "state" / "pointcloud"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Could not find pointcloud folder from '{raw}'. Expected either a direct "
        "state/pointcloud path or a recording/replay directory containing it."
    )


def resolve_recording_root_from_pointcloud_root(root: Path) -> Optional[Path]:
    try:
        if root.name == "pointcloud" and root.parent.name == "state":
            return root.parent.parent
    except Exception:
        return None
    return None


def list_sensors(root: Path) -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_sensor_steps(sensor_dir: Path) -> list[int]:
    steps: set[int] = set()
    for ext in (".npy", ".ply", ".pcd"):
        for path in sensor_dir.glob(f"*{ext}"):
            stem = path.stem
            if stem.isdigit():
                steps.add(int(stem))
    return sorted(steps)


def resolve_sensor(root: Path, sensor_name: str) -> Path:
    sensors = list_sensors(root)
    if not sensors:
        raise RuntimeError(f"No sensor folders found in '{root}'.")
    if sensor_name:
        sensor_dir = root / sensor_name
        if not sensor_dir.is_dir():
            raise FileNotFoundError(
                f"Sensor '{sensor_name}' not found in '{root}'. Available: {', '.join(sensors)}"
            )
        return sensor_dir
    if len(sensors) == 1:
        return root / sensors[0]
    raise RuntimeError(
        "Multiple pointcloud sensors found. Please pass --sensor. "
        f"Available: {', '.join(sensors)}"
    )


def resolve_step(steps: list[int], requested: str) -> int:
    if not steps:
        raise RuntimeError("No pointcloud steps were found for the selected sensor.")
    text = str(requested).strip().lower()
    if text == "latest":
        return steps[-1]
    if text == "first":
        return steps[0]
    value = int(text)
    if value not in steps:
        raise ValueError(f"Step {value} not available. Available range: {steps[0]}..{steps[-1]}")
    return value


def find_pointcloud_file(sensor_dir: Path, step: int) -> Path:
    for ext in (".npy", ".ply", ".pcd"):
        path = sensor_dir / f"{step:08d}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No pointcloud file found for step {step:08d} in '{sensor_dir}'.")


def find_metadata_file(sensor_dir: Path, step: int) -> Optional[Path]:
    path = sensor_dir / f"{step:08d}_meta.json"
    return path if path.exists() else None


def load_pointcloud_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=True)
        return np.asarray(array)

    o3d = require_open3d()
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if cloud.has_colors():
        colors = np.asarray(cloud.colors)
        if colors.size and colors.max() <= 1.0:
            colors = colors * 255.0
        return np.hstack([points, colors.astype(np.float32)])
    return points.astype(np.float32)


def reduce_points(points: np.ndarray, stride: int, max_points: int) -> np.ndarray:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Invalid pointcloud shape {pts.shape}; expected Nx3+.")
    if stride > 1:
        pts = pts[::stride]
    if max_points > 0 and len(pts) > max_points:
        idx = np.random.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts


def points_to_open3d(points: np.ndarray):
    o3d = require_open3d()
    pts = np.asarray(points, dtype=np.float32)
    cloud = o3d.geometry.PointCloud()
    if pts.size == 0:
        cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        return cloud

    cloud.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
    if pts.shape[1] >= 6:
        colors = pts[:, 3:6].astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    elif pts.shape[1] >= 4:
        intensity = pts[:, 3].astype(np.float64)
        if intensity.size and intensity.max() > 1.0:
            intensity = intensity / (float(intensity.max()) + 1e-9)
        gray = np.stack([intensity, intensity, intensity], axis=1)
        cloud.colors = o3d.utility.Vector3dVector(np.clip(gray, 0.0, 1.0))
    return cloud


def load_metadata(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None


def load_bboxes3d(recording_root: Optional[Path], step: int) -> list[dict]:
    if recording_root is None:
        return []
    path = recording_root / "state" / "bboxes3d" / f"{int(step):08d}.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


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
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    idx = sum(ord(ch) for ch in str(label)) % 20
    color = cmap(idx)
    return float(color[0]), float(color[1]), float(color[2])


def collect_bbox_line_sets(recording_root: Optional[Path], step: int, metadata: Optional[dict]):
    o3d = require_open3d()
    geometries = []
    for entry in load_bboxes3d(recording_root, step):
        corners = np.asarray(entry.get("corners"), dtype=np.float32)
        if corners.shape != (8, 3):
            continue
        local_corners = transform_world_points_to_sensor(corners, metadata)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(local_corners.astype(np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(BOX_EDGE_INDEXES, dtype=np.int32))
        color = np.asarray(label_color_rgb01(entry.get("class", "object")), dtype=np.float64)
        colors = np.repeat(color.reshape(1, 3), len(BOX_EDGE_INDEXES), axis=0)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)
    return geometries
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def print_listing(root: Path) -> None:
    sensors = list_sensors(root)
    if not sensors:
        print(f"No pointcloud sensors found in {root}")
        return
    print(f"Pointcloud root: {root}")
    for sensor in sensors:
        sensor_dir = root / sensor
        steps = list_sensor_steps(sensor_dir)
        if not steps:
            print(f"- {sensor}: no steps")
            continue
        print(f"- {sensor}: {len(steps)} step(s), first={steps[0]:08d}, last={steps[-1]:08d}")


def visualize_single(
    recording_root: Optional[Path],
    sensor_dir: Path,
    sensor_name: str,
    step: int,
    stride: int,
    max_points: int,
    point_size: float,
    show_frame: bool,
    show_bboxes3d: bool,
) -> None:
    pointcloud_path = find_pointcloud_file(sensor_dir, step)
    metadata = load_metadata(find_metadata_file(sensor_dir, step))
    points = reduce_points(load_pointcloud_array(pointcloud_path), stride=stride, max_points=max_points)
    cloud = points_to_open3d(points)

    print(f"Sensor: {sensor_name}")
    print(f"Step: {step:08d}")
    print(f"File: {pointcloud_path}")
    print(f"Points shown: {len(points)}")
    if metadata:
        print("Metadata:")
        print(json.dumps(metadata, indent=2))

    o3d = require_open3d()
    geometries = [cloud]
    if show_bboxes3d:
        geometries.extend(collect_bbox_line_sets(recording_root, step, metadata))
    if show_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.75))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"PatoSim PointCloud - {sensor_name} - {step:08d}")
    for geometry in geometries:
        vis.add_geometry(geometry)
    render = vis.get_render_option()
    render.point_size = float(point_size)
    vis.run()
    vis.destroy_window()


def iter_step_files(sensor_dir: Path, steps: Iterable[int]) -> Iterable[tuple[int, Path, Optional[Path]]]:
    for step in steps:
        yield step, find_pointcloud_file(sensor_dir, step), find_metadata_file(sensor_dir, step)


def visualize_sequence(
    recording_root: Optional[Path],
    sensor_dir: Path,
    sensor_name: str,
    steps: list[int],
    stride: int,
    max_points: int,
    point_size: float,
    show_frame: bool,
    fps: float,
    show_bboxes3d: bool,
) -> None:
    o3d = require_open3d()
    if not steps:
        raise RuntimeError("No steps available for playback.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"PatoSim PointCloud Playback - {sensor_name}")
    render = vis.get_render_option()
    render.point_size = float(point_size)

    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.75) if show_frame else None
    point_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud)
    if frame_mesh is not None:
        vis.add_geometry(frame_mesh)
    bbox_geometries = []

    dt = 1.0 / max(0.1, float(fps))
    print(f"Playing {len(steps)} steps for sensor '{sensor_name}' at {fps:.2f} FPS")
    for step, pc_path, meta_path in iter_step_files(sensor_dir, steps):
        pts = reduce_points(load_pointcloud_array(pc_path), stride=stride, max_points=max_points)
        current = points_to_open3d(pts)
        point_cloud.points = current.points
        point_cloud.colors = current.colors
        vis.update_geometry(point_cloud)
        for geometry in bbox_geometries:
            vis.remove_geometry(geometry, reset_bounding_box=False)
        bbox_geometries = []
        if frame_mesh is not None:
            vis.update_geometry(frame_mesh)
        meta = load_metadata(meta_path)
        if show_bboxes3d:
            bbox_geometries = collect_bbox_line_sets(recording_root, step, meta)
            for geometry in bbox_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
                vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        print(
            f"\rstep={step:08d} points={len(pts):6d} file={pc_path.name}"
            + (" meta=yes" if meta else " meta=no"),
            end="",
            flush=True,
        )
        time.sleep(dt)
    print()
    vis.run()
    vis.destroy_window()


def main() -> int:
    args = parse_args()
    root = resolve_pointcloud_root(args.path)
    recording_root = resolve_recording_root_from_pointcloud_root(root)

    if args.list:
        print_listing(root)
        return 0

    sensor_dir = resolve_sensor(root, args.sensor)
    sensor_name = sensor_dir.name
    steps = list_sensor_steps(sensor_dir)

    if args.play:
        visualize_sequence(
            recording_root=recording_root,
            sensor_dir=sensor_dir,
            sensor_name=sensor_name,
            steps=steps,
            stride=max(1, int(args.stride)),
            max_points=max(0, int(args.max_points)),
            point_size=float(args.point_size),
            show_frame=bool(args.show_frame),
            fps=float(args.fps),
            show_bboxes3d=not bool(args.hide_bboxes3d),
        )
        return 0

    step = resolve_step(steps, args.step)
    visualize_single(
        recording_root=recording_root,
        sensor_dir=sensor_dir,
        sensor_name=sensor_name,
        step=step,
        stride=max(1, int(args.stride)),
        max_points=max(0, int(args.max_points)),
        point_size=float(args.point_size),
        show_frame=bool(args.show_frame),
        show_bboxes3d=not bool(args.hide_bboxes3d),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
