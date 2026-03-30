#!/usr/bin/env python3
"""
Validate replay outputs without launching Isaac Sim.

Checks:
- common state files exist
- pointcloud files are present and contain relevant data
- split annotation folders exist when requested
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


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
    parser.add_argument("--path", type=str, default=os.path.join(DATA_DIR, "replays"))
    parser.add_argument("--pc_min_points", type=int, default=32)
    parser.add_argument("--pc_min_extent", type=float, default=0.05)
    parser.add_argument("--pc_require_spread", type=parse_bool, default=True)
    parser.add_argument("--require_annotations", type=parse_bool, default=True)
    return parser.parse_args()


def resolve_replay_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isdir(os.path.join(path, "state")):
        return path
    candidates = sorted(
        [p for p in glob.glob(os.path.join(path, "*")) if os.path.isdir(os.path.join(p, "state"))],
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No replay directory found in: {path}")
    return candidates[-1]


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


def analyze_pointcloud_array(
    value: Any,
    min_points: int = 32,
    min_extent: float = 0.05,
    require_spread: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {
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


def main() -> int:
    args = parse_args()
    replay_path = resolve_replay_path(args.path)

    bootstrap_repo_paths()
    from omni.ext.patosim.reader import Reader

    reader = Reader(replay_path)
    common_steps = len(reader.steps)
    print(f"[validate_replay] Replay: {replay_path}")
    print(f"[validate_replay] Common steps: {common_steps}")
    if common_steps == 0:
        print("[validate_replay] No common state files found.")
        return 1

    invalid_pc = []
    valid_pc_entries = 0
    for index, step in enumerate(reader.steps):
        pc_state = reader.read_state_dict_pointcloud(index)
        for name, value in pc_state.items():
            info = analyze_pointcloud_array(
                value,
                min_points=args.pc_min_points,
                min_extent=args.pc_min_extent,
                require_spread=args.pc_require_spread,
            )
            if info["valid"]:
                valid_pc_entries += 1
            else:
                invalid_pc.append((step, name, info["reason"], info["num_points"], info["extent"]))

    print(f"[validate_replay] Valid pointcloud entries: {valid_pc_entries}")
    if invalid_pc:
        print("[validate_replay] Invalid pointcloud entries:")
        for step, name, reason, num_points, extent in invalid_pc[:50]:
            print(
                f"  step={step:08d} sensor={name} reason={reason} "
                f"points={num_points} extent={extent:.4f}"
            )

    annotation_files = {
        "bboxes2d": glob.glob(os.path.join(replay_path, "state", "bboxes2d", "*.json")),
        "bboxes3d": glob.glob(os.path.join(replay_path, "state", "bboxes3d", "*.json")),
        "classes": glob.glob(os.path.join(replay_path, "state", "classes", "*.json")),
        "semantic": glob.glob(os.path.join(replay_path, "state", "semantic", "*.json")),
    }
    for key, files in annotation_files.items():
        print(f"[validate_replay] {key}: {len(files)} files")

    if args.require_annotations:
        if not all(annotation_files.values()):
            print("[validate_replay] Missing one or more split annotation folders/files.")
            return 1

    if invalid_pc:
        print("[validate_replay] Replay validation failed due to invalid pointcloud entries.")
        return 1

    print("[validate_replay] Replay outputs look consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
