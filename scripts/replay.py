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
Replay and render a single recording with Isaac Sim python runner.
"""

import argparse
import glob
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


if "MOBILITY_GEN_DATA" in os.environ:
    DATA_DIR = os.environ["MOBILITY_GEN_DATA"]
else:
    DATA_DIR = os.path.expanduser("~/MobilityGenData")


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
    parser.add_argument("--input", type=str, default=os.path.join(DATA_DIR, "recordings"))
    parser.add_argument("--output", type=str, default=os.path.join(DATA_DIR, "replays"))
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
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--verbose", type=parse_bool, default=False)
    args = parser.parse_args()
    return args


def build_isaac_env() -> dict:
    env = os.environ.copy()
    for key in (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "CONDA_EXE",
        "CONDA_PYTHON_EXE",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def warn_if_inotify_pressure(prefix: str) -> None:
    limit_path = Path("/proc/sys/fs/inotify/max_user_watches")
    proc_dir = Path("/proc")
    if not limit_path.exists() or not proc_dir.exists():
        return

    try:
        limit = int(limit_path.read_text(encoding="utf-8").strip())
    except Exception:
        return
    if limit <= 0:
        return

    usage = 0
    try:
        for pid_dir in proc_dir.iterdir():
            if not pid_dir.name.isdigit():
                continue
            fdinfo_dir = pid_dir / "fdinfo"
            if not fdinfo_dir.is_dir():
                continue
            for fdinfo in fdinfo_dir.iterdir():
                try:
                    text = fdinfo.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if "inotify" not in text:
                    continue
                usage += text.count("\ninotify ")
                if text.startswith("inotify "):
                    usage += 1
    except Exception:
        return

    if usage < int(limit * 0.9):
        return

    print(
        f"{prefix} Warning: inotify watches near/exceeded limit ({usage}/{limit}). "
        "Isaac may fail with errno=28 (No space left on device)."
    )
    print(
        f"{prefix} Tip: stop high-watch processes (for example tracker miner) and/or "
        "raise fs.inotify.max_user_watches."
    )


def main() -> int:
    args = parse_args()

    recording_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)

    if os.path.isdir(recording_path) and os.path.isdir(os.path.join(recording_path, "state")) is False:
        candidates = sorted(
            [p for p in glob.glob(os.path.join(recording_path, "*")) if os.path.isdir(p)],
            key=os.path.getmtime,
        )
        if not candidates:
            print(f"[replay] No recording directories found in: {recording_path}")
            return 1
        recording_path = candidates[-1]

    name = os.path.basename(recording_path)
    if output_path.endswith("/replays") or output_path.endswith(os.path.sep + "replays") or (
        os.path.isdir(output_path) and os.path.isdir(os.path.join(output_path, "state")) is False
    ):
        output_path = os.path.join(output_path, name)

    if not os.path.isdir(recording_path):
        print(f"[replay] Input recording path not found: {recording_path}")
        return 1

    warn_if_inotify_pressure("[replay]")

    repo_root = Path(__file__).resolve().parents[1]
    runner_path = repo_root / "app" / "python.sh"
    replay_impl_path = repo_root / "scripts" / "replay_implementation.py"

    if not runner_path.exists():
        print(f"[replay] Isaac python runner not found: {runner_path}")
        print("[replay] Run link_app.sh first and ensure app/python.sh exists.")
        return 1

    input_steps = len(glob.glob(os.path.join(recording_path, "state", "common", "*.npy")))
    output_steps = len(glob.glob(os.path.join(output_path, "state", "common", "*.npy")))

    if input_steps == output_steps:
        print(f"Skipping {name} as it is already replayed")
        return 0

    print(f"Replaying {name}")
    cmd = [
        str(runner_path),
        str(replay_impl_path),
        "--input_path",
        recording_path,
        "--output_path",
        output_path,
        "--render_interval",
        str(args.render_interval),
        "--render_rt_subframes",
        str(args.render_rt_subframes),
        "--rgb_enabled",
        str(args.rgb_enabled),
        "--segmentation_enabled",
        str(args.segmentation_enabled),
        "--instance_id_segmentation_enabled",
        str(args.instance_id_segmentation_enabled),
        "--normals_enabled",
        str(args.normals_enabled),
        "--depth_enabled",
        str(args.depth_enabled),
        "--pc_enabled",
        str(args.pc_enabled),
        "--pc_format",
        str(args.pc_format),
        "--annotations_enabled",
        str(args.annotations_enabled),
        "--pc_interval",
        str(args.pc_interval),
        "--overwrite",
        str(args.overwrite),
        "--verbose",
        str(args.verbose),
    ]

    try:
        subprocess.run(cmd, cwd=str(repo_root), check=True, env=build_isaac_env())
    except KeyboardInterrupt:
        print("\n[replay] Interrupted by user (Ctrl+C).")
        return 130

    output_steps_after = len(glob.glob(os.path.join(output_path, "state", "common", "*.npy")))
    if input_steps > 0 and output_steps_after == 0:
        print(
            "[replay] Replay finished without generating state/common files. "
            "Check Isaac logs for extension/module errors."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
