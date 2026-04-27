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
import PIL.Image
import numpy as np
import shutil
import json

try:
    import open3d as o3d
except Exception:
    o3d = None

from omni.ext.patosim.config import Config
from omni.ext.patosim.occupancy_map import OccupancyMap


class Writer:
    """Persist sensor outputs and scene metadata using the replay layout.

    The writer keeps each modality in its own folder under ``state/`` so
    lightweight online recording and heavier offline replay passes can reuse
    the same directory structure.
    """

    def __init__(self, path: str):
        self.path = path

    def write_state_dict_common(self, state_dict: dict, step: int):
        # ``common`` is the lightweight per-step snapshot: poses, controls and
        # scalar state that can be recorded frequently without render cost.
        dict_folder = os.path.join(self.path, "state", "common")
        if not os.path.exists(dict_folder):
            os.makedirs(dict_folder)
        state_dict_path = os.path.join(dict_folder, f"{step:08d}.npy")
        np.save(state_dict_path, state_dict)

    def write_state_dict_rgb(self, state_rgb: dict, step: int, sonar_ref=None):
        has_sonar_rgb = False
        for name, value in state_rgb.items():
            if value is not None:
                if "sonar" in str(name).lower():
                    has_sonar_rgb = True
                image_folder = os.path.join(self.path, "state", "rgb", name)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, f"{step:08d}.jpg")
                image = PIL.Image.fromarray(value)
                image.save(image_path)
        if sonar_ref is None or not has_sonar_rgb:
            return
        try:
            preview = sonar_ref.render_polar_preview(size=512)
        except Exception:
            preview = None
        if preview is None:
            return
        preview_np = np.asarray(preview, dtype=np.uint8)
        output_folder = os.path.join(self.path, "state", "rgb", "sonar_polar")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, f"{step:08d}.jpg")
        try:
            import cv2

            preview_bgr = cv2.cvtColor(preview_np, cv2.COLOR_RGBA2BGR)
            ok, encoded = cv2.imencode(".jpg", preview_bgr)
            if ok:
                with open(output_path, "wb") as handle:
                    handle.write(encoded.tobytes())
                return
        except Exception:
            pass
        PIL.Image.fromarray(preview_np[..., :3]).save(output_path)

    def write_state_dict_segmentation(self, state_segmentation: dict, step: int):
        for name, value in state_segmentation.items():
            if value is not None:
                image_folder = os.path.join(self.path, "state", "segmentation", name)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, f"{step:08d}.png")
                image = PIL.Image.fromarray(value)
                image.save(image_path)

    def write_state_dict_instance_id_segmentation(self, state_segmentation: dict, step: int):
        for name, value in state_segmentation.items():
            if value is not None:
                image_folder = os.path.join(self.path, "state", "instance_id_segmentation", name)
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, f"{step:08d}.png")
                image = PIL.Image.fromarray(value)
                image.save(image_path)

    def write_state_dict_depth(self, state_np: dict, step: int):
        for name, value in state_np.items():
            if value is not None:
                output_folder = os.path.join(self.path, "state", "depth", name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Depth is stored as inverse-depth in 16-bit PNG to keep files
                # compact while remaining easy to visualize and decode later.
                inverse_depth = 1.0 / (1.0 + value)
                inverse_depth = (65535 * inverse_depth).astype(np.uint16)
                image = PIL.Image.fromarray(inverse_depth, "I;16")
                
                output_path = os.path.join(output_folder, f"{step:08d}.png")

                image.save(output_path)

    def write_state_dict_normals(self, state_np: dict, step: int):
        for name, value in state_np.items():
            if value is not None:
                output_folder = os.path.join(self.path, "state", "normals", name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_path = os.path.join(output_folder, f"{step:08d}.npy")
                np.save(output_path, value)

    def _save_ply(self, points: np.ndarray, path: str):
        """Write a simple ASCII PLY file with XYZ (and optional intensity).

        This is a minimal writer to avoid adding a heavy dependency. It writes
        an ASCII PLY with vertex properties x y z [intensity].
        """
        if points is None:
            return
        # The project normalizes pointcloud buffers to Nx3/Nx4/Nx6/Nx7, where
        # extra columns may encode intensity and/or RGB.
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points must be Nx3 or Nx4")

        # Prefer Open3D when available: it handles binary pointcloud formats
        # efficiently and preserves RGB visualization nicely.
        if o3d is not None:
            try:
                # build point cloud
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
                if pts.shape[1] >= 6:
                    # rgb in cols 3-5
                    rgb = pts[:, 3:6]
                    rgb = np.asarray(rgb)
                    # normalize to 0..1 if necessary
                    if rgb.max() > 1.0:
                        rgb = rgb / 255.0
                    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
                elif pts.shape[1] == 4:
                    # intensity only: map to grayscale color
                    intensity = pts[:, 3]
                    intensity = np.asarray(intensity).astype(np.float64)
                    if intensity.max() > 1.0:
                        intensity = intensity / (intensity.max() + 1e-6)
                    gray = np.stack([intensity, intensity, intensity], axis=1)
                    pc.colors = o3d.utility.Vector3dVector(gray)

                # write binary PLY by default
                o3d.io.write_point_cloud(path, pc, write_ascii=False)
                return
            except Exception:
                # fallback to ASCII writer below
                pass

        # Fallback path keeps replay export working even on machines without
        # Open3D. This path is slower and simpler, but dependency-free.
        with open(path, "w") as f:
            header = [
                "ply",
                "format ascii 1.0",
                f"element vertex {pts.shape[0]}",
                "property float x",
                "property float y",
                "property float z",
            ]
            # support intensity (4) or rgb (6) or rgb+intensity (7)
            if pts.shape[1] == 4:
                header.append("property float intensity")
            elif pts.shape[1] == 6:
                header.extend(["property uchar red", "property uchar green", "property uchar blue"])
            elif pts.shape[1] >= 7:
                header.extend(["property uchar red", "property uchar green", "property uchar blue", "property float intensity"])
            header.append("end_header")
            f.write("\n".join(header) + "\n")
            for p in pts:
                if pts.shape[1] == 4:
                    f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")
                elif pts.shape[1] == 6:
                    # rgb may be floats [0-1] or ints [0-255]
                    r, g, b = p[3], p[4], p[5]
                    if float(r) <= 1.0 and float(g) <= 1.0 and float(b) <= 1.0:
                        r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    f.write(f"{p[0]} {p[1]} {p[2]} {int(r)} {int(g)} {int(b)}\n")
                elif pts.shape[1] >= 7:
                    r, g, b = p[3], p[4], p[5]
                    if float(r) <= 1.0 and float(g) <= 1.0 and float(b) <= 1.0:
                        r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    f.write(f"{p[0]} {p[1]} {p[2]} {int(r)} {int(g)} {int(b)} {p[6]}\n")
                else:
                    f.write(f"{p[0]} {p[1]} {p[2]}\n")

    def write_annotations(self, annotations: dict, step: int):
        """Write annotations into split folders under state/.

        Output layout:
          - state/bboxes2d/<step>.json
          - state/bboxes3d/<step>.json
          - state/classes/<step>.json
          - state/semantic/<step>.json
        """
        payload = annotations if isinstance(annotations, dict) else {}
        self._write_json_state_entry("bboxes2d", payload.get("bboxes2d", []), step)
        self._write_json_state_entry("bboxes3d", payload.get("bboxes3d", []), step)
        self._write_json_state_entry("classes", payload.get("classes", []), step)
        self._write_json_state_entry("semantic", payload.get("semantic", {}), step)

    def _write_json_state_entry(self, folder_name: str, data, step: int):
        # Keeping each annotation family in its own folder makes staged replay
        # simpler: passes can regenerate only what is missing.
        folder = os.path.join(self.path, "state", folder_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        output_path = os.path.join(folder, f"{step:08d}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def write_pointcloud_metadata(self, metadata: dict, step: int):
        """Write per-sensor metadata for pointclouds.

        metadata is expected to be a dict mapping sensor full names to a small
        dict with keys like 'position' and 'orientation'. We write each
        sensor's metadata next to the pointcloud files as
        state/pointcloud/<sensor_name>/<step>_meta.json
        """
        for sensor_name, meta in metadata.items():
            output_folder = os.path.join(self.path, "state", "pointcloud", sensor_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            meta_path = os.path.join(output_folder, f"{step:08d}_meta.json")
            try:
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                # Best-effort: ignore failures
                pass

    def write_state_dict_pointcloud(self, state_pc: dict, step: int, save_format: str = "npy"):
        """Write point-cloud data to disk.

        Each point-cloud value is expected to be a Nx3 or Nx4 numpy array.
        By default we save as .npy; set save_format to 'ply' to write ASCII
        PLY files alongside the .npy (or instead).
        """
        for name, value in state_pc.items():
            if value is not None:
                output_folder = os.path.join(self.path, "state", "pointcloud", name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                if save_format == "npy":
                    # ``.npy`` is the most faithful format for replay because it
                    # preserves arbitrary extra columns without reinterpretation.
                    output_path = os.path.join(output_folder, f"{step:08d}.npy")
                    np.save(output_path, value)
                elif save_format == "ply":
                    output_path = os.path.join(output_folder, f"{step:08d}.ply")
                    try:
                        # if open3d is available, _save_ply will write binary PLY
                        self._save_ply(value, output_path)
                    except Exception:
                        # Fallback: also save raw npy
                        np.save(os.path.join(output_folder, f"{step:08d}.npy"), value)
                elif save_format == "pcd":
                    output_path = os.path.join(output_folder, f"{step:08d}.pcd")
                    try:
                        if o3d is not None:
                            pc = o3d.geometry.PointCloud()
                            pc.points = o3d.utility.Vector3dVector(np.asarray(value)[:, :3].astype(np.float64))
                            # Keep PCD export aligned with the same conventions
                            # used for PLY: RGB if present, otherwise grayscale
                            # from intensity for quick inspection.
                            if np.asarray(value).shape[1] >= 6:
                                rgb = np.asarray(value)[:, 3:6]
                                if rgb.max() > 1.0:
                                    rgb = rgb / 255.0
                                pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
                            elif np.asarray(value).shape[1] == 4:
                                intensity = np.asarray(value)[:, 3]
                                if intensity.max() > 1.0:
                                    intensity = intensity / (intensity.max() + 1e-6)
                                gray = np.stack([intensity, intensity, intensity], axis=1)
                                pc.colors = o3d.utility.Vector3dVector(gray)
                            o3d.io.write_point_cloud(output_path, pc, write_ascii=False)
                        else:
                            # fallback to PLY ASCII if open3d not available
                            self._save_ply(value, os.path.join(output_folder, f"{step:08d}.ply"))
                    except Exception:
                        np.save(os.path.join(output_folder, f"{step:08d}.npy"), value)
                else:
                    # Unknown formats fall back to ``.npy`` so recording never
                    # silently drops a pointcloud.
                    output_path = os.path.join(output_folder, f"{step:08d}.npy")
                    np.save(output_path, value)

    def write_stage(self):
        from omni.ext.patosim.utils.global_utils import save_stage

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        save_stage(
            os.path.join(self.path, "stage.usd")
        )

    def copy_stage(self, input_path: str):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        shutil.copyfile(input_path, os.path.join(self.path, "stage.usd"))

    def write_config(self, config: Config):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open(os.path.join(self.path, "config.json"), 'w') as f:
            f.write(config.to_json())

    def write_occupancy_map(self, occupancy_map: OccupancyMap):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        occupancy_map.save_ros(os.path.join(self.path, "occupancy_map"))

    def copy_init(self, other_path: str, overwrite: bool = False, verbose: bool = False):
        """Copy initial artifacts (stage/config/occupancy_map) from another recording.

        Args:
            other_path: source recording directory
            overwrite: if True, remove destination path before copying
            verbose: if True, print progress messages
        """
        if overwrite and os.path.exists(self.path):
            try:
                if verbose:
                    print(f"[Writer] Overwrite requested: removing existing output path {self.path}")
                shutil.rmtree(self.path)
            except Exception:
                # best-effort: continue
                pass

        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:
                pass

        # These artifacts define the replay context independently from the
        # per-step sensor data, so they can be copied once and reused.
        try:
            src_stage = os.path.join(other_path, "stage.usd")
            dst_stage = os.path.join(self.path, "stage.usd")
            if os.path.exists(src_stage):
                shutil.copyfile(src_stage, dst_stage)
                if verbose:
                    print(f"[Writer] Copied stage.usd to {dst_stage}")
        except Exception:
            # best-effort
            if verbose:
                print(f"[Writer] Failed to copy stage.usd from {other_path}")

        try:
            src_cfg = os.path.join(other_path, "config.json")
            dst_cfg = os.path.join(self.path, "config.json")
            if os.path.exists(src_cfg):
                shutil.copyfile(src_cfg, dst_cfg)
                if verbose:
                    print(f"[Writer] Copied config.json to {dst_cfg}")
        except Exception:
            if verbose:
                print(f"[Writer] Failed to copy config.json from {other_path}")

        # ``occupancy_map`` is directory-shaped and may already exist in a
        # staged replay output, so copying is intentionally merge-friendly.
        try:
            src_occ = os.path.join(other_path, "occupancy_map")
            dst_occ = os.path.join(self.path, "occupancy_map")
            if os.path.exists(src_occ):
                if os.path.exists(dst_occ):
                    if verbose:
                        print(f"[Writer] Merging occupancy_map from {src_occ} into existing {dst_occ}")
                    # merge files from src into dst (overwrite existing files)
                    for root, dirs, files in os.walk(src_occ):
                        rel = os.path.relpath(root, src_occ)
                        dest_dir = os.path.join(dst_occ, rel) if rel != '.' else dst_occ
                        os.makedirs(dest_dir, exist_ok=True)
                        for f in files:
                            try:
                                shutil.copyfile(os.path.join(root, f), os.path.join(dest_dir, f))
                                if verbose:
                                    print(f"[Writer] Copied {os.path.join(root, f)} -> {os.path.join(dest_dir, f)}")
                            except Exception:
                                # ignore individual file copy failures
                                if verbose:
                                    print(f"[Writer] Failed to copy {os.path.join(root, f)}")
                                pass
                else:
                    # typical case: simply copy the whole tree
                    try:
                        shutil.copytree(src_occ, dst_occ)
                        if verbose:
                            print(f"[Writer] Copied occupancy_map tree to {dst_occ}")
                    except Exception:
                        # fallback to manual copy
                        if verbose:
                            print(f"[Writer] copytree failed, falling back to manual copy for occupancy_map")
                        for root, dirs, files in os.walk(src_occ):
                            rel = os.path.relpath(root, src_occ)
                            dest_dir = os.path.join(dst_occ, rel) if rel != '.' else dst_occ
                            os.makedirs(dest_dir, exist_ok=True)
                            for f in files:
                                try:
                                    shutil.copyfile(os.path.join(root, f), os.path.join(dest_dir, f))
                                    if verbose:
                                        print(f"[Writer] Copied {os.path.join(root, f)} -> {os.path.join(dest_dir, f)}")
                                except Exception:
                                    if verbose:
                                        print(f"[Writer] Failed to copy {os.path.join(root, f)}")
                                    pass
        except Exception:
            if verbose:
                print(f"[Writer] Error while copying occupancy_map from {other_path}")
