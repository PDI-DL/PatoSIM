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


import PIL.Image
import glob
import numpy as np
import os
from collections import OrderedDict
import json
try:
    import open3d as o3d
except Exception:
    o3d = None


from omni.ext.patosim.occupancy_map import OccupancyMap
from omni.ext.patosim.config import Config


class Reader:
    """Read recordings stored in the writer's split-folder replay layout.

    ``common`` drives the canonical timeline. Other modalities may be dense or
    sparse, but are looked up using the same step numbers when available.
    """

    def __init__(self, recording_path: str):
        self.recording_path = recording_path
        
        state_dict_paths = glob.glob(os.path.join(
            self.recording_path, "state", "common", "*.npy"
        ))

        # ``state/common`` defines the step index used by the rest of the API.
        steps = [int(os.path.basename(path).split('.')[0]) for path in state_dict_paths]
        self.steps = sorted(steps)

        # Each sensor gets its own subfolder inside each modality. The reader
        # discovers them dynamically instead of relying on a hard-coded schema.
        self.rgb_folders = glob.glob(os.path.join(self.recording_path, "state", "rgb", "*"))
        self.segmentation_folders = glob.glob(os.path.join(self.recording_path, "state", "segmentation", "*"))
        self.instance_id_segmentation_folders = glob.glob(
            os.path.join(self.recording_path, "state", "instance_id_segmentation", "*")
        )
        self.depth_folders = glob.glob(os.path.join(self.recording_path, "state", "depth", "*"))
        self.normals_folders = glob.glob(os.path.join(self.recording_path, "state", "normals", "*"))
        self.pointcloud_folders = glob.glob(os.path.join(self.recording_path, "state", "pointcloud", "*"))
        self.bboxes2d_paths = glob.glob(os.path.join(self.recording_path, "state", "bboxes2d", "*.json"))
        self.bboxes3d_paths = glob.glob(os.path.join(self.recording_path, "state", "bboxes3d", "*.json"))
        self.classes_paths = glob.glob(os.path.join(self.recording_path, "state", "classes", "*.json"))
        self.semantic_paths = glob.glob(os.path.join(self.recording_path, "state", "semantic", "*.json"))
        self.annotation_paths = glob.glob(os.path.join(self.recording_path, "state", "annotations", "*.json"))

        self.rgb_names = [os.path.basename(folder) for folder in self.rgb_folders]
        self.segmentation_names = [os.path.basename(folder) for folder in self.segmentation_folders]
        self.instance_id_segmentation_names = [
            os.path.basename(folder) for folder in self.instance_id_segmentation_folders
        ]
        self.depth_names = [os.path.basename(folder) for folder in self.depth_folders]
        self.normals_names = [os.path.basename(folder) for folder in self.normals_folders]
        self.pointcloud_names = [os.path.basename(folder) for folder in self.pointcloud_folders]
        annotation_sources = (
            self.bboxes2d_paths
            + self.bboxes3d_paths
            + self.classes_paths
            + self.semantic_paths
            + self.annotation_paths
        )
        # Split annotation folders are the preferred layout, but ``annotations``
        # is still scanned for backward compatibility with older recordings.
        self.annotation_steps = sorted(
            {
                int(os.path.basename(path).split(".")[0])
                for path in annotation_sources
                if os.path.basename(path).split(".")[0].isdigit()
            }
        )

    def read_config(self) -> Config:
        with open(os.path.join(self.recording_path, "config.json"), 'r') as f:
            config = Config.from_json(f.read())
        return config

    def read_occupancy_map(self):
        return OccupancyMap.from_ros_yaml(os.path.join(self.recording_path, "occupancy_map", "map.yaml"))
    
    def read_rgb(self, name: str, index: int):
        step = self.steps[index]
        image = PIL.Image.open(os.path.join(self.recording_path, "state", "rgb", name, f"{step:08d}.jpg"))
        return np.asarray(image)
    
    def read_state_dict_rgb(self, index: int):
        rgb_dict = OrderedDict()
        for name in self.rgb_names:
            data = self.read_rgb(name, index)
            rgb_dict[name] = data
        return rgb_dict
    
    def read_segmentation(self, name: str, index: int):
        step = self.steps[index]
        image = PIL.Image.open(os.path.join(self.recording_path, "state", "segmentation", name, f"{step:08d}.png"))
        return np.asarray(image)
    
    def read_normals(self, name: str, index: int):
        step = self.steps[index]
        data = np.load(
            os.path.join(self.recording_path, "state", "normals", name, f"{step:08d}.npy")
        )
        return data

    def read_state_dict_segmentation(self, index: int):
        segmentation_dict = OrderedDict()
        for name in self.segmentation_names:
            data = self.read_segmentation(name, index)
            segmentation_dict[name] = data
        return segmentation_dict

    def read_instance_id_segmentation(self, name: str, index: int):
        step = self.steps[index]
        image = PIL.Image.open(
            os.path.join(
                self.recording_path,
                "state",
                "instance_id_segmentation",
                name,
                f"{step:08d}.png",
            )
        )
        return np.asarray(image)

    def read_state_dict_instance_id_segmentation(self, index: int):
        segmentation_dict = OrderedDict()
        for name in self.instance_id_segmentation_names:
            data = self.read_instance_id_segmentation(name, index)
            segmentation_dict[name] = data
        return segmentation_dict
    
    def read_depth(self, name: str, index: int, eps=1e-6):
        step = self.steps[index]
        image = PIL.Image.open(os.path.join(self.recording_path, "state", "depth", name, f"{step:08d}.png")).convert("I;16")
        # Inverse the compact encoding used by ``Writer.write_state_dict_depth``.
        depth = 65535 / (np.asarray(image).astype(np.float32) + eps) - 1.0
        return depth
    
    def read_state_dict_depth(self, index: int):
        depth_dict = OrderedDict()
        for name in self.depth_names:
            data = self.read_depth(name, index)
            depth_dict[name] = data
        return depth_dict

    def read_state_dict_normals(self, index: int):
        normals_dict = OrderedDict()
        for name in self.normals_names:
            data = self.read_normals(name, index)
            normals_dict[name] = data
        return normals_dict

    def read_pointcloud(self, name: str, index: int):
        step = self.steps[index]
        npy_path = os.path.join(self.recording_path, "state", "pointcloud", name, f"{step:08d}.npy")
        ply_path = os.path.join(self.recording_path, "state", "pointcloud", name, f"{step:08d}.ply")
        pcd_path = os.path.join(self.recording_path, "state", "pointcloud", name, f"{step:08d}.pcd")
        if os.path.exists(npy_path):
            return np.load(npy_path, allow_pickle=True)
        if os.path.exists(ply_path) or os.path.exists(pcd_path):
            # Prefer Open3D for robust binary PLY/PCD support when available.
            if o3d is not None:
                try:
                    read_path = ply_path if os.path.exists(ply_path) else pcd_path
                    pcd = o3d.io.read_point_cloud(read_path)
                    pts = np.asarray(pcd.points)
                    # Colors are expanded back into extra columns so downstream
                    # viewers can treat them similarly to numpy-exported clouds.
                    if pcd.has_colors():
                        cols = np.asarray(pcd.colors)
                        # Normalize back to the writer's common convention.
                        if cols.max() <= 1.0:
                            cols = (cols * 255.0).astype(np.float32)
                        pts = np.hstack([pts, cols])
                    return pts
                except Exception:
                    pass
            # Dependency-free fallback for simple ASCII PLY files.
            try:
                with open(ply_path, 'r') as f:
                    # skip header
                    line = f.readline()
                    while line and 'end_header' not in line:
                        line = f.readline()
                    # read the remaining lines as floats
                    data = np.loadtxt(f)
                    if data.ndim == 1:
                        data = data.reshape(-1, 3)
                    return data
            except Exception:
                return None
        return None

    def read_pointcloud_metadata(self, name: str, index: int):
        step = self.steps[index]
        meta_path = os.path.join(self.recording_path, "state", "pointcloud", name, f"{step:08d}_meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def read_state_dict_pointcloud(self, index: int):
        pc_dict = OrderedDict()
        for name in self.pointcloud_names:
            data = self.read_pointcloud(name, index)
            pc_dict[name] = data
        return pc_dict

    def read_state_dict_common(self, index: int):
        step = self.steps[index]
        state_dict = np.load(os.path.join(self.recording_path, "state", "common", f"{step:08d}.npy"), allow_pickle=True).item()
        return state_dict

    def read_annotations(self, index: int):
        step = self.steps[index]
        bboxes2d = self._read_json_state_entry("bboxes2d", step, default=[])
        bboxes3d = self._read_json_state_entry("bboxes3d", step, default=[])
        classes = self._read_json_state_entry("classes", step, default=[])
        semantic = self._read_json_state_entry("semantic", step, default={})

        # Prefer the split layout introduced for staged replay. If at least one
        # component exists, synthesize a unified annotation dict on the fly.
        if any(
            (
                bboxes2d not in (None, []),
                bboxes3d not in (None, []),
                classes not in (None, []),
                semantic not in (None, {}),
            )
        ):
            return {
                "step": int(step),
                "bboxes2d": bboxes2d if isinstance(bboxes2d, list) else [],
                "bboxes3d": bboxes3d if isinstance(bboxes3d, list) else [],
                "classes": classes if isinstance(classes, list) else [],
                "semantic": semantic if isinstance(semantic, dict) else {},
            }

        ann_path = os.path.join(self.recording_path, "state", "annotations", f"{step:08d}.json")
        if not os.path.exists(ann_path):
            return None
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _read_json_state_entry(self, folder_name: str, step: int, default=None):
        path = os.path.join(self.recording_path, "state", folder_name, f"{step:08d}.json")
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def read_bboxes2d(self, index: int):
        step = self.steps[index]
        data = self._read_json_state_entry("bboxes2d", step, default=None)
        if data is not None:
            return data
        annotations = self.read_annotations(index)
        if isinstance(annotations, dict):
            return annotations.get("bboxes2d", [])
        return []

    def read_bboxes3d(self, index: int):
        step = self.steps[index]
        data = self._read_json_state_entry("bboxes3d", step, default=None)
        if data is not None:
            return data
        annotations = self.read_annotations(index)
        if isinstance(annotations, dict):
            return annotations.get("bboxes3d", [])
        return []

    def read_classes(self, index: int):
        step = self.steps[index]
        data = self._read_json_state_entry("classes", step, default=None)
        if data is not None:
            return data
        annotations = self.read_annotations(index)
        if isinstance(annotations, dict):
            return annotations.get("classes", [])
        return []

    def read_semantic_annotations(self, index: int):
        step = self.steps[index]
        data = self._read_json_state_entry("semantic", step, default=None)
        if isinstance(data, dict):
            return data
        annotations = self.read_annotations(index)
        if not isinstance(annotations, dict):
            return None
        semantic = annotations.get("semantic")
        if isinstance(semantic, dict):
            return semantic
        return None

    def read_state_dict_flat(self, index: int):
        # Legacy helper that flattens everything into one dict. Handy for older
        # viewers, but note that keys from different modalities may coexist.
        state_dict = self.read_state_dict_common(index)
        rgb_dict = self.read_state_dict_rgb(index)
        segmentation_dict = self.read_state_dict_segmentation(index)
        instance_id_segmentation_dict = self.read_state_dict_instance_id_segmentation(index)
        depth_dict = self.read_state_dict_depth(index)
        normals_dict = self.read_state_dict_normals(index)
        pc_dict = self.read_state_dict_pointcloud(index)

        full_dict = OrderedDict()
        full_dict.update(state_dict)
        full_dict.update(rgb_dict)
        full_dict.update(segmentation_dict)
        full_dict.update(instance_id_segmentation_dict)
        full_dict.update(depth_dict)
        full_dict.update(normals_dict)
        full_dict.update(pc_dict)
        return full_dict

    def read_state_dict(self, index: int):
        # Structured form used by newer tools: modalities stay separated, and
        # pointcloud metadata travels alongside the numeric arrays.
        pointcloud_metadata = OrderedDict()
        for name in self.pointcloud_names:
            pointcloud_metadata[name] = self.read_pointcloud_metadata(name, index)

        return OrderedDict(
            [
                ("common", self.read_state_dict_common(index)),
                ("rgb", self.read_state_dict_rgb(index)),
                ("segmentation", self.read_state_dict_segmentation(index)),
                ("instance_id_segmentation", self.read_state_dict_instance_id_segmentation(index)),
                ("depth", self.read_state_dict_depth(index)),
                ("normals", self.read_state_dict_normals(index)),
                ("pointcloud", self.read_state_dict_pointcloud(index)),
                ("pointcloud_metadata", pointcloud_metadata),
                ("bboxes2d", self.read_bboxes2d(index)),
                ("bboxes3d", self.read_bboxes3d(index)),
                ("classes", self.read_classes(index)),
                ("semantic", self.read_semantic_annotations(index) or {}),
            ]
        )
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __getitem__(self, index: int):
        return self.read_state_dict(index)
