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


from omni.ext.mobility_gen.occupancy_map import OccupancyMap
from omni.ext.mobility_gen.config import Config


class Reader:

    def __init__(self, recording_path: str):
        self.recording_path = recording_path
        
        state_dict_paths = glob.glob(os.path.join(
            self.recording_path, "state", "common", "*.npy"
        ))

        steps = [int(os.path.basename(path).split('.')[0]) for path in state_dict_paths]
        self.steps = sorted(steps)

        self.rgb_folders = glob.glob(os.path.join(self.recording_path, "state", "rgb", "*"))
        self.segmentation_folders = glob.glob(os.path.join(self.recording_path, "state", "segmentation", "*"))
        self.depth_folders = glob.glob(os.path.join(self.recording_path, "state", "depth", "*"))
        self.normals_folders = glob.glob(os.path.join(self.recording_path, "state", "normals", "*"))
        self.pointcloud_folders = glob.glob(os.path.join(self.recording_path, "state", "pointcloud", "*"))
        self.annotation_paths = glob.glob(os.path.join(self.recording_path, "state", "annotations", "*.json"))

        self.rgb_names = [os.path.basename(folder) for folder in self.rgb_folders]
        self.segmentation_names = [os.path.basename(folder) for folder in self.segmentation_folders]
        self.depth_names = [os.path.basename(folder) for folder in self.depth_folders]
        self.normals_names = [os.path.basename(folder) for folder in self.normals_folders]
        self.pointcloud_names = [os.path.basename(folder) for folder in self.pointcloud_folders]
        self.annotation_steps = sorted(
            [
                int(os.path.basename(path).split(".")[0])
                for path in self.annotation_paths
                if os.path.basename(path).split(".")[0].isdigit()
            ]
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
    
    def read_depth(self, name: str, index: int, eps=1e-6):
        step = self.steps[index]
        image = PIL.Image.open(os.path.join(self.recording_path, "state", "depth", name, f"{step:08d}.png")).convert("I;16")
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
            # Prefer open3d if available for robust PLY/PCD reading
            if o3d is not None:
                try:
                    read_path = ply_path if os.path.exists(ply_path) else pcd_path
                    pcd = o3d.io.read_point_cloud(read_path)
                    pts = np.asarray(pcd.points)
                    # attempt to read colors
                    if pcd.has_colors():
                        cols = np.asarray(pcd.colors)
                        # colors 0..1 -> convert to 0..255 for compatibility with writer's assumptions
                        if cols.max() <= 1.0:
                            cols = (cols * 255.0).astype(np.float32)
                        pts = np.hstack([pts, cols])
                    return pts
                except Exception:
                    pass
            # Fallback minimal ASCII PLY reader (assumes x y z [intensity])
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
        ann_path = os.path.join(self.recording_path, "state", "annotations", f"{step:08d}.json")
        if not os.path.exists(ann_path):
            return None
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def read_semantic_annotations(self, index: int):
        annotations = self.read_annotations(index)
        if not isinstance(annotations, dict):
            return None
        semantic = annotations.get("semantic")
        if isinstance(semantic, dict):
            return semantic
        return None

    def read_state_dict(self, index: int):

        state_dict = self.read_state_dict_common(index)
        rgb_dict = self.read_state_dict_rgb(index)
        segmentation_dict = self.read_state_dict_segmentation(index)
        depth_dict = self.read_state_dict_depth(index)
        normals_dict = self.read_state_dict_normals(index)
        pc_dict = self.read_state_dict_pointcloud(index)

        full_dict = OrderedDict()
        full_dict.update(state_dict)
        full_dict.update(rgb_dict)
        full_dict.update(segmentation_dict)
        full_dict.update(depth_dict)
        full_dict.update(normals_dict)
        full_dict.update(pc_dict)

        return full_dict
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def __getitem__(self, index: int):
        return self.read_state_dict(index)
