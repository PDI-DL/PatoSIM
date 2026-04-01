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


import numpy as np


def vector_angle(w: np.ndarray, v: np.ndarray):
    delta_angle = np.arctan2(
        w[1] * v[0] - w[0] * v[1], 
        w[0] * v[0] + w[1] * v[1]
    )
    return delta_angle


def nearest_point_on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    a2b = b - a
    a2c = c - a
    a2b_mag = np.sqrt(np.sum(a2b**2))
    a2b_norm = a2b / (a2b_mag + 1e-6)
    dist = np.dot(a2c, a2b_norm)
    if dist < 0:
        return a, dist
    elif dist > a2b_mag:
        return b, dist
    else:
        return a + a2b_norm * dist, dist
    

class PathHelper:
    def __init__(self, points: np.ndarray):
        self.points = points
        self._init_point_distances()

    def _init_point_distances(self):
        self._point_distances = np.zeros(len(self.points))
        length = 0.
        for i in range(0, len(self.points) - 1):
            self._point_distances[i] = length
            a = self.points[i]
            b = self.points[i + 1]
            dist = np.sqrt(np.sum((a - b)**2))
            length += dist
        self._point_distances[-1] = length

    def point_distances(self):
        return self._point_distances

    def get_path_length(self):
        length = 0.
        for i in range(1, len(self.points)):
            a = self.points[i - 1]
            b = self.points[i]
            dist = np.sqrt(np.sum((a - b)**2))
            length += dist
        return length
    
    def points_x(self):
        return self.points[:, 0]
    
    def points_y(self):
        return self.points[:, 1]
    
    def get_segment_by_distance(self, distance):

        for i in range(0, len(self.points) - 1):
            d_a = self._point_distances[i]
            d_b = self._point_distances[i + 1]

            if distance < d_b:
                return (i, i + 1)
            
        i = len(self.points) - 2

        return (i, i + 1)

    def get_point_by_distance(self, distance):
        a_idx, b_idx = self.get_segment_by_distance(distance)
        a, b = self.points[a_idx], self.points[b_idx]
        a_dist, b_dist = self._point_distances[a_idx], self._point_distances[b_idx]
        u = (distance - a_dist) / ((b_dist - a_dist) + 1e-6)
        u = np.clip(u, 0., 1.)
        return a + u * (b - a)
    
    def find_nearest(self, point):
        min_pt_dist_to_seg = 1e9
        min_pt_seg = None
        min_pt = None
        min_pt_dist_along_path = None

        for a_idx in range(0, len(self.points) - 1):
            b_idx = a_idx + 1
            a = self.points[a_idx]
            b = self.points[b_idx]
            nearest_pt, dist_along_seg = nearest_point_on_segment(a, b, point)
            dist_to_seg = np.sqrt(np.sum((point - nearest_pt)**2))

            if dist_to_seg < min_pt_dist_to_seg:
                min_pt_seg = (a_idx, b_idx)
                min_pt_dist_to_seg = dist_to_seg
                min_pt = nearest_pt
                min_pt_dist_along_path = self._point_distances[a_idx] + dist_along_seg

        
        return min_pt, min_pt_dist_along_path, min_pt_seg, min_pt_dist_to_seg


class PathHelper3D:
    """PathHelper para caminhos XYZ 3D (array N×3 float32)."""

    def __init__(self, points: np.ndarray):
        assert points.ndim == 2 and points.shape[1] >= 3, \
            "PathHelper3D requer array (N, 3)"
        self.points = np.asarray(points[:, :3], dtype=np.float32)
        self._cumulative_distances = [0.0]
        for i in range(1, len(self.points)):
            d = float(np.linalg.norm(self.points[i] - self.points[i - 1]))
            self._cumulative_distances.append(self._cumulative_distances[-1] + d)

    @property
    def total_length(self) -> float:
        return self._cumulative_distances[-1]

    def find_nearest(self, point: np.ndarray):
        """Retorna (ponto_mais_proximo_3d, dist_ao_longo_caminho, (seg_i, seg_j), dist_ao_seg)."""
        pt = np.asarray(point[:3], dtype=np.float32)
        best_dist = float("inf")
        best_s = 0.0
        best_pt = self.points[0].copy()
        best_seg = (0, min(1, len(self.points) - 1))
        for i in range(len(self.points) - 1):
            a, b = self.points[i], self.points[i + 1]
            ab = b - a
            ab_len_sq = float(np.dot(ab, ab))
            t = float(np.clip(np.dot(pt - a, ab) / ab_len_sq, 0.0, 1.0)) \
                if ab_len_sq > 1e-12 else 0.0
            proj = a + t * ab
            dist = float(np.linalg.norm(pt - proj))
            if dist < best_dist:
                best_dist = dist
                seg_len = float(np.linalg.norm(ab))
                best_s = self._cumulative_distances[i] + t * seg_len
                best_pt = proj
                best_seg = (i, i + 1)
        return best_pt, best_s, best_seg, best_dist

    def get_point_by_distance(self, distance: float) -> np.ndarray:
        total = self._cumulative_distances[-1]
        distance = float(np.clip(distance, 0.0, total))
        for i in range(len(self._cumulative_distances) - 1):
            if self._cumulative_distances[i + 1] >= distance - 1e-9:
                seg_start = self._cumulative_distances[i]
                seg_end = self._cumulative_distances[i + 1]
                seg_len = seg_end - seg_start
                if seg_len < 1e-12:
                    return self.points[i].copy()
                t = (distance - seg_start) / seg_len
                return self.points[i] + t * (self.points[i + 1] - self.points[i])
        return self.points[-1].copy()
