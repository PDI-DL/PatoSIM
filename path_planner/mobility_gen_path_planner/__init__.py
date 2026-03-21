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

from typing import Tuple, List
import numpy as np
import random
from dataclasses import dataclass
import os
import importlib
import importlib.util

# Try the normal import first; if the compiled extension was built for a
# different Python ABI (e.g. cpython-3.10) the standard import may fail on
# another interpreter (e.g. Python 3.11). In that case, look for a
# corresponding .so in the package directory and load it directly.
try:
    import mobility_gen_path_planner._mobility_gen_path_planner_C as _C
except ModuleNotFoundError:
    _C = None
    pkg_dir = os.path.dirname(__file__)
    # find a .so file that starts with the expected module name
    for fname in os.listdir(pkg_dir):
        if fname.startswith("_mobility_gen_path_planner_C") and fname.endswith(".so"):
            so_path = os.path.join(pkg_dir, fname)
            spec = importlib.util.spec_from_file_location("mobility_gen_path_planner._mobility_gen_path_planner_C", so_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _C = module
            break
    if _C is None:
        # re-raise the original error for visibility
        raise


@dataclass
class GeneratePathsOutput:
    visited: np.ndarray
    distance_to_start: np.ndarray
    prev_i: np.ndarray
    prev_j: np.ndarray

    def unroll_path(self, end: Tuple[int, int]) -> np.ndarray:
        end = np.array([end[0], end[1]], dtype=np.int64)
        path = _C.unroll_path(end, self.prev_i, self.prev_j)
        return np.array(path)
    
    def get_valid_end_points(self):
        return np.where(self.visited != 0)
    
    def sample_random_end_point(self) -> Tuple[int, int]:
        i, j = self.get_valid_end_points()
        index = random.randint(0, len(i) - 1)
        return (int(i[index]), int(j[index]))

    def sample_random_path(self) -> np.ndarray:
        end = self.sample_random_end_point()
        return self.unroll_path(end)
    

def generate_paths(start: Tuple[int, int], freespace: np.ndarray) -> GeneratePathsOutput:

    start = np.array([start[0], start[1]], dtype=np.int64)
    freespace = freespace.astype(np.uint8)
    visited = np.zeros(freespace.shape, dtype=np.uint8)
    distance_to_start = np.zeros(freespace.shape, dtype=np.float64)
    prev_i = -np.ones((freespace.shape), dtype=np.int64)
    prev_j = -np.ones((freespace.shape), dtype=np.int64)

    _C.generate_paths(
        start,
        freespace,
        visited,
        distance_to_start,
        prev_i,
        prev_j
    )

    return GeneratePathsOutput(
        visited=visited,
        distance_to_start=distance_to_start,
        prev_i=prev_i,
        prev_j=prev_j
    )