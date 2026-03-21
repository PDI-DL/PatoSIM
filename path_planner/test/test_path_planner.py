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


import pytest
import numpy as np

import mobility_gen_path_planner._mobility_gen_path_planner_C as _C

from mobility_gen_path_planner import generate_paths


def test_plan_all_paths():


    start = np.array([0, 0], dtype=np.int64)
    freespace = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]).astype(np.uint8)

    visited = np.zeros((3, 3), dtype=np.uint8)

    distance_to_start = np.zeros((3, 3), dtype=np.float64)

    prev_i = -np.ones((3, 3), dtype=np.int64)
    prev_j = -np.ones((3, 3), dtype=np.int64)


    _C.generate_paths(
        start,
        freespace,
        visited,
        distance_to_start,
        prev_i,
        prev_j
    )

    print(distance_to_start)


def test_plan_all_paths_v2():


    start = np.array([0, 0], dtype=np.int64)
    freespace = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]).astype(np.uint8)

    visited = np.zeros((3, 3), dtype=np.uint8)

    distance_to_start = np.zeros((3, 3), dtype=np.float64)

    prev_i = -np.ones((3, 3), dtype=np.int64)
    prev_j = -np.ones((3, 3), dtype=np.int64)


    _C.generate_paths(
        start,
        freespace,
        visited,
        distance_to_start,
        prev_i,
        prev_j
    )

    print(distance_to_start)


def test_plan_all_paths_api_v2():

    start = (0, 0)
    freespace = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    output = generate_paths(start, freespace)

    print(output)


def test_unroll_path():

    start = (0, 0)
    freespace = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    output = generate_paths(start, freespace)

    path = output.unroll_path(end=(2, 2))

    print(path)

if __name__ == "__main__":
    test_plan_all_paths()