/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <algorithm>


namespace py = pybind11;


struct Point {
    int i;
    int j;

    bool operator==(const Point &other) const {
        return (i == other.i) && (j == other.j);
    }
};


struct PriorityQueueItem {
    float priority;
    Point point;
};


struct PriorityQueueCompare {
    bool operator()(
        const PriorityQueueItem &a, 
        const PriorityQueueItem &b
    ) {
        return a.priority > b.priority;
    }
};


float get_distance(Point &start, Point &end) {
    float di = end.i - start.i;
    float dj = end.j - start.j;
    return std::sqrt(di*di + dj*dj);
}


std::vector<Point> get_children(
    Point &point,
    py::detail::unchecked_reference<uint8_t, 2> freespace,
    py::detail::unchecked_mutable_reference<uint8_t, 2> visited,
    int nrows,
    int ncols
) {

    // Initialize candidate locations
    std::vector<Point> candidates = {
        // Top row
        { point.i - 1, point.j - 1},
        { point.i - 1, point.j },
        { point.i - 1, point.j + 1},

        // Middle row (exclude self)
        { point.i, point.j - 1},
        { point.i, point.j + 1},

        // Bottom row
        { point.i + 1, point.j - 1},
        { point.i + 1, point.j },
        { point.i + 1, point.j + 1}
    };

    // Initialize output children
    std::vector<Point> children;
    children.reserve(8);

    // Filter candidates and populate output
    for (int i = 0; i < 8; i++) {

        auto candidate = candidates[i];

        // Filter out of bounds
        if (candidate.i < 0) continue;
        if (candidate.i >= nrows) continue;
        if (candidate.j < 0) continue;
        if (candidate.j >= ncols) continue;

        // Filter non-freespace
        if (!freespace(candidate.i, candidate.j)) continue;

        // Filter visited
        if (visited(candidate.i, candidate.j)) continue;

        children.push_back(candidate);
    }

    return children;
}

void generate_paths(
    py::array_t<int64_t> start_np,
    py::array_t<uint8_t> freespace,
    py::array_t<uint8_t> visited,
    py::array_t<double> distance_to_start,
    py::array_t<int64_t> prev_i,
    py::array_t<int64_t> prev_j
) {

    int nrows = freespace.shape(0);
    int ncols = freespace.shape(1);
    auto _start_np = start_np.unchecked<1>();

    Point start = {
        _start_np(0), _start_np(1)
    };

    auto _freespace = freespace.unchecked<2>();
    auto _visited = visited.mutable_unchecked<2>();
    auto _distance_to_start = distance_to_start.mutable_unchecked<2>();
    auto _prev_i = prev_i.mutable_unchecked<2>();
    auto _prev_j = prev_j.mutable_unchecked<2>();

    std::priority_queue<
        PriorityQueueItem,
        std::vector<PriorityQueueItem>,
        PriorityQueueCompare
    > q;

    // Initialize
    _distance_to_start(start.i, start.j) = 0.0;
    q.push({0., start});
    _visited(start.i, start.j) = true;

    // Search
    while (!q.empty()) {
        
        PriorityQueueItem node = q.top();

        //std::cout << (int)_visited(node.point.i, node.point.j) << "," << node.priority << "," << node.point.i << "," << node.point.j << std::endl;

        q.pop();

        // Add children
        std::vector<Point> children = get_children(
            node.point,
            _freespace,
            _visited,
            nrows,
            ncols
        );

        for (unsigned int i = 0; i < children.size(); i++) {

            Point child = children[i];
            
            // Mark parent node, for unrolling path
            _prev_i(child.i, child.j) = node.point.i;
            _prev_j(child.i, child.j) = node.point.j;

            // Mark distance to start
            
            float child_distanace_to_start = (
                _distance_to_start(node.point.i, node.point.j) + 
                get_distance(node.point, child)
            );

            _distance_to_start(child.i, child.j) = child_distanace_to_start;

            _visited(child.i, child.j) = true;

            q.push({child_distanace_to_start, child});
        }
    }
}

std::vector<std::pair<int, int>> unroll_path(
    py::array_t<int64_t> end_np,
    py::array_t<int64_t> prev_i,
    py::array_t<int64_t> prev_j
) {

    auto _prev_i = prev_i.unchecked<2>();
    auto _prev_j = prev_j.unchecked<2>();

    auto _end_np = end_np.unchecked<1>();

    std::vector<std::pair<int, int>> path;
    std::pair<int, int> point = {_end_np(0), _end_np(1)};

    while (1) {
        path.push_back(point);

        if (_prev_i(point.first, point.second) < 0) {
            break; // end of path
        }

        point = {
            _prev_i(point.first, point.second),
            _prev_j(point.first, point.second)
        };
    }

    std::reverse(path.begin(), path.end());

    return path;
}

PYBIND11_MODULE(_mobility_gen_path_planner_C, m) {
    m.doc() = "MobilityGen Path Planner C++ Bindings";
    m.def("generate_paths", &generate_paths, "Generate paths");
    m.def("unroll_path", &unroll_path, "Unroll a path");
}