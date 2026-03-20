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


import random
import numpy as np
import math


from omni.ext.mobility_gen.occupancy_map import OccupancyMap
from omni.ext.mobility_gen.types import Pose2d, Point2d


class PoseSampler:
    """An abstract base class for 2D pose samplers.

    This is an abstract base class for 2D pose samplers that
    can generate poses based on an occupancy map.
    """

    def sample_px(self, occupancy_map: OccupancyMap) -> Pose2d:
        """Sample a 2D pose, with (x, y) in pixel coordinates.

        This method should be implemented by PoseSampler
        implementations.

        Args:
            occupancy_map (OccupancyMap): An occupancy map that the 
                pose sampler may use.           

        Raises:
            NotImplementedError: The method is not implemented.

        Returns:
            Pose2d: The sampled pose in (x, y) pixel and (theta) 
                world coordinates.
        """
        raise NotImplementedError
    
    def sample(self, occupancy_map: OccupancyMap) -> Pose2d:
        """Sample a 2D pose in world coordinates.

        Args:
            occupancy_map (OccupancyMap): An occupancy map that the
                pose sampler may use.       

        Returns:
            Pose2d: The sampled 2D pose.
        """
        pose_px = self.sample_px(occupancy_map)
        world_pt = occupancy_map.pixel_to_world(Point2d(pose_px.x, pose_px.y))
        return Pose2d(world_pt.x, world_pt.y, pose_px.theta)


class UniformPoseSampler(PoseSampler):
    """A pose sampler that samples poses uniformly.

    This pose sampler samples poses by selecting any freespace 
    in the occupancy map with equal probability.
    """

    def sample_px(self, occupancy_map: OccupancyMap) -> Pose2d:
        freespace = occupancy_map.freespace_mask()
        coords = np.argwhere(freespace)
        random_index = np.random.randint(0, len(coords))
        pixel = coords[random_index]
        theta = random.uniform(-math.pi, math.pi)
        pixel = Pose2d(x=pixel[1], y=pixel[0], theta=theta)
        return pixel


class GridPoseSampler(PoseSampler):
    """A pose sampler that samples poses using grid partitioning.

    This pose sampler samples poses by

    1. Splitting the occupancy map into grid regions specified by "grid_size_meters"
    2. Sampling a grid region uniformly
    3. Sampling a final pose uniformly from the freespace inside the sampled region.

    """

    grid_size_meters: float

    def __init__(self, grid_size_meters: float):
        self.grid_size_meters = grid_size_meters

    def sample_px(self, occupancy_map: OccupancyMap) -> Pose2d:
        num_grid_x = math.ceil(occupancy_map.width_meters() / self.grid_size_meters)
        num_grid_y = math.ceil(occupancy_map.height_meters() / self.grid_size_meters)

        block_x = random.randint(0, num_grid_x - 1)
        block_y = random.randint(0, num_grid_y - 1)

        block_size_px = int(occupancy_map.width_pixels() * self.grid_size_meters / occupancy_map.width_meters())

        block_x_min = block_x * block_size_px
        block_y_min = block_y * block_size_px

        mask = occupancy_map.freespace_mask()

        block_mask = np.zeros_like(mask)
        block_mask[block_x_min:block_x_min+block_size_px] = True
        block_mask[block_y_min:block_y_min+block_size_px] = True

        net_mask = block_mask & mask

        # TODO: check no unoccupied

        coords = np.argwhere(net_mask)
        random_index = np.random.randint(0, len(coords))
        pixel = coords[random_index]
        pixel = Point2d(x=pixel[1], y=pixel[0])
        theta = random.uniform(-math.pi, math.pi)

        return Pose2d(pixel.x, pixel.y, theta)
    