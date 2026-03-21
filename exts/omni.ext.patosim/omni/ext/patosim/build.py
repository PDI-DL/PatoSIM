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

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import open_stage
import isaacsim.core.api.objects as objects
from isaacsim.core.utils.stage import add_reference_to_stage


from omni.ext.patosim.occupancy_map import OccupancyMap
from omni.ext.patosim.config import Config
from omni.ext.patosim.utils.global_utils import new_stage, new_world, set_viewport_camera
from omni.ext.patosim.scenarios import Scenario, SCENARIOS
from omni.ext.patosim.robots import ROBOTS
from omni.ext.patosim.reader import Reader


def _validate_scene_usd_path(scene_path: str) -> str:
    scene_path = str(scene_path or "").strip()
    if scene_path == "":
        raise RuntimeError(
            "build_scenario_from_config: 'scene_usd' is empty. "
            "Please set the USD Path / URL in the UI before clicking Build."
        )

    lower = scene_path.lower()
    supported_suffixes = (".usd", ".usda", ".usdc", ".usdz")
    if not lower.endswith(supported_suffixes):
        raise RuntimeError(
            "build_scenario_from_config: unsupported scene asset path "
            f"'{scene_path}'. Expected a USD asset ({', '.join(supported_suffixes)}), "
            "not a zip/archive or another file type."
        )

    if "://" not in scene_path and not os.path.exists(scene_path):
        raise RuntimeError(
            "build_scenario_from_config: scene asset does not exist at "
            f"'{scene_path}'."
        )

    return scene_path


def load_scenario(path: str) -> Scenario:
    reader = Reader(path)
    config = reader.read_config()
    robot_type = ROBOTS.get(config.robot_type)
    scenario_type = SCENARIOS.get(config.scenario_type)
    open_stage(os.path.join(path, "stage.usd"))
    prim_utils.delete_prim("/World/robot")
    new_world(physics_dt=robot_type.physics_dt)
    occupancy_map = reader.read_occupancy_map()
    robot = robot_type.build("/World/robot")
    chase_camera_path = robot.build_chase_camera()
    set_viewport_camera(chase_camera_path)
    robot_type = ROBOTS.get(config.robot_type)
    occupancy_map = OccupancyMap.from_ros_yaml(
        ros_yaml_path=os.path.join(path, "occupancy_map", "map.yaml")
    )
    scenario = scenario_type.from_robot_occupancy_map(robot, occupancy_map)
    return scenario


async def build_scenario_from_config(config: Config):
    from omni.ext.patosim.utils.occupancy_map_utils import occupancy_map_generate_from_prim_async

    robot_type = ROBOTS.get(config.robot_type)
    scenario_type = SCENARIOS.get(config.scenario_type)
    scene_usd = _validate_scene_usd_path(getattr(config, "scene_usd", ""))
    new_stage()
    world = new_world(physics_dt=robot_type.physics_dt)
    await world.initialize_simulation_context_async()
    add_reference_to_stage(scene_usd, "/World/scene")
    objects.GroundPlane("/World/ground_plane", visible=False)
    robot = robot_type.build("/World/robot")
    occupancy_map = await occupancy_map_generate_from_prim_async(
        "/World/scene",
        cell_size=robot.occupancy_map_cell_size,
        z_min=robot.occupancy_map_z_min,
        z_max=robot.occupancy_map_z_max
    )
    if getattr(occupancy_map, "data", None) is None or occupancy_map.data.size == 0:
        raise RuntimeError(
            "build_scenario_from_config: occupancy map generation returned an empty map. "
            f"Scene asset='{scene_usd}'. Check whether the USD loaded successfully and "
            "whether the referenced stage contains visible geometry under /World/scene."
        )
    chase_camera_path = robot.build_chase_camera()
    set_viewport_camera(chase_camera_path)
    scenario = scenario_type.from_robot_occupancy_map(robot, occupancy_map)
    return scenario
