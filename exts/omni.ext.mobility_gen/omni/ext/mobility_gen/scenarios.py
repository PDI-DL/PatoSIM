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
from typing import Tuple

from mobility_gen_path_planner import generate_paths

from omni.ext.mobility_gen.utils.path_utils import PathHelper, vector_angle
from omni.ext.mobility_gen.utils.registry import Registry
from omni.ext.mobility_gen.common import Module, Buffer
from omni.ext.mobility_gen.robots import Robot
from omni.ext.mobility_gen.occupancy_map import OccupancyMap

import omni.ext.mobility_gen.pose_samplers as pose_samplers
import omni.ext.mobility_gen.inputs as inputs
import omni.replicator.core as rep




class Scenario(Module):

    def __init__(self, 
            robot: Robot, 
            occupancy_map: OccupancyMap
        ):
        self.robot = robot
        self.occupancy_map = occupancy_map
        self.buffered_occupancy_map = occupancy_map.buffered_meters(self.robot.occupancy_map_radius)

    @classmethod
    def from_robot_occupancy_map(cls, robot: Robot, occupancy_map: OccupancyMap):
        return cls(robot, occupancy_map)
    
    def reset(self):
        raise NotImplementedError
    
    def step(self, step_size: float) -> bool:
        raise NotImplementedError


SCENARIOS = Registry[Scenario]()


@SCENARIOS.register()
class KeyboardTeleoperationScenario(Scenario):

    def __init__(self, 
            robot: Robot, 
            occupancy_map: OccupancyMap
        ):
        super().__init__(robot, occupancy_map)
        self.keyboard = inputs.Keyboard()
        self.pose_sampler = pose_samplers.UniformPoseSampler()

    def reset(self):
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)

    def step(self, step_size):

        self.update_state()

        buttons = self.keyboard.buttons.get_value()

        w_val = float(buttons[0])
        a_val = float(buttons[1])
        s_val = float(buttons[2])
        d_val = float(buttons[3])

        linear_velocity = (w_val - s_val) * self.robot.keyboard_linear_velocity_gain
        angular_velocity = (a_val - d_val) * self.robot.keyboard_angular_velocity_gain

        self.robot.action.set_value(np.array([linear_velocity, angular_velocity]))

        self.robot.write_action(step_size)

        self.update_state()

        return True
    

@SCENARIOS.register()
class GamepadTeleoperationScenario(Scenario):

    def __init__(self, 
            robot: Robot, 
            occupancy_map: OccupancyMap
        ):
        super().__init__(robot, occupancy_map)
        self.gamepad = inputs.Gamepad()
        self.pose_sampler = pose_samplers.UniformPoseSampler()

    def reset(self):
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)

    def step(self, step_size: float):

        self.gamepad.update_state()

        axes = self.gamepad.axes.get_value()
        linear_velocity = axes[0] * self.robot.gamepad_linear_velocity_gain
        angular_velocity = axes[3] * self.robot.gamepad_angular_velocity_gain

        self.robot.action.set_value(np.array([linear_velocity, angular_velocity]))
        self.robot.write_action(step_size)

        self.update_state()

        return True
    

@SCENARIOS.register()
class RandomAccelerationScenario(Scenario):

    def __init__(self, 
            robot: Robot, 
            occupancy_map: OccupancyMap
        ):
        super().__init__(robot, occupancy_map)
        self.pose_sampler = pose_samplers.GridPoseSampler(robot.random_action_grid_pose_sampler_grid_size)
        self.is_alive = True
        self.collision_occupancy_map = occupancy_map.buffered(robot.occupancy_map_collision_radius)

    def reset(self):
        self.robot.action.set_value(np.zeros(2))
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)
        self.is_alive = True

    def step(self, step_size: float):

        self.update_state()

        current_action = self.robot.action.get_value()

        linear_velocity = current_action[0] + step_size * np.random.randn(1) * self.robot.random_action_linear_acceleration_std
        angular_velocity = current_action[1] + step_size * np.random.randn(1) * self.robot.random_action_angular_acceleration_std
        
        linear_velocity = np.clip(linear_velocity, *self.robot.random_action_linear_velocity_range)[0]
        angular_velocity = np.clip(angular_velocity, *self.robot.random_action_angular_velocity_range)[0]

        self.robot.action.set_value(np.array([linear_velocity, angular_velocity]))
        self.robot.write_action(step_size)

        self.update_state()

        # Check out of bounds or collision
        pose = self.robot.get_pose_2d()
        if not self.collision_occupancy_map.check_world_point_in_bounds(pose):
            self.is_alive = False
        elif not self.collision_occupancy_map.check_world_point_in_freespace(pose):
            self.is_alive = False

        return self.is_alive



@SCENARIOS.register()
class RandomPathFollowingScenario(Scenario):
    def __init__(self, 
            robot: Robot, 
            occupancy_map: OccupancyMap
        ):
        super().__init__(robot, occupancy_map)
        self.pose_sampler = pose_samplers.UniformPoseSampler()
        self.is_alive = True
        self.target_path = Buffer()
        self.collision_occupancy_map = occupancy_map.buffered(robot.occupancy_map_collision_radius)

    def set_random_target_path(self):
        current_pose = self.robot.get_pose_2d()

        start_px = self.occupancy_map.world_to_pixel_numpy(np.array([
            [current_pose.x, current_pose.y]
        ]))
        freespace = self.buffered_occupancy_map.freespace_mask()

        start = (start_px[0, 1], start_px[0, 0])

        output = generate_paths(start, freespace)
        end = output.sample_random_end_point()
        path = output.unroll_path(end)
        path = path[:, ::-1] # y,x -> x,y coordinates
        path = self.occupancy_map.pixel_to_world_numpy(path)
        self.target_path.set_value(path)
        self.target_path_helper = PathHelper(path)

    def reset(self):
        self.robot.action.set_value(np.zeros(2))
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)
        self.set_random_target_path()
        self.is_alive = True
    
    def step(self, step_size: float):

        self.update_state()
        target_path = self.target_path.get_value()
        current_pose = self.robot.get_pose_2d()

        if not self.collision_occupancy_map.check_world_point_in_bounds(current_pose):
            self.is_alive = False
            return self.is_alive
        elif not self.collision_occupancy_map.check_world_point_in_freespace(current_pose):
            self.is_alive = False
            return self.is_alive
    
        pt_robot = np.array([current_pose.x, current_pose.y])
        pt_path, pt_path_length, _, _ = self.target_path_helper.find_nearest(pt_robot)
        pt_target = self.target_path_helper.get_point_by_distance(distance=
            pt_path_length + self.robot.path_following_target_point_offset_meters
        )

        path_end = target_path[-1]
        dist_to_target = np.sqrt(np.sum((pt_robot - path_end)**2))

        if dist_to_target < self.robot.path_following_stop_distance_threshold:
            self.set_random_target_path()
        else:
            vec_robot_unit = np.array([np.cos(current_pose.theta), np.sin(current_pose.theta)])
            vec_target = (pt_target - pt_robot)
            vec_target_unit = vec_target / np.sqrt(np.sum(vec_target**2))
            d_theta = vector_angle(vec_robot_unit, vec_target_unit)

            if abs(d_theta) > self.robot.path_following_forward_angle_threshold:
                linear_velocity = 0.
            else:
                linear_velocity = self.robot.path_following_speed

            angular_gain: float = self.robot.path_following_angular_gain
            angular_velocity = - angular_gain * d_theta
            self.robot.action.set_value(np.array([linear_velocity, angular_velocity]))

        self.robot.write_action(step_size=step_size)

        return self.is_alive
    ###############################################
    #novo cenario de coleta de dados se necessario#
    ###############################################
    @SCENARIOS.register()
    class KeyboardTeleoperationWithSensors(Scenario):
        """
        Envolve o KeyboardTeleoperationScenario: reaproveita teleop e só monta câmeras + writer.
        """
        def __init__(self, robot: Robot, occupancy_map: OccupancyMap,
                    mount_bev=True, mount_rs=True, mount_zed=True, out_dir="/tmp/mg_dataset"):
            super().__init__(robot, occupancy_map)
            # instancia o cenário base (mesma assinatura)
            self.base = KeyboardTeleoperationScenario(robot, occupancy_map)
            self.mount_bev = mount_bev
            self.mount_rs = mount_rs
            self.mount_zed = mount_zed
            self.out_dir = out_dir
            self._writer = None
            self._cam_paths = []

        def reset(self):
            self.base.reset()  # isso chama pose sampler e etc.

            # monta câmeras no robô (métodos da sua classe)
            if self.mount_bev:
                self._cam_paths.append(self.robot.mount_bev_camera_topdown())
            if self.mount_rs:
                self._cam_paths.append(self.robot.mount_realsense_rgbd())
            if self.mount_zed:
                l, r = self.robot.mount_zed_stereo()
                self._cam_paths += [l, r]

            # liga o writer do Replicator
            self._writer = rep.WriterRegistry.get("BasicWriter")
            self._writer.initialize(
                output_dir=self.out_dir,
                rgb=True,
                semantic_segmentation=True,
                instance_segmentation=True,
                distance_to_camera=True,
            )
            with rep.trigger.on_frame(num_frames=-1):
                prims = []
                for p in self._cam_paths:
                    if isinstance(p, tuple):
                        prims += [rep.get.prims(path=p[0]), rep.get.prims(path=p[1])]
                    else:
                        prims.append(rep.get.prims(path=p))
                self._writer.attach(prims)

        def step(self, step_size):
            return self.base.step(step_size)
