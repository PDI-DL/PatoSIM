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

from __future__ import annotations  # <-- DEIXE AQUI (linha 3), ou REMOVA em Python 3.10+
# Standard imports
import numpy as np
import os
import math
#from typing import List, Type, Tuple, Union
from typing import Optional, Sequence, Union, Dict, List, Tuple, Type, List



# Isaac Sim Imports
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.api.robots.robot import Robot as _Robot
from isaacsim.core.prims import Articulation as _ArticulationView
from isaacsim.robot.wheeled_robots.robots import WheeledRobot as _WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
import isaacsim.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.stage import add_reference_to_stage
#from omni.isaac.core.utils.prims import set_world_pose
from omni.isaac.core.utils.rotations import euler_angles_to_quat

# Extension imports
from omni.ext.mobility_gen.common import Buffer, Module
from omni.ext.mobility_gen.sensors import Sensor, HawkCamera
from omni.ext.mobility_gen.utils.global_utils import get_stage, get_world
from omni.ext.mobility_gen.utils.stage_utils import stage_get_prim, stage_add_camera, stage_add_usd_ref
from omni.ext.mobility_gen.utils.prim_utils import prim_rotate_x, prim_rotate_y, prim_rotate_z, prim_translate
from omni.ext.mobility_gen.types import Pose2d
from omni.ext.mobility_gen.utils.registry import Registry
# RELATIVOS (recomendado)
# from .registry import ROBOTS
# from .robot import Robot


# dentro da classe ForkliftC
from pxr import UsdGeom, Gf
import omni.usd
from omni.isaac.core.utils.rotations import euler_angles_to_quat



#=========================================================
#  BASE CLASSES
#=========================================================


class Robot(Module):
    """Abstract base class for robots

    This class defines an abstract base class for robots.

    Robot implementations must subclass this class, define the 
    required class parameters and abstract methods.

    The two main abstract methods subclasses must define are the build() and write_action()
    methods.
    
    Parameters:
        physics_dt (float): The physics time step the robot requires (in seconds).
            This may need to be modified depending on the underlying controller's
            
        z_offset (float): A z offset to use when spawning the robot to ensure it
            drops at an appropriate height when initializing.
        chase_camera_base_path (str):  A path (in the USD stage) relative to the 
            base path to use as a parent when defining the "chase" camera.  Typically,
            this is the same as the path to the transform used for determining
            the 2D pose of the robot.
        chase_camera_x_offset (str):  The x offset of the chase camera.  Typically this is
            negative, to locate the camera behind the robot.
        chase_camera_z_offset (str): The z offset of the chase camera.  Typically this is
            positive, to locate the camera above the robot.
        chase_camera_tile_angle (float):  The tilt angle of the chase camera.  Typically
            this does not need to be modified.
        front_camera_type (Type[Sensor]):  The configurable sensor class to attach
            at the front camera for the robot.  This should be a final sensor class (like HawkCamera)
            that can be built using the class method HawkCamera.build(prim_path).
        front_camera_base_path (str):  The relative path (in the USD stage) relative to the 
            robot prim path to use as the basis for creating the front camera XForm.
        front_camera_rotation (Tuple[float, float, float]):  The (x, y, z) rotation to apply
            when building the XForm for the front camera.
        front_camera_translation (Tuple[float, float, float]):  The (x, y, z) rotation to apply
            when building the XForm for the front camera.

    """

    physics_dt: float

    z_offset: float

    chase_camera_base_path: str
    chase_camera_x_offset: float
    chase_camera_z_offset: float
    chase_camera_tilt_angle: float

    occupancy_map_radius: float
    occupancy_map_z_min: float
    occupancy_map_z_max: float
    occupancy_map_cell_size: float
    occupancy_map_collision_radius: float

    front_camera_type: Type[Sensor]
    front_camera_base_path: str
    front_camera_rotation: Tuple[float, float, float]
    front_camera_translation: Tuple[float, float, float]

    keyboard_linear_velocity_gain: float
    keyboard_angular_velocity_gain: float

    gamepad_linear_velocity_gain: float
    gamepad_angular_velocity_gain: float

    random_action_linear_velocity_range: Tuple[float, float]
    random_action_angular_velocity_range: Tuple[float, float]
    random_action_linear_acceleration_std: float
    random_action_angular_acceleration_std: float
    random_action_grid_pose_sampler_grid_size: float


    path_following_speed: float
    path_following_angular_gain: float
    path_following_stop_distance_threshold: float
    path_following_forward_angle_threshold = math.pi
    path_following_target_point_offset_meters: float


    def __init__(self,
            prim_path: str,
            robot: _Robot,
            articulation_view: _ArticulationView,
            front_camera: Sensor
        ):
        self.prim_path = prim_path
        self.robot = robot
        self.articulation_view = articulation_view

        self.action = Buffer(np.zeros(2))
        self.position = Buffer()
        self.orientation = Buffer()
        self.joint_positions = Buffer()
        self.joint_velocities = Buffer()
        self.front_camera = front_camera

    @classmethod
    def build_front_camera(cls, prim_path):
        
        # Add camera
        camera_path = os.path.join(prim_path, cls.front_camera_base_path)
        front_camera_xform = XFormPrim(camera_path)

        stage = get_stage()
        front_camera_prim = stage_get_prim(stage, camera_path)
        prim_rotate_x(front_camera_prim, cls.front_camera_rotation[0])
        prim_rotate_y(front_camera_prim, cls.front_camera_rotation[1])
        prim_rotate_z(front_camera_prim, cls.front_camera_rotation[2])
        prim_translate(front_camera_prim, cls.front_camera_translation)

        return cls.front_camera_type.build(prim_path=camera_path)

    def build_chase_camera(self) -> str:

        stage = get_stage()

        camera_path = os.path.join(self.prim_path, self.chase_camera_base_path, "chase_camera")
        stage_add_camera(stage, 
            camera_path, 
            focal_length=10, horizontal_aperature=30, vertical_aperature=30
        )
        camera_prim = stage_get_prim(stage, camera_path)
        prim_rotate_x(camera_prim, self.chase_camera_tilt_angle)
        prim_rotate_y(camera_prim, 0)
        prim_rotate_z(camera_prim, -90)
        prim_translate(camera_prim, (self.chase_camera_x_offset, 0., self.chase_camera_z_offset))

        return camera_path
    
    @classmethod
    def build(cls, prim_path: str) -> "Robot":
        raise NotImplementedError
    
    def write_action(self, step_size: float):
        raise NotImplementedError
    
    def update_state(self):
        pos, ori = self.robot.get_local_pose()
        self.position.set_value(pos)
        self.orientation.set_value(ori)
        self.joint_positions.set_value(self.robot.get_joint_positions())
        self.joint_velocities.set_value(self.robot.get_joint_velocities())
        super().update_state()

    def write_replay_data(self):
        self.robot.set_local_pose(
            self.position.get_value(),
            self.orientation.get_value()
        )
        self.articulation_view.set_joint_positions(
            self.joint_positions.get_value()
        )
        super().write_replay_data()

    def set_pose_2d(self, pose: Pose2d):
        self.articulation_view.initialize()
        self.robot.set_world_velocity(np.array([0., 0., 0., 0., 0., 0.]))
        self.robot.post_reset()
        position, orientation = self.robot.get_local_pose()
        position[0] = pose.x
        position[1] = pose.y
        position[2] = self.z_offset
        orientation = rot_utils.euler_angles_to_quats(np.array([0., 0., pose.theta]))
        self.robot.set_local_pose(
            position, orientation
        )
    
    def get_pose_2d(self) -> Pose2d:
        position, orientation = self.robot.get_local_pose()
        theta = rot_utils.quats_to_euler_angles(orientation)[2]
        return Pose2d(
            x=position[0],
            y=position[1],
            theta=theta
        )
    

class WheeledRobot(Robot):

    # Wheeled robot parameters
    wheel_dof_names: List[str]
    usd_url: str
    chassis_subpath: str
    wheel_radius: float
    wheel_base: float

    def __init__(self,
            prim_path: str,
            robot: _WheeledRobot,
            articulation_view: _ArticulationView,
            controller: DifferentialController,
            front_camera: Sensor | None = None
        ):
        super().__init__(
            prim_path=prim_path,
            robot=robot,
            articulation_view=articulation_view,
            front_camera=front_camera
        )
        self.controller = controller
        self.robot = robot
        
    @classmethod
    def build(cls, prim_path: str) -> "WheeledRobot":

        world = get_world()

        robot = world.scene.add(_WheeledRobot(
            prim_path,
            wheel_dof_names=cls.wheel_dof_names,
            create_robot=True,
            usd_path=cls.usd_url
        ))

        view = _ArticulationView(
            os.path.join(prim_path, cls.chassis_subpath)
        )

        world.scene.add(view)

        controller = DifferentialController(
            name="controller",
            wheel_radius=cls.wheel_radius,
            wheel_base=cls.wheel_base
        )
        
        camera = cls.build_front_camera(prim_path)

        return cls(
            prim_path=prim_path,
            robot=robot,
            articulation_view=view,
            controller=controller,
            front_camera=camera
        )
    
    def write_action(self, step_size: float):
        self.robot.apply_wheel_actions(
            self.controller.forward(
                command=self.action.get_value()
            )
        )


class IsaacLabRobot(Robot):

    usd_url: str
    articulation_path: str

    def __init__(self, 
            prim_path: str, 
            robot: _Robot,
            articulation_view: _ArticulationView,
            controller: Union[H1FlatTerrainPolicy, SpotFlatTerrainPolicy],
            front_camera: Sensor | None = None
        ):
        super().__init__(prim_path, robot, articulation_view, front_camera)
        self.controller = controller

    @classmethod
    def build_policy(cls, prim_path: str):
        raise NotImplementedError

    @classmethod
    def build(cls, prim_path: str):
        stage = get_stage()
        world = get_world()

        stage_add_usd_ref(
            stage=stage,
            path=prim_path,
            usd_path=cls.usd_url
        )
        
        robot = _Robot(prim_path=prim_path)

        world.scene.add(robot)

        # Articulation
        view = _ArticulationView(
            os.path.join(prim_path, cls.articulation_path)
        )

        world.scene.add(view)

        # Controller
        controller = cls.build_policy(prim_path)

        prim = stage_get_prim(stage, prim_path)        
        prim_translate(prim, (0, 0, cls.z_offset))


        camera = cls.build_front_camera(prim_path)

        return cls(
            prim_path=prim_path, 
            robot=robot, 
            articulation_view=view, 
            controller=controller,
            front_camera=camera
        )
    
    def write_action(self, step_size):
        action = self.action.get_value()
        command = np.array([action[0], 0., action[1]])
        self.controller.forward(step_size, command)

    def set_pose_2d(self, pose):
        super().set_pose_2d(pose)
        self.controller.initialize()


#=========================================================
#  FINAL CLASSES
#=========================================================

ROBOTS = Registry[Robot]()


@ROBOTS.register()
class JetbotRobot(WheeledRobot):

    physics_dt: float = 0.005

    z_offset: float = 0.1

    chase_camera_base_path = "chassis"
    chase_camera_x_offset: float = -0.5
    chase_camera_z_offset: float = 0.5
    chase_camera_tilt_angle: float = 60.

    occupancy_map_radius: float = 0.25
    occupancy_map_z_min: float = 0.05
    occupancy_map_z_max: float = 0.5
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.25

    front_camera_base_path = "chassis/rgb_camera/front_hawk"
    front_camera_rotation = (0., 0., 0.)
    front_camera_translation = (0., 0., 0.)
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 0.25
    keyboard_angular_velocity_gain: float = 1.0

    gamepad_linear_velocity_gain: float = 0.25
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 0.25)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 1.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    path_following_speed: float = 0.25
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.5
    path_following_forward_angle_threshold = math.pi / 4
    path_following_target_point_offset_meters: float = 1.0

    wheel_dof_names: List[str] = ["left_wheel_joint", "right_wheel_joint"]
    usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Jetbot/jetbot.usd"
    chassis_subpath: str = "chassis"
    wheel_base: float = 0.1125
    wheel_radius: float = 0.03
    

@ROBOTS.register()
class CarterRobot(WheeledRobot):

    physics_dt: float = 0.005

    z_offset: float = 0.25

    chase_camera_base_path = "chassis_link"
    chase_camera_x_offset: float = -1.5
    chase_camera_z_offset: float = 0.8
    chase_camera_tilt_angle: float = 60.

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 0.62
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    front_camera_base_path = "chassis_link/front_hawk/front_hawk"
    front_camera_rotation = (0., 0., 0.)
    front_camera_translation = (0., 0., 0.)
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0

    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 5.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    path_following_speed: float = 1.0
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.5
    path_following_forward_angle_threshold = math.pi / 4
    path_following_target_point_offset_meters: float = 1.0

    wheel_dof_names: List[str] = ["joint_wheel_left", "joint_wheel_right"]
    usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Carter/nova_carter_sensors.usd"
    chassis_subpath: str = "chassis_link"
    wheel_base = 0.413
    wheel_radius = 0.14


@ROBOTS.register()
class H1Robot(IsaacLabRobot):

    physics_dt: float = 0.005

    z_offset: float = 1.05

    chase_camera_base_path = "pelvis"
    chase_camera_x_offset: float = -1.5
    chase_camera_z_offset: float = 0.8
    chase_camera_tilt_angle: float = 60.

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 2.0
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    front_camera_base_path = "d435_left_imager_link/front_camera/front"
    front_camera_rotation = (0., 250., 90.)
    front_camera_translation = (-0.06, 0., 0.)
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0

    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 5.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0
    
    path_following_speed: float = 1.0
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.5
    path_following_forward_angle_threshold = math.pi / 4
    path_following_target_point_offset_meters: float = 1.0

    usd_url = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Unitree/H1/h1.usd"
    articulation_path = "pelvis"
    controller_z_offset: float = 1.05

    @classmethod
    def build_policy(cls, prim_path: str):
        return H1FlatTerrainPolicy(
            prim_path=prim_path,
            position=np.array([0., 0., cls.controller_z_offset])
        )


@ROBOTS.register()
class SpotRobot(IsaacLabRobot):

    physics_dt: float = 0.005
    z_offset: float = 0.7

    chase_camera_base_path = "body"
    chase_camera_x_offset: float = -1.5
    chase_camera_z_offset: float = 0.8
    chase_camera_tilt_angle: float = 60.

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = 0.1
    occupancy_map_z_max: float = 0.62
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    front_camera_base_path = "body/front_camera"
    front_camera_rotation = (180, 180, 180)
    front_camera_translation = (0.44, 0.075, 0.01)
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0

    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 5.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0
    
    path_following_speed: float = 1.0
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold:1.0

    @classmethod
    def build_policy(cls, prim_path: str):
        return SpotFlatTerrainPolicy(
            prim_path=prim_path,
            position=np.array([0., 0., cls.controller_z_offset])
        )
    
    ####################################################################################
    #Colocar aqui a nova definição do robo forklift# Preliminar faltam ajustes e testes#
    ####################################################################################

#Falta colocar as cameras do nuscenes
# Front image - [ ]  
# Front left image - [ ]  
# Front Right image - [ ]  
# Back image - [ ]  
# Back right image - [ ]  
# Back left image - [ ]  
@ROBOTS.register()
class ForkliftC(Robot):
    """
    Adaptador MobilityGen para o asset USD 'forklift_c' (empilhadeira).

    Ação base: self.action = [v, w]  (m/s, rad/s) → Ackermann → steer (rad).
    Lift/Tilt ficam em 0 por padrão (exponha depois no cenário, se quiser).
    """

    # ===== Parâmetros exigidos pela base Robot =====
    physics_dt: float = 1.0 / 120.0
    z_offset: float = 0.10

    chase_camera_base_path: str = ""
    chase_camera_x_offset: float = -1.5
    chase_camera_z_offset: float = 0.9
    chase_camera_tilt_angle: float = 50.0

    occupancy_map_radius: float = 1.5
    occupancy_map_z_min: float = -0.20
    occupancy_map_z_max: float = 2.50
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.5

    front_camera_type = HawkCamera
    front_camera_base_path: str = "sensors/front_cam"
    front_camera_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    front_camera_translation: Tuple[float, float, float] = (0.55, 0.0, 1.40)

    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0
    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 5.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    path_following_speed: float = 1.0
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.2
    path_following_forward_angle_threshold: float = math.pi
    path_following_target_point_offset_meters: float = 0.0

    # ===== Específicos da empilhadeira =====
    usd_path: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd"
    wheel_radius: float = 0.165
    wheel_base: float = 1.10
    track_width: float = 0.95
    steer_limit_rad: float = math.radians(40.0)
    rear_diff_drive: bool = False

    prefer_names = dict(
        steer=["steer", "steering", "front_axle_steer"],
        rear_left_drive=["rear_left_wheel", "rear_l_wheel", "rl_wheel", "drive_left"],
        rear_right_drive=["rear_right_wheel", "rear_r_wheel", "rr_wheel", "drive_right"],
        fork_prismatic=["fork", "lift", "prismatic"],
        mast_tilt=["tilt", "mast_tilt"],
    )

    # ---------- helpers precisão / quat ----------
    @staticmethod
    def _set_vec_by_precision(op: UsdGeom.XformOp, xyz: Tuple[float, float, float]):
        x, y, z = map(float, xyz)
        try:
            prec = op.GetPrecision()
        except Exception:
            prec = UsdGeom.XformOp.PrecisionDouble
        (op.GetAttr().Set(Gf.Vec3f(x, y, z)) if prec == UsdGeom.XformOp.PrecisionFloat
         else op.GetAttr().Set(Gf.Vec3d(x, y, z)))

    @staticmethod
    def _set_quat_by_precision(op: UsdGeom.XformOp, quat_xyzw: Sequence[float]):
        qx, qy, qz, qw = map(float, quat_xyzw)  # (x,y,z,w)
        try:
            prec = op.GetPrecision()
        except Exception:
            prec = UsdGeom.XformOp.PrecisionDouble
        (op.GetAttr().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz))) if prec == UsdGeom.XformOp.PrecisionFloat
         else op.GetAttr().Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz))))

    # ----------- construção -----------
    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        usd_path: Optional[str] = None,
        enable_bev: bool = True,
        enable_bev_front: bool = True,
        enable_realsense: bool = True,
        enable_zed: bool = True,
        steer_names: Optional[List[str]] = None,
        rear_left_drive_name: Optional[str] = None,
        rear_right_drive_name: Optional[str] = None,
        fork_name: Optional[str] = None,
        mast_tilt_name: Optional[str] = None,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        orientation_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        debug: bool = False,
    ) -> "ForkliftC":

        usd = usd_path or cls.usd_path
        if not usd or not isinstance(usd, str):
            raise RuntimeError("Defina 'usd_path' válido para o forklift_c.usd.")

        # Referencia e posiciona
        add_reference_to_stage(usd, prim_path)
        cls._set_local_pose_static(prim_path=prim_path, position=position, orientation_xyzw=orientation_xyzw)

        # Só ArticulationView (estável). NÃO criar Articulation para evitar warnings/__del__ issues.
        art_view = _ArticulationView(prim_paths_expr=prim_path, name="forklift_art")
        try:
            art_view.initialize()
        except Exception:
            pass

        # Câmera frontal padrão (base Robot usa)
        front_cam = cls.build_front_camera(prim_path)

        # Instância do Robot-base
        self = cls(
            prim_path=prim_path,
            robot=art_view,
            articulation_view=art_view,
            front_camera=front_cam,
        )

        # Estado DOF (adiado)
        self._debug = debug
        self._dof_names: List[str] = []
        self._idx = dict(steer=[], rear_left=None, rear_right=None, fork=None, mast_tilt=None)
        self._pending_dof = dict(
            steer_names=steer_names,
            rear_left_drive_name=rear_left_drive_name,
            rear_right_drive_name=rear_right_drive_name,
            fork_name=fork_name,
            mast_tilt_name=mast_tilt_name,
        )
        self._dofs_ready = False

        # 6) Câmeras auxiliares
        self.cam_paths = {}
        if enable_bev:
            self.cam_paths["bev_topdown"] = self.mount_bev_camera_topdown()
        if enable_bev_front:
            self.cam_paths["bev_front_down"] = self.mount_bev_camera_forward_down()
        if enable_realsense:
            self.cam_paths["realsense"] = self.mount_realsense_rgbd()
        if enable_zed:
            l, r = self.mount_zed_stereo()
            self.cam_paths["zed_left"], self.cam_paths["zed_right"] = l, r

        # 7) Chase camera
        try: self.build_chase_camera()
        except Exception: pass

        # 8) Render products (define resolução aqui, não na câmera!)
        self.build_render_products()
        return self
    
    def _ensure_initialized(self):
        try:
            self.articulation_view.initialize()
        except Exception:
            pass

    # ===== Descoberta de DOFs (lazy) =====
    def _get_dof_names_safe(self) -> Optional[List[str]]:
        """Tenta vários getters de nomes; retorna lista ou None se ainda indisponível."""
        for getter in ("get_dof_names", "get_joint_names"):
            if hasattr(self.articulation_view, getter):
                g = getattr(self.articulation_view, getter)
                try:
                    names = g() if callable(g) else g
                    if names:
                        return list(names)
                except Exception:
                    pass
        return None

    def _try_discover_dofs(self):
        """Descobre DOFs quando a sim já expôs os nomes; não lança exceção se ainda não deu."""
        names = self._get_dof_names_safe()
        if not names:
            return False

        self._dof_names = names

        def find(tokens: List[str]) -> List[int]:
            out = []
            for i, nm in enumerate(names):
                low = str(nm).lower()
                if any(t in low for t in tokens):
                    out.append(i)
            return out

        pn = self.prefer_names
        ov = self._pending_dof or {}

        # steer
        if ov.get("steer_names"):
            self._idx["steer"] = [names.index(n) for n in ov["steer_names"]]
        else:
            self._idx["steer"] = find([t.lower() for t in pn["steer"]])

        # rodas L/R
        if ov.get("rear_left_drive_name"):
            self._idx["rear_left"] = names.index(ov["rear_left_drive_name"])
        else:
            c = find([t.lower() for t in pn["rear_left_drive"]])
            self._idx["rear_left"] = c[0] if c else None

        if ov.get("rear_right_drive_name"):
            self._idx["rear_right"] = names.index(ov["rear_right_drive_name"])
        else:
            c = find([t.lower() for t in pn["rear_right_drive"]])
            self._idx["rear_right"] = c[0] if c else None

        # fork
        if ov.get("fork_name"):
            self._idx["fork"] = names.index(ov["fork_name"])
        else:
            c = find([t.lower() for t in pn["fork_prismatic"]])
            self._idx["fork"] = c[0] if c else None

        # tilt
        if ov.get("mast_tilt_name"):
            self._idx["mast_tilt"] = names.index(ov["mast_tilt_name"])
        else:
            c = find([t.lower() for t in pn["mast_tilt"]])
            self._idx["mast_tilt"] = c[0] if c else None

        # avisos úteis
        if len(self._idx["steer"]) == 0:
            print("[ForkliftC] Aviso: juntas de direção não encontradas por heurística.")
        if self._idx["rear_left"] is None or self._idx["rear_right"] is None:
            print("[ForkliftC] Aviso: rodas traseiras motrizes não identificadas.")
        if self._idx["fork"] is None:
            print("[ForkliftC] Aviso: junta prismática do garfo não encontrada.")
        if self._idx["mast_tilt"] is None:
            print("[ForkliftC] Aviso: junta de tilt não encontrada.")

        self._dofs_ready = True
        if self._debug:
            self.print_dofs()
        return True

    # ----------- Ação -----------
    def write_action(self, step_size: float):
        self._ensure_initialized()

        # descobre DOFs na primeira oportunidade
        if not self._dofs_ready:
            if not self._try_discover_dofs():
                # ainda não há nomes → não faz nada neste frame (evita crash)
                # opcional: log leve, apenas uma vez
                if not hasattr(self, "_warned_init"):
                    print("[ForkliftC] Aguardando inicialização do ArticulationView para mapear DOFs…")
                    self._warned_init = True
                return

        # lê ação [v,w]
        v, w = 0.0, 0.0
        try:
            vals = self.action.get_value()
            v = float(vals[0]); w = float(vals[1])
        except Exception:
            pass

        steer = self._ackermann_steer(v, w)

        n = self.articulation_view.num_dofs
        pos = self.articulation_view.get_joint_positions()
        vel = self.articulation_view.get_joint_velocities()

        def _ensure_1d(x, fill=0.0):
            if x is None:
                return np.full(n, fill, dtype=np.float32), "1d"
            x = np.array(x, dtype=np.float32)
            return (x[0].copy(), "2d") if x.ndim == 2 else (x.copy(), "1d")

        pos, pos_mode = _ensure_1d(pos, 0.0)
        vel, vel_mode = _ensure_1d(vel, 0.0)

        # steer → posição
        for j in self._idx["steer"]:
            if j is not None:
                pos[j] = steer

        if pos_mode == "1d":
            try:
                self.articulation_view.set_joint_positions(pos)
            except Exception:
                self.articulation_view.set_joint_positions(np.expand_dims(pos, 0))
        else:
            self.articulation_view.set_joint_positions(np.expand_dims(pos, 0))

        # tração traseira → ω = v / R
        R = float(self.wheel_radius)
        if self._idx["rear_left"] is not None:
            vel[self._idx["rear_left"]] = v / R
        if self._idx["rear_right"] is not None:
            vel[self._idx["rear_right"]] = v / R

        # garfo/tilt = 0 por padrão
        if self._idx["fork"] is not None:
            vel[self._idx["fork"]] = 0.0
        if self._idx["mast_tilt"] is not None:
            vel[self._idx["mast_tilt"]] = 0.0

        if vel_mode == "1d":
            try:
                self.articulation_view.set_joint_velocities(vel)
            except Exception:
                self.articulation_view.set_joint_velocities(np.expand_dims(vel, 0))
        else:
            self.articulation_view.set_joint_velocities(np.expand_dims(vel, 0))

    # ----------- Pose (LOCAL) -----------
    @staticmethod
    def _set_local_pose_static(prim_path: str, position, orientation_xyzw):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)

        ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
        t_op = ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        o_op = ops.get(UsdGeom.XformOp.TypeOrient)    or xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

        ForkliftC._set_vec_by_precision(t_op, tuple(position))
        ForkliftC._set_quat_by_precision(o_op, tuple(orientation_xyzw))

    # ----------- Utils -----------
    def _ackermann_steer(self, v: float, w: float) -> float:
        if abs(v) < 1e-4:
            return 0.0
        steer = math.atan((w * self.wheel_base) / max(abs(v), 1e-4))
        steer = steer if v >= 0.0 else -steer
        return max(-self.steer_limit_rad, min(self.steer_limit_rad, steer))

    # ----------- Debug -----------
    def print_dofs(self):
        print("==== DOF names (index : name) ====")
        for i, n in enumerate(self._dof_names):
            print(f"{i:03d} : {n}")
        print("---- mapeamento atual ----")
        print("steer idx:", self._idx["steer"])
        print("rear_left idx:", self._idx["rear_left"])
        print("rear_right idx:", self._idx["rear_right"])
        print("fork idx:", self._idx["fork"])
        print("mast_tilt idx:", self._idx["mast_tilt"])

    # ----------- Câmeras (iguais às suas, com precisão segura) -----------
    #####################################################
    ###REVER A IMPLEMENTACAO E PASSAR PARA SENSORES #####
    #####################################################
    def _ensure_cam_xform(self, cam_path: str) -> UsdGeom.Xformable:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(cam_path)
        return UsdGeom.Xformable(prim)

    def mount_bev_camera_topdown(
        self,
        rel_path="sensors/bev_topdown",
        height_m=5.0,
        view_width_m=14.0,
        view_height_m=14.0,
    ) -> str:
        stage = omni.usd.get_context().get_stage()
        cam_path = f"{self.prim_path}/{rel_path}"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
        cam.CreateHorizontalApertureAttr(view_width_m * 10.0)   # USD usa mm → 1m = 1000mm → *10 = cm (compatível com schema)
        cam.CreateVerticalApertureAttr(view_height_m * 10.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))

        xform = self._ensure_cam_xform(cam_path)
        ops = xform.GetOrderedXformOps()
        t_op = ops[0] if ops else xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        self._set_vec_by_precision(t_op, (0.0, 0.0, float(height_m)))
        return cam_path


    def mount_bev_camera_forward_down(
        self,
        rel_path="sensors/bev_front_down",
        forward_m=0.8, height_m=1.8,
        pitch_down_deg=55.0, fov_deg=70.0,
    ) -> str:
        stage = omni.usd.get_context().get_stage()
        cam_path = f"{self.prim_path}/{rel_path}"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

        # focal a partir do FOV (apertura “equivalente” de 36mm)
        horiz_ap_mm = 36.0
        focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(fov_deg) * 0.5)
        cam.CreateHorizontalApertureAttr(horiz_ap_mm)
        cam.CreateVerticalApertureAttr(horiz_ap_mm)  # mantém aspecto no Render Product
        cam.CreateFocalLengthAttr(focal_mm)

        xform = self._ensure_cam_xform(cam_path)
        ops = xform.GetOrderedXformOps()
        t_op = ops[0] if ops else xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        o_op = (ops[1] if len(ops) > 1 else xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble))
        self._set_vec_by_precision(t_op, (float(forward_m), 0.0, float(height_m)))
        q_xyzw = rot_utils.euler_angles_to_quats(np.array([0.0, -math.radians(pitch_down_deg), 0.0]))
        self._set_quat_by_precision(o_op, q_xyzw)
        return cam_path


    def mount_realsense_rgbd(
        self,
        rel_path="sensors/realsense",
        forward_m=0.4, height_m=1.2, pitch_down_deg=10.0,
        hfov_deg=69.0,
    ) -> str:
        stage = omni.usd.get_context().get_stage()
        cam_path = f"{self.prim_path}/{rel_path}"
        cam = UsdGeom.Camera.Define(stage, cam_path)

        horiz_ap_mm = 36.0
        focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(hfov_deg) * 0.5)
        cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        cam.CreateHorizontalApertureAttr(horiz_ap_mm)
        cam.CreateVerticalApertureAttr(horiz_ap_mm)
        cam.CreateFocalLengthAttr(focal_mm)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

        xform = self._ensure_cam_xform(cam_path)
        ops = xform.GetOrderedXformOps()
        t_op = ops[0] if ops else xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        o_op = (ops[1] if len(ops) > 1 else xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble))
        self._set_vec_by_precision(t_op, (float(forward_m), 0.0, float(height_m)))
        q_xyzw = rot_utils.euler_angles_to_quats(np.array([0.0, -math.radians(pitch_down_deg), 0.0]))
        self._set_quat_by_precision(o_op, q_xyzw)
        return cam_path


    def mount_zed_stereo(
        self,
        base_rel="sensors/zed",
        forward_m=0.5, height_m=1.3,
        baseline_m=0.12, hfov_deg=90.0,
    ) -> Tuple[str, str]:
        stage = omni.usd.get_context().get_stage()
        left_path  = f"{self.prim_path}/{base_rel}/left"
        right_path = f"{self.prim_path}/{base_rel}/right"

        def _make(path, y_offset):
            cam = UsdGeom.Camera.Define(stage, path)
            horiz_ap_mm = 36.0
            focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(hfov_deg) * 0.5)
            cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
            cam.CreateHorizontalApertureAttr(horiz_ap_mm)
            cam.CreateVerticalApertureAttr(horiz_ap_mm)
            cam.CreateFocalLengthAttr(focal_mm)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
            xform = self._ensure_cam_xform(path)
            ops = xform.GetOrderedXformOps()
            t_op = ops[0] if ops else xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
            self._set_vec_by_precision(t_op, (float(forward_m), float(y_offset), float(height_m)))
            return path

        l = _make(left_path,  -baseline_m / 2.0)
        r = _make(right_path, +baseline_m / 2.0)
        return l, r

    def build_render_products(self, resolutions: Optional[Dict[str, Tuple[int,int]]] = None) -> Dict[str, object]:
        """
        Cria Render Products do Replicator para cada câmera do robô.
        Ex.: resolutions={"bev_topdown": (1024,1024), "realsense": (1280,720)}
        """
        try:
            import omni.replicator.core as rep
        except Exception as e:
            print(f"[ForkliftC] Replicator indisponível, pulando render products: {e}")
            self.render_products = {}
            return {}

        # defaults
        res = {
            "bev_topdown": (1024, 1024),
            "bev_front_down": (1280, 720),
            "realsense": (1280, 720),
            "zed_left": (1280, 720),
            "zed_right": (1280, 720),
        }
        if resolutions:
            res.update(resolutions)

        products = {}
        for key, path in self.cam_paths.items():
            # NOTE: cam_paths["zed_left"] e ["zed_right"] já são strings separadas
            if isinstance(path, str):
                products[key] = rep.create.render_product(path, resolution=res.get(key, (1280, 720)))
        self.render_products = products
        return products




# @ROBOTS.register()
# class ForkliftC(Robot):
#     """
#     Adaptador MobilityGen para o asset USD 'forklift_c' (empilhadeira).

#     Integração com a base Robot:
#       - build(cls, prim_path): cria e retorna a instância pronta p/ uso nos cenários
#       - write_action(self, step_size): lê self.action=[v,w] e comanda a articulação

#     Modelo de ação:
#       self.action = [v, w]  (m/s, rad/s)  → Ackermann → steer (rad)
#       Garfo (lift) e Tilt são mantidos 0 por padrão (podem ser expostos depois conforme cenário).
#     """

#     # ===== Parâmetros exigidos pela base Robot =====
#     physics_dt: float = 1.0 / 120.0
#     z_offset: float = 0.10  # útil ao spawn para cair no chão sem interpenetração

#     # câmera "chase" (traseira elevada)
#     chase_camera_base_path: str = ""         # vazio = no prim raiz; ajuste se tiver XForm específico (ex.: "base_link")
#     chase_camera_x_offset: float = -1.5
#     chase_camera_z_offset: float = 0.9
#     chase_camera_tilt_angle: float = 50.0

#     # mapa de ocupância (default razoável em indoor)
#     occupancy_map_radius: float = 1.5
#     occupancy_map_z_min: float = -0.20
#     occupancy_map_z_max: float = 2.50
#     occupancy_map_cell_size: float = 0.05
#     occupancy_map_collision_radius: float = 0.5

#     # câmera frontal "configurável" (a base chama build_front_camera)
#     front_camera_type = HawkCamera            # troque se usar outro sensor final
#     front_camera_base_path: str = "sensors/front_cam"
#     front_camera_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
#     front_camera_translation: Tuple[float, float, float] = (0.55, 0.0, 1.40)

#     # ganhos teleop (teclado/gamepad) lidos pelos cenários padrão
#     keyboard_linear_velocity_gain: float = 1.0
#     keyboard_angular_velocity_gain: float = 1.0
#     gamepad_linear_velocity_gain: float = 1.0
#     gamepad_angular_velocity_gain: float = 1.0

#     # ruído/limites da ação aleatória (usado por alguns cenários)
#     random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 1.0)
#     random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
#     random_action_linear_acceleration_std: float = 5.0
#     random_action_angular_acceleration_std: float = 5.0
#     random_action_grid_pose_sampler_grid_size: float = 5.0

#     # seguindo trajetória (seu PID/ganhos podem ser calibrados depois)
#     path_following_speed: float = 1.0
#     path_following_angular_gain: float = 1.0
#     path_following_stop_distance_threshold: float = 0.2
#     path_following_forward_angle_threshold: float = math.pi
#     path_following_target_point_offset_meters: float = 0.0

#     # ===== Parâmetros específicos da empilhadeira =====
#     usd_path: str =  "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd"   # ajuste pro seu asset (omniverse://... ou relativo)
#     wheel_radius: float = 0.165
#     wheel_base: float = 1.10
#     track_width: float = 0.95
#     steer_limit_rad: float = math.radians(40.0)
#     rear_diff_drive: bool = False  # manter false (eixo único), mas suportado se quisesse variar

#     # Heurística p/ DOFs (pode sobrepor via kwargs no build)
#     prefer_names = dict(
#         steer=["steer", "steering", "front_axle_steer"],
#         rear_left_drive=["rear_left_wheel", "rear_l_wheel", "rl_wheel", "drive_left"],
#         rear_right_drive=["rear_right_wheel", "rear_r_wheel", "rr_wheel", "drive_right"],
#         fork_prismatic=["fork", "lift", "prismatic"],
#         mast_tilt=["tilt", "mast_tilt"],
#     )

#     # ----------- construção (Classe → Instância), padrão MobilityGen -----------
#     @classmethod
#     def build(
#         cls,
#         prim_path: str,
#         *,
#         usd_path: Optional[str] = None,
#         # flags extras de câmera (opcionais)
#         enable_bev: bool = True,
#         enable_bev_front: bool = True,
#         enable_realsense: bool = True,
#         enable_zed: bool = True,
#         # overrides DOF (strings exatas)
#         steer_names: Optional[List[str]] = None,
#         rear_left_drive_name: Optional[str] = None,
#         rear_right_drive_name: Optional[str] = None,
#         fork_name: Optional[str] = None,
#         mast_tilt_name: Optional[str] = None,
#         # pose inicial (world/local do prim raiz)
#         position: Sequence[float] = (0.0, 0.0, 0.0),
#         orientation_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
#         # debug
#         debug: bool = False,
#     ) -> "ForkliftC":
#         """
#         Constrói a empilhadeira e retorna a instância pronta para os cenários padrão.
#         """
#         usd = usd_path or cls.usd_path
#         if not usd or not isinstance(usd, str):
#             raise RuntimeError("Defina 'usd_path' válido para o forklift_c.usd (Copy Path do Content Browser).")

#         # 1) Referencia o asset e posiciona o prim
#         add_reference_to_stage(usd, prim_path)
#         cls._set_local_pose_static(prim_path=prim_path, position=position, orientation_xyzw=orientation_xyzw)

#         # 2) Cria a visão de articulação (controle de juntas)
#         art_view = _ArticulationView(prim_paths_expr=prim_path, name="forklift_art")
#         try:
#             art_view.initialize()  # ok se a sim já estiver pronta
#         except Exception:
#             # sim ainda não está pronta; o cenário chamará initialize() em set_pose_2d()
#             pass


#         # 3) Constrói a câmera frontal padrão (usada pelos cenários)
#         front_cam = cls.build_front_camera(prim_path)

#         # 4) Instancia o Robot base
#         self = cls(
#             prim_path=prim_path,
#             robot=art_view,               # usamos o próprio ArticulationView como "robot" com API mínima
#             articulation_view=art_view,
#             front_camera=front_cam,
#         )

#         # 5) Estado interno e DOFs
#         self._debug = debug
#         self._dof_names: List[str] = []
#         self._idx = dict(steer=[], rear_left=None, rear_right=None, fork=None, mast_tilt=None)

#         self._discover_dofs(
#             art_view,
#             steer_names=steer_names,
#             rear_left_drive_name=rear_left_drive_name,
#             rear_right_drive_name=rear_right_drive_name,
#             fork_name=fork_name,
#             mast_tilt_name=mast_tilt_name,
#         )
#         if debug:
#             self.print_dofs()

#         # 6) Câmeras auxiliares (opcionais, úteis p/ BEV/rgbd/estéreo)
#         self.cam_paths: Dict[str, Union[str, Tuple[str, str]]] = {}
#         if enable_bev:
#             self.cam_paths["bev_topdown"] = self.mount_bev_camera_topdown()
#         if enable_bev_front:
#             self.cam_paths["bev_front_down"] = self.mount_bev_camera_forward_down()
#         if enable_realsense:
#             self.cam_paths["realsense"] = self.mount_realsense_rgbd()
#         if enable_zed:
#             l, r = self.mount_zed_stereo()
#             self.cam_paths["zed_left"], self.cam_paths["zed_right"] = l, r

#         # 7) Câmera de perseguição (útil p/ visualização)
#         try:
#             self.build_chase_camera()
#         except Exception:
#             pass

#         return self
#     def _ensure_initialized(self):
#         try:
#             self.articulation_view.initialize()
#         except Exception:
#             pass
#     # ----------- Ação (cenários chamam write_action(step_size)) -----------
#     def write_action(self, step_size: float):
#         """
#         Lê self.action = [v, w] (m/s, rad/s), converte para steer (Ackermann) e aplica:
#          - steer: alvo de POSIÇÃO
#          - rodas traseiras: alvo de VELOCIDADE (ω = v / R)
#          - garfo/tilt: velocidade 0 (padrão; pode expandir cenário para comandar)
#         """
#         self._ensure_initialized()
#         v, w = 0.0, 0.0
#         try:
#             vals = self.action.get_value()
#             v = float(vals[0]); w = float(vals[1])
#         except Exception:
#             pass

#         steer = self._ackermann_steer(v, w)

#         # estado atual (1D/2D safe)
#         n = self.articulation_view.num_dofs
#         pos = self.articulation_view.get_joint_positions()
#         vel = self.articulation_view.get_joint_velocities()

#         def _ensure_1d(x, fill=0.0):
#             if x is None:
#                 return np.full(n, fill, dtype=np.float32), "1d"
#             x = np.array(x, dtype=np.float32)
#             return (x[0].copy(), "2d") if x.ndim == 2 else (x.copy(), "1d")

#         pos, pos_mode = _ensure_1d(pos, 0.0)
#         vel, vel_mode = _ensure_1d(vel, 0.0)

#         # steer → posição-alvo
#         for j in self._idx["steer"]:
#             if j is not None:
#                 pos[j] = steer

#         # commit posição
#         if pos_mode == "1d":
#             try:
#                 self.articulation_view.set_joint_positions(pos)
#             except Exception:
#                 self.articulation_view.set_joint_positions(np.expand_dims(pos, 0))
#         else:
#             self.articulation_view.set_joint_positions(np.expand_dims(pos, 0))

#         # tração traseira → velocidade (ω = v / R)
#         R = float(self.wheel_radius)
#         if self._idx["rear_left"] is not None:
#             vel[self._idx["rear_left"]] = v / R
#         if self._idx["rear_right"] is not None:
#             vel[self._idx["rear_right"]] = v / R

#         # garfo / tilt → 0 por padrão (ou exponha buffers próprios depois)
#         if self._idx["fork"] is not None:
#             vel[self._idx["fork"]] = 0.0
#         if self._idx["mast_tilt"] is not None:
#             vel[self._idx["mast_tilt"]] = 0.0

#         # commit velocidade
#         if vel_mode == "1d":
#             try:
#                 self.articulation_view.set_joint_velocities(vel)
#             except Exception:
#                 self.articulation_view.set_joint_velocities(np.expand_dims(vel, 0))
#         else:
#             self.articulation_view.set_joint_velocities(np.expand_dims(vel, 0))

#     # ----------- Utilidades de pose -----------
    
#     @staticmethod
#     def _set_local_pose_static(prim_path: str, position, orientation_xyzw):
#         """
#         Define a pose LOCAL do prim (Translate + Orient), convertendo para o tipo
#         exato que o atributo USD espera (Vec3d/Vec3f e Quatd/Quatf).
#         """
#         stage = omni.usd.get_context().get_stage()
#         prim = stage.GetPrimAtPath(prim_path)
#         xform = UsdGeom.Xformable(prim)

#         # pega (ou cria) ops
#         ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
#         t_op = ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp()
#         o_op = ops.get(UsdGeom.XformOp.TypeOrient)    or xform.AddOrientOp()

#         # normaliza inputs
#         px, py, pz = [float(v) for v in position]
#         qx, qy, qz, qw = [float(v) for v in orientation_xyzw]

#         # --- translate: respeita o tipo do atributo (Vec3d x Vec3f)
#         t_attr = t_op.GetOpAttr()
#         t_typ = str(t_attr.GetTypeName())
#         if "float3" in t_typ or "GfVec3f" in t_typ:
#             t_attr.Set(Gf.Vec3f(px, py, pz))
#         else:
#             t_attr.Set(Gf.Vec3d(px, py, pz))

#         # --- orient: respeita o tipo do atributo (Quatd x Quatf)
#         o_attr = o_op.GetOpAttr()
#         o_typ = str(o_attr.GetTypeName())
#         if "quatf" in o_typ or "GfQuatf" in o_typ:
#             o_attr.Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
#         else:
#             o_attr.Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))


#     def _ackermann_steer(self, v: float, w: float) -> float:
#         """Converte v (m/s) e yaw-rate w (rad/s) em ângulo de direção clampado."""
#         if abs(v) < 1e-4:
#             return 0.0
#         steer = math.atan((w * self.wheel_base) / max(abs(v), 1e-4))
#         steer = steer if v >= 0.0 else -steer
#         return max(-self.steer_limit_rad, min(self.steer_limit_rad, steer))

#     # ----------- Descoberta de DOFs -----------
#     def _discover_dofs(
#         self,
#         art_view: _ArticulationView,
#         *,
#         steer_names: Optional[List[str]] = None,
#         rear_left_drive_name: Optional[str] = None,
#         rear_right_drive_name: Optional[str] = None,
#         fork_name: Optional[str] = None,
#         mast_tilt_name: Optional[str] = None,
#     ):
#         dof_names = art_view.get_dof_names()
#         self._dof_names = list(dof_names)

#         def _find_indices_by_tokens(tokens: List[str]) -> List[int]:
#             idxs = []
#             for i, name in enumerate(dof_names):
#                 low = name.lower()
#                 if any(t in low for t in tokens):
#                     idxs.append(i)
#             return idxs

#         # STEER
#         if steer_names:
#             self._idx["steer"] = [dof_names.index(n) for n in steer_names]
#         else:
#             self._idx["steer"] = _find_indices_by_tokens([t.lower() for t in self.prefer_names["steer"]])

#         # DRIVE L/R
#         if rear_left_drive_name:
#             self._idx["rear_left"] = dof_names.index(rear_left_drive_name)
#         else:
#             c = _find_indices_by_tokens([t.lower() for t in self.prefer_names["rear_left_drive"]])
#             self._idx["rear_left"] = c[0] if c else None

#         if rear_right_drive_name:
#             self._idx["rear_right"] = dof_names.index(rear_right_drive_name)
#         else:
#             c = _find_indices_by_tokens([t.lower() for t in self.prefer_names["rear_right_drive"]])
#             self._idx["rear_right"] = c[0] if c else None

#         # FORK
#         if fork_name:
#             self._idx["fork"] = dof_names.index(fork_name)
#         else:
#             c = _find_indices_by_tokens([t.lower() for t in self.prefer_names["fork_prismatic"]])
#             self._idx["fork"] = c[0] if c else None

#         # MAST TILT
#         if mast_tilt_name:
#             self._idx["mast_tilt"] = dof_names.index(mast_tilt_name)
#         else:
#             c = _find_indices_by_tokens([t.lower() for t in self.prefer_names["mast_tilt"]])
#             self._idx["mast_tilt"] = c[0] if c else None

#         # avisos úteis
#         if len(self._idx["steer"]) == 0:
#             print("[ForkliftC] Aviso: juntas de direção não encontradas por heurística.")
#         if self._idx["rear_left"] is None or self._idx["rear_right"] is None:
#             print("[ForkliftC] Aviso: rodas traseiras motrizes não encontradas por heurística.")
#         if self._idx["fork"] is None:
#             print("[ForkliftC] Aviso: junta prismática do garfo não encontrada.")
#         if self._idx["mast_tilt"] is None:
#             print("[ForkliftC] Aviso: junta de tilt do mastro não encontrada.")

#     def print_dofs(self):
#         print("==== DOF names (index : name) ====")
#         for i, n in enumerate(self._dof_names):
#             print(f"{i:03d} : {n}")
#         print("---- mapeamento atual ----")
#         print("steer idx:", self._idx["steer"])
#         print("rear_left idx:", self._idx["rear_left"])
#         print("rear_right idx:", self._idx["rear_right"])
#         print("fork idx:", self._idx["fork"])
#         print("mast_tilt idx:", self._idx["mast_tilt"])

#     # ----------- Câmeras auxiliares (opcionais) -----------
#     def _ensure_cam_xform(self, cam_path: str) -> UsdGeom.Xformable:
#         stage = omni.usd.get_context().get_stage()
#         prim = stage.GetPrimAtPath(cam_path)
#         return UsdGeom.Xformable(prim)

#     def mount_bev_camera_topdown(
#         self,
#         rel_path="sensors/bev_topdown",
#         height_m=5.0,
#         view_width_m=14.0,
#         view_height_m=14.0,
#         resolution=(1024, 1024),
#     ) -> str:
#         stage = omni.usd.get_context().get_stage()
#         cam_path = f"{self.prim_path}/{rel_path}"
#         cam = UsdGeom.Camera.Define(stage, cam_path)
#         cam.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
#         cam.CreateHorizontalApertureAttr(view_width_m * 10.0)
#         cam.CreateVerticalApertureAttr(view_height_m * 10.0)
#         cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
#         cam.CreateResolutionAttr(Gf.Vec2i(int(resolution[0]), int(resolution[1])))
#         xform = self._ensure_cam_xform(cam_path)
#         if not xform.GetOrderedXformOps():
#             xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, height_m))
#         else:
#             xform.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.0, 0.0, height_m))
#         return cam_path

#     def mount_bev_camera_forward_down(
#         self,
#         rel_path="sensors/bev_front_down",
#         forward_m=0.8, height_m=1.8,
#         pitch_down_deg=55.0, fov_deg=70.0,
#         resolution=(1280, 720),
#     ) -> str:
#         stage = omni.usd.get_context().get_stage()
#         cam_path = f"{self.prim_path}/{rel_path}"
#         cam = UsdGeom.Camera.Define(stage, cam_path)
#         cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
#         cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
#         cam.CreateResolutionAttr(Gf.Vec2i(int(resolution[0]), int(resolution[1])))
#         horiz_ap_mm = 36.0
#         focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(fov_deg) * 0.5)
#         cam.CreateHorizontalApertureAttr(horiz_ap_mm)
#         cam.CreateVerticalApertureAttr(horiz_ap_mm * (resolution[1] / resolution[0]))
#         cam.CreateFocalLengthAttr(focal_mm)
#         xform = self._ensure_cam_xform(cam_path)
#         if not xform.GetOrderedXformOps():
#             xform.AddTranslateOp().Set(Gf.Vec3d(forward_m, 0.0, height_m))
#             q = rot_utils.euler_angles_to_quats(np.array([0.0, -math.radians(pitch_down_deg), 0.0]), degrees=False)
#             # q = [w,x,y,z]
#             xform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
#         else:
#             ops = xform.GetOrderedXformOps()
#             ops[0].Set(Gf.Vec3d(forward_m, 0.0, height_m))
#             if len(ops) > 1:
#                 q = rot_utils.euler_angles_to_quats(np.array([0.0, -math.radians(pitch_down_deg), 0.0]), degrees=False)
#                 ops[1].Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
#         return cam_path

#     def mount_realsense_rgbd(
#         self,
#         rel_path="sensors/realsense",
#         forward_m=0.4, height_m=1.2, pitch_down_deg=10.0,
#         hfov_deg=69.0, resolution=(1280, 720),
#     ) -> str:
#         stage = omni.usd.get_context().get_stage()
#         cam_path = f"{self.prim_path}/{rel_path}"
#         cam = UsdGeom.Camera.Define(stage, cam_path)
#         horiz_ap_mm = 36.0
#         focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(hfov_deg) * 0.5)
#         cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
#         cam.CreateHorizontalApertureAttr(horiz_ap_mm)
#         cam.CreateVerticalApertureAttr(horiz_ap_mm * (resolution[1] / resolution[0]))
#         cam.CreateFocalLengthAttr(focal_mm)
#         cam.CreateResolutionAttr(Gf.Vec2i(int(resolution[0]), int(resolution[1])))
#         cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
#         xform = self._ensure_cam_xform(cam_path)
#         if not xform.GetOrderedXformOps():
#             xform.AddTranslateOp().Set(Gf.Vec3d(forward_m, 0.0, height_m))
#             q = rot_utils.euler_angles_to_quats(np.array([0.0, -math.radians(pitch_down_deg), 0.0]), degrees=False)
#             xform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
#         else:
#             ops = xform.GetOrderedXformOps()
#             ops[0].Set(Gf.Vec3d(forward_m, 0.0, height_m))
#             if len(ops) > 1:
#                 q = rot_utils.euler_angles_to_quats(np.array([0.0, -math.radians(pitch_down_deg), 0.0]), degrees=False)
#                 ops[1].Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
#         return cam_path

#     def mount_zed_stereo(
#         self,
#         base_rel="sensors/zed",
#         forward_m=0.5, height_m=1.3,
#         baseline_m=0.12, hfov_deg=90.0,
#         resolution=(1280, 720),
#     ) -> Tuple[str, str]:
#         stage = omni.usd.get_context().get_stage()
#         left_path  = f"{self.prim_path}/{base_rel}/left"
#         right_path = f"{self.prim_path}/{base_rel}/right"

#         def _make_cam(path, y_offset):
#             cam = UsdGeom.Camera.Define(stage, path)
#             horiz_ap_mm = 36.0
#             focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(hfov_deg) * 0.5)
#             cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
#             cam.CreateHorizontalApertureAttr(horiz_ap_mm)
#             cam.CreateVerticalApertureAttr(horiz_ap_mm * (resolution[1] / resolution[0]))
#             cam.CreateFocalLengthAttr(focal_mm)
#             cam.CreateResolutionAttr(Gf.Vec2i(int(resolution[0]), int(resolution[1])))
#             cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
#             xform = self._ensure_cam_xform(path)
#             if not xform.GetOrderedXformOps():
#                 xform.AddTranslateOp().Set(Gf.Vec3d(forward_m, y_offset, height_m))
#             else:
#                 xform.GetOrderedXformOps()[0].Set(Gf.Vec3d(forward_m, y_offset, height_m))
#             return path

#         l = _make_cam(left_path,  -baseline_m / 2.0)
#         r = _make_cam(right_path, +baseline_m / 2.0)
#         return l, r

# @ROBOTS.register()
# class ForkliftC(Robot):
#     """
#     Adaptador MobilityGen para o asset USD 'forklift_c' (Isaac Sim).

#     Ação (modo direto): [v_mps, steer_rad, lift_mps, tilt_radps]
#     Compatível com scenarios padrão que usam self.action=[v,w] e write_action(dt).
#     """

#     # ===== Parâmetros gerais =====
#     physics_dt: float = 1.0 / 120.0
#     occupancy_map_z_min: float = -0.2
#     occupancy_map_z_max: float = 3.0

#     # Caminho do USD (ajuste para seu asset)
#     usd_path: str = "Robots/Forklift/forklift_c.usd"
#     prim_path: str = "/World/ForkliftC"

#     # Geometria Ackermann
#     wheel_radius: float = 0.165
#     wheel_base: float = 1.1
#     track_width: float = 0.95
#     steer_limit_rad: float = math.radians(40.0)

#     rear_diff_drive: bool = False  # True → rpm L/R diferentes (se quiser evoluir)

#     # Heurística de nomes de DOF
#     prefer_names = dict(
#         steer=["steer", "steering", "front_axle_steer"],
#         rear_left_drive=["rear_left_wheel", "rear_l_wheel", "rl_wheel", "drive_left"],
#         rear_right_drive=["rear_right_wheel", "rear_r_wheel", "rr_wheel", "drive_right"],
#         fork_prismatic=["fork", "lift", "prismatic"],
#         mast_tilt=["tilt", "mast_tilt"],
#     )

#     # Ganhos usados por KeyboardTeleoperationScenario
#     keyboard_linear_velocity_gain: float = 1.0
#     keyboard_angular_velocity_gain: float = 1.0

#     def __init__(self, **kwargs):
#         """
#         kwargs úteis:
#           usd_path/prim_path: override de caminhos.
#           enable_bev/_bev_front/_realsense/_zed: se True, monta câmeras no build().
#           steer_names/rear_*_name/fork_name/mast_tilt_name: DOFs explícitos.
#         """
#         # overrides de caminhos
#         self._usd_path = kwargs.pop("usd_path", None)
#         self._prim_path = kwargs.pop("prim_path", None)
#         self._debug = kwargs.pop("debug", False)

#         # Flags de câmera (ligadas por padrão; desligue no build() se quiser)
#         self._enable_bev = kwargs.pop("enable_bev", True)
#         self._enable_bev_front = kwargs.pop("enable_bev_front", True)
#         self._enable_realsense = kwargs.pop("enable_realsense", True)
#         self._enable_zed = kwargs.pop("enable_zed", True)

#         # Overrides de DOFs
#         self._override = dict(
#             steer_names=kwargs.pop("steer_names", None),  # List[str]
#             rear_left_drive_name=kwargs.pop("rear_left_drive_name", None),
#             rear_right_drive_name=kwargs.pop("rear_right_drive_name", None),
#             fork_name=kwargs.pop("fork_name", None),
#             mast_tilt_name=kwargs.pop("mast_tilt_name", None),
#         )

#         super().__init__(**kwargs)

#         self._art: Optional[ArticulationView] = None
#         self._mode: Optional[str] = None

#         # índices/dofs preenchidos após build()
#         self._dof_names: List[str] = []
#         self._idx = dict(
#             steer=[],             # 1 ou 2 DOFs de direção
#             rear_left=None,
#             rear_right=None,
#             fork=None,
#             mast_tilt=None,
#         )

#         # paths das câmeras criadas
#         self.cam_paths: Dict[str, Union[str, Tuple[str, str]]] = {}

#     # ---------------------- BUILD ----------------------
#     def build(
#         self,
#         usd_path: Optional[str] = None,
#         prim_path: Optional[str] = None,
#         world=None,
#         position: Sequence[float] = (0.0, 0.0, 0.0),
#         orientation_xyzw: Sequence[float] = (0, 0, 0, 1),
#         enable_bev: Optional[bool] = None,
#         enable_bev_front: Optional[bool] = None,
#         enable_realsense: Optional[bool] = None,
#         enable_zed: Optional[bool] = None,
#         **kwargs,
#     ):
#         """
#         Spawna o USD, inicializa ArticulationView, descobre DOFs e, opcionalmente, monta câmeras.
#         Precedência: flag passada no build() > valor do __init__.
#         """
#         usd = usd_path or self._usd_path or self.usd_path
#         prim = prim_path or self._prim_path or self.prim_path
#         if not usd or not isinstance(usd, str):
#             raise RuntimeError("Defina 'usd_path' (Copy Path no Content Browser).")

#         # 1) referencia o asset no stage e define pose local
#         add_reference_to_stage(usd, prim)
#         self._set_local_pose(prim_path=prim, position=position, orientation_xyzw=orientation_xyzw)

#         # 2) cria ArticulationView
#         self._art = ArticulationView(prim_paths_expr=prim, name="forklift_art")
#         self._art.initialize()
#         self._mode = "articulation"

#         # 3) mapeia DOFs
#         self._discover_dofs(**self._override)
#         if self._debug:
#             self.print_dofs()

#         # 4) monta câmeras conforme flags
#         use_bev        = self._pick_flag(enable_bev,        self._enable_bev)
#         use_bev_front  = self._pick_flag(enable_bev_front,  self._enable_bev_front)
#         use_rs         = self._pick_flag(enable_realsense,  self._enable_realsense)
#         use_zed        = self._pick_flag(enable_zed,        self._enable_zed)

#         self.cam_paths.clear()
#         if use_bev:
#             self.cam_paths["bev_topdown"] = self.mount_bev_camera_topdown()
#         if use_bev_front:
#             self.cam_paths["bev_front_down"] = self.mount_bev_camera_forward_down()
#         if use_rs:
#             self.cam_paths["realsense"] = self.mount_realsense_rgbd()
#         if use_zed:
#             l, r = self.mount_zed_stereo()
#             self.cam_paths["zed_left"] = l
#             self.cam_paths["zed_right"] = r

#         return self

#     @staticmethod
#     def _pick_flag(arg_val: Optional[bool], init_val: bool) -> bool:
#         """Escolhe entre valor passado no build() e default definido no __init__."""
#         return init_val if arg_val is None else bool(arg_val)

#     # ---------------------- AÇÃO ----------------------
#     def write_action(self, action: Union[Sequence[float], Dict[str, float], float, int]):
#         """
#         Aplica ação. Modos:
#           - write_action(dt): lê self.action=[v,w], converte p/ steer (Ackermann)
#           - [v, steer, (lift), (tilt)]
#           - [v, w]  / {"v":..,"w":..}  (converte w→steer)
#           - {"v":..,"steer":..,(lift),(tilt)}
#         """
#         if self._art is None:
#             raise RuntimeError("Robot not built. Chame build() antes de write_action().")

#         # ---- interpretar entrada ----
#         v = 0.0; steer = 0.0; lift_v = 0.0; tilt_v = 0.0

#         if isinstance(action, (int, float)):
#             # cenário: action=dt; lê [v,w]
#             w = 0.0
#             try:
#                 a = getattr(self, "action", None)
#                 if a is not None:
#                     vals = a.get_value()
#                     v = float(vals[0]); w = float(vals[1])
#             except Exception:
#                 v, w = 0.0, 0.0
#             steer = self._ackermann_steer(v, w)

#         elif isinstance(action, dict):
#             v = float(action.get("v", 0.0))
#             if "steer" in action:
#                 steer = float(action.get("steer", 0.0))
#             else:
#                 steer = self._ackermann_steer(v, float(action.get("w", 0.0)))
#             lift_v = float(action.get("lift", 0.0))
#             tilt_v = float(action.get("tilt", 0.0))

#         else:
#             seq = list(action)
#             if len(seq) == 2:
#                 v, w = float(seq[0]), float(seq[1])
#                 steer = self._ackermann_steer(v, w)
#             else:
#                 v = float(seq[0]) if len(seq) > 0 else 0.0
#                 steer = float(seq[1]) if len(seq) > 1 else 0.0
#                 lift_v = float(seq[2]) if len(seq) > 2 else 0.0
#                 tilt_v = float(seq[3]) if len(seq) > 3 else 0.0

#         # clamp final
#         steer = max(-self.steer_limit_rad, min(self.steer_limit_rad, steer))

#         # ---- estado atual (1D/2D safe) ----
#         n = self._art.num_dofs
#         pos = self._art.get_joint_positions()
#         vel = self._art.get_joint_velocities()

#         def _ensure_1d(x, fill=0.0):
#             if x is None:
#                 return np.full(n, fill, dtype=np.float32), "1d"
#             x = np.array(x, dtype=np.float32)
#             return (x[0].copy(), "2d") if x.ndim == 2 else (x.copy(), "1d")

#         pos, pos_mode = _ensure_1d(pos, 0.0)
#         vel, vel_mode = _ensure_1d(vel, 0.0)

#         # ---- direção: alvo de posição ----
#         for j in self._idx["steer"]:
#             if j is not None:
#                 pos[j] = steer

#         if pos_mode == "1d":
#             try:
#                 self._art.set_joint_positions(pos)
#             except Exception:
#                 self._art.set_joint_positions(np.expand_dims(pos, 0))
#         else:
#             self._art.set_joint_positions(np.expand_dims(pos, 0))

#         # ---- tração traseira: ω = v / R ----
#         R = float(self.wheel_radius)
#         if self.rear_diff_drive and (self._idx["rear_left"] is not None) and (self._idx["rear_right"] is not None):
#             vel[self._idx["rear_left"]]  = v / R
#             vel[self._idx["rear_right"]] = v / R
#         else:
#             if self._idx["rear_left"]  is not None:
#                 vel[self._idx["rear_left"]]  = v / R
#             if self._idx["rear_right"] is not None:
#                 vel[self._idx["rear_right"]] = v / R

#         # ---- garfo (prismática) e tilt (revoluto) ----
#         if self._idx["fork"] is not None:
#             vel[self._idx["fork"]] = lift_v
#         if self._idx["mast_tilt"] is not None:
#             vel[self._idx["mast_tilt"]] = tilt_v

#         if vel_mode == "1d":
#             try:
#                 self._art.set_joint_velocities(vel)
#             except Exception:
#                 self._art.set_joint_velocities(np.expand_dims(vel, 0))
#         else:
#             self._art.set_joint_velocities(np.expand_dims(vel, 0))

#     # ---------------------- Pose ----------------------
#     def _set_local_pose(self, prim_path: str, position, orientation_xyzw):
#         """
#         Define pose LOCAL do prim (Translate + Orient XformOps).
#         Se o prim é filho direto de /World, local == world.
#         """
#         stage = omni.usd.get_context().get_stage()
#         prim = stage.GetPrimAtPath(prim_path)
#         xform = UsdGeom.Xformable(prim)

#         # Reusa ops existentes, senão cria
#         ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
#         t_op = ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp()
#         o_op = ops.get(UsdGeom.XformOp.TypeOrient)    or xform.AddOrientOp()

#         px, py, pz = [float(v) for v in position]
#         qx, qy, qz, qw = [float(v) for v in orientation_xyzw]

#         t_op.Set(Gf.Vec3d(px, py, pz))
#         o_op.Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))

#     # ---------------------- TELEOP / UTILS ----------------------
#     def set_pose_2d(self, pose_xytheta):
#         """
#         Define a pose planar (x, y, yaw) para o prim raiz do robô.
#         Mantém Z=0 (ajuste se seu terreno exigir offset).
#         """
#         x, y, yaw = float(pose_xytheta[0]), float(pose_xytheta[1]), float(pose_xytheta[2])
#         qx = qy = 0.0
#         qz = math.sin(0.5 * yaw); qw = math.cos(0.5 * yaw)
#         self._set_local_pose(
#             prim_path=self.prim_path,
#             position=(x, y, 0.0),
#             orientation_xyzw=(qx, qy, qz, qw),
#         )

#     def _ackermann_steer(self, v: float, w: float) -> float:
#         """Converte v (m/s) e yaw-rate w (rad/s) em ângulo de direção (rad)."""
#         if abs(v) < 1e-4:
#             return 0.0
#         steer = math.atan((w * self.wheel_base) / max(abs(v), 1e-4))
#         return max(-self.steer_limit_rad, min(self.steer_limit_rad, steer if v >= 0 else -steer))

#     # ====================== CÂMERAS ======================
#     def _ensure_cam_xform(self, cam_path: str) -> UsdGeom.Xformable:
#         """Retorna o Xformable da câmera para atualizar ops sem empilhar."""
#         stage = omni.usd.get_context().get_stage()
#         prim = stage.GetPrimAtPath(cam_path)
#         return UsdGeom.Xformable(prim)

#     def mount_bev_camera_topdown(
#         self,
#         rel_path="sensors/bev_topdown",
#         height_m=5.0,
#         view_width_m=14.0,
#         view_height_m=14.0,
#         resolution=(1024, 1024),
#     ) -> str:
#         """Câmera ortográfica top-down (ideal para GT BEV). m/px = view_width_m / resolution[0]."""
#         stage = omni.usd.get_context().get_stage()
#         cam_path = f"{self.prim_path}/{rel_path}"
#         cam = UsdGeom.Camera.Define(stage, cam_path)

#         cam.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
#         cam.CreateHorizontalApertureAttr(view_width_m * 10.0)
#         cam.CreateVerticalApertureAttr(view_height_m * 10.0)
#         cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
#         cam.CreateResolutionAttr(Gf.Vec2i(int(resolution[0]), int(resolution[1])))

#         xform = self._ensure_cam_xform(cam_path)
#         if not xform.GetOrderedXformOps():
#             xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, height_m))
#         else:
#             ops = xform.GetOrderedXformOps()
#             ops[0].Set(Gf.Vec3d(0.0, 0.0, height_m))
#         return cam_path

#     def mount_bev_camera_forward_down(
#         self,
#         rel_path="sensors/bev_front_down",
#         forward_m=0.8,
#         height_m=1.8,
#         pitch_down_deg=55.0,
#         fov_deg=70.0,
#         resolution=(1280, 720),
#     ) -> str:
#         """Câmera perspectiva à frente e acima, inclinada para baixo."""
#         stage = omni.usd.get_context().get_stage()
#         cam_path = f"{self.prim_path}/{rel_path}"
#         cam = UsdGeom.Camera.Define(stage, cam_path)

#         cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
#         cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
#         cam.CreateResolutionAttr(Gf.Vec2i(int(resolution[0]), int(resolution[1])))

#         # focal a partir do FOV (aperture 36mm)
#         horiz_ap_mm = 36.0
#         focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(fov_deg) * 0.5)
#         cam.CreateHorizontalApertureAttr(horiz_ap_mm)
#         cam.CreateVerticalApertureAttr(horiz_ap_mm * (resolution[1] / resolution[0]))
#         cam.CreateFocalLengthAttr(focal_mm)

#         xform = self._ensure_cam_xform(cam_path)
#         if not xform.GetOrderedXformOps():
#             xform.AddTranslateOp().Set(Gf.Vec3d(forward_m, 0.0, height_m))
#             qx, qy, qz, qw = euler_angles_to_quat(np.array([0.0, -math.radians(pitch_down_deg), 0.0]))
#             xform.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
#         else:
#             ops = xform.GetOrderedXformOps()
#             ops[0].Set(Gf.Vec3d(forward_m, 0.0, height_m))
#             if len(ops) > 1:
#                 qx, qy, qz, qw = euler_angles_to_quat(np.array([0.0, -math.radians(pitch_down_deg), 0.0]))
#                 ops[1].Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
#         return cam_path

#     def mount_realsense_rgbd(
#         self,
#         rel_path="sensors/realsense",
#         forward_m=0.4, height_m=1.2, pitch_down_deg=10.0,
#         hfov_deg=69.0, resolution=(1280, 720),
#     ) -> str:
#         """Câmera “tipo RealSense”: RGB + Depth via Replicator (distance_to_camera=True)."""
#         stage = omni.usd.get_context().get_stage()
#         cam_path = f"{self.prim_path}/{rel_path}"
#         cam = UsdGeom.Camera.Define(stage, cam_path)

#         horiz_ap_mm = 36.0
#         focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(hfov_deg) * 0.5)
#         cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
#         cam.CreateHorizontalApertureAttr(horiz_ap_mm)
#         cam.CreateVerticalApertureAttr(horiz_ap_mm * (resolution[1] / resolution[0]))
#         cam.CreateFocalLengthAttr(focal_mm)
#         cam.CreateResolutionAttr(Gf.Vec2i(int(resolution[0]), int(resolution[1])))
#         cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

#         xform = self._ensure_cam_xform(cam_path)
#         if not xform.GetOrderedXformOps():
#             xform.AddTranslateOp().Set(Gf.Vec3d(forward_m, 0.0, height_m))
#             qx, qy, qz, qw = euler_angles_to_quat(np.array([0.0, -math.radians(pitch_down_deg), 0.0]))
#             xform.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
#         else:
#             ops = xform.GetOrderedXformOps()
#             ops[0].Set(Gf.Vec3d(forward_m, 0.0, height_m))
#             if len(ops) > 1:
#                 qx, qy, qz, qw = euler_angles_to_quat(np.array([0.0, -math.radians(pitch_down_deg), 0.0]))
#                 ops[1].Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
#         return cam_path

#     def mount_zed_stereo(
#         self,
#         base_rel="sensors/zed",
#         forward_m=0.5,
#         height_m=1.3,
#         baseline_m=0.12,
#         hfov_deg=90.0,
#         resolution=(1280, 720),
#     ) -> Tuple[str, str]:
#         """Par estéreo “tipo ZED”: duas câmeras com baseline no eixo Y (left/right)."""
#         stage = omni.usd.get_context().get_stage()
#         left_path  = f"{self.prim_path}/{base_rel}/left"
#         right_path = f"{self.prim_path}/{base_rel}/right"

#         def _make_cam(path, y_offset):
#             cam = UsdGeom.Camera.Define(stage, path)
#             horiz_ap_mm = 36.0
#             focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(hfov_deg) * 0.5)
#             cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
#             cam.CreateHorizontalApertureAttr(horiz_ap_mm)
#             cam.CreateVerticalApertureAttr(horiz_ap_mm * (resolution[1] / resolution[0]))
#             cam.CreateFocalLengthAttr(focal_mm)
#             cam.CreateResolutionAttr(Gf.Vec2i(int(resolution[0]), int(resolution[1])))
#             cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))

#             xform = self._ensure_cam_xform(path)
#             if not xform.GetOrderedXformOps():
#                 xform.AddTranslateOp().Set(Gf.Vec3d(forward_m, y_offset, height_m))
#             else:
#                 ops = xform.GetOrderedXformOps()
#                 ops[0].Set(Gf.Vec3d(forward_m, y_offset, height_m))
#             return path

#         l = _make_cam(left_path,  -baseline_m / 2.0)
#         r = _make_cam(right_path, +baseline_m / 2.0)
#         return l, r

