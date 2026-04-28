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
from isaacsim.core.prims import SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.api.robots.robot import Robot as _Robot
from isaacsim.core.prims import Articulation as _ArticulationView
from isaacsim.robot.wheeled_robots.robots import WheeledRobot as _WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController

from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.rotations import quat_to_rot_matrix, euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage

# Extension imports
from omni.ext.patosim.common import Buffer, Module
from omni.ext.patosim.sensors import Sensor, HawkCamera
from omni.ext.patosim.utils.global_utils import get_stage, get_world
from omni.ext.patosim.utils.stage_utils import stage_get_prim, stage_add_camera, stage_add_usd_ref
from omni.ext.patosim.utils.prim_utils import prim_rotate_x, prim_rotate_y, prim_rotate_z, prim_translate
from omni.ext.patosim.types import Pose2d, Pose3d
from omni.ext.patosim.utils.registry import Registry
from omni.ext.patosim.underwater_physics import BlueROVHydrodynamicsConfig, BlueROVUnderwaterPhysics
from types import SimpleNamespace
from omni.ext.patosim.oceansim.utils.assets_utils import get_oceansim_assets_path

from .rear_simple_controller import RearDriveSimpleController
# sensores novos (do seu sensors.py)
from omni.ext.patosim.sensors import (
    HawkCamera,
    RealSenseRGBDCamera,
    ZedStereoCamera,
    OceanSimUWCamera,
    OceanSimStereoUWCamera,
    OceanSimImagingSonar,
    OceanSimDVL,
    OceanSimBarometer,
    #nuscenes
)
# convenience imports for the Forklift builder
from omni.ext.patosim.sensors import (
    Camera,
    Lidar,
    ensure_single_usd_reference,
    _define_camera_prim,
    _quat_from_euler_xyz,
    _xform_orient_quat,
    _xform_translate,
    _resolve_asset_path,
)
#lidar 

# dentro da classe ForkliftC
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Usd, Sdf
import omni.usd



#=========================================================
#  BASE CLASSES
#=========================================================

#posso modificar a classe base para criar mais cameras e masi vistas das cameras 
# no momento nao vou me preocupar com os modelos das cameras apenas com conteudo depois colo os usds no local correto
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
    chase_camera_y_offset: float
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

        mount_path = os.path.join(self.prim_path, self.chase_camera_base_path, "chase_camera_mount")
        XFormPrim(mount_path)
        mount_prim = stage_get_prim(stage, mount_path)
        prim_translate(
            mount_prim,
            (
                self.chase_camera_x_offset,
                self.chase_camera_y_offset,
                self.chase_camera_z_offset,
            ),
        )

        camera_path = os.path.join(mount_path, "chase_camera")
        stage_add_camera(stage, 
            camera_path, 
            focal_length=10, horizontal_aperature=30, vertical_aperature=30
        )
        camera_prim = stage_get_prim(stage, camera_path)
        prim_rotate_x(camera_prim, self.chase_camera_tilt_angle)
        prim_rotate_y(camera_prim, 0)
        prim_rotate_z(camera_prim, -90)

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

    def set_pointcloud_enabled(self, enabled: bool):
        super().set_pointcloud_enabled(enabled)

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
        # Defensive: articulation_view / internal ISAAC state may not be
        # fully initialized during some startup sequences. Wrap calls
        # that can raise into try/except so the extension does not crash
        # unnecessarily; best-effort to set pose.
        try:
            if self.articulation_view is not None:
                try:
                    self.articulation_view.initialize()
                except Exception:
                    # continue even if initialize fails; we may still set pose
                    pass
        except Exception:
            pass

        try:
            # zero velocities before teleporting pose
            try:
                self.robot.set_world_velocity(np.array([0., 0., 0., 0., 0., 0.]))
            except Exception:
                pass

            # set local pose using robot API; wrap in try to avoid bubbling ISAAC internals
            try:
                position, orientation = self.robot.get_local_pose()
                position[0] = pose.x
                position[1] = pose.y
                position[2] = self.z_offset
                orientation = rot_utils.euler_angles_to_quats(np.array([0., 0., pose.theta]))
                self.robot.set_local_pose(position, orientation)
            except Exception:
                # last-resort: ignore failures here (caller should handle robot availability)
                pass
        except Exception:
            pass
    
    def get_pose_2d(self) -> Pose2d:
        position, orientation = self.robot.get_local_pose()
        theta = rot_utils.quats_to_euler_angles(orientation)[2]
        return Pose2d(
            x=position[0],
            y=position[1],
            theta=theta
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
class OceanSimROVRobot(Robot):
    """ROV BlueROV do OceanSim integrado ao pipeline de Robot do PatoSim."""

    physics_dt: float = 0.01
    z_offset: float = -0.8

    chase_camera_base_path: str = ""
    # Chase camera tuned for the BlueROV scene:
    # keep the vehicle framed without starting too far away.
    chase_camera_x_offset: float = -2.0
    chase_camera_y_offset: float = 0.0
    chase_camera_z_offset: float = 2.0
    chase_camera_tilt_angle: float = 45.0

    occupancy_map_radius: float = 1.0
    occupancy_map_z_min: float = -5.0
    occupancy_map_z_max: float = 5.0
    occupancy_map_cell_size: float = 0.1
    occupancy_map_collision_radius: float = 0.5

    front_camera_type: Type[Sensor] = OceanSimUWCamera
    front_camera_base_path: str = "sensors/uw_front"
    front_camera_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    front_camera_translation: Tuple[float, float, float] = (0.3, 0.0, 0.1)

    keyboard_linear_velocity_gain: float = 45.0
    keyboard_angular_velocity_gain: float = 22.0
    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0
    teleop_linear_speed_gain: float = 1.2
    teleop_angular_speed_gain: float = 1.4
    teleop_velocity_assist_gain: float = 1.0

    usd_relative_path: str = "bluerov/BROV_low.usd"
    fallback_assets_root: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "assets",
            "models",
        )
    )

    initial_translation: Tuple[float, float, float] = (-2.0, 0.0, -0.8)
    initial_orientation_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    mass_kg: float = 5.0
    linear_damping: float = 10.0
    angular_damping: float = 10.0
    disable_gravity: bool = True
    sleep_threshold: float = 0.0
    stabilization_threshold: float = 0.0
    max_linear_velocity: float = 5.0
    max_angular_velocity: float = 720.0

    sonar_translation: Tuple[float, float, float] = (0.3, 0.0, 0.3)
    sonar_orientation_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mono_camera_translation: Tuple[float, float, float] = (0.3, 0.0, 0.1)
    stereo_left_translation: Tuple[float, float, float] = (0.3, -0.05, 0.1)
    stereo_right_translation: Tuple[float, float, float] = (0.3, 0.05, 0.1)
    dvl_translation: Tuple[float, float, float] = (0.0, 0.0, -0.1)
    lidar_translation: Tuple[float, float, float] = (0.32, 0.0, 0.18)
    lidar_orientation_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    uw_camera_focal_length: float = 2.1
    uw_camera_clipping_range: Tuple[float, float] = (0.1, 100.0)
    water_surface_z: float = 1.43389
    water_profile_path: Optional[str] = None
    sonar_max_range: float = 10.0
    dvl_max_range: float = 10.0
    enable_dvl_debug_lines: bool = False
    enable_front_camera: bool = True
    enable_stereo_camera: bool = False
    enable_lidar: bool = False
    enable_sonar: bool = True
    enable_dvl: bool = True
    enable_barometer: bool = True
    fluid_density_kg_m3: float = 1025.0
    gravity_m_s2: float = 9.81
    displaced_volume_m3: Optional[float] = None
    buoyancy_factor: float = 1.02
    vehicle_height_m: float = 0.34
    center_of_buoyancy_body_m: Tuple[float, float, float] = (0.0, 0.0, 0.05)
    linear_drag_coeffs: Tuple[float, float, float] = (8.0, 10.0, 12.0)
    quadratic_drag_coeffs: Tuple[float, float, float] = (3.0, 4.0, 5.0)
    angular_drag_coeffs: Tuple[float, float, float] = (0.8, 0.8, 1.1)
    angular_quadratic_drag_coeffs: Tuple[float, float, float] = (0.1, 0.1, 0.15)
    thruster_max_force_newtons: float = 40.0
    thruster_time_constant_s: float = 0.12

    def __init__(
        self,
        prim_path: str,
        robot: SingleRigidPrim,
        articulation_view: _ArticulationView | None,
        front_camera: Sensor | None = None,
    ):
        super().__init__(
            prim_path=prim_path,
            robot=robot,
            articulation_view=articulation_view,
            front_camera=front_camera,
        )
        self.action = Buffer(np.zeros(6, dtype=np.float32))
        self.linear_velocity = Buffer()
        self.angular_velocity = Buffer()

        self.front_camera = front_camera
        self.front_stereo: Optional[OceanSimStereoUWCamera] = None
        self.lidar: Optional[Lidar] = None
        self.sonar: Optional[OceanSimImagingSonar] = None
        self.dvl: Optional[OceanSimDVL] = None
        self.barometer: Optional[OceanSimBarometer] = None
        self.underwater_physics: Optional[BlueROVUnderwaterPhysics] = None

        stage = get_stage()
        prim = stage_get_prim(stage, prim_path)
        self._force_api = PhysxSchema.PhysxForceAPI.Apply(prim)
        self._force_enabled_attr = self._force_api.CreateForceEnabledAttr()
        self._world_frame_enabled_attr = self._force_api.CreateWorldFrameEnabledAttr()
        self._mode_attr = self._force_api.CreateModeAttr()
        self._force_attr = self._force_api.CreateForceAttr()
        self._torque_attr = self._force_api.CreateTorqueAttr()
        self._force_enabled_attr.Set(True)
        self._world_frame_enabled_attr.Set(True)
        self._mode_attr.Set("force")
        self._force_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self._torque_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))

    @classmethod
    def _resolve_usd_path(cls) -> str:
        candidate_roots: List[str] = []
        candidate_roots.append(cls.fallback_assets_root)
        try:
            candidate_roots.append(get_oceansim_assets_path())
        except Exception:
            pass

        usd_candidates: List[str] = []
        for candidate in (
            cls.usd_relative_path,
            "bluerov/BROV_low.usd",
        ):
            candidate_str = str(candidate or "").strip()
            if candidate_str and candidate_str not in usd_candidates:
                usd_candidates.append(candidate_str)

        for candidate in usd_candidates:
            if os.path.isabs(candidate) and os.path.exists(candidate):
                return os.path.normpath(candidate)

        for root in candidate_roots:
            root_str = str(root or "").strip()
            if not root_str:
                continue
            normalized_root = os.path.normpath(root_str)
            for candidate in usd_candidates:
                if os.path.isabs(candidate):
                    continue
                usd_path = os.path.normpath(os.path.join(normalized_root, candidate))
                if os.path.exists(usd_path):
                    return usd_path

        attempted = [
            os.path.normpath(os.path.join(str(root).strip(), candidate))
            for root in candidate_roots
            if str(root or "").strip()
            for candidate in usd_candidates
            if not os.path.isabs(candidate)
        ]
        raise FileNotFoundError(
            "Nao foi possivel localizar o asset do ROV do OceanSim. "
            f"Tentado: {attempted}"
        )

    @classmethod
    def _initial_orientation(cls) -> np.ndarray:
        return euler_angles_to_quat(
            np.array(cls.initial_orientation_euler_deg, dtype=np.float32),
            degrees=True,
        )

    @classmethod
    def _build_underwater_physics_model(cls) -> BlueROVUnderwaterPhysics:
        physics_cfg = BlueROVHydrodynamicsConfig(
            mass_kg=float(cls.mass_kg),
            water_surface_z=float(cls.water_surface_z),
            fluid_density_kg_m3=float(cls.fluid_density_kg_m3),
            gravity_m_s2=float(cls.gravity_m_s2),
            displaced_volume_m3=cls.displaced_volume_m3,
            buoyancy_factor=float(cls.buoyancy_factor),
            vehicle_height_m=float(cls.vehicle_height_m),
            center_of_buoyancy_body_m=tuple(float(v) for v in cls.center_of_buoyancy_body_m),
            linear_drag_coeffs=tuple(float(v) for v in cls.linear_drag_coeffs),
            quadratic_drag_coeffs=tuple(float(v) for v in cls.quadratic_drag_coeffs),
            angular_drag_coeffs=tuple(float(v) for v in cls.angular_drag_coeffs),
            angular_quadratic_drag_coeffs=tuple(float(v) for v in cls.angular_quadratic_drag_coeffs),
            thruster_max_force_newtons=float(cls.thruster_max_force_newtons),
            thruster_time_constant_s=float(cls.thruster_time_constant_s),
        )
        return BlueROVUnderwaterPhysics.create_default(physics_cfg)

    @classmethod
    def _apply_oceansim_physics_profile(cls, prim_path: str) -> None:
        """Replica o setup físico básico do exemplo SensorExample do OceanSim."""
        stage = get_stage()
        prim = stage_get_prim(stage, prim_path)
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid_api.CreateDisableGravityAttr(bool(cls.disable_gravity))
        rigid_api.GetLinearDampingAttr().Set(float(cls.linear_damping))
        rigid_api.GetAngularDampingAttr().Set(float(cls.angular_damping))
        try:
            rigid_api.CreateSleepThresholdAttr().Set(float(cls.sleep_threshold))
        except Exception:
            pass
        try:
            rigid_api.CreateStabilizationThresholdAttr().Set(float(cls.stabilization_threshold))
        except Exception:
            pass
        try:
            rigid_api.CreateMaxLinearVelocityAttr().Set(float(cls.max_linear_velocity))
        except Exception:
            pass
        try:
            rigid_api.CreateMaxAngularVelocityAttr().Set(float(cls.max_angular_velocity))
        except Exception:
            pass

        try:
            collider = SingleGeometryPrim(prim_path=prim_path, collision=True)
            try:
                collider.set_collision_approximation("boundingCube")
            except Exception:
                pass
        except Exception:
            pass

    @classmethod
    def build(cls, prim_path: str) -> "OceanSimROVRobot":
        world = get_world()

        usd_path = cls._resolve_usd_path()
        add_reference_to_stage(usd_path, prim_path)
        cls._apply_oceansim_physics_profile(prim_path)

        robot = SingleRigidPrim(
            prim_path=prim_path,
            mass=float(cls.mass_kg),
            translation=np.asarray(cls.initial_translation, dtype=np.float32),
            orientation=cls._initial_orientation(),
        )
        try:
            world.scene.add(robot)
        except Exception:
            pass

        instance = cls(
            prim_path=prim_path,
            robot=robot,
            articulation_view=None,
            front_camera=None,
        )
        instance.underwater_physics = cls._build_underwater_physics_model()

        if cls.enable_front_camera:
            instance.front_camera = OceanSimUWCamera.build(
                os.path.join(prim_path, "sensors/uw_front"),
                name="UW_Camera_Front",
                translation=np.asarray(cls.mono_camera_translation, dtype=np.float32),
                focal_length=cls.uw_camera_focal_length,
                clipping_range=cls.uw_camera_clipping_range,
                water_profile_path=cls.water_profile_path,
            )
        if cls.enable_stereo_camera:
            instance.front_stereo = OceanSimStereoUWCamera.build(
                os.path.join(prim_path, "sensors/uw_stereo"),
                left_translation=np.asarray(cls.stereo_left_translation, dtype=np.float32),
                right_translation=np.asarray(cls.stereo_right_translation, dtype=np.float32),
                focal_length=cls.uw_camera_focal_length,
                clipping_range=cls.uw_camera_clipping_range,
                water_profile_path=cls.water_profile_path,
            )
        if cls.enable_lidar:
            lidar_path = os.path.join(prim_path, "sensors/lidar")
            instance.lidar = Lidar.build(lidar_path)
            _xform_translate(lidar_path, tuple(float(v) for v in cls.lidar_translation))
            qw, qx, qy, qz = _quat_from_euler_xyz(*cls.lidar_orientation_euler_deg)
            _xform_orient_quat(lidar_path, (qw, qx, qy, qz))
            try:
                instance.lidar.enable_lidar()
            except Exception:
                pass
        if cls.enable_sonar:
            instance.sonar = OceanSimImagingSonar.build(
                os.path.join(prim_path, "sensors/sonar"),
                translation=np.asarray(cls.sonar_translation, dtype=np.float32),
                orientation=euler_angles_to_quat(
                    np.array(cls.sonar_orientation_euler_deg, dtype=np.float32),
                    degrees=True,
                ),
                max_range=float(cls.sonar_max_range),
            )
        if cls.enable_dvl:
            instance.dvl = OceanSimDVL.build(
                rigid_body_path=prim_path,
                translation=np.asarray(cls.dvl_translation, dtype=np.float32),
                max_range=float(cls.dvl_max_range),
                add_debug_lines=bool(cls.enable_dvl_debug_lines),
            )
        if cls.enable_barometer:
            instance.barometer = OceanSimBarometer.build(
                os.path.join(prim_path, "sensors/barometer"),
                water_surface_z=float(cls.water_surface_z),
            )
        return instance

    def write_action(self, step_size: float):
        action = np.asarray(self.action.get_value(), dtype=np.float32).reshape(-1)
        if action.size < 6:
            padded = np.zeros(6, dtype=np.float32)
            padded[: action.size] = action
            action = padded

        force_body = np.asarray(action[:3], dtype=np.float32)
        torque_body = np.asarray(action[3:6], dtype=np.float32)

        if self.underwater_physics is not None:
            linear_velocity_world = np.zeros(3, dtype=np.float32)
            angular_velocity_world = np.zeros(3, dtype=np.float32)
            world_position = np.zeros(3, dtype=np.float32)
            rot_m = np.eye(3, dtype=np.float32)

            try:
                world_position, orientation = self.robot.get_world_pose()
                world_position = np.asarray(world_position, dtype=np.float32).reshape(3)
                rot_m = np.asarray(quat_to_rot_matrix(orientation), dtype=np.float32)
            except Exception:
                pass
            try:
                linear_velocity_world = np.asarray(self.robot.get_linear_velocity(), dtype=np.float32).reshape(3)
            except Exception:
                pass
            try:
                angular_velocity_world = np.asarray(self.robot.get_angular_velocity(), dtype=np.float32).reshape(3)
            except Exception:
                pass

            linear_velocity_body = rot_m.T @ linear_velocity_world
            angular_velocity_body = rot_m.T @ angular_velocity_world
            net_force_body, net_torque_body = self.underwater_physics.step(
                desired_wrench_body=np.concatenate([force_body, torque_body], axis=0),
                linear_velocity_body=linear_velocity_body,
                angular_velocity_body=angular_velocity_body,
                world_z=float(world_position[2]),
                dt=float(max(step_size, self.physics_dt)),
            )
            force_world = rot_m @ net_force_body
            torque_world = rot_m @ net_torque_body
        else:
            force_world = force_body
            torque_world = torque_body

        self._force_attr.Set(Gf.Vec3f(float(force_world[0]), float(force_world[1]), float(force_world[2])))
        self._torque_attr.Set(Gf.Vec3f(float(torque_world[0]), float(torque_world[1]), float(torque_world[2])))

    def apply_body_velocity_command(
        self,
        linear_body: np.ndarray,
        angular_body: np.ndarray,
    ) -> None:
        linear_body = np.asarray(linear_body, dtype=np.float32).reshape(3)
        angular_body = np.asarray(angular_body, dtype=np.float32).reshape(3)

        try:
            _, orientation = self.robot.get_world_pose()
            rot_m = np.asarray(quat_to_rot_matrix(orientation), dtype=np.float32)
            linear_world = rot_m @ linear_body
            angular_world = rot_m @ angular_body
        except Exception:
            linear_world = linear_body
            angular_world = angular_body

        try:
            self.robot.set_linear_velocity(linear_world)
        except Exception:
            pass
        try:
            self.robot.set_angular_velocity(angular_world)
        except Exception:
            pass

    def update_state(self):
        try:
            pos, ori = self.robot.get_local_pose()
            self.position.set_value(pos)
            self.orientation.set_value(ori)
        except Exception:
            self.position.set_value(None)
            self.orientation.set_value(None)

        try:
            self.linear_velocity.set_value(np.asarray(self.robot.get_linear_velocity(), dtype=np.float32))
        except Exception:
            self.linear_velocity.set_value(None)

        try:
            self.angular_velocity.set_value(np.asarray(self.robot.get_angular_velocity(), dtype=np.float32))
        except Exception:
            self.angular_velocity.set_value(None)

        self.joint_positions.set_value(np.zeros(0, dtype=np.float32))
        self.joint_velocities.set_value(np.zeros(0, dtype=np.float32))
        Module.update_state(self)

    def write_replay_data(self):
        try:
            if self.position.get_value() is not None and self.orientation.get_value() is not None:
                self.robot.set_local_pose(
                    self.position.get_value(),
                    self.orientation.get_value(),
                )
        except Exception:
            pass
        Module.write_replay_data(self)

    def set_pose_2d(self, pose: Pose2d):
        position = np.array([pose.x, pose.y, self.z_offset], dtype=np.float32)
        orientation = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, pose.theta], dtype=np.float32))
        self.set_pose_3d(position, orientation)

    def set_pose_3d(self, position: np.ndarray, orientation: np.ndarray):
        position = np.asarray(position, dtype=np.float32)
        orientation = np.asarray(orientation, dtype=np.float32)

        try:
            self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        except Exception:
            pass
        try:
            self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        except Exception:
            pass
        try:
            self.robot.set_world_velocity(np.zeros(6, dtype=np.float32))
        except Exception:
            pass
        try:
            self.robot.set_local_pose(position, orientation)
        except Exception:
            pass

        self.action.set_value(np.zeros(6, dtype=np.float32))
        if self.underwater_physics is not None:
            self.underwater_physics.reset()
        self.write_action(self.physics_dt)

    def get_pose_3d(self) -> Pose3d:
        position, orientation = self.robot.get_local_pose()
        return Pose3d(
            position=np.asarray(position, dtype=np.float32),
            orientation=np.asarray(orientation, dtype=np.float32),
        )

    def get_yaw_from_orientation(self) -> float:
        """Extrai yaw (rotação em torno do Z global) da orientação atual.

        Assume quaternion no formato wxyz (convenção Isaac Sim).
        """
        try:
            pose = self.get_pose_3d()
            ori = np.asarray(pose.orientation, dtype=np.float32)
            # Isaac Sim: orientation = [w, x, y, z]
            w, x, y, z = float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return float(np.arctan2(siny_cosp, cosy_cosp))
        except Exception:
            return 0.0
