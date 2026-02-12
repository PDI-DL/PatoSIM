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
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController

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
from types import SimpleNamespace

from .rear_simple_controller import RearDriveSimpleController
# sensores novos (do seu sensors.py)
from omni.ext.mobility_gen.sensors import (
    HawkCamera,
    BevTopDownCamera,
    BevFrontDownCamera,
    RealSenseRGBDCamera,
    ZedStereoCamera,
    #nuscenes
)
# convenience imports for the Forklift builder
from omni.ext.mobility_gen.sensors import (
    Camera,
    Lidar,
    _define_camera_prim,
    _quat_from_euler_xyz,
    _xform_orient_quat,
    _xform_translate,
)
#lidar 

# dentro da classe ForkliftC
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Usd, Sdf
import omni.usd
from omni.isaac.core.utils.rotations import euler_angles_to_quat



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

# ===============================
#  Forklift C: tração + direção traseiras (sem Ackermann)
# ===============================
# =================================================================================================
# ROBÔ: 4 rodas, tração + direção NAS TRASEIRAS (sem Ackermann), compatível com MobilityGen/Robot
# =================================================================================================
@ROBOTS.register()
class FourWheelRearSteerRobot_V1(Robot):
    """
    - Dianteiras: livres (sem comando de ω).
    - Traseiras: recebem ω (iguais) e δ (mesmo sinal) via juntas de direção.
    - Sem Ackermann: comportamento de "carro com esterço traseiro".
    """

    # ===== Sim / sensores =====
    physics_dt: float = 0.005
    z_offset: float = 0.25

    # Third-person chase camera behind and above the forklift.
    chase_camera_base_path: str = "body"
    chase_camera_x_offset: float = -4.0
    chase_camera_z_offset: float = 2.0
    chase_camera_tilt_angle: float = 35.0

    # ===== Occupancy Map (usado pelo builder) =====
    occupancy_map_radius: float = 1.5
    occupancy_map_z_min: float = 0.05
    occupancy_map_z_max: float = 1.2
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.3

    # ===== Câmera frontal =====
    front_camera_type = HawkCamera
    front_camera_base_path = "sensors/rgb_camera/front_camera"
    front_camera_rotation = (0., 0., 0.)
    # Front stereo camera mounted near the mast, facing forward.
    front_camera_translation = (1.10, 0.0, 1.65)

    # ===== Teleop =====
    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0

    # ===== Geometria =====
    wheel_base: float = 1.65
    track_width: float = 1.25
    wheel_radius: float = 0.5

    # Ângulo máximo (rad) — aumente se o USD permitir
    max_steer_angle: float = math.radians(80.0)

    # ===== Juntas (ajuste para seu USD) =====
    rear_wheel_dof_names: List[str] = ["left_back_wheel_joint", "right_back_wheel_joint"]
    steering_dof_names:   List[str] = ["left_rotator_joint", "right_rotator_joint"]  # yaw traseiro

    # ===== USD =====
    usd_url: str = ("http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
                    "Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd")
    chassis_subpath: str = ""

    # ===== Drives de direção (posição) =====
    steer_stiffness: float = 1.6e5
    steer_damping:   float = 3.0e3
    steer_max_force: float = 8.0e6

    # Mesmo sentido nos dois lados (auto-calib pode inverter se detectar erro)
    steering_axis_signs: Tuple[float, float] = (1.0, 1.0)

    # Limite efetivo após ler/ajustar limites do USD (rad)
    effective_steer_limit: float = math.radians(80.0)

    def __init__(self,
                 prim_path: str,
                 robot: _WheeledRobot,
                 articulation_view: _ArticulationView,
                 front_camera: Sensor | None = None):
        super().__init__(prim_path=prim_path, robot=robot, articulation_view=articulation_view, front_camera=front_camera)
        # Explicit sensor handles used by extension preview/writer.
        self.front_stereo: Optional[Sensor] = front_camera
        self.fisheye_left: Optional[Camera] = None
        self.fisheye_right: Optional[Camera] = None
        self.lidar: Optional[Lidar] = None
        self.controller = RearDriveSimpleController(self.wheel_radius, self.max_steer_angle)
        self._rear_idx: np.ndarray | None = None
        self._steer_idx: np.ndarray | None = None
        self._last_cmd: np.ndarray | None = None
        self._auto_done = False
        self._auto_frames = 45
        self._frame = 0
        self._rebind_warmup_frames = 0
        self._warned_missing_rear = False
        self._warned_missing_steer = False
        self._warned_transient_reset = False

    # ---------- Build ----------
    @classmethod
    def build(cls, prim_path: str) -> "FourWheelRearSteerRobot":
        world = get_world()
        robot = world.scene.add(_WheeledRobot(
            prim_path,
            wheel_dof_names=cls.rear_wheel_dof_names,   # só as traseiras recebem ω
            create_robot=True,
            usd_path=cls.usd_url
        ))
        view_path = os.path.join(prim_path, cls.chassis_subpath) if cls.chassis_subpath else prim_path
        view = _ArticulationView(view_path)
        world.scene.add(view)

        camera = cls.build_front_camera(prim_path)
        instance = cls(
            prim_path=prim_path,
            robot=robot,
            articulation_view=view,
            front_camera=camera
        )
        instance._attach_aux_sensors()
        return instance

    def _attach_aux_sensors(self):
        """Attach fisheye pair + lidar and expose a single front stereo handle."""
        self.front_stereo = self.front_camera

        # Left fisheye
        try:
            left_path = os.path.join(self.prim_path, "sensors/fisheye_left")
            _define_camera_prim(left_path)
            self.fisheye_left = Camera(left_path, (640, 480))
            _xform_translate(left_path, (1.05, -0.38, 1.58))
            qw, qx, qy, qz = _quat_from_euler_xyz(0.0, 0.0, 28.0)
            _xform_orient_quat(left_path, (qw, qx, qy, qz))
        except Exception:
            self.fisheye_left = None

        # Right fisheye
        try:
            right_path = os.path.join(self.prim_path, "sensors/fisheye_right")
            _define_camera_prim(right_path)
            self.fisheye_right = Camera(right_path, (640, 480))
            _xform_translate(right_path, (1.05, 0.38, 1.58))
            qw, qx, qy, qz = _quat_from_euler_xyz(0.0, 0.0, -28.0)
            _xform_orient_quat(right_path, (qw, qx, qy, qz))
        except Exception:
            self.fisheye_right = None

        # Roof lidar
        try:
            lidar_path = os.path.join(self.prim_path, "sensors/lidar")
            self.lidar = Lidar(lidar_path)
            _xform_translate(lidar_path, (0.25, 0.0, 2.10))
            self.lidar.enable_lidar()
        except Exception:
            self.lidar = None

        # Ensure RGB streams are available for preview/recording.
        try:
            if hasattr(self.front_stereo, "left"):
                self.front_stereo.left.enable_rgb_rendering()
            if hasattr(self.front_stereo, "right"):
                self.front_stereo.right.enable_rgb_rendering()
        except Exception:
            pass
        try:
            if self.fisheye_left is not None:
                self.fisheye_left.enable_rgb_rendering()
        except Exception:
            pass
        try:
            if self.fisheye_right is not None:
                self.fisheye_right.enable_rgb_rendering()
        except Exception:
            pass

    # ---------- Helpers ----------
    def _get_dof_names(self) -> List[str]:
        names = []
        for attr in ("get_dof_names", "dof_names", "_dof_names"):
            if hasattr(self.articulation_view, attr):
                try:
                    val = getattr(self.articulation_view, attr)
                    names = list(val() if callable(val) else val)
                    if names:
                        return names
                except Exception:
                    pass
        return names

    def _resolve_indices(self, names: List[str]) -> np.ndarray:
        idx = []
        for n in names:
            try: idx.append(self.articulation_view.get_dof_index(n))
            except Exception: idx.append(-1)
        return np.asarray(idx, np.int32)

    def _auto_find_rear_wheel_dofs(self) -> Optional[List[str]]:
        names = self._get_dof_names()
        if not names:
            return None

        def is_rear_wheel(name: str) -> bool:
            s = name.lower()
            return (("rear" in s or "back" in s) and ("wheel" in s))

        cands = [n for n in names if is_rear_wheel(n)]
        lefts = [n for n in cands if "left" in n.lower()]
        rights = [n for n in cands if "right" in n.lower()]
        if lefts and rights:
            return [lefts[0], rights[0]]
        return cands[:2] if len(cands) >= 2 else None

    def _auto_find_rear_steer_dofs(self) -> Optional[List[str]]:
        # tenta localizar DOFs traseiros de esterço por nome
        names = self._get_dof_names()
        if not names:
            return None
        def is_steer(n: str) -> bool:
            s = n.lower()
            return (("rear" in s or "back" in s) and ("steer" in s or "swivel" in s or "yaw" in s or "rotator" in s))
        cands = [n for n in names if is_steer(n)]
        lefts  = [n for n in cands if "left"  in n.lower()]
        rights = [n for n in cands if "right" in n.lower()]
        if lefts and rights:
            return [lefts[0], rights[0]]
        return cands[:2] if len(cands) >= 2 else None

    def _force_limits_and_drive_in_usd(self, joint_names: List[str]):
        """Abre limites (±max_steer_angle, em graus no USD) e aplica Drive angular PD."""
        stage = omni.usd.get_context().get_stage()
        max_deg = math.degrees(self.max_steer_angle)
        for prim in stage.TraverseAll():
            if prim.GetTypeName() != "PhysicsRevoluteJoint":
                continue
            if prim.GetName() not in joint_names:
                continue
            try:
                rj = UsdPhysics.RevoluteJoint(prim)
                (rj.CreateLowerLimitAttr() if not rj.GetLowerLimitAttr() else rj.GetLowerLimitAttr()).Set(-max_deg)
                (rj.CreateUpperLimitAttr() if not rj.GetUpperLimitAttr() else rj.GetUpperLimitAttr()).Set(+max_deg)
            except Exception:
                pass
            try:
                drv = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drv: drv = UsdPhysics.DriveAPI.Apply(prim, "angular")
                (drv.GetStiffnessAttr() if drv.GetStiffnessAttr() else drv.CreateStiffnessAttr()).Set(self.steer_stiffness)
                (drv.GetDampingAttr()   if drv.GetDampingAttr()   else drv.CreateDampingAttr()).Set(self.steer_damping)
                (drv.GetMaxForceAttr()  if drv.GetMaxForceAttr()  else drv.CreateMaxForceAttr()).Set(self.steer_max_force)
            except Exception:
                pass

    def _ensure_indices_limits_and_drive(self) -> bool:
        try:
            self.articulation_view.initialize()
        except Exception:
            # Em alguns frames (especialmente após reset), initialize pode falhar
            # mesmo com índices já válidos. Nesses casos, seguimos.
            if self._rear_idx is not None and self._steer_idx is not None:
                return True

        # rodas traseiras
        if self._rear_idx is None:
            rear_names = list(self.rear_wheel_dof_names)
            idx = self._resolve_indices(rear_names)
            if (idx < 0).any():
                auto = self._auto_find_rear_wheel_dofs()
                if auto:
                    rear_names = auto
                    idx = self._resolve_indices(rear_names)
            if (idx < 0).any():
                if not self._warned_missing_rear:
                    available = self._get_dof_names()
                    print(
                        "[FourWheelRearSteerRobot_V1] rear wheel indices not found "
                        f"(resolved {idx}) for names {rear_names}. "
                        f"Available DOFs (first 20): {available[:20]}"
                    )
                    self._warned_missing_rear = True
                return False
            self._warned_missing_rear = False
            self.rear_wheel_dof_names = rear_names
            self._rear_idx = idx

        # direção traseira
        if self._steer_idx is None:
            steer_names = list(self.steering_dof_names)
            idx = self._resolve_indices(steer_names)
            if (idx < 0).any():
                auto = self._auto_find_rear_steer_dofs()
                if auto:
                    steer_names = auto
                    idx = self._resolve_indices(steer_names)
            if (idx < 0).any():
                if not self._warned_missing_steer:
                    available = self._get_dof_names()
                    print(
                        "[FourWheelRearSteerRobot_V1] steer indices not found "
                        f"(resolved {idx}) for names {steer_names}. "
                        f"Available DOFs (first 20): {available[:20]}"
                    )
                    self._warned_missing_steer = True
                return False
            self._warned_missing_steer = False
            self.steering_dof_names = steer_names
            self._steer_idx = idx

            # abre limites/drives no USD também
            self._force_limits_and_drive_in_usd(self.steering_dof_names)

        # reforça limites via AV (rad) e guarda limite efetivo
        try:
            lo_all, up_all = self.articulation_view.get_dof_limits()
            lo = np.array([lo_all[i] for i in self._steer_idx], np.float32)
            up = np.array([up_all[i] for i in self._steer_idx], np.float32)
            req = float(self.max_steer_angle)
            need = (np.abs(lo) < req) | (np.abs(up) < req)
            if np.any(need):
                self.articulation_view.set_dof_limits(
                    joint_indices=self._steer_idx,
                    lower_limits=np.where(need, -req, lo).astype(np.float32),
                    upper_limits=np.where(need,  req, up).astype(np.float32),
                )
            lo_all, up_all = self.articulation_view.get_dof_limits()
            lo = np.array([lo_all[i] for i in self._steer_idx], np.float32)
            up = np.array([up_all[i] for i in self._steer_idx], np.float32)
            self.effective_steer_limit = float(min(self.max_steer_angle, np.min(np.abs([lo, up]))))
        except Exception:
            self.effective_steer_limit = float(self.max_steer_angle)

        # drive de posição via AV (rad)
        try:
            if hasattr(self.articulation_view, "set_dof_position_drive_properties"):
                self.articulation_view.set_dof_position_drive_properties(
                    joint_indices=self._steer_idx,
                    stiffness=np.full(len(self._steer_idx), self.steer_stiffness, np.float32),
                    damping=np.full(len(self._steer_idx),   self.steer_damping,   np.float32),
                    max_forces=np.full(len(self._steer_idx), self.steer_max_force, np.float32),
                )
        except Exception:
            pass
        return True
    # ---------- ciclo de vida ----------
    def post_reset(self):
        self._last_cmd = None
        self._auto_done = False
        self._frame = 0
        self._rebind_warmup_frames = max(self._rebind_warmup_frames, 60)
        self._warned_transient_reset = False
        try:
            self.articulation_view.initialize()
        except Exception:
            pass
    # ---------- controle ----------
    def write_action(self, step_size: float):
        def _is_transient_articulation_error(exc: Exception) -> bool:
            text = str(exc)
            return ("NoneType" in text and "joint_positions" in text) or (
                "Physics Simulation View is not created yet" in text
            )

        # após reset, dê alguns frames para a physics view/articulation controller subir
        if self._rebind_warmup_frames > 0:
            self._rebind_warmup_frames -= 1

        if not self._ensure_indices_limits_and_drive():
            return

        # 1) rodas traseiras
        action = self.action.get_value()
        if action is None or len(action) < 2:
            return
        v_mps = float(action[0])
        steer_cmd = float(action[1])
        wheel_action, steer_targets = self.controller.forward(np.array([v_mps, steer_cmd], dtype=np.float32))
        try:
            self.robot.apply_wheel_actions(wheel_action)
        except Exception as exc:
            if _is_transient_articulation_error(exc):
                if not self._warned_transient_reset:
                    print(
                        "[FourWheelRearSteerRobot_V1] transient articulation state after reset; "
                        "waiting for physics view rebind..."
                    )
                    self._warned_transient_reset = True
                self._rear_idx = None
                self._steer_idx = None
                self._rebind_warmup_frames = max(self._rebind_warmup_frames, 20)
                return
            # fallback: aplica velocidade direto no articulation_view
            try:
                omega = v_mps / max(float(self.wheel_radius), 1e-6)
                wheel_vel = np.array([omega, omega], dtype=np.float32)
                self.articulation_view.set_joint_velocity_targets(
                    velocities=wheel_vel,
                    joint_indices=self._rear_idx,
                )
            except Exception as exc2:
                if _is_transient_articulation_error(exc2):
                    self._rear_idx = None
                    self._steer_idx = None
                    self._rebind_warmup_frames = max(self._rebind_warmup_frames, 20)
                    return
                print(f"[FourWheelRearSteerRobot_V1] wheel command failed: {exc2}")
                return

        # 2) direção traseira: δ igual nos dois lados
        steer_targets = (np.asarray(self.steering_axis_signs, np.float32) * steer_targets).astype(np.float32)
        lim = float(self.effective_steer_limit)
        steer_targets = np.clip(steer_targets, -lim, +lim).astype(np.float32)
        try:
            self.articulation_view.set_joint_position_targets(
                positions=steer_targets, joint_indices=self._steer_idx
            )
        except Exception as exc:
            if _is_transient_articulation_error(exc):
                self._rear_idx = None
                self._steer_idx = None
                self._rebind_warmup_frames = max(self._rebind_warmup_frames, 20)
                return
            print(f"[FourWheelRearSteerRobot_V1] set_joint_position_targets failed: {exc}")
            return
        self._warned_transient_reset = False

        # 3) auto calibra sinais nos primeiros frames (se medição divergir do comando)
        try:
            q_all = self.articulation_view.get_joint_positions()
            q_all = np.array(q_all)[0] if np.ndim(q_all) == 2 else np.array(q_all)
            q_now = q_all[self._steer_idx].astype(np.float32)
        except Exception:
            q_now = None

        if (not self._auto_done) and (q_now is not None):
            if self._last_cmd is not None and np.max(np.abs(self._last_cmd)) > 1e-3:
                exp = np.sign(self._last_cmd)
                meas = np.sign(q_now + 1e-9)
                flip = [meas[i] != exp[i] for i in (0, 1)]
                if any(flip):
                    s = list(self.steering_axis_signs)
                    for i in (0, 1):
                        if flip[i]:
                            s[i] = -s[i]
                    self.steering_axis_signs = (float(s[0]), float(s[1]))
                    print(f"[RearSteer] auto-calib: invertendo sinais -> {self.steering_axis_signs}")
            self._frame += 1
            if self._frame >= self._auto_frames:
                self._auto_done = True

        self._last_cmd = steer_targets.copy()

        # Debug: print indices and commands (small-volume) so we can trace when wheels not moving
        try:
            print(
                "[FourWheelRearSteerRobot_V1] "
                f"rear_idx={self._rear_idx}, steer_idx={self._steer_idx}, "
                f"v_mps={v_mps:.3f}, steer_cmd={steer_cmd:.3f}, steer_targets={steer_targets}"
            )
        except Exception:
            pass


#  @ROBOTS.register()
# class ForkliftCRobot(Robot):
#     """
#     Forklift C (empilhadeira) com:
#       - Tração somente nas rodas traseiras
#       - Direção apenas nas juntas 'rotator' (mesmo sentido no mundo)
#       - Sem Ackermann: mapeamento direto (v_mps, delta_rad) -> (omega_traseiras, posição_rotators)
#     Ação esperada: np.array([v_mps, delta_rad], float32)
#     """

#     # ====== Sim / sensores ======
#     physics_dt: float = 0.005
#     z_offset: float = 0.25

#     chase_camera_base_path: str = "body"
#     chase_camera_x_offset: float = -2.0
#     chase_camera_z_offset: float = 2.0
#     chase_camera_tilt_angle: float = 45.0

#     occupancy_map_radius: float = 1.5
#     occupancy_map_z_min: float = 0.05
#     occupancy_map_z_max: float = 1.2
#     occupancy_map_cell_size: float = 0.05
#     occupancy_map_collision_radius: float = 0.5


#     front_camera_base_path = "chassis/rgb_camera/front_hawk"
#     front_camera_rotation = (0., 0., 0.)
#     front_camera_translation = (0., 0., 0.)
#     front_camera_type = HawkCamera

#     # ====== Geometria ======
#     wheel_base: float = 1.65
#     track_width: float = 1.25
#     wheel_radius: float = 0.5

#     # ====== Juntas ======
#     rear_wheel_dof_names = ["left_back_wheel_joint", "right_back_wheel_joint"]
#     front_wheel_dof_names = ["left_front_wheel_joint", "right_front_wheel_joint"]
#     steering_dof_names     = ["left_rotator_joint", "right_rotator_joint"]

#     # ====== USD ======
#     usd_url = ("http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
#                "Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd")
#     chassis_subpath: str = ""  # se o root de articulação for um filho, ex.: "Chassis"

#     # ====== Interface p/ Teleop (compatível com sua cena) ======
#     keyboard_linear_velocity_gain: float = 1.0
#     keyboard_angular_velocity_gain: float = 1.0
#     # A cena lê max_steer_angle; deixe em rad e ajuste como preferir
#     max_steer_angle: float = 1.20  # ~68.8°

#     # ====== Garantias de direção e suavização ======
#     min_steer_abs_deg: float = 45.0  # garantia de curso mínimo (>= 45°)
#     steering_axis_signs: Tuple[float, float] = (1.0, -1.0)  # ajuste se eixos locais estiverem espelhados
#     max_wheel_accel_radps2: float = 30.0  # rampa nas rodas traseiras (evita trancos)

#     def __init__(self, prim_path, robot, articulation_view, front_camera=None):
#         super().__init__(prim_path, robot, articulation_view, front_camera)
#         self.robot = robot

#         # Cache de índices
#         self._rear_idx = None
#         self._front_idx = None
#         self._steer_idx = None

#         # Estado de rampa para rodas traseiras
#         self._rear_omega_cmd = np.zeros(2, dtype=np.float32)

#     # ---------- Inicialização / garantias ----------
#     def _ensure_joint_indices(self):
#         try:
#             self.articulation_view.initialize()
#         except Exception:
#             pass

#         def _idx_or_raise(names):
#             idxs = [self.articulation_view.get_dof_index(n) for n in names]
#             missing = [names[i] for i, v in enumerate(idxs) if v is None or v < 0]
#             if missing:
#                 raise RuntimeError(
#                     f"Juntas não encontradas na ArticulationView: {missing}. "
#                     f"Verifique root e nomes no USD."
#                 )
#             return np.asarray(idxs, dtype=np.int32)

#         self._rear_idx  = _idx_or_raise(self.rear_wheel_dof_names)
#         self._front_idx = _idx_or_raise(self.front_wheel_dof_names)
#         self._steer_idx = _idx_or_raise(self.steering_dof_names)

#     def _ensure_steering_limits(self):
#         """
#         Garante limites de direção >= max(45°, max_steer_angle).
#         Se a API suportar set_dof_limits, amplia; caso contrário, documente no USD.
#         """
#         required_abs = max(math.radians(self.min_steer_abs_deg), float(self.max_steer_angle))
#         try:
#             lower, upper = self.articulation_view.get_dof_limits()  # arrays [ndof], [ndof] ou [[lo,up],...]
#             # Normaliza formas diferentes de retorno
#             # Tenta tratar como vetores "lower[i], upper[i]"
#             lo = np.array([lower[i] for i in self._steer_idx], dtype=np.float32)
#             up = np.array([upper[i] for i in self._steer_idx], dtype=np.float32)

#             needs_update = (np.abs(lo) < required_abs) | (np.abs(up) < required_abs)
#             if np.any(needs_update):
#                 new_lo = np.where(needs_update, -required_abs, lo)
#                 new_up = np.where(needs_update,  +required_abs, up)
#                 self.articulation_view.set_dof_limits(
#                     joint_indices=self._steer_idx,
#                     lower_limits=new_lo,
#                     upper_limits=new_up
#                 )
#         except AttributeError:
#             # Sem suporte na versão — ajuste no USD (recommended: ±max_steer_angle, no mínimo ±45°)
#             pass

#     def post_reset(self):
#         super().post_reset()
#         if (self._rear_idx is None) or (self._front_idx is None) or (self._steer_idx is None):
#             self._ensure_joint_indices()
#             self._ensure_steering_limits()

#     # ---------- Construção ----------
#     @classmethod
#     def build(cls, prim_path: str) -> "ForkliftCRobot":
#         world = get_world()

#         robot = world.scene.add(_WheeledRobot(
#             prim_path,
#             wheel_dof_names=cls.rear_wheel_dof_names + cls.front_wheel_dof_names,
#             create_robot=True,
#             usd_path=cls.usd_url
#         ))

#         view_path = os.path.join(prim_path, cls.chassis_subpath) if cls.chassis_subpath else prim_path
#         view = _ArticulationView(view_path)
#         world.scene.add(view)

#         camera = cls.build_front_camera(prim_path)
#         return cls(prim_path=prim_path, robot=robot, articulation_view=view, front_camera=camera)

#     # ---------- Controle (malha aberta) ----------
#     def write_action(self, step_size: float):
#         """
#         Entrada: (v_mps, delta_rad)
#           - v_mps     : velocidade linear desejada nas rodas traseiras [m/s]
#           - delta_rad : ângulo de direção desejado [rad], mesmo para os dois rotators
#         Saída:
#           - rodas traseiras: velocidade alvo (rad/s) com rampa
#           - rodas dianteiras: velocidade alvo zero
#           - direção: posição alvo (rad) com clamp pelos limites garantidos
#         """
#         if (self._rear_idx is None) or (self._front_idx is None) or (self._steer_idx is None):
#             self._ensure_joint_indices()
#             self._ensure_steering_limits()

#         v_mps, delta_rad = self.action.get_value()
#         v_mps    = float(v_mps)
#         delta_in = float(delta_rad)

#         # Respeita o maior entre (curso garantido) e (max_steer_angle declarado)
#         steer_abs_max = max(math.radians(self.min_steer_abs_deg), float(self.max_steer_angle))
#         delta_cmd = float(np.clip(delta_in, -steer_abs_max, +steer_abs_max))

#         # Mesmo sentido no mundo (ajustável via steering_axis_signs, se um eixo local estiver invertido)
#         steer_targets = np.asarray(self.steering_axis_signs, dtype=np.float32) * delta_cmd
#         self.articulation_view.set_joint_position_targets(
#             positions=steer_targets.astype(np.float32),
#             joint_indices=self._steer_idx
#         )

#         # Tração apenas traseira: omega = v / R
#         omega_des = float(v_mps / max(self.wheel_radius, 1e-6))
#         omega_pair = np.array([omega_des, omega_des], dtype=np.float32)

#         dt = float(step_size) if step_size else self.physics_dt
#         max_domega = float(self.max_wheel_accel_radps2) * dt
#         domega = np.clip(omega_pair - self._rear_omega_cmd, -max_domega, +max_domega)
#         self._rear_omega_cmd = self._rear_omega_cmd + domega

#         # Aplica velocidades
#         self.articulation_view.set_joint_velocity_targets(
#             velocities=self._rear_omega_cmd.astype(np.float32),
#             joint_indices=self._rear_idx
#         )
#         # Dianteiras paradas (sem tração)
#         self.articulation_view.set_joint_velocity_targets(
#             velocities=np.zeros(2, dtype=np.float32),
#             joint_indices=self._front_idx
#         )

#     # ---------- Observações técnicas ----------
#     # - Cinemática típica de empilhadeira (rear-wheel steering):
#     #     R ≈ L / tan(delta), yaw_rate ≈ v * tan(delta) / L.
#     # - Se notar que os rotators giram em sentidos opostos no mundo, ajuste:
#     #     steering_axis_signs = (1.0, -1.0)  # ou vice-versa
#     # - Caso a API não permita ampliar limites, configure ±max_steer_angle (>= 45°) diretamente no USD.


# # # ---------- Subclasse para o asset Forklift ----------
# # @ROBOTS.register()
# # class ForkliftCRobot(FourWheeledRearSteerRobot):
# #     # Física/render
# #     physics_dt: float = 0.005
# #     z_offset: float = 0.25

# #     # DOFs (ordem: [FL, FR, RL, RR])
# #     wheel_dof_names = [
# #         "left_front_wheel_joint", "right_front_wheel_joint",
# #         "left_back_wheel_joint",  "right_back_wheel_joint",
# #     ]
# #     # DUAS juntas TRASEIRAS (yaw)
# #     steering_dof_names = ["left_rotator_joint", "right_rotator_joint"]

# #     # Asset
# #     usd_url = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd"
# #     chassis_subpath: str = ""

# #     # Geometria/limites (ângulo alto para curvas fáceis; o USD pode limitar)
# #     wheel_radius: float = 0.30
# #     max_steer_angle: float = 1.90
# #     steer_axis_signs: Tuple[float, float] = (-1.0, +1.0)  # virar MESMO LADO no mundo
# #     invert_steering: bool = F
# @ROBOTS.register()
# class ForkliftCRobot(Robot):
#     """
#     Forklift C (empilhadeira) com:
#       - Tração somente nas rodas traseiras
#       - Direção apenas nas juntas 'rotator' (mesmo sentido no mundo)
#       - Sem Ackermann: mapeamento direto (v_mps, delta_rad) -> (omega_traseiras, posição_rotators)
#     Ação esperada: np.array([v_mps, delta_rad], float32)
#     """

#     # ====== Sim / sensores ======
#     physics_dt: float = 0.005
#     z_offset: float = 0.25

#     chase_camera_base_path: str = "body"
#     chase_camera_x_offset: float = -2.0
#     chase_camera_z_offset: float = 2.0
#     chase_camera_tilt_angle: float = 45.0

#     occupancy_map_radius: float = 1.5
#     occupancy_map_z_min: float = 0.05
#     occupancy_map_z_max: float = 1.2
#     occupancy_map_cell_size: float = 0.05
#     occupancy_map_collision_radius: float = 0.5


#     front_camera_base_path = "chassis/rgb_camera/front_hawk"
#     front_camera_rotation = (0., 0., 0.)
#     front_camera_translation = (0., 0., 0.)
#     front_camera_type = HawkCamera

#     # ====== Geometria ======
#     wheel_base: float = 1.65
#     track_width: float = 1.25
#     wheel_radius: float = 0.5

#     # ====== Juntas ======
#     rear_wheel_dof_names = ["left_back_wheel_joint", "right_back_wheel_joint"]
#     front_wheel_dof_names = ["left_front_wheel_joint", "right_front_wheel_joint"]
#     steering_dof_names     = ["left_rotator_joint", "right_rotator_joint"]

#     # ====== USD ======
#     usd_url = ("http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
#                "Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd")
#     chassis_subpath: str = ""  # se o root de articulação for um filho, ex.: "Chassis"

#     # ====== Interface p/ Teleop (compatível com sua cena) ======
#     keyboard_linear_velocity_gain: float = 1.0
#     keyboard_angular_velocity_gain: float = 1.0
#     # A cena lê max_steer_angle; deixe em rad e ajuste como preferir
#     max_steer_angle: float = 1.20  # ~68.8°

#     # ====== Garantias de direção e suavização ======
#     min_steer_abs_deg: float = 45.0  # garantia de curso mínimo (>= 45°)
#     steering_axis_signs: Tuple[float, float] = (1.0, -1.0)  # ajuste se eixos locais estiverem espelhados
#     max_wheel_accel_radps2: float = 30.0  # rampa nas rodas traseiras (evita trancos)

#     def __init__(self, prim_path, robot, articulation_view, front_camera=None):
#         super().__init__(prim_path, robot, articulation_view, front_camera)
#         self.robot = robot

#         # Cache de índices
#         self._rear_idx = None
#         self._front_idx = None
#         self._steer_idx = None

#         # Estado de rampa para rodas traseiras
#         self._rear_omega_cmd = np.zeros(2, dtype=np.float32)

#     # ---------- Inicialização / garantias ----------
#     def _ensure_joint_indices(self):
#         try:
#             self.articulation_view.initialize()
#         except Exception:
#             pass

#         def _idx_or_raise(names):
#             idxs = [self.articulation_view.get_dof_index(n) for n in names]
#             missing = [names[i] for i, v in enumerate(idxs) if v is None or v < 0]
#             if missing:
#                 raise RuntimeError(
#                     f"Juntas não encontradas na ArticulationView: {missing}. "
#                     f"Verifique root e nomes no USD."
#                 )
#             return np.asarray(idxs, dtype=np.int32)

#         self._rear_idx  = _idx_or_raise(self.rear_wheel_dof_names)
#         self._front_idx = _idx_or_raise(self.front_wheel_dof_names)
#         self._steer_idx = _idx_or_raise(self.steering_dof_names)

#     def _ensure_steering_limits(self):
#         """
#         Garante limites de direção >= max(45°, max_steer_angle).
#         Se a API suportar set_dof_limits, amplia; caso contrário, documente no USD.
#         """
#         required_abs = max(math.radians(self.min_steer_abs_deg), float(self.max_steer_angle))
#         try:
#             lower, upper = self.articulation_view.get_dof_limits()  # arrays [ndof], [ndof] ou [[lo,up],...]
#             # Normaliza formas diferentes de retorno
#             # Tenta tratar como vetores "lower[i], upper[i]"
#             lo = np.array([lower[i] for i in self._steer_idx], dtype=np.float32)
#             up = np.array([upper[i] for i in self._steer_idx], dtype=np.float32)

#             needs_update = (np.abs(lo) < required_abs) | (np.abs(up) < required_abs)
#             if np.any(needs_update):
#                 new_lo = np.where(needs_update, -required_abs, lo)
#                 new_up = np.where(needs_update,  +required_abs, up)
#                 self.articulation_view.set_dof_limits(
#                     joint_indices=self._steer_idx,
#                     lower_limits=new_lo,
#                     upper_limits=new_up
#                 )
#         except AttributeError:
#             # Sem suporte na versão — ajuste no USD (recommended: ±max_steer_angle, no mínimo ±45°)
#             pass

#     def post_reset(self):
#         super().post_reset()
#         if (self._rear_idx is None) or (self._front_idx is None) or (self._steer_idx is None):
#             self._ensure_joint_indices()
#             self._ensure_steering_limits()

#     # ---------- Construção ----------
#     @classmethod
#     def build(cls, prim_path: str) -> "ForkliftCRobot":
#         world = get_world()

#         robot = world.scene.add(_WheeledRobot(
#             prim_path,
#             wheel_dof_names=cls.rear_wheel_dof_names + cls.front_wheel_dof_names,
#             create_robot=True,
#             usd_path=cls.usd_url
#         ))

#         view_path = os.path.join(prim_path, cls.chassis_subpath) if cls.chassis_subpath else prim_path
#         view = _ArticulationView(view_path)
#         world.scene.add(view)

#         camera = cls.build_front_camera(prim_path)
#         return cls(prim_path=prim_path, robot=robot, articulation_view=view, front_camera=camera)

#     # ---------- Controle (malha aberta) ----------
#     def write_action(self, step_size: float):
#         """
#         Entrada: (v_mps, delta_rad)
#           - v_mps     : velocidade linear desejada nas rodas traseiras [m/s]
#           - delta_rad : ângulo de direção desejado [rad], mesmo para os dois rotators
#         Saída:
#           - rodas traseiras: velocidade alvo (rad/s) com rampa
#           - rodas dianteiras: velocidade alvo zero
#           - direção: posição alvo (rad) com clamp pelos limites garantidos
#         """
#         if (self._rear_idx is None) or (self._front_idx is None) or (self._steer_idx is None):
#             self._ensure_joint_indices()
#             self._ensure_steering_limits()

#         v_mps, delta_rad = self.action.get_value()
#         v_mps    = float(v_mps)
#         delta_in = float(delta_rad)

#         # Respeita o maior entre (curso garantido) e (max_steer_angle declarado)
#         steer_abs_max = max(math.radians(self.min_steer_abs_deg), float(self.max_steer_angle))
#         delta_cmd = float(np.clip(delta_in, -steer_abs_max, +steer_abs_max))

#         # Mesmo sentido no mundo (ajustável via steering_axis_signs, se um eixo local estiver invertido)
#         steer_targets = np.asarray(self.steering_axis_signs, dtype=np.float32) * delta_cmd
#         self.articulation_view.set_joint_position_targets(
#             positions=steer_targets.astype(np.float32),
#             joint_indices=self._steer_idx
#         )

#         # Tração apenas traseira: omega = v / R
#         omega_des = float(v_mps / max(self.wheel_radius, 1e-6))
#         omega_pair = np.array([omega_des, omega_des], dtype=np.float32)

#         dt = float(step_size) if step_size else self.physics_dt
#         max_domega = float(self.max_wheel_accel_radps2) * dt
#         domega = np.clip(omega_pair - self._rear_omega_cmd, -max_domega, +max_domega)
#         self._rear_omega_cmd = self._rear_omega_cmd + domega

#         # Aplica velocidades
#         self.articulation_view.set_joint_velocity_targets(
#             velocities=self._rear_omega_cmd.astype(np.float32),
#             joint_indices=self._rear_idx
#         )
#         # Dianteiras paradas (sem tração)
#         self.articulation_view.set_joint_velocity_targets(
#             velocities=np.zeros(2, dtype=np.float32),
#             joint_indices=self._front_idx
#         )

    # ---------- Observações técnicas ----------
    # - Cinemática típica de empilhadeira (rear-wheel steering):
    #     R ≈ L / tan(delta), yaw_rate ≈ v * tan(delta) / L.
    # - Se notar que os rotators giram em sentidos opostos no mundo, ajuste:
    #     steering_axis_signs = (1.0, -1.0)  # ou vice-versa
    # - Caso a API não permita ampliar limites, configure ±max_steer_angle (>= 45°) diretamente no USD.


# # ---------- Subclasse para o asset Forklift ----------
# @ROBOTS.register()
# class ForkliftCRobot(FourWheeledRearSteerRobot):
#     # Física/render
#     physics_dt: float = 0.005
#     z_offset: float = 0.25

#     # DOFs (ordem: [FL, FR, RL, RR])
#     wheel_dof_names = [
#         "left_front_wheel_joint", "right_front_wheel_joint",
#         "left_back_wheel_joint",  "right_back_wheel_joint",
#     ]
#     # DUAS juntas TRASEIRAS (yaw)
#     steering_dof_names = ["left_rotator_joint", "right_rotator_joint"]

#     # Asset
#     usd_url = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd"
#     chassis_subpath: str = ""

#     # Geometria/limites (ângulo alto para curvas fáceis; o USD pode limitar)
#     wheel_radius: float = 0.30
#     max_steer_angle: float = 1.90
#     steer_axis_signs: Tuple[float, float] = (-1.0, +1.0)  # virar MESMO LADO no mundo
#     invert_steering: bool = False
#     keyboard_linear_velocity_gain: float = 1.7

#     # ==== Requisitos do builder (evita erros de atributos ausentes) ====
#     # Occupancy map
#     occupancy_map_radius: float = 1.5
#     occupancy_map_z_min: float = 0.05
#     occupancy_map_z_max: float = 1.20
#     occupancy_map_cell_size: float = 0.05
#     occupancy_map_collision_radius: float = 0.50

#     # Chase camera (usado por build_scenario_from_config → robot.build_chase_camera())
#     chase_camera_base_path: str = "body"
#     chase_camera_x_offset: float = -1.2
#     chase_camera_z_offset: float = 7.3
#     chase_camera_tilt_angle: float = 60.0

#@ROBOTS.register()
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
    occupancy_map_cell_size: float = 0.07
    occupancy_map_collision_radius: float = 0.25

    front_camera_base_path = "chassis/rgb_camera/front_hawk"
    front_camera_rotation = (0., 0., 0.)
    front_camera_translation = (0., 0., 0.)
    front_camera_type = HawkCamera

    keyboard_linear_velocity_gain: float = 1.0
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
    

#@ROBOTS.register()
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


#@ROBOTS.register()
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


#@ROBOTS.register()
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

#@ROBOTS.register()
class ForkliftRobot(FourWheelRearSteerRobot_V1):
    """
    Concrete Forklift robot that attaches:
      - front stereo camera (ZedStereoCamera)
      - left and right fisheye cameras (simple Camera wrappers)
      - 360 3D Lidar (Lidar wrapper)

    Provides a simple control-mode selector: 'manual' or 'auto'.
    In 'auto' mode the robot follows a 2D path (list of (x,y) world points)
    using a lightweight pure-pursuit style controller. This keeps the
    mobility_gen recording APIs compatible because sensors expose
    the same Buffers (tags) used by the writer/reader.
    """

    # auto navigation defaults
    auto_max_speed: float = 1.0
    auto_ang_gain: float = 1.5
    auto_arrival_tol: float = 0.25

    def __init__(self, prim_path: str, robot: _WheeledRobot, articulation_view: _ArticulationView, front_camera: Sensor | None = None):
        super().__init__(prim_path=prim_path, robot=robot, articulation_view=articulation_view, front_camera=front_camera)
        # sensor placeholders
        self.front_stereo: Optional[Sensor] = None
        self.fisheye_left: Optional[Camera] = None
        self.fisheye_right: Optional[Camera] = None
        self.lidar: Optional[object] = None

        # control mode and path
        self.control_mode: str = "manual"  # or 'auto'
        self._auto_path: Optional[List[Tuple[float, float]]] = None
        self._auto_idx: int = 0

    @classmethod
    def build(cls, prim_path: str) -> "ForkliftRobot":
        # Use base builder to create articulation and robot
        base = super().build(prim_path)
        # attach extra sensors under <prim_path>/sensors
        try:
            # front stereo
            from omni.ext.mobility_gen.sensors import ZedStereoCamera, Camera, Lidar
            front_path = os.path.join(prim_path, "sensors/front_stereo")
            base.front_stereo = ZedStereoCamera.build(front_path)

            # left fisheye
            left_path = os.path.join(prim_path, "sensors/fisheye_left")
            base.fisheye_left = Camera(left_path, (640, 480))
            # position/rotate to left
            try:
                from omni.ext.mobility_gen.sensors import _xform_translate, _xform_orient_quat
                _xform_translate(left_path, (1.0, -0.45, 1.2))
                qw, qx, qy, qz = _quat_from_euler_xyz(0.0, 0.0, 20.0)
                _xform_orient_quat(left_path, (qw, qx, qy, qz))
            except Exception:
                pass

            # right fisheye
            right_path = os.path.join(prim_path, "sensors/fisheye_right")
            base.fisheye_right = Camera(right_path, (640, 480))
            try:
                _xform_translate(right_path, (1.0, 0.45, 1.2))
                qw, qx, qy, qz = _quat_from_euler_xyz(0.0, 0.0, -20.0)
                _xform_orient_quat(right_path, (qw, qx, qy, qz))
            except Exception:
                pass

            # lidar
            lidar_path = os.path.join(prim_path, "sensors/lidar")
            base.lidar = Lidar(lidar_path)
            try:
                base.lidar.enable_lidar()
            except Exception:
                # best-effort
                pass

            # enable rgb rendering on cameras so writer picks up frames
            try:
                if hasattr(base.front_stereo, "left"):
                    base.front_stereo.left.enable_rgb_rendering()
                    base.front_stereo.right.enable_rgb_rendering()
            except Exception:
                pass
            try:
                base.fisheye_left.enable_rgb_rendering()
                base.fisheye_right.enable_rgb_rendering()
            except Exception:
                pass
        except Exception:
            # sensors are optional at build-time if environment lacks assets
            pass

        return base

    def set_control_mode(self, mode: str):
        assert mode in ("manual", "auto"), "mode must be 'manual' or 'auto'"
        self.control_mode = mode

    def set_auto_path(self, path: List[Tuple[float, float]]):
        """Provide a list of (x,y) world points for the auto controller to follow."""
        self._auto_path = path
        self._auto_idx = 0

    def plan_path_from_occupancy_map(self, occupancy_map) -> List[Tuple[float, float]]:
        """Plan a path using the repository's path_planner and set it as the auto path.

        This is a convenience wrapper so scenarios can call:
            robot.plan_path_from_occupancy_map(occupancy_map)

        The method does a best-effort import of the planner (it may require
        that the `path_planner` package is available in sys.path or that
        `scenarios.py` has inserted the repo path into sys.path as done in
        the project). If the planner is unavailable a RuntimeError is raised.
        """
        try:
            from mobility_gen_path_planner import generate_paths
        except Exception as e:
            raise RuntimeError("mobility_gen path_planner not importable in this Python environment") from e

        # current robot pose in world
        pose = self.get_pose_2d()
        # convert to pixel coordinates for planner
        start_px = occupancy_map.world_to_pixel_numpy(np.array([[pose.x, pose.y]]))
        # planner expects (row,col) as (y,x) ordering; follow the convention used
        # in the existing scenarios code (swap indices)
        start = (int(start_px[0, 1]), int(start_px[0, 0]))

        freespace = occupancy_map.buffered_meters(self.occupancy_map_radius).freespace_mask()
        output = generate_paths(start, freespace)
        end = output.sample_random_end_point()
        path_px = output.unroll_path(end)
        # planner returns (y,x) pixel coords -> swap to (x,y)
        path_px = path_px[:, ::-1]
        # convert pixel path to world coords and set as auto path
        path_world = occupancy_map.pixel_to_world_numpy(path_px)
        path_list = [(float(x), float(y)) for x, y in path_world]
        self.set_auto_path(path_list)
        return path_list

    def _compute_auto_command(self) -> Tuple[float, float]:
        """Simple go-to-point controller returning (v_mps, delta_rad)."""
        if (not self._auto_path) or (self._auto_idx >= len(self._auto_path)):
            return (0.0, 0.0)

        # current pose
        pose = self.get_pose_2d()
        tx, ty = self._auto_path[self._auto_idx]
        dx = tx - pose.x
        dy = ty - pose.y
        dist = math.hypot(dx, dy)
        ang_to_target = math.atan2(dy, dx)
        # heading error
        dtheta = (ang_to_target - pose.theta + math.pi) % (2 * math.pi) - math.pi

        # arrival
        if dist < float(self.auto_arrival_tol):
            self._auto_idx += 1
            return (0.0, 0.0)

        v = float(np.clip(dist * 0.6, -self.auto_max_speed, self.auto_max_speed))
        delta = float(np.clip(self.auto_ang_gain * dtheta, -self.max_steer_angle, +self.max_steer_angle))
        return (v, delta)

    def write_action(self, step_size: float):
        # if auto mode, compute and overwrite action
        if self.control_mode == "auto":
            cmd = self._compute_auto_command()
            self.action.set_value(np.array([cmd[0], cmd[1]], dtype=np.float32))

        # delegate to base implementation which applies wheel/steer
        super().write_action(step_size)




#@ROBOTS.register()
class ForkliftRobotV3(FourWheelRearSteerRobot_V1):
    """
    Versão V3 simplificada do Forklift:
    - Herda diretamente de Robot.
    - Usa um comando de ação compatível com os cenários de empilhadeira:
        action = [v_mps, delta_rad]
      onde v_mps é a velocidade linear desejada e delta_rad é o ângulo de direção
      traseiro (mesmo sinal nas duas juntas de direção).
    - Controla as rodas e juntas traseiras via ArticulationView (sem editar prims USD).
    - Anexa:
        - Câmera estéreo frontal (ZedStereoCamera).
        - Duas câmeras fisheye (esquerda/direita) usando Camera.
        - Um Lidar 3D para nuvem de pontos usando Lidar.
    """

    # ===== Sim / sensores básicos =====
    physics_dt: float = 0.005
    z_offset: float = 0.25

    # Third-person chase camera behind and above the forklift.
    chase_camera_base_path: str = "body"
    chase_camera_x_offset: float = -4.0
    chase_camera_z_offset: float = 2.0
    chase_camera_tilt_angle: float = 35.0

    # ===== Occupancy Map =====
    occupancy_map_radius: float = 1.5
    occupancy_map_z_min: float = 0.05
    occupancy_map_z_max: float = 1.2
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.3

    # ===== Teleop / random actions / path following (mantidos compatíveis com cenários) =====
    keyboard_linear_velocity_gain: float = 1.0
    keyboard_angular_velocity_gain: float = 1.0

    gamepad_linear_velocity_gain: float = 1.0
    gamepad_angular_velocity_gain: float = 1.0

    random_action_linear_velocity_range: Tuple[float, float] = (-0.3, 0.25)
    random_action_angular_velocity_range: Tuple[float, float] = (-0.75, 0.75)
    random_action_linear_acceleration_std: float = 1.0
    random_action_angular_acceleration_std: float = 5.0
    random_action_grid_pose_sampler_grid_size: float = 5.0

    path_following_speed: float = 0.5
    path_following_angular_gain: float = 1.0
    path_following_stop_distance_threshold: float = 0.5
    path_following_forward_angle_threshold = math.pi
    path_following_target_point_offset_meters: float = 1.0

    # ===== Geometria / juntas / asset =====
    wheel_base: float = 1.65
    track_width: float = 1.25
    wheel_radius: float = 0.5

    max_steer_angle: float = math.radians(80.0)

    # Nomes das juntas conforme o USD de forklift_c.usd
    rear_wheel_dof_names: List[str] = ["left_back_wheel_joint", "right_back_wheel_joint"]
    steering_dof_names:   List[str] = ["left_rotator_joint", "right_rotator_joint"]

    usd_url: str = (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/4.2/Isaac/Robots/Forklift/forklift_c.usd"
    )
    chassis_subpath: str = ""  # articulação raiz é o próprio prim do robô

    # Câmera frontal "oficial" para compatibilidade com Robot.build_front_camera
    front_camera_type: Type[Sensor] = ZedStereoCamera
    front_camera_base_path: str = "sensors/front_stereo"
    front_camera_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    front_camera_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __init__(
        self,
        prim_path: str,
        robot: _WheeledRobot,
        articulation_view: _ArticulationView,
        front_camera: Sensor | None = None,
    ):
        # Usa o construtor da classe-base FourWheelRearSteerRobot_V1 (que por sua vez
        # chama Robot.__init__) para preservar a integração com o _WheeledRobot
        # e a lógica de índices/limites já validada.
        super().__init__(prim_path=prim_path, robot=robot, articulation_view=articulation_view, front_camera=front_camera)

        # Sensores expostos de forma explícita (usados pela extensão e pelo Writer)
        self.front_stereo: Optional[Sensor] = front_camera  # ZedStereoCamera
        self.fisheye_left: Optional[Camera] = None
        self.fisheye_right: Optional[Camera] = None
        self.lidar: Optional[Lidar] = None

        # Cache de índices de DOF (rodas e direção)
        self._rear_idx: Optional[np.ndarray] = None
        self._steer_idx: Optional[np.ndarray] = None

    # ------------- Construção -------------
    @classmethod
    def build(cls, prim_path: str) -> "ForkliftRobotV3":
        """
        Constrói o Forklift V3 reutilizando o pipeline de construção já testado
        de FourWheelRearSteerRobot_V1 (que instancia o _WheeledRobot,
        ArticulationView e configura limites/índices de DOF), e em seguida
        apenas "embrulha" essa instância em ForkliftRobotV3 para adicionar
        sensores extras e uma write_action mais simples.
        """
        base = FourWheelRearSteerRobot_V1.build(prim_path)

        instance = cls(
            prim_path=base.prim_path,
            robot=base.robot,
            articulation_view=base.articulation_view,
            front_camera=base.front_camera,
        )

        # Reaproveita índices já computados pela base, se existirem
        try:
            instance._rear_idx = getattr(base, "_rear_idx", None)
            instance._steer_idx = getattr(base, "_steer_idx", None)
            instance.front_stereo = getattr(base, "front_stereo", base.front_camera)
            instance.fisheye_left = getattr(base, "fisheye_left", None)
            instance.fisheye_right = getattr(base, "fisheye_right", None)
            instance.lidar = getattr(base, "lidar", None)
        except Exception:
            pass

        # Completa sensores auxiliares ausentes.
        if instance.fisheye_left is None or instance.fisheye_right is None or instance.lidar is None:
            instance._attach_aux_sensors()

        return instance

    def _setup_dof_indices(self):
        """Resolve índices de DOF para rodas traseiras e juntas de direção."""
        try:
            self.articulation_view.initialize()
        except Exception:
            pass

        def _resolve(names: List[str]) -> np.ndarray:
            idxs: List[int] = []
            for n in names:
                i = -1
                try:
                    i_val = self.articulation_view.get_dof_index(n)
                    # Alguns wrappers retornam None quando o DOF não existe;
                    # convertemos isso explicitamente para -1 para evitar erros
                    # no np.asarray(..., dtype=np.int32).
                    if i_val is None:
                        i = -1
                    else:
                        i = int(i_val)
                except Exception:
                    i = -1
                idxs.append(i)
            arr = np.asarray(idxs, dtype=np.int32)
            if (arr < 0).any():
                print(f"[ForkliftRobotV3] Warning: some DOF names not found: {names} -> {arr}")
            return arr

        self._rear_idx = _resolve(self.rear_wheel_dof_names)
        self._steer_idx = _resolve(self.steering_dof_names)

    def _attach_aux_sensors(self):
        """Anexa câmeras fisheye e Lidar em caminhos fixos sob o prim do robô."""
        # Paths baseados em self.prim_path para manter a cena organizada
        # Já temos self.front_camera (ZedStereoCamera) via build_front_camera,
        # mas expomos também em self.front_stereo para ficar explícito.
        self.front_stereo = self.front_camera

        # Câmera fisheye esquerda
        try:
            left_path = os.path.join(self.prim_path, "sensors/fisheye_left")
            _define_camera_prim(left_path)
            self.fisheye_left = Camera(left_path, (640, 480))
            _xform_translate(left_path, (1.05, -0.38, 1.58))
            qw, qx, qy, qz = _quat_from_euler_xyz(0.0, 0.0, 28.0)
            _xform_orient_quat(left_path, (qw, qx, qy, qz))
        except Exception:
            self.fisheye_left = None

        # Câmera fisheye direita
        try:
            right_path = os.path.join(self.prim_path, "sensors/fisheye_right")
            _define_camera_prim(right_path)
            self.fisheye_right = Camera(right_path, (640, 480))
            _xform_translate(right_path, (1.05, 0.38, 1.58))
            qw, qx, qy, qz = _quat_from_euler_xyz(0.0, 0.0, -28.0)
            _xform_orient_quat(right_path, (qw, qx, qy, qz))
        except Exception:
            self.fisheye_right = None

        # Lidar 3D no topo
        try:
            lidar_path = os.path.join(self.prim_path, "sensors/lidar")
            self.lidar = Lidar(lidar_path)
            _xform_translate(lidar_path, (0.25, 0.0, 2.10))
            try:
                self.lidar.enable_lidar()
            except Exception:
                pass
        except Exception:
            self.lidar = None

        # Garante que câmeras tenham RGB habilitado para gravação
        try:
            if hasattr(self.front_stereo, "left"):
                self.front_stereo.left.enable_rgb_rendering()
            if hasattr(self.front_stereo, "right"):
                self.front_stereo.right.enable_rgb_rendering()
        except Exception:
            pass
        try:
            if self.fisheye_left is not None:
                self.fisheye_left.enable_rgb_rendering()
        except Exception:
            pass
        try:
            if self.fisheye_right is not None:
                self.fisheye_right.enable_rgb_rendering()
        except Exception:
            pass

    # ------------- Controle -------------
    def write_action(self, step_size: float):
        """
        Interpreta self.action como [v_mps, delta_rad] e:
        - Converte v_mps em velocidades angulares nas rodas traseiras (ω = v/R).
        - Aplica delta_rad como alvo de posição igual nas duas juntas de direção.

        Isso é compatível com:
        - KeyboardTeleoperationScenario_forklift / _forklift_2 (cenários que já
          publicam [v_mps, delta_rad]).
        - RandomAccelerationScenario / RandomPathFollowingScenario, desde que
          configurados para empurrar a mesma convenção de ação.
        """
        try:
            action = self.action.get_value()
        except Exception:
            action = None

        if action is None or len(action) < 2:
            return

        v_mps = float(action[0])
        delta = float(action[1])

        # Garante limites de direção básicos (sem alterar USD)
        delta = float(np.clip(delta, -self.max_steer_angle, self.max_steer_angle))

        # Garante índices/limites usando a lógica robusta da classe-base
        try:
            if getattr(self, "_rear_idx", None) is None or getattr(self, "_steer_idx", None) is None:
                if hasattr(self, "_ensure_indices_limits_and_drive"):
                    self._ensure_indices_limits_and_drive()
        except Exception:
            pass

        if getattr(self, "_rear_idx", None) is None or getattr(self, "_steer_idx", None) is None:
            # sem índices válidos, não aplicamos ação
            return

        # 1) Rodas traseiras: velocidade angular ω = v / R
        try:
            omega = v_mps / max(self.wheel_radius, 1e-6)
            wheel_vel = np.array([omega, omega], dtype=np.float32)
            self.articulation_view.set_joint_velocity_targets(
                velocities=wheel_vel,
                joint_indices=self._rear_idx,
            )
        except Exception:
            pass

        # 2) Direção traseira: mesmo ângulo nas duas juntas
        try:
            steer_targets = np.array([delta, delta], dtype=np.float32)
            self.articulation_view.set_joint_position_targets(
                positions=steer_targets,
                joint_indices=self._steer_idx,
            )
        except Exception:
            pass
