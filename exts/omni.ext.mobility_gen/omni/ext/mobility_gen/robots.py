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
    # @classmethod
    # def build_front_camera(cls, prim_path):
    #     """
    #     Cria um mount seguro para a câmera fora da hierarquia rígida do `body`,
    #     garantindo um Xform puro (sem colliders/rigid body) e sem imports extras.
    #     """

    #     # Evita montar a câmera dentro de "body/*".
    #     # Sempre usamos um mount sob "<robot>/sensors/front_camera".
    #     camera_mount_rel = "sensors/front_camera"
    #     camera_path = os.path.join(prim_path, camera_mount_rel)

    #     # Garante que o mount é um Xform puro (isso NÃO adiciona física nem colisores)
    #     front_camera_xform = XFormPrim(camera_path)

    #     # Posiciona o mount (transform relativo ao robô)
    #     stage = get_stage()
    #     cam_prim = stage_get_prim(stage, camera_path)
    #     prim_rotate_x(cam_prim, cls.front_camera_rotation[0])
    #     prim_rotate_y(cam_prim, cls.front_camera_rotation[1])
    #     prim_rotate_z(cam_prim, cls.front_camera_rotation[2])
    #     prim_translate(cam_prim, cls.front_camera_translation)

    #     # Constrói o sensor dentro do mount Xform.
    #     # Como o mount NÃO é um RigidBody e está FORA do `body`,
    #     # não ocorre o erro de "RigidBody filho de RigidBody".
    #     return cls.front_camera_type.build(prim_path=camera_path)
    

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

    chase_camera_base_path: str = "body"
    chase_camera_x_offset: float = -5.0#
    chase_camera_z_offset: float =  -10.0
    chase_camera_tilt_angle: float = 60.0

    # ===== Occupancy Map (usado pelo builder) =====
    occupancy_map_radius: float = 1.5
    occupancy_map_z_min: float = 0.05
    occupancy_map_z_max: float = 1.2
    occupancy_map_cell_size: float = 0.05
    occupancy_map_collision_radius: float = 0.3

    # ===== Câmera frontal =====
    front_camera_type = ZedStereoCamera
    front_camera_base_path = "rgb_camera/front_camera"
    front_camera_rotation = (0., 0., 0.)
    front_camera_translation = (10., 0., 10.)

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
        self.controller = RearDriveSimpleController(self.wheel_radius, self.max_steer_angle)
        self._rear_idx: np.ndarray | None = None
        self._steer_idx: np.ndarray | None = None
        self._last_cmd: np.ndarray | None = None
        self._auto_done = False
        self._auto_frames = 45
        self._frame = 0

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
        return cls(
            prim_path=prim_path,
            robot=robot,
            articulation_view=view,
            front_camera=camera
        )

    # ---------- Helpers ----------
    def _resolve_indices(self, names: List[str]) -> np.ndarray:
        idx = []
        for n in names:
            try: idx.append(self.articulation_view.get_dof_index(n))
            except Exception: idx.append(-1)
        return np.asarray(idx, np.int32)

    def _auto_find_rear_steer_dofs(self) -> Optional[List[str]]:
        # tenta localizar DOFs traseiros de esterço por nome
        names = []
        for attr in ("get_dof_names", "dof_names", "_dof_names"):
            if hasattr(self.articulation_view, attr):
                try:
                    val = getattr(self.articulation_view, attr)
                    names = list(val() if callable(val) else val); break
                except Exception:
                    pass
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

    def _ensure_indices_limits_and_drive(self):
        try: self.articulation_view.initialize()
        except Exception: pass

        # rodas traseiras
        if self._rear_idx is None:
            idx = self._resolve_indices(self.rear_wheel_dof_names)
            if (idx < 0).any():
                raise RuntimeError(f"[RearSteer] Juntas traseiras não encontradas: {self.rear_wheel_dof_names}")
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
                raise RuntimeError(f"[RearSteer] Juntas de direção não encontradas. Ajuste steering_dof_names. (atual={self.steering_dof_names})")
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
    # ---------- ciclo de vida ----------
    def post_reset(self):
        super().post_reset()
        self._ensure_indices_limits_and_drive()
        self._last_cmd = None
        self._auto_done = False
        self._frame = 0
    # ---------- controle ----------
    def write_action(self, step_size: float):
        # 1) rodas traseiras: ω = v/R
        wheel_action, steer_targets = self.controller.forward(self.action.get_value())
        self.robot.apply_wheel_actions(wheel_action)

        # 2) direção traseira: δ igual nos dois lados
        self._ensure_indices_limits_and_drive()
        steer_targets = (np.asarray(self.steering_axis_signs, np.float32) * steer_targets).astype(np.float32)
        lim = float(self.effective_steer_limit)
        steer_targets = np.clip(steer_targets, -lim, +lim).astype(np.float32)
        self.articulation_view.set_joint_position_targets(
            positions=steer_targets, joint_indices=self._steer_idx
        )

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




