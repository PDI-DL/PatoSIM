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
import math

from mobility_gen_path_planner import generate_paths

from omni.ext.mobility_gen.utils.path_utils import PathHelper, vector_angle
from omni.ext.mobility_gen.utils.registry import Registry
from omni.ext.mobility_gen.common import Module, Buffer
from omni.ext.mobility_gen.robots import Robot
from omni.ext.mobility_gen.occupancy_map import OccupancyMap

import omni.ext.mobility_gen.pose_samplers as pose_samplers
import omni.ext.mobility_gen.inputs as inputs
import omni.replicator.core as rep
# >>> NOVO: interface do teclado do Kit para teclas extras




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

#@SCENARIOS.register()
class RandomPathFollowingScenario(Scenario):
    """
    Path following compatível com FourWheelRearSteerRobot_V1:
      action = [v_mps, delta_rad]
    - v: velocidade para avançar ao longo do caminho
    - delta: ângulo de direção (traseiro) limitado por effective_steer_limit ou max_steer_angle
    """

    def __init__(self, robot: Robot, occupancy_map: OccupancyMap):
        super().__init__(robot, occupancy_map)
        self.pose_sampler = pose_samplers.UniformPoseSampler()
        self.is_alive = True
        self.target_path = Buffer()
        self.collision_occupancy_map = occupancy_map.buffered(robot.occupancy_map_collision_radius)

        # parâmetros do seguidor de caminho vindos do robô (com fallbacks seguros)
        self._v_nom = float(getattr(self.robot, "path_following_speed", 1.0))
        self._k_ang = float(getattr(self.robot, "path_following_angular_gain", 1.5))
        self._stop_dist = float(getattr(self.robot, "path_following_stop_distance_threshold", 0.5))
        self._fwd_ang_th = float(getattr(self.robot, "path_following_forward_angle_threshold", np.pi))
        self._lookahead = float(getattr(self.robot, "path_following_target_point_offset_meters", 1.0))

        # limite de direção (rad)
        self._delta_lim = float(getattr(self.robot, "effective_steer_limit",
                                  getattr(self.robot, "max_steer_angle", 0.6)))

        # filtragem leve para delta (evita chattering)
        self._delta = 0.0
        self._alpha_delta = 0.35  # 0..1 (0=mais suave)

    # ---------- utilidades ----------
    def _set_random_target_path(self):
        """Gera um caminho aleatório a partir da pose atual."""
        current_pose = self.robot.get_pose_2d()

        start_px = self.occupancy_map.world_to_pixel_numpy(
            np.array([[current_pose.x, current_pose.y]])
        )
        freespace = self.buffered_occupancy_map.freespace_mask()
        start = (start_px[0, 1], start_px[0, 0])

        output = generate_paths(start, freespace)
        end = output.sample_random_end_point()
        path = output.unroll_path(end)
        path = path[:, ::-1]  # (y,x) -> (x,y)
        path = self.occupancy_map.pixel_to_world_numpy(path)

        self.target_path.set_value(path)
        self._helper = PathHelper(path)

    # ---------- ciclo de vida ----------
    def reset(self):
        # zera e posiciona
        self.robot.action.set_value(np.zeros(2, dtype=np.float32))
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)

        # novo caminho e estado
        self._set_random_target_path()
        self._delta = 0.0
        self.is_alive = True

    # ---------- passo ----------
    def step(self, step_size: float) -> bool:
        self.update_state()

        # checagens de segurança
        current_pose = self.robot.get_pose_2d()
        if not self.collision_occupancy_map.check_world_point_in_bounds(current_pose):
            self.is_alive = False
            return False
        if not self.collision_occupancy_map.check_world_point_in_freespace(current_pose):
            self.is_alive = False
            return False

        # ponto mais próximo e ponto de lookahead
        path = self.target_path.get_value()
        pt_robot = np.array([current_pose.x, current_pose.y], dtype=np.float32)
        _, s_near, _, _ = self._helper.find_nearest(pt_robot)
        pt_target = self._helper.get_point_by_distance(distance=s_near + self._lookahead)

        # condição de chegada
        dist_to_goal = float(np.linalg.norm(pt_robot - path[-1]))
        if dist_to_goal < self._stop_dist:
            self._set_random_target_path()
            # após replanejar, nada a fazer nesse frame
            self.robot.action.set_value(np.array([0.0, 0.0], dtype=np.float32))
            self.robot.write_action(step_size)
            return True

        # orientação atual e rumo até alvo
        v_robot = np.array([np.cos(current_pose.theta), np.sin(current_pose.theta)], dtype=np.float32)
        v_targ = (pt_target - pt_robot).astype(np.float32)
        n = float(np.linalg.norm(v_targ))
        if n < 1e-6:
            # alvo degenerado; avance pouco e mantenha delta atual
            v_cmd = 0.2 * self._v_nom
            delta_cmd = self._delta
        else:
            v_targ_unit = v_targ / n
            d_theta = vector_angle(v_robot, v_targ_unit)  # [-pi, pi], positivo = alvo à esquerda

            # política simples: se muito “de costas”, pare para corrigir
            v_cmd = 0.0 if abs(d_theta) > self._fwd_ang_th else self._v_nom

            # mapeia erro angular direto para ângulo de direção traseiro
            delta_raw = self._k_ang * d_theta
            delta_cmd = float(np.clip(delta_raw, -self._delta_lim, self._delta_lim))

            # filtragem (evita oscilações no drive da junta)
            self._delta = (1.0 - self._alpha_delta) * self._delta + self._alpha_delta * delta_cmd
            delta_cmd = self._delta

        # publica ação e aplica
        self.robot.action.set_value(np.array([v_cmd, delta_cmd], dtype=np.float32))
        self.robot.write_action(step_size)
        return True

@SCENARIOS.register()
class KeyboardTeleoperationScenario_forklift_2(Scenario):

    """
    Teleop p/ FourWheelRearSteerRobot_V2
    Mapeamento:
      [0]=W, [1]=A, [2]=S, [3]=D, [4]=Space(FREIO), [5]=C(RECENTRA)
    Publica ação: [v_mps, delta_rad]
    """

    # constantes do teleop
    _default_steer_rate    = 1.5   # rad/s (taxa de variação de delta)
    _default_brake_acc     = 5.0   # m/s^2 (freio ao segurar Space)
    _default_center_relax  = 2.5   # rad/s (recentra ao segurar C)
    _warmup_frames_total   = 20    # reaquece drives pós-reset por N frames

    def __init__(self, robot: Robot, occupancy_map: OccupancyMap):
        super().__init__(robot, occupancy_map)
        self.keyboard = inputs.Keyboard()
        self.pose_sampler = pose_samplers.UniformPoseSampler()

        # ganhos vindos do robô (se definidos)
        self._lin_gain   = float(getattr(self.robot, "keyboard_linear_velocity_gain", 1.0))
        self._steer_rate = float(getattr(self.robot, "keyboard_angular_velocity_gain", self._default_steer_rate))

        # estados internos
        self._v_cmd   = 0.0
        self._delta   = 0.0
        self._warmup  = 0

    def reset(self):
        # posiciona o robô e deixa a classe do robô reconfigurar drives pós-reset
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)

        # zera ação e dá um "prime" imediato para acordar drives
        self._v_cmd = 0.0
        self._delta = 0.0
        self.robot.action.set_value(np.array([0.0, 0.0], dtype=np.float32))
        self.robot.write_action(getattr(self.robot, "physics_dt", 0.005))

        # ativa warmup por alguns frames (força publicação contínua)
        self._warmup = self._warmup_frames_total

    def step(self, step_size: float) -> bool:
        self.update_state()

        # limite de direção PUXADO DO ROBÔ a cada frame (pode mudar após reset)
        delta_lim = float(getattr(self.robot, "effective_steer_limit",
                           getattr(self.robot, "max_steer_angle", 0.6)))

        # leitura dos botões
        buttons = self.keyboard.buttons.get_value()
        def btn(i: int) -> float:
            return float(buttons[i]) if i < len(buttons) else 0.0

        w = btn(0); a = btn(1); s = btn(2); d = btn(3)
        space = btn(4)  # freio
        cent  = btn(5)  # recentra

        # alvo de velocidade (antes do freio)
        v_target = (w - s) * self._lin_gain

        # freio (Space): desacelera até zero com taxa fixa
        if space > 0.5:
            dv = self._default_brake_acc * float(step_size)
            if abs(self._v_cmd) <= dv:
                self._v_cmd = 0.0
            else:
                self._v_cmd -= np.sign(self._v_cmd) * dv
        else:
            self._v_cmd = v_target

        # integra direção com taxa limitada
        self._delta += (a - d) * self._steer_rate * float(step_size)

        # recentra quando 'C' está pressionado
        if cent > 0.5:
            relax = self._default_center_relax * float(step_size)
            if abs(self._delta) <= relax:
                self._delta = 0.0
            else:
                self._delta -= np.sign(self._delta) * relax

        # clamp no limite efetivo
        self._delta = float(np.clip(self._delta, -delta_lim, delta_lim))

        # publica ação e aplica no robô
        self.robot.action.set_value(np.array([self._v_cmd, self._delta], dtype=np.float32))
        self.robot.write_action(step_size)

        # warmup: nos primeiros frames pós-reset, publicamos mesmo parados
        if self._warmup > 0:
            self._warmup -= 1
            # reforça uma publicação redundante (direção+rodas) para reaquecer drives
            self.robot.write_action(step_size)

        self.update_state()
        return True

@SCENARIOS.register()
class KeyboardTeleoperationScenario_forklift(Scenario):
    """
    Teleop p/ FourWheelRearSteerRobot
    Mapeamento:
      [0]=W, [1]=A, [2]=S, [3]=D, [4]=Space(FREIO), [5]=C(RECENTRA)
    Ação publicada: [v_mps, delta_rad]

    v     = (W - S) * robot.keyboard_linear_velocity_gain
    delta = integrado por (A - D) * steer_rate, limitado por effective_steer_limit (ou max_steer_angle)
    """

    # parâmetros adicionais do teleop
    _default_steer_rate = 1.5      # rad/s (taxa de mudança do ângulo de direção)
    _default_brake_acc  = 5.0      # m/s^2 (desaceleração quando Space está pressionado)
    _default_center_relax = 2.5    # rad/s (taxa para recentrar quando C está pressionado)

    def __init__(self, robot, occupancy_map):
        super().__init__(robot, occupancy_map)
        self.keyboard = inputs.Keyboard()
        self.pose_sampler = pose_samplers.UniformPoseSampler()

        # ganhos do próprio robô (se existir) + defaults locais
        self._lin_gain   = float(getattr(self.robot, "keyboard_linear_velocity_gain", 1.0))
        self._steer_rate = float(getattr(self.robot, "keyboard_angular_velocity_gain",
                                         self._default_steer_rate))
        self._brake_acc  = self._default_brake_acc
        self._center_relax = self._default_center_relax

        # limite efetivo de direção (rad)
        self._delta_lim = float(getattr(self.robot, "effective_steer_limit",
                                 getattr(self.robot, "max_steer_angle", 0.6)))

        # estados
        self._v_cmd = 0.0
        self._delta = 0.0

    def reset(self):
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)
        self._v_cmd = 0.0
        self._delta = 0.0
        self.robot.action.set_value(np.array([0.0, 0.0], dtype=np.float32))

    def step(self, step_size: float) -> bool:
        self.update_state()

        buttons = self.keyboard.buttons.get_value()
        def btn(i: int) -> float:
            return float(buttons[i]) if i < len(buttons) else 0.0

        w = btn(0); a = btn(1); s = btn(2); d = btn(3)
        space = btn(4)   # freio
        cent  = btn(5)   # recentra

        # alvo de velocidade sem freio
        v_target = (w - s) * self._lin_gain

        # freio (Space): desacelera até zero com taxa fixa
        if space > 0.5:
            dv = self._brake_acc * float(step_size)
            if abs(self._v_cmd) <= dv:
                self._v_cmd = 0.0
            else:
                self._v_cmd -= np.sign(self._v_cmd) * dv
        else:
            self._v_cmd = v_target

        # integra direção com taxa limitada
        self._delta += (a - d) * self._steer_rate * float(step_size)
        # recentra quando 'C' está pressionado
        if cent > 0.5:
            relax = self._center_relax * float(step_size)
            if abs(self._delta) <= relax:
                self._delta = 0.0
            else:
                self._delta -= np.sign(self._delta) * relax

        # clamp no limite efetivo
        self._delta = float(np.clip(self._delta, -self._delta_lim, self._delta_lim))

        # publica ação e aplica no robô
        self.robot.action.set_value(np.array([self._v_cmd, self._delta], dtype=np.float32))
        self.robot.write_action(step_size)

        self.update_state()
        return True

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







