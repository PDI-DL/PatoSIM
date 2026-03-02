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
import os
import sys
import warnings

# Ensure the local `path_planner` package is importable when running inside Isaac Sim.
# Isaac's embedded Python doesn't include the repository layout on sys.path by default,
# so add the project's `path_planner` folder to sys.path if it exists.
_HERE = os.path.dirname(__file__)
# move up to the repository root: mobility_gen -> omni/ext/mobility_gen -> omni/ext -> omni.ext.mobility_gen -> exts -> <repo root>
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
_PATH_PLANNER = os.path.join(_REPO_ROOT, "path_planner")
if os.path.isdir(_PATH_PLANNER) and _PATH_PLANNER not in sys.path:
    sys.path.insert(0, _PATH_PLANNER)

try:
    import mobility_gen_path_planner as _planner_pkg
    from mobility_gen_path_planner import generate_paths as _generate_paths
    try:
        _planner_backend = "cpp" if getattr(_planner_pkg, "_C", None) is not None else "python"
    except Exception:
        _planner_backend = "python"
except Exception as exc:
    _planner_pkg = None
    _generate_paths = None
    _planner_backend = "unavailable"
    warnings.warn(
        f"mobility_gen_path_planner is unavailable ({exc}). "
        "RandomPathFollowingScenario will use robot planner fallback when possible.",
        RuntimeWarning,
    )

from omni.ext.mobility_gen.utils.path_utils import PathHelper, vector_angle
from omni.ext.mobility_gen.utils.registry import Registry
from omni.ext.mobility_gen.common import Module, Buffer
from omni.ext.mobility_gen.robots import Robot
from omni.ext.mobility_gen.occupancy_map import OccupancyMap
from omni.ext.mobility_gen.types import Point2d

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

#Naofunciona bem ainda
@SCENARIOS.register()
class RandomPathFollowingScenarioRearSteer(Scenario):
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
        # limite prático para path following (evita esterço exagerado em manobras automáticas)
        self._delta_cmd_lim = float(
            np.clip(
                getattr(self.robot, "path_following_max_steer_command", 0.45),
                0.12,
                self._delta_lim,
            )
        )

        # filtragem leve para delta (evita chattering)
        self._delta = 0.0
        self._alpha_delta = 0.35  # 0..1 (0=mais suave)
        self._delta_rate_limit = float(getattr(self.robot, "path_following_delta_rate_limit", 1.2))
        self._lookahead_min = float(getattr(self.robot, "path_following_lookahead_min", 0.6))
        self._lookahead_max = float(getattr(self.robot, "path_following_lookahead_max", 2.5))
        self._planner_min_goal_distance_m = float(
            getattr(self.robot, "path_following_min_goal_distance_m", 4.0)
        )
        self._path_smoothing_iterations = int(
            getattr(self.robot, "path_following_smoothing_iterations", 2)
        )
        self._safety_points = int(getattr(self.robot, "path_following_safety_points", 8))
        self._safety_margin = float(getattr(self.robot, "path_following_safety_margin", 0.35))
        self._max_curve_speed_factor = float(
            getattr(self.robot, "path_following_max_curve_speed_factor", 1.0)
        )
        self._min_curve_speed_factor = float(
            getattr(self.robot, "path_following_min_curve_speed_factor", 0.45)
        )
        self._min_v_cmd = float(getattr(self.robot, "path_following_min_speed", 0.18))
        self._last_v_cmd = 0.0
        self._blocked_steps = 0
        self._planner_info_printed = False

    # ---------- utilidades ----------
    def _planner_backend_info(self) -> str:
        if _planner_backend == "cpp":
            return "mobility_gen_path_planner (C++ backend)"
        if _planner_backend == "python":
            return "mobility_gen_path_planner (python wrapper)"
        return "planner unavailable"

    def _smooth_path(self, path_world: np.ndarray) -> np.ndarray:
        if path_world is None or len(path_world) < 3:
            return path_world
        out = path_world.astype(np.float32)
        for _ in range(max(0, self._path_smoothing_iterations)):
            if len(out) < 3:
                break
            refined = [out[0]]
            for i in range(len(out) - 1):
                p = out[i]
                q = out[i + 1]
                refined.append(0.75 * p + 0.25 * q)
                refined.append(0.25 * p + 0.75 * q)
            refined.append(out[-1])
            out = np.asarray(refined, dtype=np.float32)
        # remove pontos quase idênticos
        filtered = [out[0]]
        for p in out[1:]:
            if np.linalg.norm(p - filtered[-1]) > 0.02:
                filtered.append(p)
        return np.asarray(filtered, dtype=np.float32)

    def _choose_endpoint(self, output) -> Tuple[int, int]:
        visited = output.visited
        ys, xs = np.where(visited != 0)
        if len(ys) == 0:
            raise RuntimeError("Path planner returned no reachable endpoints.")

        dists = output.distance_to_start[ys, xs]
        min_px = max(4.0, self._planner_min_goal_distance_m / max(self.occupancy_map.resolution, 1e-6))
        valid = dists >= min_px
        if np.any(valid):
            ys = ys[valid]
            xs = xs[valid]
            dists = dists[valid]

        # Prefere endpoints mais longos para evitar caminhos curtos/instáveis
        order = np.argsort(dists)
        tail_start = int(0.6 * len(order))
        candidate_idx = order[tail_start:] if tail_start < len(order) else order
        choice = int(np.random.choice(candidate_idx))
        return int(ys[choice]), int(xs[choice])

    def _is_segment_free(self, a_xy: np.ndarray, b_xy: np.ndarray) -> bool:
        for t in np.linspace(0.0, 1.0, max(2, self._safety_points)):
            p = (1.0 - t) * a_xy + t * b_xy
            q = Point2d(x=float(p[0]), y=float(p[1]))
            if not self.collision_occupancy_map.check_world_point_in_freespace(q):
                return False
        return True

    def _compute_dynamic_lookahead(self, heading_error: float) -> float:
        lookahead = self._lookahead + 0.5 * abs(self._last_v_cmd) - 0.35 * abs(heading_error)
        return float(np.clip(lookahead, self._lookahead_min, self._lookahead_max))

    def _set_random_target_path(self):
        """Gera um caminho aleatório a partir da pose atual."""
        if not self._planner_info_printed:
            print(f"[RandomPathFollowingRearSteer] planner backend: {self._planner_backend_info()}")
            self._planner_info_printed = True

        # Caminho primário: planner C++ (via mobility_gen_path_planner)
        # Fallback: planner do robô (quando disponível)
        current_pose = self.robot.get_pose_2d()
        start_px = self.occupancy_map.world_to_pixel_numpy(
            np.array([[current_pose.x, current_pose.y]], dtype=np.float32)
        )
        freespace = self.buffered_occupancy_map.freespace_mask()
        start = (int(start_px[0, 1]), int(start_px[0, 0]))  # planner usa (row, col) = (y, x)

        try:
            if _generate_paths is not None:
                output = _generate_paths(start, freespace)
                end = self._choose_endpoint(output)
                path_px = output.unroll_path(end)
                path_px = path_px[:, ::-1]  # (y,x) -> (x,y)
                path = self.occupancy_map.pixel_to_world_numpy(path_px)
            elif hasattr(self.robot, "plan_path_from_occupancy_map"):
                path_list = self.robot.plan_path_from_occupancy_map(self.occupancy_map)
                path = np.asarray(path_list, dtype=np.float32)
            else:
                raise RuntimeError(
                    "mobility_gen_path_planner is unavailable and no robot fallback planner was found. "
                    "Install with Isaac python: ./app/python.sh -m pip install -e path_planner"
                )
        except Exception as exc:
            if hasattr(self.robot, "plan_path_from_occupancy_map"):
                path_list = self.robot.plan_path_from_occupancy_map(self.occupancy_map)
                path = np.asarray(path_list, dtype=np.float32)
                print(f"[RandomPathFollowingRearSteer] planner fallback via robot after error: {exc}")
            else:
                raise

        path = self._smooth_path(path.astype(np.float32))
        if path is None or len(path) < 2:
            raise RuntimeError("Generated path is invalid (needs at least 2 points).")

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
        self._last_v_cmd = 0.0
        self._blocked_steps = 0
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

        # ponto mais próximo e ponto de lookahead dinâmico
        path = self.target_path.get_value()
        pt_robot = np.array([current_pose.x, current_pose.y], dtype=np.float32)
        _, s_near, _, _ = self._helper.find_nearest(pt_robot)
        if s_near is None:
            self._set_random_target_path()
            return True
        pt_target_near = self._helper.get_point_by_distance(distance=s_near + self._lookahead)
        v_robot = np.array([np.cos(current_pose.theta), np.sin(current_pose.theta)], dtype=np.float32)
        v_targ_near = (pt_target_near - pt_robot).astype(np.float32)
        n_near = float(np.linalg.norm(v_targ_near))
        if n_near > 1e-6:
            d_theta_near = float(vector_angle(v_robot, v_targ_near / n_near))
        else:
            d_theta_near = 0.0
        lookahead = self._compute_dynamic_lookahead(d_theta_near)
        pt_target = self._helper.get_point_by_distance(distance=s_near + lookahead)

        # condição de chegada
        dist_to_goal = float(np.linalg.norm(pt_robot - path[-1]))
        if dist_to_goal < self._stop_dist:
            self._set_random_target_path()
            # após replanejar, nada a fazer nesse frame
            self.robot.action.set_value(np.array([0.0, 0.0], dtype=np.float32))
            self.robot.write_action(step_size)
            return True

        # orientação atual e rumo até alvo
        v_targ = (pt_target - pt_robot).astype(np.float32)
        n = float(np.linalg.norm(v_targ))
        if n < 1e-6:
            # alvo degenerado; avance pouco e mantenha delta atual
            v_cmd = 0.2 * self._v_nom
            delta_cmd = self._delta
        else:
            v_targ_unit = v_targ / n
            d_theta = vector_angle(v_robot, v_targ_unit)  # [-pi, pi], positivo = alvo à esquerda

            # mapeia erro angular direto para ângulo de direção traseiro
            delta_target = float(np.clip(self._k_ang * d_theta, -self._delta_cmd_lim, self._delta_cmd_lim))

            # filtragem + limitação da taxa de esterço (curva mais suave)
            delta_filtered = (1.0 - self._alpha_delta) * self._delta + self._alpha_delta * delta_target
            max_delta_step = max(1e-4, self._delta_rate_limit * float(step_size))
            delta_step = float(np.clip(delta_filtered - self._delta, -max_delta_step, +max_delta_step))
            self._delta = float(np.clip(self._delta + delta_step, -self._delta_cmd_lim, +self._delta_cmd_lim))
            delta_cmd = self._delta

            # velocidade adaptativa: reduz quando há grande heading error / grande esterço
            heading_factor = max(0.0, math.cos(abs(float(d_theta))))
            steer_ratio = min(1.0, abs(delta_cmd) / (self._delta_cmd_lim + 1e-6))
            curve_factor = self._max_curve_speed_factor - (
                (self._max_curve_speed_factor - self._min_curve_speed_factor) * steer_ratio
            )
            v_cmd = self._v_nom * heading_factor * curve_factor
            if v_cmd > 0.0:
                v_cmd = max(self._min_v_cmd, v_cmd)
            if abs(d_theta) > self._fwd_ang_th:
                v_cmd = 0.0

            # safety gate: se segmento à frente não está livre, reduz/paralisa e replaneja se persistir
            ahead_dist = max(self._safety_margin, abs(self._last_v_cmd) * 0.8 + self._safety_margin)
            ahead_target = pt_robot + ahead_dist * v_robot
            safe_corridor = self._is_segment_free(pt_robot, pt_target) and self._is_segment_free(pt_robot, ahead_target)
            if not safe_corridor:
                self._blocked_steps += 1
                v_cmd *= 0.35
                if v_cmd < 0.08:
                    v_cmd = 0.0
                if self._blocked_steps > 20:
                    self._set_random_target_path()
                    self._blocked_steps = 0
            else:
                self._blocked_steps = 0

        # publica ação e aplica
        self.robot.action.set_value(np.array([v_cmd, delta_cmd], dtype=np.float32))
        self.robot.write_action(step_size)
        self._last_v_cmd = float(v_cmd)
        return True

#@SCENARIOS.register()
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
        self._lin_gain   = float(getattr(self.robot, "keyboard_linear_velocity_gain", 1.2))
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
            # buttons may be None if the input system isn't initialized; guard it
            if buttons is None:
                return 0.0
            try:
                return float(buttons[i]) if i < len(buttons) else 0.0
            except Exception:
                return 0.0

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


#@SCENARIOS.register()
class KeyboardTeleoperationScenario_ForkliftV3(Scenario):
    """
    Teleop dedicado para ForkliftRobotV3 (rear-steer Ackermann-like).

    Mapeamento de botões:
      [0]=W, [1]=A, [2]=S, [3]=D, [4]=Space (freio), [5]=C (recentra)

    Publica ação: [v_mps, delta_rad]
      v_mps    = (W - S) * robot.keyboard_linear_velocity_gain
      delta_rad: integrado de (A - D) * steer_rate, limitado por
                 robot.effective_steer_limit (se existir) ou robot.max_steer_angle.
    """

    _default_steer_rate = 1.5      # rad/s
    _default_brake_acc = 5.0       # m/s^2
    _default_center_relax = 2.5    # rad/s
    _warmup_frames_total = 20

    def __init__(self, robot: Robot, occupancy_map: OccupancyMap):
        super().__init__(robot, occupancy_map)
        self.keyboard = inputs.Keyboard()
        self.pose_sampler = pose_samplers.UniformPoseSampler()

        self._lin_gain = float(getattr(self.robot, "keyboard_linear_velocity_gain", 1.0))
        self._steer_rate = float(getattr(self.robot, "keyboard_angular_velocity_gain", self._default_steer_rate))

        self._v_cmd = 0.0
        self._delta = 0.0
        self._warmup = 0

    def _delta_limit(self) -> float:
        # limite efetivo de direção vindo do robô, com fallbacks seguros
        return float(getattr(self.robot, "effective_steer_limit", getattr(self.robot, "max_steer_angle", 0.6)))

    def reset(self):
        # posiciona o robô usando o sampler de pose no mapa
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)

        # zera comando e publica uma vez para "acordar" drives
        self._v_cmd = 0.0
        self._delta = 0.0
        self.robot.action.set_value(np.array([0.0, 0.0], dtype=np.float32))
        try:
            dt = float(getattr(self.robot, "physics_dt", 0.005))
        except Exception:
            dt = 0.005
        self.robot.write_action(dt)

        self._warmup = self._warmup_frames_total

    def step(self, step_size: float) -> bool:
        self.update_state()

        delta_lim = self._delta_limit()

        buttons = self.keyboard.buttons.get_value()

        def btn(i: int) -> float:
            if buttons is None:
                return 0.0
            try:
                return float(buttons[i]) if i < len(buttons) else 0.0
            except Exception:
                return 0.0

        w = btn(0)
        a = btn(1)
        s = btn(2)
        d = btn(3)
        space = btn(4)
        cent = btn(5)

        # velocidade alvo sem freio
        v_target = (w - s) * self._lin_gain

        # freio
        if space > 0.5:
            dv = self._default_brake_acc * float(step_size)
            if abs(self._v_cmd) <= dv:
                self._v_cmd = 0.0
            else:
                self._v_cmd -= np.sign(self._v_cmd) * dv
        else:
            self._v_cmd = v_target

        # integra direção
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

        # warmup: primeiros frames após reset reforçam publicação
        if self._warmup > 0:
            self._warmup -= 1
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
    _warmup_frames_total = 20

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
        self._warmup = 0

    def reset(self):
        pose = self.pose_sampler.sample(self.buffered_occupancy_map)
        self.robot.set_pose_2d(pose)
        try:
            self.robot.post_reset()
        except Exception:
            pass
        self._v_cmd = 0.0
        self._delta = 0.0
        self.robot.action.set_value(np.array([0.0, 0.0], dtype=np.float32))
        try:
            dt = float(getattr(self.robot, "physics_dt", 0.005))
        except Exception:
            dt = 0.005
        self.robot.write_action(dt)
        self._warmup = self._warmup_frames_total

    def step(self, step_size: float) -> bool:
        self.update_state()

        buttons = self.keyboard.buttons.get_value()
        def btn(i: int) -> float:
            # defensive: buttons can be None or malformed; return 0.0 in that case
            if buttons is None:
                return 0.0
            try:
                return float(buttons[i]) if i < len(buttons) else 0.0
            except Exception:
                return 0.0

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
        if self._warmup > 0:
            self._warmup -= 1
            self.robot.write_action(step_size)

        self.update_state()
        return True

#@SCENARIOS.register()
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
    

#@SCENARIOS.register()
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
    

#@SCENARIOS.register()
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



#@SCENARIOS.register()
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
