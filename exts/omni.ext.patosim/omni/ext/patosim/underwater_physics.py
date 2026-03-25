from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from omni.ext.patosim.common import Buffer, Module


def _as_vec3(values: Sequence[float], default: Sequence[float]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float32).reshape(3)
    except Exception:
        arr = np.asarray(default, dtype=np.float32).reshape(3)
    return arr


def _safe_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 1.0e-6, norms, 1.0)
    return vectors / norms


@dataclass
class BlueROVHydrodynamicsConfig:
    mass_kg: float
    water_surface_z: float = 0.0
    fluid_density_kg_m3: float = 1025.0
    gravity_m_s2: float = 9.81
    displaced_volume_m3: Optional[float] = None
    buoyancy_factor: float = 1.02
    vehicle_height_m: float = 0.34
    center_of_buoyancy_body_m: Sequence[float] = (0.0, 0.0, 0.05)
    linear_drag_coeffs: Sequence[float] = (14.0, 18.0, 20.0)
    quadratic_drag_coeffs: Sequence[float] = (6.0, 8.0, 10.0)
    angular_drag_coeffs: Sequence[float] = (1.4, 1.4, 1.8)
    angular_quadratic_drag_coeffs: Sequence[float] = (0.2, 0.2, 0.3)
    thruster_max_force_newtons: float = 18.0
    thruster_time_constant_s: float = 0.12


class BlueROVUnderwaterPhysics(Module):
    """Simple underwater dynamics model for BlueROV-style 6DOF control."""

    def __init__(
        self,
        config: BlueROVHydrodynamicsConfig,
        thruster_positions_body_m: np.ndarray,
        thruster_directions_body: np.ndarray,
        thruster_names: Sequence[str],
    ):
        self.config = config
        self.thruster_positions_body_m = np.asarray(thruster_positions_body_m, dtype=np.float32).reshape(-1, 3)
        self.thruster_directions_body = _safe_normalize_rows(thruster_directions_body)
        self.thruster_names = list(thruster_names)
        self._thruster_count = int(self.thruster_positions_body_m.shape[0])

        self._center_of_buoyancy_body = _as_vec3(
            config.center_of_buoyancy_body_m,
            default=(0.0, 0.0, 0.05),
        )
        self._linear_drag = _as_vec3(config.linear_drag_coeffs, default=(14.0, 18.0, 20.0))
        self._quadratic_drag = _as_vec3(config.quadratic_drag_coeffs, default=(6.0, 8.0, 10.0))
        self._angular_drag = _as_vec3(config.angular_drag_coeffs, default=(1.4, 1.4, 1.8))
        self._angular_quadratic_drag = _as_vec3(
            config.angular_quadratic_drag_coeffs,
            default=(0.2, 0.2, 0.3),
        )
        self._thruster_max_force = float(max(1.0e-3, config.thruster_max_force_newtons))
        self._thruster_time_constant_s = float(max(1.0e-3, config.thruster_time_constant_s))
        self._displaced_volume_m3 = float(
            config.displaced_volume_m3
            if config.displaced_volume_m3 is not None
            else (config.mass_kg / max(config.fluid_density_kg_m3, 1.0e-6)) * float(config.buoyancy_factor)
        )

        self._allocation_matrix = self._build_allocation_matrix(
            self.thruster_positions_body_m,
            self.thruster_directions_body,
        )
        self._allocation_pinv = np.linalg.pinv(self._allocation_matrix, rcond=1.0e-4)
        self._thruster_forces = np.zeros(self._thruster_count, dtype=np.float32)

        self.desired_wrench_body = Buffer(np.zeros(6, dtype=np.float32))
        self.body_linear_velocity = Buffer(np.zeros(3, dtype=np.float32))
        self.body_angular_velocity = Buffer(np.zeros(3, dtype=np.float32))
        self.thruster_forces = Buffer(np.zeros(self._thruster_count, dtype=np.float32))
        self.thruster_commands = Buffer(np.zeros(self._thruster_count, dtype=np.float32))
        self.thrust_force_body = Buffer(np.zeros(3, dtype=np.float32))
        self.thrust_torque_body = Buffer(np.zeros(3, dtype=np.float32))
        self.drag_force_body = Buffer(np.zeros(3, dtype=np.float32))
        self.drag_torque_body = Buffer(np.zeros(3, dtype=np.float32))
        self.buoyancy_force_body = Buffer(np.zeros(3, dtype=np.float32))
        self.buoyancy_torque_body = Buffer(np.zeros(3, dtype=np.float32))
        self.net_force_body = Buffer(np.zeros(3, dtype=np.float32))
        self.net_torque_body = Buffer(np.zeros(3, dtype=np.float32))
        self.submerged_fraction = Buffer(0.0)

    @classmethod
    def create_default(cls, config: BlueROVHydrodynamicsConfig) -> "BlueROVUnderwaterPhysics":
        c = float(np.cos(np.deg2rad(45.0)))
        s = float(np.sin(np.deg2rad(45.0)))

        horizontal_positions = np.asarray(
            [
                [0.22, 0.16, 0.00],
                [0.22, -0.16, 0.00],
                [-0.22, 0.16, 0.00],
                [-0.22, -0.16, 0.00],
            ],
            dtype=np.float32,
        )
        horizontal_directions = np.asarray(
            [
                [c, s, 0.0],
                [c, -s, 0.0],
                [c, -s, 0.0],
                [c, s, 0.0],
            ],
            dtype=np.float32,
        )

        vertical_positions = np.asarray(
            [
                [0.16, 0.14, 0.08],
                [0.16, -0.14, 0.08],
                [-0.16, 0.14, 0.08],
                [-0.16, -0.14, 0.08],
            ],
            dtype=np.float32,
        )
        vertical_directions = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        thruster_positions = np.concatenate([horizontal_positions, vertical_positions], axis=0)
        thruster_directions = np.concatenate([horizontal_directions, vertical_directions], axis=0)
        thruster_names = (
            "h_front_left",
            "h_front_right",
            "h_rear_left",
            "h_rear_right",
            "v_front_left",
            "v_front_right",
            "v_rear_left",
            "v_rear_right",
        )
        return cls(config, thruster_positions, thruster_directions, thruster_names)

    @staticmethod
    def _build_allocation_matrix(positions_body_m: np.ndarray, directions_body: np.ndarray) -> np.ndarray:
        columns = []
        for position, direction in zip(positions_body_m, directions_body):
            moment = np.cross(position, direction)
            column = np.concatenate([direction, moment], axis=0)
            columns.append(column.astype(np.float32))
        return np.stack(columns, axis=1)

    def reset(self) -> None:
        self._thruster_forces = np.zeros(self._thruster_count, dtype=np.float32)
        self.thruster_forces.set_value(self._thruster_forces.copy())
        self.thruster_commands.set_value(np.zeros(self._thruster_count, dtype=np.float32))
        self.desired_wrench_body.set_value(np.zeros(6, dtype=np.float32))
        self.body_linear_velocity.set_value(np.zeros(3, dtype=np.float32))
        self.body_angular_velocity.set_value(np.zeros(3, dtype=np.float32))
        self.thrust_force_body.set_value(np.zeros(3, dtype=np.float32))
        self.thrust_torque_body.set_value(np.zeros(3, dtype=np.float32))
        self.drag_force_body.set_value(np.zeros(3, dtype=np.float32))
        self.drag_torque_body.set_value(np.zeros(3, dtype=np.float32))
        self.buoyancy_force_body.set_value(np.zeros(3, dtype=np.float32))
        self.buoyancy_torque_body.set_value(np.zeros(3, dtype=np.float32))
        self.net_force_body.set_value(np.zeros(3, dtype=np.float32))
        self.net_torque_body.set_value(np.zeros(3, dtype=np.float32))
        self.submerged_fraction.set_value(0.0)

    def _compute_submerged_fraction(self, world_z: float) -> float:
        half_height = max(1.0e-3, float(self.config.vehicle_height_m) * 0.5)
        z_top = world_z + half_height
        z_bottom = world_z - half_height
        water_surface_z = float(self.config.water_surface_z)

        if z_bottom >= water_surface_z:
            return 0.0
        if z_top <= water_surface_z:
            return 1.0
        return float(np.clip((water_surface_z - z_bottom) / max(2.0 * half_height, 1.0e-6), 0.0, 1.0))

    def _compute_buoyancy(self, world_z: float) -> tuple[np.ndarray, np.ndarray, float]:
        submerged_fraction = self._compute_submerged_fraction(world_z)
        buoyancy_magnitude = (
            float(self.config.fluid_density_kg_m3)
            * float(self.config.gravity_m_s2)
            * self._displaced_volume_m3
            * submerged_fraction
        )
        weight_magnitude = float(self.config.mass_kg) * float(self.config.gravity_m_s2)
        hydrostatic_force_z = buoyancy_magnitude - weight_magnitude
        force_body = np.array([0.0, 0.0, hydrostatic_force_z], dtype=np.float32)
        torque_body = np.cross(
            self._center_of_buoyancy_body,
            np.array([0.0, 0.0, buoyancy_magnitude], dtype=np.float32),
        ).astype(np.float32)
        return force_body, torque_body, submerged_fraction

    def _compute_drag(
        self,
        linear_velocity_body: np.ndarray,
        angular_velocity_body: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        linear_velocity_body = np.asarray(linear_velocity_body, dtype=np.float32).reshape(3)
        angular_velocity_body = np.asarray(angular_velocity_body, dtype=np.float32).reshape(3)

        drag_force = -(
            self._linear_drag * linear_velocity_body
            + self._quadratic_drag * np.abs(linear_velocity_body) * linear_velocity_body
        )
        drag_torque = -(
            self._angular_drag * angular_velocity_body
            + self._angular_quadratic_drag * np.abs(angular_velocity_body) * angular_velocity_body
        )
        return drag_force.astype(np.float32), drag_torque.astype(np.float32)

    def _mix_thrusters(self, desired_wrench_body: np.ndarray) -> np.ndarray:
        desired_wrench_body = np.asarray(desired_wrench_body, dtype=np.float32).reshape(6)
        thruster_force_targets = self._allocation_pinv @ desired_wrench_body
        return np.clip(
            thruster_force_targets,
            -self._thruster_max_force,
            self._thruster_max_force,
        ).astype(np.float32)

    def step(
        self,
        desired_wrench_body: np.ndarray,
        linear_velocity_body: np.ndarray,
        angular_velocity_body: np.ndarray,
        world_z: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        desired_wrench_body = np.asarray(desired_wrench_body, dtype=np.float32).reshape(6)
        linear_velocity_body = np.asarray(linear_velocity_body, dtype=np.float32).reshape(3)
        angular_velocity_body = np.asarray(angular_velocity_body, dtype=np.float32).reshape(3)
        dt = float(max(dt, 1.0e-4))

        desired_thruster_forces = self._mix_thrusters(desired_wrench_body)
        alpha = float(np.clip(dt / (self._thruster_time_constant_s + dt), 0.0, 1.0))
        self._thruster_forces = self._thruster_forces + alpha * (desired_thruster_forces - self._thruster_forces)

        thrust_wrench_body = (self._allocation_matrix @ self._thruster_forces).astype(np.float32)
        thrust_force_body = thrust_wrench_body[:3]
        thrust_torque_body = thrust_wrench_body[3:]

        buoyancy_force_body, buoyancy_torque_body, submerged_fraction = self._compute_buoyancy(float(world_z))
        drag_force_body, drag_torque_body = self._compute_drag(linear_velocity_body, angular_velocity_body)

        net_force_body = (thrust_force_body + buoyancy_force_body + drag_force_body).astype(np.float32)
        net_torque_body = (thrust_torque_body + buoyancy_torque_body + drag_torque_body).astype(np.float32)

        self.desired_wrench_body.set_value(desired_wrench_body.copy())
        self.body_linear_velocity.set_value(linear_velocity_body.copy())
        self.body_angular_velocity.set_value(angular_velocity_body.copy())
        self.thruster_forces.set_value(self._thruster_forces.copy())
        self.thruster_commands.set_value((self._thruster_forces / self._thruster_max_force).astype(np.float32))
        self.thrust_force_body.set_value(thrust_force_body.copy())
        self.thrust_torque_body.set_value(thrust_torque_body.copy())
        self.drag_force_body.set_value(drag_force_body.copy())
        self.drag_torque_body.set_value(drag_torque_body.copy())
        self.buoyancy_force_body.set_value(buoyancy_force_body.copy())
        self.buoyancy_torque_body.set_value(buoyancy_torque_body.copy())
        self.net_force_body.set_value(net_force_body.copy())
        self.net_torque_body.set_value(net_torque_body.copy())
        self.submerged_fraction.set_value(float(submerged_fraction))

        return net_force_body, net_torque_body
