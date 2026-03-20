# controllers/rear_drive_simple.py
import numpy as np
from isaacsim.core.utils.types import ArticulationAction

# ---------------- Controlador mínimo: ω=v/R nas duas traseiras; δ igual nas duas ----------------
class RearDriveSimpleController:
    def __init__(self, wheel_radius: float, max_steer_angle: float):
        self.R = float(max(wheel_radius, 1e-6))
        self.max_delta = float(max_steer_angle)

    def forward(self, command: np.ndarray):
        v_mps, delta = float(command[0]), float(command[1])
        omega = v_mps / self.R
        wheel_action = ArticulationAction(
            joint_velocities=np.array([omega, omega], dtype=np.float32)
        )
        steer_targets = np.clip(np.array([delta, delta], np.float32),
                                -self.max_delta, +self.max_delta)
        return wheel_action, steer_targets
        return wheel_action, steer_targets