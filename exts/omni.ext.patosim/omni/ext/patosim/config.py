import json
from typing import Literal
from dataclasses import dataclass, asdict


@dataclass
class Config:
    scenario_type: str
    robot_type: str
    scene_usd: str
    dataset_object_enabled: bool = False
    dataset_object_usd: str = ""
    dataset_object_reflectivity: float = 1.5
    water_profile_path: str = ""
    waypoint_path: str = ""
    apply_sonar_reflectivity_to_world: bool = True
    rov_linear_speed: float = 0.75
    rov_angular_speed: float = 0.9
    enable_dvl_debug_lines: bool = False
    enable_rov_front_camera: bool = True
    enable_rov_stereo_camera: bool = False
    enable_rov_lidar: bool = True
    enable_rov_sonar: bool = True
    enable_rov_dvl: bool = True
    enable_rov_barometer: bool = True
    dataset_object_position: tuple = (0.0, 0.0, 0.0)
    dataset_object_scale: float = 1.0
    dataset_object_rotation_euler_deg: tuple = (0.0, 0.0, 0.0)
    rov_operating_depth: float = -2.0
    occupancy_map_z_half: float = 3.0
    sonar_save_raw_npy: bool = True
    sonar_save_png16: bool = False
    sonar_save_polar_png: bool = False
    sonar_save_jpeg: bool = True

    # "raw" preserves true exponential distance falloff (best for dataset
    # generation); "range"/"all" are visualization-oriented presets that trade
    # distance information for contrast. See docs/plano_upgrade_simulacao_subaquatica.md §2.2.
    sonar_normalizing_method: str = "raw"

    # Underwater lidar profile, calibrated against real subsea laser scanners
    # (Voyis Insight family) rather than a generic terrestrial lidar. See
    # docs/plano_upgrade_simulacao_subaquatica.md §4.
    lidar_range_profile: str = "insight_micro"
    lidar_min_range: float = 0.13
    lidar_max_range: float = 7.0
    # Beer-Lambert style attenuation coefficient (1/m) for the lidar's
    # wavelength through water; higher = more turbid. Independent of which
    # hardware range profile is selected.
    lidar_attenuation_coeff: float = 0.35

    # ROV gamepad/joystick teleoperation tuning.
    rov_gamepad_linear_gain: float = 1.0
    rov_gamepad_vertical_gain: float = 1.0
    rov_gamepad_angular_gain: float = 1.0
    rov_gamepad_deadzone: float = 0.08
    rov_gamepad_expo: float = 0.3
    # Highest-leverage underwater_physics.py "feel" parameter, exposed here so
    # it can be tuned per-dataset without editing robots.py. Other drag/
    # buoyancy coefficients remain robots.py class attrs (already centralized
    # there) -- see docs/plano_upgrade_simulacao_subaquatica.md §Fase 3.
    rov_thruster_max_force_newtons: float = 40.0

    # Occupancy map mode: "2d_band" (single depth band, legacy behaviour) or
    # "3d_stack" (multiple depth bands so navigation can reason about
    # obstacles above/below the ROV's current depth).
    occupancy_map_mode: str = "2d_band"
    occupancy_map_num_bands: int = 1

    # Fields typed as tuple: JSON has no tuple type, so a naive round-trip
    # through json.dumps/json.loads silently turns them into lists, breaking
    # equality/type checks against a freshly-constructed Config. from_json
    # restores them explicitly.
    _TUPLE_FIELDS = ("dataset_object_position", "dataset_object_rotation_euler_deg")

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(data: str):
        raw = json.loads(data)
        for field in Config._TUPLE_FIELDS:
            if field in raw and raw[field] is not None:
                raw[field] = tuple(raw[field])
        return Config(**raw)
