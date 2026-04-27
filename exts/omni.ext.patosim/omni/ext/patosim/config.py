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
    enable_rov_lidar: bool = False
    enable_rov_sonar: bool = True
    enable_rov_dvl: bool = True
    enable_rov_barometer: bool = True
    dataset_object_position: tuple = (0.0, 0.0, 0.0)
    rov_operating_depth: float = -2.0

    def to_json(self):
        return json.dumps(asdict(self), indent=2)
    
    @staticmethod
    def from_json(data: str):
        return Config(**json.loads(data))
