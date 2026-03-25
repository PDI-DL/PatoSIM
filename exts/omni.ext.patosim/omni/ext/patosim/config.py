import json
from typing import Literal
from dataclasses import dataclass, asdict


@dataclass
class Config:
    scenario_type: str
    robot_type: str
    scene_usd: str
    water_profile_path: str = ""
    waypoint_path: str = ""
    rov_linear_speed: float = 0.75
    rov_angular_speed: float = 0.9
    enable_dvl_debug_lines: bool = False
    enable_rov_front_camera: bool = True
    enable_rov_stereo_camera: bool = False
    enable_rov_sonar: bool = True
    enable_rov_dvl: bool = True
    enable_rov_barometer: bool = True

    def to_json(self):
        return json.dumps(asdict(self), indent=2)
    
    @staticmethod
    def from_json(data: str):
        return Config(**json.loads(data))
