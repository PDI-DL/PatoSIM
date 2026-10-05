# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Real unit tests against the production omni.ext.patosim.config.Config
dataclass, including the new fields added by this pass."""

from omni.ext.patosim.config import Config


def _minimal_config(**overrides):
    base = dict(scenario_type="OceanSimROVTeleoperationScenario", robot_type="OceanSimROVRobot", scene_usd="x.usd")
    base.update(overrides)
    return Config(**base)


def test_roundtrip_preserves_all_fields():
    cfg = _minimal_config(
        sonar_normalizing_method="all",
        lidar_range_profile="insight_pro",
        lidar_min_range=0.5,
        lidar_max_range=15.0,
        lidar_attenuation_coeff=0.12,
        rov_gamepad_linear_gain=1.5,
        occupancy_map_mode="3d_stack",
        occupancy_map_num_bands=5,
    )
    restored = Config.from_json(cfg.to_json())
    assert restored == cfg


def test_new_defaults_match_the_fix_intent():
    cfg = _minimal_config()
    # "raw" preserves true distance falloff -- the actual bug fix default.
    assert cfg.sonar_normalizing_method == "raw"
    # Lidar enabled by default, calibrated to a real subsea scanner envelope.
    assert cfg.enable_rov_lidar is True
    assert cfg.lidar_range_profile == "insight_micro"
    assert 0.0 < cfg.lidar_min_range < cfg.lidar_max_range
    # Occupancy map mode defaults to the pre-existing (non-breaking) behaviour.
    assert cfg.occupancy_map_mode == "2d_band"
    assert cfg.occupancy_map_num_bands == 1


def test_from_json_rejects_unknown_fields():
    import json

    cfg = _minimal_config()
    data = json.loads(cfg.to_json())
    data["totally_unknown_field"] = 123
    try:
        Config.from_json(json.dumps(data))
    except TypeError:
        pass
    else:
        raise AssertionError("Config.from_json should reject unknown fields (dataclass strictness)")
