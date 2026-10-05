# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Real unit tests against the production omni.ext.patosim.unit_scale_math
module used by build.py's asset-scale fix."""

from omni.ext.patosim.unit_scale_math import unit_scale_ratio


def test_same_units_is_identity():
    assert unit_scale_ratio(1.0, 1.0) == 1.0
    assert unit_scale_ratio(0.01, 0.01) == 1.0


def test_centimeter_asset_into_meter_stage_is_scaled_down():
    """This is exactly the reported bug scenario: a scan/photogrammetry
    asset authored in centimeters (metersPerUnit=0.01) referenced into a
    meter-authored stage (metersPerUnit=1.0) must be scaled down by 0.01,
    or it renders 100x too large."""
    scale = unit_scale_ratio(source_meters_per_unit=0.01, target_meters_per_unit=1.0)
    assert abs(scale - 0.01) < 1e-9


def test_meter_asset_into_centimeter_stage_is_scaled_up():
    scale = unit_scale_ratio(source_meters_per_unit=1.0, target_meters_per_unit=0.01)
    assert abs(scale - 100.0) < 1e-6


def test_invalid_inputs_fall_back_to_identity():
    assert unit_scale_ratio(0.0, 1.0) == 1.0
    assert unit_scale_ratio(1.0, 0.0) == 1.0
    assert unit_scale_ratio(-1.0, 1.0) == 1.0
