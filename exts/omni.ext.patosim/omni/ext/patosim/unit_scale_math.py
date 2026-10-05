# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure, dependency-free stage-unit compensation math.

Deliberately has NO omni/pxr imports so it can be unit-tested without an
Isaac Sim runtime (see tests/unit/test_unit_scale_math.py). build.py's
_compute_asset_unit_scale reads the source/target metersPerUnit via pxr and
delegates the actual ratio computation here.
"""


def unit_scale_ratio(source_meters_per_unit: float, target_meters_per_unit: float) -> float:
    """Scale factor to apply to an asset referenced from a stage authored in
    `source_meters_per_unit` units into a stage authored in
    `target_meters_per_unit` units, so the asset keeps its real-world size.

    Example: an asset authored in centimeters (metersPerUnit=0.01)
    referenced into a stage authored in meters (metersPerUnit=1.0) needs
    scale = 0.01/1.0 = 0.01 applied, or it will appear 100x too large.
    """
    source_mpu = float(source_meters_per_unit)
    target_mpu = float(target_meters_per_unit)
    if source_mpu <= 0.0 or target_mpu <= 0.0:
        return 1.0
    return source_mpu / target_mpu
