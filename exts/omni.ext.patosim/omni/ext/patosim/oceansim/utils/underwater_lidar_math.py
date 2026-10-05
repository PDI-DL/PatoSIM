# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure, dependency-free math for the underwater lidar attenuation model.

Deliberately has NO omni/carb/pxr/warp imports so it can be unit-tested
without an Isaac Sim runtime (see tests/unit/test_underwater_lidar_math.py).
sensors.py's Lidar class delegates its underwater post-processing here.
"""

import numpy as np


def apply_underwater_lidar_profile(
    points: np.ndarray,
    min_range: float,
    max_range: float,
    attenuation_coeff: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Clip a local-frame point cloud to [min_range, max_range] and apply a
    Beer-Lambert style return-probability dropout with distance.

    Args:
        points: Nx3 (or NxK, K>=3) array, columns 0:3 = local xyz. Column 3,
            if present, is treated as intensity/reflectance and is also
            attenuated.
        min_range/max_range: usable range envelope in meters (real subsea
            laser scanners are short-range compared to terrestrial lidar).
        attenuation_coeff: Beer-Lambert coefficient (1/m); higher = more
            turbid water, faster falloff.
        rng: numpy random Generator used for the stochastic dropout (passed
            in so behaviour is reproducible/testable).

    Returns:
        The filtered/attenuated points, same column layout as input.
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return pts

    dist = np.linalg.norm(pts[:, :3], axis=1)
    in_range = (dist >= min_range) & (dist <= max_range)
    if not np.any(in_range):
        return pts[in_range]

    survival_prob = np.exp(-attenuation_coeff * dist)
    keep = in_range & (rng.random(dist.shape[0]) < survival_prob)
    out = pts[keep]
    if out.shape[1] >= 4:
        out = out.copy()
        out[:, 3] = out[:, 3] * np.exp(-attenuation_coeff * dist[keep])
    return out
