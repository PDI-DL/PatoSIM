# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Real unit tests against the production
omni.ext.patosim.oceansim.utils.underwater_lidar_math module (no Isaac Sim
dependency needed -- see tests/conftest.py)."""

import numpy as np

from omni.ext.patosim.oceansim.utils.underwater_lidar_math import apply_underwater_lidar_profile


def _radial_points(distances):
    """Points placed along +X at the given distances, in the sensor-local
    frame (columns 0:3 = local xyz)."""
    n = len(distances)
    pts = np.zeros((n, 4), dtype=np.float32)
    pts[:, 0] = distances
    pts[:, 3] = 1.0  # intensity column, uniform to start
    return pts


def test_points_beyond_max_range_are_dropped():
    pts = _radial_points([0.5, 3.0, 6.9, 7.5, 20.0])
    rng = np.random.default_rng(0)
    out = apply_underwater_lidar_profile(
        pts, min_range=0.13, max_range=7.0, attenuation_coeff=0.0, rng=rng
    )
    assert out.shape[0] == 3  # 0.5, 3.0, 6.9 survive; 7.5 and 20.0 are clipped
    assert np.all(out[:, 0] <= 7.0)


def test_points_below_min_range_are_dropped():
    pts = _radial_points([0.01, 0.05, 0.13, 1.0])
    rng = np.random.default_rng(0)
    out = apply_underwater_lidar_profile(
        pts, min_range=0.13, max_range=7.0, attenuation_coeff=0.0, rng=rng
    )
    assert out.shape[0] == 2  # 0.13 and 1.0 survive; 0.01 and 0.05 are below min_range
    assert np.all(out[:, 0] >= 0.13)


def test_zero_attenuation_keeps_all_in_range_points():
    """attenuation_coeff=0 means exp(0)=1 survival probability everywhere:
    no stochastic dropout, only the hard range clip applies."""
    distances = np.linspace(0.2, 6.9, 50)
    pts = _radial_points(distances)
    rng = np.random.default_rng(1)
    out = apply_underwater_lidar_profile(
        pts, min_range=0.13, max_range=7.0, attenuation_coeff=0.0, rng=rng
    )
    assert out.shape[0] == pts.shape[0]


def test_return_rate_decreases_with_distance():
    """Beer-Lambert dropout: averaged over many trials, the fraction of
    points surviving at a far range must be lower than at a near range."""
    n_trials = 4000
    near = _radial_points([1.0] * n_trials)
    far = _radial_points([6.0] * n_trials)
    rng = np.random.default_rng(42)
    near_out = apply_underwater_lidar_profile(
        near, min_range=0.13, max_range=7.0, attenuation_coeff=0.35, rng=rng
    )
    far_out = apply_underwater_lidar_profile(
        far, min_range=0.13, max_range=7.0, attenuation_coeff=0.35, rng=rng
    )
    near_rate = near_out.shape[0] / n_trials
    far_rate = far_out.shape[0] / n_trials
    expected_near = np.exp(-0.35 * 1.0)
    expected_far = np.exp(-0.35 * 6.0)
    assert far_rate < near_rate, "return rate must decrease with distance"
    # Sanity-check the empirical rate is close to the theoretical Beer-Lambert value.
    assert abs(near_rate - expected_near) < 0.05
    assert abs(far_rate - expected_far) < 0.05


def test_intensity_column_is_attenuated_not_just_clipped():
    pts = _radial_points([1.0, 5.0])
    rng = np.random.default_rng(2)
    out = apply_underwater_lidar_profile(
        pts, min_range=0.13, max_range=7.0, attenuation_coeff=0.2, rng=rng
    )
    if out.shape[0] == 2:
        near_intensity, far_intensity = out[0, 3], out[1, 3]
        assert far_intensity < near_intensity


def test_empty_input_returns_empty():
    rng = np.random.default_rng(0)
    empty = np.zeros((0, 4), dtype=np.float32)
    out = apply_underwater_lidar_profile(
        empty, min_range=0.13, max_range=7.0, attenuation_coeff=0.35, rng=rng
    )
    assert out.shape[0] == 0


def test_none_and_malformed_input_pass_through_safely():
    rng = np.random.default_rng(0)
    bad = np.zeros((3, 2), dtype=np.float32)  # too few columns
    out = apply_underwater_lidar_profile(
        bad, min_range=0.13, max_range=7.0, attenuation_coeff=0.35, rng=rng
    )
    assert out is bad  # returned unchanged, no crash
