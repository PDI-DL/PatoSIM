# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the sonar distance-falloff bug fix.

IMPORTANT LIMITATION: the real sonar normalization runs as NVIDIA Warp GPU
kernels in oceansim/utils/ImagingSonar_kernels.py, which requires an Isaac
Sim / CUDA runtime not available in this plain-Python test environment
(`import warp` fails here -- confirmed: no GPU/Isaac runtime in this
sandbox). This test is therefore a **numpy mirror** of the exact same
per-element arithmetic as the three kernels (make_sonar_map_raw/_range/_all),
copied by hand from ImagingSonarSensor.py / ImagingSonar_kernels.py as of
this fix. It validates the ALGORITHM's behavior, not the actual GPU
execution -- an Isaac Sim headless run (see
docs/plano_upgrade_simulacao_subaquatica.md) is still needed to confirm the
real kernel dispatch produces the same result end-to-end.
"""

import numpy as np


def _make_sonar_map_raw(intensity, offset=0.0, gain=1.0):
    """Mirrors make_sonar_map_raw: no normalization by any maximum."""
    out = intensity.copy()
    out = out + offset
    out = out * gain
    return np.clip(out, 0.0, 1.0)


def _make_sonar_map_range(intensity, offset=0.0, gain=1.0):
    """Mirrors make_sonar_map_range: each range ring (row) divided by its
    own maximum."""
    out = intensity.copy()
    row_max = out.max(axis=1, keepdims=True)
    nonzero = row_max[:, 0] != 0
    out[nonzero] = out[nonzero] / row_max[nonzero]
    out = out + offset
    out = out * gain
    return np.clip(out, 0.0, 1.0)


def _make_sonar_map_all(intensity, offset=0.0, gain=1.0):
    """Mirrors make_sonar_map_all: whole frame divided by its global max."""
    out = intensity.copy()
    global_max = out.max()
    if global_max != 0:
        out = out / global_max
    out = out + offset
    out = out * gain
    return np.clip(out, 0.0, 1.0)


def _synthetic_intensity_grid(n_ranges=20, n_azimuths=8, attenuation=0.3, max_range=10.0):
    """A grid of range rings each containing one identical reflective target
    (same reflectivity/cos_theta), only the range attenuation differs by
    row -- exactly `compute_intensity`'s exp(-attenuation*dist) term, summed
    per bin (binning_method="sum" is the project default)."""
    ranges = np.linspace(0.5, max_range, n_ranges)
    reflectivity = 1.0
    per_point_intensity = reflectivity * np.exp(-attenuation * ranges)
    # binning_method="sum": assume the same number of points landed in every
    # range ring's bins so only the attenuation term varies row to row.
    n_points_per_bin = 3.0
    grid = np.tile((per_point_intensity * n_points_per_bin)[:, None], (1, n_azimuths))
    return ranges, grid


def test_raw_mode_preserves_monotonic_distance_decay():
    """This is the core regression test for the bug: with normalizing_method
    unset/at its old default ("range"), a target's peak brightness no longer
    decreases with distance. "raw" must preserve it."""
    ranges, grid = _synthetic_intensity_grid()
    raw = _make_sonar_map_raw(grid)
    peak_per_range = raw.max(axis=1)
    # Strictly decreasing (allow equal only at numerical precision) as range increases.
    assert np.all(np.diff(peak_per_range) <= 1e-9), (
        "raw mode must preserve monotonic decay of intensity with distance"
    )
    # And it must actually vary substantially across the range span (not
    # flattened by clamping) -- this is the actual user complaint being fixed.
    assert peak_per_range[0] - peak_per_range[-1] > 0.05, (
        "raw mode intensity range-span collapsed -- distance information lost"
    )


def test_range_mode_flattens_distance_information():
    """Documents the OLD (pre-fix default) behavior that caused the bug:
    per-range-ring normalization makes every ring's peak ~1.0 regardless of
    true distance, which is exactly why it must not be the default for
    dataset generation."""
    ranges, grid = _synthetic_intensity_grid()
    range_normalized = _make_sonar_map_range(grid)
    peak_per_range = range_normalized.max(axis=1)
    # Every non-empty ring gets normalized to (close to) its own max -> ~1.0
    # everywhere, i.e. near-zero spread across ranges.
    assert peak_per_range.max() - peak_per_range.min() < 1e-6, (
        "range mode should flatten distance-dependent peak brightness (this is the bug being fixed)"
    )


def test_all_mode_preserves_relative_distance_ordering_but_saturates_near():
    """'all' mode (global max) preserves ordering (still useful as an
    alternative preset) but saturates the closest ring to 1.0, unlike raw."""
    ranges, grid = _synthetic_intensity_grid()
    all_normalized = _make_sonar_map_all(grid)
    peak_per_range = all_normalized.max(axis=1)
    assert np.all(np.diff(peak_per_range) <= 1e-9)
    assert peak_per_range[0] == 1.0  # nearest/strongest ring saturates to the global max


def test_raw_mode_is_new_sensor_default():
    """Confirms sensors.py's OceanSimImagingSonar defaults to normalizing_method='raw'."""
    src_path = __file__.replace(
        "tests/unit/test_sonar_normalization_math.py",
        "exts/omni.ext.patosim/omni/ext/patosim/sensors.py",
    )
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert 'self._normalizing_method = "raw"' in src, (
        "OceanSimImagingSonar's default normalizing_method regressed away from 'raw'"
    )
