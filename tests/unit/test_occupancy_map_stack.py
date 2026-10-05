# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Real unit tests against the production
omni.ext.patosim.occupancy_map.OccupancyMapStack (the new Fase 4a
multi-band occupancy layer)."""

import numpy as np

from omni.ext.patosim.occupancy_map import OccupancyMap, OccupancyMapStack


def _band(all_free: bool, resolution: float = 0.25, size: int = 20):
    freespace = np.full((size, size), all_free, dtype=bool)
    occupied = np.full((size, size), not all_free, dtype=bool)
    origin = (-(size * resolution) / 2.0, -(size * resolution) / 2.0, 0.0)
    return OccupancyMap.from_masks(freespace, occupied, resolution=resolution, origin=origin)


def test_band_index_for_z_picks_nearest_band():
    stack = OccupancyMapStack(
        bands=[_band(True), _band(True), _band(True)],
        z_centers=[-5.0, -2.0, 1.0],
        band_half_height=1.5,
    )
    assert stack.band_index_for_z(-5.2) == 0
    assert stack.band_index_for_z(-3.4) == 1  # closer to -2.0 than -5.0
    assert stack.band_index_for_z(1.4) == 2
    assert stack.band_index_for_z(-0.4) == 2  # |−0.4−(−2.0)|=1.6 vs |−0.4−1.0|=1.4 -> nearer to 1.0


def test_band_for_z_returns_the_matching_occupancy_map():
    free_band = _band(True)
    occupied_band = _band(False)
    stack = OccupancyMapStack(
        bands=[occupied_band, free_band], z_centers=[-5.0, -1.0], band_half_height=2.0
    )
    assert stack.band_for_z(-1.1) is free_band
    assert stack.band_for_z(-5.1) is occupied_band


def test_has_vertical_clearance_true_when_all_neighbor_bands_free():
    bands = [_band(True), _band(True), _band(True)]
    stack = OccupancyMapStack(bands=bands, z_centers=[-6.0, -3.0, 0.0], band_half_height=1.5)
    assert stack.has_vertical_clearance(0.0, 0.0, -3.0, num_neighbor_bands=1) is True


def test_has_vertical_clearance_false_when_a_neighbor_band_is_occupied():
    bands = [_band(False), _band(True), _band(True)]  # bottom band fully occupied
    stack = OccupancyMapStack(bands=bands, z_centers=[-6.0, -3.0, 0.0], band_half_height=1.5)
    # current band (-3.0) is free, but checking 1 neighbor band down hits the occupied band.
    assert stack.has_vertical_clearance(0.0, 0.0, -3.0, num_neighbor_bands=1) is False
    # checking only the current band (no neighbors) should be True.
    assert stack.has_vertical_clearance(0.0, 0.0, -3.0, num_neighbor_bands=0) is True


def test_save_and_load_roundtrip(tmp_path):
    bands = [_band(True), _band(False), _band(True)]
    stack = OccupancyMapStack(bands=bands, z_centers=[-6.0, -3.0, 0.0], band_half_height=1.5)
    out_dir = str(tmp_path / "occupancy_map_3d")
    stack.save(out_dir)

    loaded = OccupancyMapStack.load(out_dir)
    assert len(loaded.bands) == 3
    assert loaded.z_centers == [-6.0, -3.0, 0.0]
    assert abs(loaded.band_half_height - 1.5) < 1e-9
    # Round-tripped through ROS PNG/YAML: occupied/free masks must match.
    assert np.array_equal(loaded.bands[0].freespace_mask(), bands[0].freespace_mask())
    assert np.array_equal(loaded.bands[1].occupied_mask(), bands[1].occupied_mask())


def test_rejects_mismatched_lengths():
    try:
        OccupancyMapStack(bands=[_band(True)], z_centers=[-1.0, -2.0], band_half_height=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("OccupancyMapStack should reject mismatched bands/z_centers lengths")
