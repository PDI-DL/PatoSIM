# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Real unit tests against the production omni.ext.patosim.gamepad_math
module used by OceanSimROVGamepadTeleoperationScenario."""

from omni.ext.patosim.gamepad_math import apply_expo, shape_axis


def test_deadzone_zeroes_small_values():
    assert shape_axis(0.05, deadzone=0.08, expo=0.3) == 0.0
    assert shape_axis(-0.08, deadzone=0.08, expo=0.3) == 0.0


def test_values_past_deadzone_are_nonzero_and_sign_preserving():
    pos = shape_axis(0.5, deadzone=0.08, expo=0.3)
    neg = shape_axis(-0.5, deadzone=0.08, expo=0.3)
    assert pos > 0.0
    assert neg < 0.0
    assert abs(pos) == abs(neg)  # symmetric response


def test_full_deflection_maps_to_full_output():
    assert abs(shape_axis(1.0, deadzone=0.08, expo=0.3) - 1.0) < 1e-6
    assert abs(shape_axis(-1.0, deadzone=0.08, expo=0.3) + 1.0) < 1e-6


def test_response_is_continuous_right_past_deadzone():
    """No jump discontinuity: value just past the deadzone boundary should
    be close to 0, not close to `deadzone`."""
    just_past = shape_axis(0.081, deadzone=0.08, expo=0.3)
    assert just_past < 0.05


def test_expo_zero_is_linear():
    assert apply_expo(0.5, expo=0.0) == 0.5
    assert apply_expo(-0.3, expo=0.0) == -0.3


def test_expo_one_is_cubic():
    assert abs(apply_expo(0.5, expo=1.0) - 0.5 ** 3) < 1e-9


def test_expo_blend_is_between_linear_and_cubic_for_fractional_stick():
    linear = 0.5
    cubic = 0.5 ** 3
    blended = apply_expo(0.5, expo=0.3)
    assert cubic < blended < linear


def test_axis_shaping_is_monotonic():
    """A physical requirement for predictable piloting: more stick
    deflection must never produce less thrust."""
    values = [shape_axis(v / 10.0, deadzone=0.08, expo=0.3) for v in range(0, 11)]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))
