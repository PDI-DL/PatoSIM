# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure, dependency-free joystick axis shaping math.

Deliberately has NO omni/carb imports so it can be unit-tested without an
Isaac Sim runtime (see tests/unit/test_gamepad_math.py). scenarios.py's
_ROVGamepadController delegates its axis shaping here.
"""


def apply_expo(value: float, expo: float) -> float:
    """Blend linear and cubic response: expo=0 is linear, expo=1 is full
    cubic. Gives finer control near center stick while keeping full range at
    the extremes -- standard RC/drone-style joystick shaping."""
    expo = max(0.0, min(1.0, float(expo)))
    return (1.0 - expo) * value + expo * (value ** 3)


def shape_axis(value: float, deadzone: float, expo: float) -> float:
    """Apply a deadzone (with continuous rescaling right past it, instead of
    a discontinuous jump from 0 to `deadzone`) followed by expo shaping."""
    deadzone = max(0.0, min(0.95, float(deadzone)))
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    scaled = (abs(value) - deadzone) / max(1e-6, 1.0 - deadzone)
    return sign * apply_expo(scaled, expo)
