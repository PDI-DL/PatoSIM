# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Pytest bootstrap for the dependency-free unit test suite.

These tests import real `omni.ext.patosim.*` modules WITHOUT an Isaac Sim
runtime. This works because `omni/ext/patosim/__init__.py` has a
PATOSIM_IMPORT_MODE=lite escape hatch that skips `from .extension import *`
(which is the only thing in the package that pulls in omni/isaacsim/pxr/carb
at import time). Only modules with zero Isaac dependencies at their own
top level are imported here (occupancy_map, config, gamepad_math,
unit_scale_math, underwater_lidar_math) -- anything that imports
omni.replicator/isaacsim.core/pxr/carb directly (extension.py, sensors.py,
robots.py, scenarios.py, build.py, the ImagingSonar* kernels) needs a real
Isaac Sim Python environment and is exercised by the manual/headless
integration checklist in docs/plano_upgrade_simulacao_subaquatica.md
instead, not by this suite.
"""

import os
import sys

os.environ.setdefault("PATOSIM_IMPORT_MODE", "lite")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXT_ROOT = os.path.join(_REPO_ROOT, "exts", "omni.ext.patosim")
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)
