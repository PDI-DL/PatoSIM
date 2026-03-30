# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility reader for the examples folder.

This module bootstraps the extension package path and re-exports the
extension Reader while keeping the older `read_state_dict()` flat output used
by examples 01-03.
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    ext_pkg_root = repo_root / "exts" / "omni.ext.patosim"
    ext_omni_ext_root = ext_pkg_root / "omni" / "ext"
    repo_root_str = str(repo_root)

    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    ext_pkg_root_str = str(ext_pkg_root)
    if ext_pkg_root.exists() and ext_pkg_root_str not in sys.path:
        sys.path.insert(0, ext_pkg_root_str)

    try:
        import omni.ext as omni_ext  # type: ignore

        ext_omni_ext_root_str = str(ext_omni_ext_root)
        if ext_omni_ext_root.exists() and ext_omni_ext_root_str not in omni_ext.__path__:
            omni_ext.__path__.append(ext_omni_ext_root_str)
    except Exception:
        pass

    return repo_root


bootstrap_repo_paths()

from omni.ext.patosim.reader import Reader as _Reader  # noqa: E402


class Reader(_Reader):
    """Backward-compatible facade for the legacy example scripts."""

    def read_state_dict(self, index: int):
        return self.read_state_dict_flat(index)
