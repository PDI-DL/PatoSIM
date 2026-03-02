#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

# Stable defaults for Kit startup:
# - Ignore persisted user layout/settings by default (can avoid native UI crashes).
# - Do not enable omni.isaac.examples unless explicitly requested.
SAFE_ARGS=()
if [[ "${MOBILITY_SAFE_START:-1}" == "1" ]]; then
    SAFE_ARGS+=(--reset-user)
fi
if [[ "${MOBILITY_CLEAR_CACHE:-0}" == "1" ]]; then
    SAFE_ARGS+=(--clear-cache)
fi
if [[ "${MOBILITY_CLEAR_DATA:-0}" == "1" ]]; then
    SAFE_ARGS+=(--clear-data)
fi

EXTRA_EXT_ARGS=()
if [[ "${MOBILITY_ENABLE_EXAMPLES:-0}" == "1" ]]; then
    EXTRA_EXT_ARGS+=(--enable omni.isaac.examples)
fi

./app/isaac-sim.sh \
    --ext-folder exts \
    --enable omni.ext.mobility_gen \
    "${SAFE_ARGS[@]}" \
    "${EXTRA_EXT_ARGS[@]}" \
    "$@"
