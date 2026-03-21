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


try:
    from .extension import *
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    # Allow utility imports (reader/writer/config/etc.) outside Kit UI context.
    if not (
        missing.startswith("omni")
        or missing.startswith("isaacsim")
        or missing.startswith("pxr")
        or missing.startswith("carb")
    ):
        raise
