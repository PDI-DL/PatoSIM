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


import omni.kit
import isaacsim.core
from pxr import Usd

from isaacsim.core.nodes.bindings import _isaacsim_core_nodes
from omni.kit.viewport.utility import get_active_viewport
from .stage_utils import stage_get_prim


def get_app():
    app = omni.kit.app.get_app()
    return app


def get_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()


def new_stage() -> Usd.Stage:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    return stage


def new_world(physics_dt: float = 0.01, stage_units_in_meters: float = 1.0) -> isaacsim.core.api.World:
    world = get_world()
    if world is not None:
        isaacsim.core.api.World.clear_instance()
    isaacsim.core.api.World(physics_dt=physics_dt, stage_units_in_meters=stage_units_in_meters)
    return isaacsim.core.api.World.instance()


def get_world() -> isaacsim.core.api.World:
    return isaacsim.core.api.World.instance()


def get_timestamp():
    return _isaacsim_core_nodes.acquire_interface().get_sim_time_monotonic()


def save_stage(path: str, default_prim: str | None = None):

    stage = get_stage()

    if default_prim is not None:
        stage.SetDefaultPrim(stage_get_prim(stage, default_prim))

    stage.Export(path)


def set_viewport_camera(path: str):

    viewport = get_active_viewport()
    viewport.camera_path = path
