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


import os
import random
import math
import numpy as np
from typing import Tuple

import isaacsim.core.utils.numpy.rotations as rot_utils
import isaacsim.core.api.objects as objects

from omni.ext.mobility_gen.utils.global_utils import get_stage
from omni.ext.mobility_gen.utils.stage_utils import stage_add_dome_light
from omni.ext.mobility_gen.utils.registry import Registry


#=========================================================
#  BASE CLASSES
#=========================================================


class SceneBuilder:
    
    @classmethod
    def build(cls, prim_path: str):
        pass

SCENE_BUILDERS = Registry[SceneBuilder]()


class RandomCubeScene(SceneBuilder):

    num_cubes: int
    x_boundary: Tuple[float, float]
    y_boundary: Tuple[float, float]
    cube_size_range: Tuple[float, float]
    color_min: Tuple[float, float, float]
    color_max: Tuple[float, float, float]

    def __init__(self, 
            prim_path: str
        ):
        self.prim_path = prim_path

    @classmethod
    def build(cls, prim_path: str):

        stage = get_stage()

        stage_add_dome_light(
            stage,
            os.path.join(prim_path, "dome_light")
        )

        for i in range(cls.num_cubes):
            size = random.uniform(*cls.cube_size_range)
            x = random.uniform(*cls.x_boundary)
            y = random.uniform(*cls.y_boundary)
            color = [random.uniform(cls.color_min[i], cls.color_max[i]) for i in range(3)]
            theta = random.uniform(-math.pi, math.pi)

            objects.FixedCuboid(
                os.path.join(prim_path, "objects", f"cube_{i}"),
                size=size,
                color=np.array(color),
                position=np.array([x, y, size/2]),
                orientation=rot_utils.euler_angles_to_quats(np.array([0., 0., theta]))
            )


@SCENE_BUILDERS.register()
class RandomCubeSceneSmall(RandomCubeScene):
    num_cubes: int = 20
    x_boundary: Tuple[float, float] = (-2., 2.)
    y_boundary: Tuple[float, float] = (-2., 2.)
    cube_size_range: Tuple[float, float] = (0.05, .45)
    color_min: Tuple[float, float, float] = (0., 0., 0.)
    color_max: Tuple[float, float, float] = (1., 1., 1.)


@SCENE_BUILDERS.register()
class RandomCubeSceneLarge(RandomCubeScene):
    num_cubes: int = 20
    x_boundary: Tuple[float, float] = (-10., 10.)
    y_boundary: Tuple[float, float] = (-10., 10.)
    cube_size_range: Tuple[float, float] = (0.3, 2.0)
    color_min: Tuple[float, float, float] = (0., 0., 0.)
    color_max: Tuple[float, float, float] = (1., 1., 1.)

