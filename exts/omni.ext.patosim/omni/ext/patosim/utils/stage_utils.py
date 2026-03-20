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

from typing import Tuple # type: ignore
import os
import PIL.Image
import tempfile # type: ignore

from pxr import (
    Gf, Sdf, Usd, UsdGeom, UsdLux, 
    UsdShade, Kind, UsdPhysics, PhysxSchema
)


def stage_add_physics(
        stage: Usd.Stage, 
        path: str
    ):
    physics = UsdPhysics.Scene.Define(stage, path)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(path))
    # physx_scene.GetEnableCCDAttr().Set(True)
    # physx_scene.GetEnableGPUDynamicsAttr().Set(False)
    # physx_scene.GetBroadphaseTypeAttr().Set("GPU")
    return physics


def stage_add_dome_light(stage: Usd.Stage, path: str, intensity: float = 1000, 
        angle: float = 180, exposure: float=0.) -> UsdLux.DomeLight:
    """Adds a dome light to a USD stage.
    
    Args:
        stage (Usd.Stage): The USD stage to modify.
        path (str): The path to add the USD prim.
        intensity (float): The intensity of the dome light (default 1000).
        angle (float): The angle of the dome light (default 180)
        exposure (float): THe exposure of the dome light (default 0)

    Returns:
        UsdLux.DomeLight:  The created Dome light.
    """

    light = UsdLux.DomeLight.Define(stage, path)

    # intensity
    light.CreateIntensityAttr().Set(intensity)
    light.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    light.CreateExposureAttr().Set(exposure)

    # cone angle
    shaping = UsdLux.ShapingAPI(light)
    shaping.Apply(light.GetPrim())
    shaping.CreateShapingConeAngleAttr().Set(angle)
    shaping.CreateShapingConeSoftnessAttr()
    shaping.CreateShapingFocusAttr()
    shaping.CreateShapingFocusTintAttr()
    shaping.CreateShapingIesFileAttr()

    return light


def stage_add_usd_ref(stage: Usd.Stage, path: str, usd_path: str) -> Usd.Prim:
    """Adds an external USD reference to a USD stage.
    
    Args:
        stage (:class:`Usd.Stage`): The USD stage to modify.
        path (str): The path to add the USD reference.
        usd_path (str): The filepath or URL of the USD reference (ie: a Nucleus
            server URL).

    Returns:
        Usd.Prim: The created USD prim.
    """

    prim_ref = stage.DefinePrim(path)
    prim_ref.GetReferences().AddReference(usd_path)

    return stage_get_prim(stage, path)


def stage_get_prim(stage: Usd.Stage, path: str) -> Usd.Prim:
    """Returns a prim at the specified path in a USD stage.
    
    Args:
        stage (Usd.Stage): The USD stage to query.
        path (str): The path of the prim.

    Returns:
        Usd.Prim:  The USD prim at the specified path.
    """
    return stage.GetPrimAtPath(path)


def stage_add_cube(
        stage: Usd.Stage, 
        path: str, 
        size: float
    ):
    cubeGeom = UsdGeom.Cube.Define(stage, path)
    cubeGeom.CreateSizeAttr(size)
    return cubeGeom

def stage_add_camera(
        stage: Usd.Stage, 
        path: str,
        focal_length: float = 35,
        horizontal_aperature: float = 20.955,
        vertical_aperature: float = 20.955,
        clipping_range: Tuple[float, float] = (0.1, 100000)
    ) -> UsdGeom.Camera:
    """Adds a camera to a USD stage.
    

    Args:
        stage (Usd.Stage): The USD stage to modify.
        path (str): The path to add the USD prim.
        focal_length (float): The focal length of the camera (default 35).
        horizontal_aperature (float): The horizontal aperature of the camera
            (default 20.955).
        vertical_aperature (float): The vertical aperature of the camera
            (default 20.955).
        clipping_range (Tuple[float, float]): The clipping range of the camera.

    returns:
        UsdGeom.Camera:  The created USD camera.
    """

    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(focal_length)
    camera.CreateHorizontalApertureAttr().Set(horizontal_aperature)
    camera.CreateVerticalApertureAttr().Set(vertical_aperature)
    camera.CreateClippingRangeAttr().Set(clipping_range)

    return camera
