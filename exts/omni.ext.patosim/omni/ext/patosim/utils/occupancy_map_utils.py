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


import enum
import numpy as np
import PIL.Image
import os

import tempfile
import omni.kit.usd
import typing as tp
from isaacsim.asset.gen.omap.bindings import _omap as _occupancy_map
from isaacsim.asset.gen.omap.utils import compute_coordinates,  update_location
from pxr import Sdf, UsdGeom, UsdPhysics, Usd, UsdShade, Kind

from ..types import Point2d
from omni.ext.patosim.utils.global_utils import (
    get_app,
    get_stage
)
from omni.ext.patosim.utils.prim_utils import prim_compute_bbox
from omni.ext.patosim.occupancy_map import (
    OccupancyMap, 
    OccupancyMapDataValue, 
    ROS_FREESPACE_THRESH_DEFAULT, 
    ROS_OCCUPIED_THRESH_DEFAULT,
    OCCUPANCY_MAP_DEFAULT_CELL_SIZE,
    OCCUPANCY_MAP_DEFAULT_Z_MAX,
    OCCUPANCY_MAP_DEFAULT_Z_MIN
)


class OccupancyMapGenerateRotation(enum.Enum):

    ROTATE_0 = 0
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3

    def degrees(self):
        if self == OccupancyMapGenerateRotation.ROTATE_0:
            return 0
        elif self == OccupancyMapGenerateRotation.ROTATE_90:
            return 90
        elif self == OccupancyMapGenerateRotation.ROTATE_180:
            return 180
        elif self == OccupancyMapGenerateRotation.ROTATE_270:
            return -90
        else:
            raise RuntimeError(f"Invalid rotation {self}.")



async def occupancy_map_generate_from_prim_async(
        prim_path: str,
        cell_size: float = OCCUPANCY_MAP_DEFAULT_CELL_SIZE,
        z_min: float = OCCUPANCY_MAP_DEFAULT_Z_MIN,
        z_max: float = OCCUPANCY_MAP_DEFAULT_Z_MAX,
        rotation: OccupancyMapGenerateRotation = OccupancyMapGenerateRotation.ROTATE_180,
        unknown_as_freespace: bool = True
    ) -> OccupancyMap:

    import warnings
    warnings.warn(
        f"[occupancy_map] Starting generation: prim_path={prim_path}, cell_size={cell_size}m, z_range=[{z_min}, {z_max}]m, rotation={rotation.name}",
        stacklevel=2
    )

    app = get_app()

    om = _occupancy_map.acquire_omap_interface()
    
    timeline = omni.timeline.get_timeline_interface()

    await app.next_update_async()
    
    stage = get_stage()
    stage_scale = UsdGeom.GetStageMetersPerUnit(stage)
    if stage_scale <= 0.0:
        raise RuntimeError(f"Invalid stage meters-per-unit value: {stage_scale!r}")
    units_per_meter = 1.0 / float(stage_scale)
    
    # Apply physics: only define the physics scene if it doesn't already exist
    try:
        existing = stage.GetPrimAtPath(Sdf.Path("/World/physicsScene"))
        if existing is None or not existing.IsValid():
            UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    except Exception:
        try:
            UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
        except Exception:
            # best-effort: if defining the physics scene fails, continue —
            # downstream code already guards for missing physics in many places
            pass
    
    await app.next_update_async()
    

    # Compute bounds for occupancy map calculation using specified prim
    prim_path = os.path.join(prim_path)
    prim_path_str = prim_path
    prim_path = stage.GetPrimAtPath(prim_path)
    
    import warnings
    warnings.warn(f"[occupancy_map] Attempting to compute bbox for prim '{prim_path_str}' (valid={prim_path is not None and prim_path.IsValid()})", stacklevel=2)

    lower_bound = upper_bound = midpoint = None
    for attempt in range(10):
        if prim_path is not None and prim_path.IsValid():
            try:
                lower_bound, upper_bound, midpoint = prim_compute_bbox(prim_path)
                size = np.linalg.norm(np.asarray(upper_bound) - np.asarray(lower_bound))
                warnings.warn(f"[occupancy_map] Attempt {attempt}: bbox size={size:.3f}m", stacklevel=2)
                if size > 1e-6:
                    warnings.warn(f"[occupancy_map] Success: valid bbox with size {size:.3f}m", stacklevel=2)
                    break
            except Exception as e:
                warnings.warn(f"[occupancy_map] Attempt {attempt}: bbox computation failed: {e}", stacklevel=2)
        await app.next_update_async()
    if lower_bound is None or upper_bound is None or midpoint is None:
        raise RuntimeError(
            f"Unable to compute a valid occupancy-map bbox for prim '{prim_path.GetPath() if prim_path and prim_path.IsValid() else prim_path_str}'. "
            "The scene may still be composing/loading, or the referenced USD may contain no visible geometry."
        )

    lower_bound = (
        lower_bound[0],
        lower_bound[1],
        z_min * units_per_meter,
    )
    upper_bound = (
        upper_bound[0],
        upper_bound[1],
        z_max * units_per_meter,
    )

    width = upper_bound[0] - lower_bound[0]
    height = upper_bound[1] - lower_bound[1]
    cell_size_stage = cell_size * units_per_meter
    origin = (lower_bound[0] - cell_size_stage, lower_bound[1] - cell_size_stage, 0)
    lower_bound = (0, 0, z_min * units_per_meter)
    upper_bound = (width + cell_size_stage, height + cell_size_stage, z_max * units_per_meter)
    
    import warnings
    warnings.warn(
        f"[occupancy_map] Bounds computed: width={width:.2f}m, height={height:.2f}m, z_range=[{lower_bound[2]:.2f}, {upper_bound[2]:.2f}]m, "
        f"origin={origin}, cell_size_stage={cell_size_stage}, units_per_meter={units_per_meter}",
        stacklevel=2
    )

    update_location(
        om,
        origin,
        lower_bound,
        upper_bound
    )
    
    await app.next_update_async()
    

    # Set cell size
    om.set_cell_size(cell_size_stage)
    
    import warnings
    warnings.warn(f"[occupancy_map] OMap interface configured with cell_size={cell_size_stage}", stacklevel=2)
    
    await app.next_update_async()
    

    # Generate occupancy map
    import warnings
    warnings.warn(f"[occupancy_map] Starting occupancy map generation: dims will be computed from bounds", stacklevel=2)
    timeline.stop()
    
    await app.next_update_async()
    
    timeline.play()
    
    await app.next_update_async()
    
    warnings.warn(f"[occupancy_map] Calling om.generate()...", stacklevel=2)
    om.generate()
    
    await app.next_update_async()
    
    timeline.stop()
    
    await app.next_update_async()
    
    warnings.warn(f"[occupancy_map] Generation complete, retrieving buffer...", stacklevel=2)

    # Format Image
    buffer = om.get_buffer()
    dims = om.get_dimensions()
    warnings.warn(f"[occupancy_map] Buffer shape: dims={dims}, buffer_size={len(buffer)}, total_cells={dims[0]*dims[1]}", stacklevel=2)
    buffer = np.array(buffer)
    buffer = np.reshape(buffer, (dims[1], dims[0]))
    occupied_mask = buffer == 1.0
    freespace_mask = buffer == 0.0
    unknown_mask = ~(occupied_mask | freespace_mask)
    warnings.warn(f"[occupancy_map] Mask counts: occupied={np.sum(occupied_mask)}, freespace={np.sum(freespace_mask)}, unknown={np.sum(unknown_mask)}", stacklevel=2)

    if unknown_as_freespace:
        freespace_mask[unknown_mask] = True
        unknown_mask = np.zeros_like(unknown_mask)

    import warnings
    occupied_pct = 100.0 * np.sum(buffer == 1.0) / buffer.size
    warnings.warn(
        f"[occupancy_map] Occupancy rate: {occupied_pct:.1f}% (occupied={np.sum(buffer == 1.0)}, freespace={np.sum(buffer == 0.0)}, unknown={np.sum(~(buffer == 1.0) & (buffer == 0.0))})",
        stacklevel=2
    )

    image = np.zeros(occupied_mask.shape, dtype=np.uint8)
    image[occupied_mask] = OccupancyMapDataValue.OCCUPIED.ros_image_value()
    image[unknown_mask] = OccupancyMapDataValue.UNKNOWN.ros_image_value()
    image[freespace_mask] = OccupancyMapDataValue.FREESPACE.ros_image_value()
    image = PIL.Image.fromarray(image)
    image = image.rotate(rotation.degrees())

    # Format Yaml
    if rotation == OccupancyMapGenerateRotation.ROTATE_0:
        top_left, top_right, bottom_left, bottom_right, image_coords = compute_coordinates(om, cell_size_stage)
    elif rotation == OccupancyMapGenerateRotation.ROTATE_270:  # -90 degrees
        top_right, bottom_right, top_left, bottom_left, image_coords = compute_coordinates(om, cell_size_stage)
    elif rotation == OccupancyMapGenerateRotation.ROTATE_90:  # 90 degrees
        bottom_left, top_left, bottom_right, top_right, image_coords = compute_coordinates(om, cell_size_stage)
    elif rotation == OccupancyMapGenerateRotation.ROTATE_180:  # 180 degrees
        bottom_right, bottom_left, top_right, top_left, image_coords = compute_coordinates(om, cell_size_stage)

    occupancy_map = OccupancyMap.from_ros_image(
        ros_image=image,
        resolution=cell_size,
        origin=[
            float(bottom_left[0] * stage_scale),
            float(bottom_left[1] * stage_scale),
            0.0
        ],
        negate=False,
        free_thresh=ROS_FREESPACE_THRESH_DEFAULT,
        occupied_thresh=ROS_OCCUPIED_THRESH_DEFAULT
    )
    
    import warnings
    warnings.warn(
        f"[occupancy_map] OccupancyMap created: resolution={cell_size}m, origin=({bottom_left[0]*stage_scale:.2f}, {bottom_left[1]*stage_scale:.2f}, 0), "
        f"image_size={image.size}, image_mode={image.mode}", 
        stacklevel=2
    )
    
    _occupancy_map.release_omap_interface(om)

    return occupancy_map


def occupancy_map_add_to_stage(
        occupancy_map: OccupancyMap,
        stage: Usd.Stage,
        path: str,
        z_offset: float = 0.0
    ) -> Usd.Prim:

    image_path = os.path.join(tempfile.mkdtemp(), "texture.png")
    image = occupancy_map.ros_image()

    # need to flip, ros uses inverted coordinates on y axis
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    image.save(image_path)

    x0, y0 = occupancy_map.top_left_pixel_world_coords()
    x1, y1 = occupancy_map.bottom_right_pixel_world_coords()

    # Add model
    modelRoot = UsdGeom.Xform.Define(stage, path)
    Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

    # Add mesh
    mesh = UsdGeom.Mesh.Define(stage, os.path.join(path, "mesh"))
    mesh.CreatePointsAttr([(x0, y0, z_offset), (x1, y0, z_offset), (x1, y1, z_offset), (x0, y1, z_offset)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.CreateExtentAttr([(x0, y0, z_offset), (x1, y1, z_offset)])

    texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.varying)
    
    texCoords.Set([(0, 0), (1, 0), (1,1), (0, 1)])

    # Add material
    material_path = os.path.join(path, "material")
    material = UsdShade.Material.Define(stage, material_path)
    pbrShader = UsdShade.Shader.Define(stage, os.path.join(material_path, "shader"))
    pbrShader.CreateIdAttr("UsdPreviewSurface")
    pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

    # Add texture to material
    stReader = UsdShade.Shader.Define(stage, os.path.join(material_path, "st_reader"))
    stReader.CreateIdAttr('UsdPrimvarReader_float2')
    diffuseTextureSampler = UsdShade.Shader.Define(stage, os.path.join(material_path, "diffuse_texture"))
    diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
    diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(image_path)
    diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
    diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

    stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
    stInput.Set('st')
    stReader.CreateInput('varname',Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
    mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)

    return modelRoot

