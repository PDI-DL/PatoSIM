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
import math
from typing import Tuple, Optional, List
from pathlib import Path

import numpy as np
import omni.usd
import omni.replicator.core as rep
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from pxr import UsdGeom, Gf
from isaacsim.core.prims import SingleXFormPrim as XFormPrim


from omni.ext.patosim.utils.global_utils import get_stage
from omni.ext.patosim.utils.stage_utils import stage_add_usd_ref
from omni.ext.patosim.common import Module, Buffer


def _mobilitygen_repo_root() -> Path:
    # sensors.py -> mobility_gen -> ext -> omni -> omni.ext.patosim -> exts -> PatoSim
    return Path(__file__).resolve().parents[5]


def _resolve_asset_path(*candidates: str) -> str:
    """Return the first existing asset path from candidates.

    Candidates can be:
      - absolute paths
      - workspace-relative paths (e.g., robot_assets/...)
      - URLs (kept as-is when no local file exists)
    """
    repo_root = _mobilitygen_repo_root()
    first_non_empty = ""
    for cand in candidates:
        text = str(cand or "").strip()
        if not text:
            continue
        if not first_non_empty:
            first_non_empty = text
        if "://" in text:
            continue
        path = Path(text)
        if not path.is_absolute():
            path = repo_root / text
        if path.exists():
            return str(path)
    return first_non_empty


def _structured_pointcloud_to_array(array: np.ndarray) -> Optional[np.ndarray]:
    # Some Isaac/Replicator backends return structured arrays with named fields
    # instead of a plain Nx3/Nx4 matrix. Normalize them here so downstream code
    # only needs to reason about contiguous XYZ[+intensity] arrays.
    if array is None:
        return None
    try:
        if getattr(array.dtype, "names", None):
            names = list(array.dtype.names or [])
            lower_map = {str(name).lower(): name for name in names}
            if not {"x", "y", "z"}.issubset(lower_map.keys()):
                return None
            cols = [
                np.asarray(array[lower_map["x"]], dtype=np.float32).reshape(-1, 1),
                np.asarray(array[lower_map["y"]], dtype=np.float32).reshape(-1, 1),
                np.asarray(array[lower_map["z"]], dtype=np.float32).reshape(-1, 1),
            ]
            if "intensity" in lower_map:
                cols.append(np.asarray(array[lower_map["intensity"]], dtype=np.float32).reshape(-1, 1))
            return np.concatenate(cols, axis=1)
    except Exception:
        return None
    return None


def _extract_pointcloud_array(payload) -> Optional[np.ndarray]:
    if payload is None:
        return None

    if isinstance(payload, dict):
        # Different annotators/backends wrap the actual pointcloud payload under
        # different keys. Search the common variants first before giving up.
        for key in (
            "point_cloud_data",
            "point_cloud",
            "data",
            "points",
            "dataCpu",
            "dataPtr",
        ):
            if key in payload:
                extracted = _extract_pointcloud_array(payload[key])
                if extracted is not None:
                    return extracted
        return None

    try:
        arr = np.asarray(payload)
    except Exception:
        return None

    structured = _structured_pointcloud_to_array(arr)
    if structured is not None:
        return structured

    if arr.ndim == 1 and arr.size >= 3:
        if arr.size % 4 == 0:
            arr = arr.reshape(-1, 4)
        elif arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        else:
            return None

    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
        return None

    return np.asarray(arr[:, : min(arr.shape[1], 4)], dtype=np.float32)


def _sanitize_pointcloud_array(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    # Keep only finite XYZ rows and force a contiguous float32 layout. This makes
    # writer/replay/viewer code much simpler and avoids backend-specific dtypes.
    if array is None:
        return None
    try:
        pts = np.asarray(array, dtype=np.float32)
    except Exception:
        return None
    if pts.ndim != 2 or pts.shape[0] <= 0 or pts.shape[1] < 3:
        return None
    finite_mask = np.all(np.isfinite(pts[:, :3]), axis=1)
    if not np.any(finite_mask):
        return None
    pts = pts[finite_mask]
    if pts.shape[0] <= 0:
        return None
    return np.ascontiguousarray(pts)


def _build_sonar_acoustic_lut() -> np.ndarray:
    anchors = np.asarray([0, 76, 178, 229, 255], dtype=np.float32)
    colors = np.asarray(
        [
            [0, 8, 24],
            [0, 64, 144],
            [0, 220, 170],
            [255, 220, 0],
            [255, 255, 255],
        ],
        dtype=np.float32,
    )
    lut = np.empty((256, 3), dtype=np.uint8)
    xs = np.arange(256, dtype=np.float32)
    for channel in range(3):
        lut[:, channel] = np.clip(
            np.interp(xs, anchors, colors[:, channel]),
            0,
            255,
        ).astype(np.uint8)
    return lut


_SONAR_ACOUSTIC_LUT = _build_sonar_acoustic_lut()


def _apply_sonar_acoustic_colormap(gray: np.ndarray) -> np.ndarray:
    gray_u8 = np.asarray(gray, dtype=np.uint8)
    return _SONAR_ACOUSTIC_LUT[gray_u8]


def _blend_overlay_rgba(base_rgba: np.ndarray, overlay_rgb: np.ndarray, overlay_alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(overlay_alpha, dtype=np.float32), 0.0, 255.0) / 255.0
    if not np.any(alpha > 0.0):
        return base_rgba
    base_rgb = np.asarray(base_rgba[..., :3], dtype=np.float32)
    overlay_rgb_f = np.asarray(overlay_rgb, dtype=np.float32)
    blended = base_rgb * (1.0 - alpha[..., None]) + overlay_rgb_f * alpha[..., None]
    out = np.array(base_rgba, copy=True)
    out[..., :3] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return out


def _normalize_asset_path(asset_path: str) -> str:
    text = str(asset_path or "").strip()
    if "://" in text:
        return text
    return os.path.normpath(text)


def _to_numpy_array(data) -> Optional[np.ndarray]:
    if data is None:
        return None
    if hasattr(data, "numpy"):
        try:
            return np.asarray(data.numpy())
        except Exception:
            pass
    try:
        return np.asarray(data)
    except Exception:
        return None


def _list_authored_reference_assets(
    prim,
    layer_ids: Optional[set[str]] = None,
) -> List[str]:
    assets: List[str] = []
    if prim is None or not prim.IsValid():
        return assets
    try:
        for spec in prim.GetPrimStack():
            try:
                layer_id = str(spec.layer.identifier)
            except Exception:
                layer_id = ""
            if layer_ids is not None and layer_id not in layer_ids:
                continue
            ref_list = getattr(spec, "referenceList", None)
            if ref_list is None:
                continue
            for field_name in ("prependedItems", "explicitItems", "addedItems", "appendedItems"):
                try:
                    items = list(getattr(ref_list, field_name, []) or [])
                except Exception:
                    items = []
                for item in items:
                    try:
                        asset = getattr(item, "assetPath", None)
                        if asset:
                            assets.append(str(asset))
                    except Exception:
                        pass
    except Exception:
        pass
    return assets


def ensure_single_usd_reference(stage, path: str, usd_path: str, context: str = "sensor"):
    """Add a USD reference, but fail if the prim already references another asset.

    This prevents silent USD overlap/stacking at the same mount slot.
    """
    prim = stage.DefinePrim(path)
    target = _normalize_asset_path(usd_path)

    # Consider only references authored on the current stage layers.
    layer_ids = set()
    try:
        root_layer = stage.GetRootLayer()
        if root_layer is not None:
            layer_ids.add(str(root_layer.identifier))
    except Exception:
        pass
    try:
        session_layer = stage.GetSessionLayer()
        if session_layer is not None:
            layer_ids.add(str(session_layer.identifier))
    except Exception:
        pass
    if len(layer_ids) == 0:
        layer_ids = None

    existing = _list_authored_reference_assets(prim, layer_ids=layer_ids)
    existing_norm = sorted({ _normalize_asset_path(ref) for ref in existing if str(ref).strip() != "" })

    if len(existing_norm) > 1:
        raise RuntimeError(
            f"[{context}] USD overlap detected at '{path}'. Existing references={existing_norm}"
        )
    if len(existing_norm) == 1 and target not in existing_norm:
        raise RuntimeError(
            f"[{context}] USD overlap detected at '{path}'. "
            f"Existing reference='{existing_norm[0]}', requested='{target}'."
        )

    # No reference yet -> add it. If same reference already exists, keep as-is.
    if len(existing_norm) == 0:
        prim.GetReferences().AddReference(usd_path)
        after = _list_authored_reference_assets(prim, layer_ids=layer_ids)
        after_norm = sorted({ _normalize_asset_path(ref) for ref in after if str(ref).strip() != "" })
        if len(after_norm) > 1 or (len(after_norm) == 1 and target not in after_norm):
            raise RuntimeError(
                f"[{context}] USD overlap detected after add at '{path}'. References={after_norm}"
            )

    return prim


class Sensor(Module):

    def build(self, prim_path: str):
        raise NotImplementedError
    
    def attach(self, prim_path: str):
        raise NotImplementedError

class Camera(Sensor):

    def __init__(self,
            prim_path: str,
            resolution: Tuple[int, int]
        ):

        self._prim_path = prim_path
        self._resolution = resolution
        self._render_product = None
        self._rgb_annotator = None
        self._segmentation_annotator = None
        self._instance_id_segmentation_annotator = None
        self._normals_annotator = None
        self._depth_annotator = None
        self._xform_prim = XFormPrim(self._prim_path)

        # Buffers are tagged so Module.state_dict_* can selectively collect only
        # the relevant sensor products for recording/replay.
        self.rgb_image = Buffer(tags=["rgb"])
        self.segmentation_image = Buffer(tags=["segmentation"])
        self.segmentation_info = Buffer()
        self.depth_image = Buffer(tags=["depth"])
        self.instance_id_segmentation_image = Buffer(tags=["segmentation"])
        self.instance_id_segmentation_info = Buffer()
        self.normals_image = Buffer(tags=['normals'])
        self.position = Buffer()
        self.orientation = Buffer()

    def enable_rendering(self):
        
        self._render_product = rep.create.render_product(
            self._prim_path,
            self._resolution,
            force_new=False
        )

    def disable_rendering(self):
        if self._render_product is None:
            return
        
        if self._rgb_annotator is not None:
            self._rgb_annotator.detach()
            self._rgb_annotator = None
        
        if self._segmentation_annotator is not None:
            self._segmentation_annotator.detach()
            self._segmentation_annotator = None

        if self._depth_annotator is not None:
            self._depth_annotator.detach()
            self._depth_annotator = None

        self._render_product.destroy()
        self._render_product = None
    
    def enable_rgb_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._rgb_annotator is not None:
            return
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self._rgb_annotator.attach(self._render_product)

    def enable_segmentation_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._segmentation_annotator is not None:
            return
        self._segmentation_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params=dict(colorize=False)
        )
        self._segmentation_annotator.attach(self._render_product)

    def enable_instance_id_segmentation_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._instance_id_segmentation_annotator is not None:
            return
        self._instance_id_segmentation_annotator = rep.AnnotatorRegistry.get_annotator(
            "instance_id_segmentation", init_params=dict(colorize=False)
        )
        self._instance_id_segmentation_annotator.attach(self._render_product)

    def enable_depth_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._depth_annotator is not None:
            return
        # Isaac returns metric distance-to-camera as a dense 2D image. If we ever
        # want RGB-D pointclouds, they should be reconstructed from this buffer.
        self._depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_camera"
        )
        self._depth_annotator.attach(self._render_product)

    def enable_normals_rendering(self):
        if self._render_product is None:
            self.enable_rendering()
        if self._normals_annotator is not None:
            return
        self._normals_annotator = rep.AnnotatorRegistry.get_annotator(
            "normals"
        )
        self._normals_annotator.attach(self._render_product)

    def update_state(self):
        if self._rgb_annotator is not None:
            self.rgb_image.set_value(
                self._rgb_annotator.get_data()[:, :, :3]
            )
        if self._segmentation_annotator is not None:
            data = self._segmentation_annotator.get_data()
            seg_image = data['data']
            seg_info = data['info']
            self.segmentation_image.set_value(seg_image)
            self.segmentation_info.set_value(seg_info)

        if self._depth_annotator is not None:
            self.depth_image.set_value(
                self._depth_annotator.get_data()
            )

        if self._instance_id_segmentation_annotator is not None:
            data = self._instance_id_segmentation_annotator.get_data()
            id_seg_image = data['data']
            id_seg_info = data['info']
            self.instance_id_segmentation_image.set_value(id_seg_image)
            self.instance_id_segmentation_info.set_value(id_seg_info)

        if self._normals_annotator is not None:
            data = self._normals_annotator.get_data()
            self.normals_image.set_value(data)
            
        position, orientation = self._xform_prim.get_world_pose()
        self.position.set_value(position)
        self.orientation.set_value(orientation)
        
        super().update_state()


# =========================================================
# LIDAR / POINTCLOUD SENSOR (skeleton)
#
# This is a small, generic Lidar sensor wrapper showing how to expose
# a point-cloud buffer to the Module/Buffer system. The implementation
# of how to obtain a point-cloud depends on which Isaac/Replicator
# or RTX Lidar API you use. The key parts are:
#  - create a Buffer(tags=["pointcloud"]) on the sensor
#  - in update_state() set the buffer value to a numpy Nx3 (or Nx4)
#    array containing points in the robot/world frame
#  - the Module.state_dict_pointcloud() helper (added to common.py)
#    collects these buffers automatically and Writer/Reader will
#    persist them.
# =========================================================


class Lidar(Sensor):
    """Generic Lidar wrapper that exposes a `pointcloud` buffer.

    Usage notes:
    - If you have a replicator annotator that produces point-clouds,
      attach it in `enable_lidar()` and call `get_data()` in
      `update_state()` to populate `self.pointcloud`.
    - If you use an RTX Lidar node or another Isaac sensor API, call
      that API in `update_state()` and set the buffer value to a
      numpy array of shape (N, 3) or (N, 4).
    """

    # Local Isaac Sim 5.1 asset used as visual 3D model for the lidar.
    usd_url: str = _resolve_asset_path(
        "robot_assets/Slamtec/RPLidar_S2e.usd",
        "robot_assets/Slamtec/RPLIDAR_S2E/Slamtec_RPLIDAR_S2E.usd",
        "/home/pdi_4/Documents/Documentos/bevlog-isaac/isaac-assets/"
        "isaac-sim-assets-robots_and_sensors-5.1.0/Assets/Isaac/5.1/"
        "Isaac/Sensors/Slamtec/RPLidar_S2e.usd",
    )
    visual_model_subpath: str = "model_3d"
    sensor_prim_subpath: str = "rtx_sensor"

    def __init__(self, prim_path: str, visual_model_path: Optional[str] = None):
        self._prim_path = prim_path
        self._sensor_prim_path = os.path.join(prim_path, self.sensor_prim_subpath)
        self._visual_model_path = visual_model_path or os.path.join(
            prim_path, self.visual_model_subpath
        )
        self._xform_prim = XFormPrim(self._prim_path)
        self.pointcloud = Buffer(tags=["pointcloud"])  # the buffer consumers will look for
        # expose position/orientation so we can persist sensor pose if needed
        self.position = Buffer()
        self.orientation = Buffer()
        self.status = Buffer("idle")
        self._rtx = None
        self._annotator = None
        self._render_product = None
        self._pointcloud_enabled = False
        self._rtx_initialized = False

    @classmethod
    def build(cls, prim_path: str) -> "Lidar":
        """Create lidar prim and attach a visual USD model under model_3d."""
        stage = get_stage()
        UsdGeom.Xform.Define(stage, prim_path)
        ensure_single_usd_reference(
            stage=stage,
            path=os.path.join(prim_path, cls.visual_model_subpath),
            usd_path=cls.usd_url,
            context="Lidar",
        )
        return cls.attach(prim_path)

    @classmethod
    def attach(cls, prim_path: str) -> "Lidar":
        return cls(
            prim_path=prim_path,
            visual_model_path=os.path.join(prim_path, cls.visual_model_subpath),
        )

    def ensure_visual_model(self):
        """Best-effort model injection for existing lidar prims."""
        stage = get_stage()
        ensure_single_usd_reference(
            stage=stage,
            path=self._visual_model_path,
            usd_path=self.usd_url,
            context="Lidar",
        )

    def enable_lidar(self):
        """Enable a Lidar source.

        Strategy:
        1) Prefer isaacsim.sensors.rtx.LidarRtx (RTX Lidar). If available,
           instantiate it for this prim and request point-cloud output.
        2) Fallback to a replicator annotator named 'point_cloud' (if present).

        After enabling, call `add_point_cloud_data_to_frame()` on RTX lidar
        to ensure point-clouds are captured into `get_current_frame()`.
        """
        # Ensure visual representation exists (same style as camera wrappers).
        self.ensure_visual_model()

        # Prefer the native RTX sensor path when available because it is the most
        # faithful 3D source and can expose pointcloud directly from the sensor.
        try:
            from isaacsim.sensors.rtx import LidarRtx

            # Use a dedicated child prim for the actual RTX sensor. The mount path
            # itself is an Xform used to position the sensor on the robot.
            self._rtx = LidarRtx(prim_path=self._sensor_prim_path, name="lidar")
            # request point cloud data on frames
            try:
                self._rtx.add_point_cloud_data_to_frame()
            except Exception:
                pass
            try:
                self._rtx.initialize()
                self._rtx_initialized = True
            except Exception:
                self._rtx_initialized = False
            # mark annotator source as RTX
            self._annotator = None
            self.status.set_value("rtx_ready")
            return
        except Exception as exc:
            # RTX Lidar not available in this environment
            self._rtx = None
            self.status.set_value(f"rtx_init_error:{type(exc).__name__}")

        # Fallback to Replicator pointcloud if the dedicated RTX sensor API is
        # not available in the current Isaac build/environment.
        try:
            self._annotator = rep.AnnotatorRegistry.get_annotator("point_cloud")
            try:
                self._render_product = rep.create.render_product(
                    self._sensor_prim_path,
                    resolution=(1, 1),
                    force_new=False,
                )
            except Exception:
                self._render_product = None
            # Many annotators expect a render product; attach to render product first
            try:
                if self._render_product is not None:
                    self._annotator.attach([self._render_product.path])
                else:
                    raise RuntimeError("render product unavailable")
            except Exception:
                # fallback to prim path
                try:
                    self._annotator.attach([self._sensor_prim_path])
                except Exception:
                    self._annotator.attach(self._sensor_prim_path)
            self.status.set_value("annotator_ready")
        except Exception:
            # no annotator available
            self._annotator = None
            self.status.set_value("no_lidar_backend")

    def disable_lidar(self):
        if self._annotator is not None:
            try:
                self._annotator.detach()
            except Exception:
                pass
            self._annotator = None
        if self._render_product is not None:
            try:
                self._render_product.destroy()
            except Exception:
                pass
            self._render_product = None
        self._pointcloud_enabled = False
        self.status.set_value("disabled")

    def set_pointcloud_enabled(self, enabled: bool):
        self._pointcloud_enabled = bool(enabled)
        if not self._pointcloud_enabled:
            self.pointcloud.set_value(None)
            self.status.set_value("pointcloud_disabled")
        elif self._rtx is not None or self._annotator is not None:
            self.status.set_value("pointcloud_enabled")

    def _extract_rtx_pointcloud(self):
        # Isaac versions differ in where RTX pointcloud data becomes visible.
        # Try the public frame first, then a couple of internal fallbacks.
        pts = None
        source = "rtx_no_pointcloud"

        try:
            if not self._rtx_initialized and hasattr(self._rtx, "initialize"):
                try:
                    self._rtx.initialize()
                    self._rtx_initialized = True
                except Exception:
                    self._rtx_initialized = False

            frame = self._rtx.get_current_frame()
            pts = _sanitize_pointcloud_array(_extract_pointcloud_array(frame))
            if pts is not None:
                return pts, "rtx_frame"
        except Exception:
            pass

        try:
            annotator = getattr(self._rtx, "_point_cloud_annotator", None)
            if annotator is not None:
                pts = _sanitize_pointcloud_array(_extract_pointcloud_array(annotator.get_data()))
                if pts is not None:
                    return pts, "rtx_annotator"
        except Exception:
            pass

        try:
            callback = getattr(self._rtx, "_data_acquisition_callback", None)
            if callable(callback):
                callback(None)
                frame = self._rtx.get_current_frame()
                pts = _sanitize_pointcloud_array(_extract_pointcloud_array(frame))
                if pts is not None:
                    return pts, "rtx_callback"
        except Exception:
            pass

        return None, source

    def update_state(self):
        if self._pointcloud_enabled:
            # If we have an annotator, try to get its data
            if self._rtx is not None:
                pts, source = self._extract_rtx_pointcloud()
                self.pointcloud.set_value(pts)
                if pts is None:
                    self.status.set_value(source)
                else:
                    self.status.set_value(f"{source}:{int(pts.shape[0])}")
            elif self._annotator is not None:
                try:
                    pc = self._annotator.get_data()
                    pts = _sanitize_pointcloud_array(_extract_pointcloud_array(pc))
                    self.pointcloud.set_value(pts)
                    if pts is None:
                        self.status.set_value("annotator_no_pointcloud")
                    else:
                        self.status.set_value(f"annotator_points:{int(pts.shape[0])}")
                except Exception:
                    # If annotator fails, leave pointcloud as-is (or set to None)
                    self.pointcloud.set_value(None)
                    self.status.set_value("annotator_error")
            else:
                # No annotator attached: device-specific API should populate
                # the buffer here.
                self.pointcloud.set_value(None)
                self.status.set_value("no_lidar_backend")
        else:
            self.pointcloud.set_value(None)

        # always update pose buffers from prim transform
        try:
            position, orientation = self._xform_prim.get_world_pose()
            self.position.set_value(position)
            self.orientation.set_value(orientation)
        except Exception:
            # best-effort: leave as None
            self.position.set_value(None)
            self.orientation.set_value(None)

        super().update_state()


#=========================================================
#  FINAL CLASSES
#=========================================================


class HawkCamera(Sensor):

    #usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/LeopardImaging/Hawk/hawk_v1.1_nominal.usd"
    usd_url: str = _resolve_asset_path(
        "robot_assets/LeopardImaging/Hawk/hawk_v1.1_nominal.usd",
        "/home/pdi_4/Documents/Documentos/bevlog-isaac/isaac-assets/"
        "isaac-sim-assets-robots_and_sensors-5.1.0/Assets/Isaac/5.1/"
        "Isaac/Sensors/LeopardImaging/Hawk/hawk_v1.1_nominal.usd",
    )
    resolution: Tuple[int, int] = (640, 400)
    left_camera_path: str = "left/camera_left"
    right_camera_path: str = "right/camera_right"

    def __init__(self, 
            left: Camera, 
            right: Camera
        ):
        self.left = left
        self.right = right
    
    @classmethod
    def build(cls, prim_path: str) -> "HawkCamera":
        
        stage = get_stage()

        ensure_single_usd_reference(
            stage=stage,
            path=prim_path,
            usd_path=cls.usd_url,
            context="HawkCamera",
        )

        return cls.attach(prim_path)
    
    @classmethod
    def attach(cls, prim_path: str) -> "HawkCamera":
        
        left_camera = Camera(os.path.join(prim_path, cls.left_camera_path), cls.resolution)
        right_camera = Camera(os.path.join(prim_path, cls.right_camera_path), cls.resolution)

        return HawkCamera(left_camera, right_camera)
    

# ================================================
# RealSense em formato HawkCamera
# ================================================
class RealSenseRGBDCamera(Sensor):
    """
    Câmera “tipo RealSense” (RGB-D) com o mesmo padrão estrutural de HawkCamera:
    - Atributos de classe: usd_url, resolution, camera_path (subprim interno)
    - build(): injeta a referência USD e delega para attach()
    - attach(): empacota o prim interno em uma Camera (sua classe)
    """
    # Ajuste para o caminho real do asset no seu servidor/Omniverse
   
    #usd_url: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Sensors/RealSense/RealSense_D435.usd"
    usd_url: str = _resolve_asset_path(
        "robot_assets/RealSense/RealSense_D435.usd",
        "/home/pdi_4/Documents/Documentos/bevlog-isaac/isaac-assets/"
        "isaac-sim-assets-robots_and_sensors-5.1.0/Assets/Isaac/5.1/"
        "Isaac/Sensors/RealSense/RealSense_D435.usd",
    )
    resolution: Tuple[int, int] = (640, 360)
    camera_path: str = "camera"  # subcaminho do prim de câmera dentro do USD referenciado

    def __init__(self, cam: Camera):
        self.cam = cam

    @classmethod
    def build(cls, prim_path: str) -> "RealSenseRGBDCamera":
        """
        Cria um Xform em `prim_path`, adiciona a referência para o USD da RealSense
        e retorna o wrapper com a Camera interna.
        """
        stage = get_stage()

        ensure_single_usd_reference(
            stage=stage,
            path=prim_path,
            usd_path=cls.usd_url,
            context="RealSenseRGBDCamera",
        )

        return cls.attach(prim_path)

    @classmethod
    def attach(cls, prim_path: str) -> "RealSenseRGBDCamera":
        """
        Resolve o caminho absoluto do prim interno da câmera (ex.: "<prim_path>/camera")
        e o empacota em uma instância da sua classe Camera, com a resolução padrão.
        """
        cam_full_path = os.path.join(prim_path, cls.camera_path)
        cam = Camera(cam_full_path, cls.resolution)
        return RealSenseRGBDCamera(cam)


# ================================================
# ZED Stereo em formato HawkCamera
# ================================================
class ZedStereoCamera(Sensor):
    """
    Par estéreo estilo ZED com o mesmo padrão estrutural de HawkCamera:
    - Atributos de classe: usd_url, resolution, left/right paths
    - build(): injeta a referência USD e delega para attach()
    - attach(): empacota os dois prims internos em duas Cameras (left/right)
    """
    # Ajuste para o caminho real do asset no seu servidor/Omniverse
    #usd_url: str = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usd"
    usd_url: str = _resolve_asset_path(
        "robot_assets/Stereolabs/ZED_X/ZED_X.usdc",
        "robot_assets/Stereolabs/ZED_X/ZED_X.usd",
        "/home/pdi_4/Documents/Documentos/bevlog-isaac/isaac-assets/"
        "isaac-sim-assets-robots_and_sensors-5.1.0/Assets/Isaac/5.1/"
        "Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usd",
    )
    resolution: Tuple[int, int] = (640, 360)
    left_camera_path: str = "left/camera_left"     # confirme no USD
    right_camera_path: str = "right/camera_right"  # confirme no USD

    def __init__(self, left: Camera, right: Camera):
        self.left = left
        self.right = right

    @classmethod
    def build(cls, prim_path: str) -> "ZedStereoCamera":
        """
        Cria um Xform em `prim_path`, adiciona a referência USD da ZED
        e retorna o wrapper estéreo com left/right.
        """
        stage = get_stage()

        ensure_single_usd_reference(
            stage=stage,
            path=prim_path,
            usd_path=cls.usd_url,
            context="ZedStereoCamera",
        )

        return cls.attach(prim_path)

    @classmethod
    def attach(cls, prim_path: str) -> "ZedStereoCamera":
        """
        Resolve os caminhos absolutos dos prims internos de cada câmera
        (ex.: "<prim_path>/left/camera_left" e "<prim_path>/right/camera_right"),
        empacota em duas instâncias Camera e retorna o par estéreo.
        """
        left_camera = Camera(os.path.join(prim_path, cls.left_camera_path), cls.resolution)
        right_camera = Camera(os.path.join(prim_path, cls.right_camera_path), cls.resolution)
        return ZedStereoCamera(left_camera, right_camera)


# =========================================================
# Helpers de câmera (como no seu código original)
# =========================================================

def _define_camera_prim(path: str) -> UsdGeom.Camera:
    """Garante/define um prim de câmera no path e o retorna."""
    stage = omni.usd.get_context().get_stage()
    return UsdGeom.Camera.Define(stage, path)

def _xform_translate(path: str, xyz: Tuple[float, float, float]):
    """Aplica (ou cria) um XformOp de Translate no prim path."""
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    ops = xf.GetOrderedXformOps()
    t_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op = op
            break
    if t_op is None:
        t_op = xf.AddTranslateOp()
    attr = t_op.GetAttr() if hasattr(t_op, "GetAttr") else t_op.GetOpAttr()
    typ = str(attr.GetTypeName())
    if "float3" in typ or "GfVec3f" in typ:
        attr.Set(Gf.Vec3f(*[float(v) for v in xyz]))
    else:
        attr.Set(Gf.Vec3d(*[float(v) for v in xyz]))

def _xform_orient_quat(path: str, quat_wxyz: Tuple[float, float, float, float]):
    """Aplica (ou cria) um XformOp de Orient (quaternion w,x,y,z) no prim path."""
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    ops = xf.GetOrderedXformOps()
    o_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            o_op = op
            break
    if o_op is None:
        o_op = xf.AddOrientOp()
    qw, qx, qy, qz = [float(q) for q in quat_wxyz]
    attr = o_op.GetAttr() if hasattr(o_op, "GetAttr") else o_op.GetOpAttr()
    typ = str(attr.GetTypeName())
    if "quatf" in typ or "GfQuatf" in typ:
        attr.Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
    else:
        attr.Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))

def _quat_from_euler_xyz(rx_deg: float, ry_deg: float, rz_deg: float):
    """Converte Euler XYZ (graus) para quaternion (w, x, y, z) com composição intrínseca."""
    rx = math.radians(rx_deg); ry = math.radians(ry_deg); rz = math.radians(rz_deg)
    cx, cy, cz = math.cos(rx/2), math.cos(ry/2), math.cos(rz/2)
    sx, sy, sz = math.sin(rx/2), math.sin(ry/2), math.sin(rz/2)
    qw = cx*cy*cz - sx*sy*sz
    qx = sx*cy*cz + cx*sy*sz
    qy = cx*sy*cz - sx*cy*sz
    qz = cx*cy*sz + sx*sy*cz
    return (qw, qx, qy, qz)




class OceanSimUWCamera(Sensor):
    """Wrapper do OceanSim UW_Camera no formato Module/Buffer do PatoSim."""

    resolution: Tuple[int, int] = (1920, 1080)

    def __init__(
        self,
        prim_path: str,
        sensor,
        *,
        water_profile_path: Optional[str] = None,
    ):
        self._prim_path = prim_path
        self._xform_prim = XFormPrim(self._prim_path)
        self._sensor = sensor
        self._water_profile_path = water_profile_path
        self._initialized = False
        self._rgb_enabled = False
        self._depth_enabled = False

        self.raw_rgb_image = Buffer(tags=["rgb"])
        self.rgb_image = Buffer(tags=["rgb"])
        self.depth_image = Buffer(tags=["depth"])
        self.position = Buffer()
        self.orientation = Buffer()

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        name: str = "UW_Camera",
        resolution: Tuple[int, int] | None = None,
        translation=None,
        orientation=None,
        focal_length: Optional[float] = None,
        clipping_range: Tuple[float, float] | None = None,
        water_profile_path: Optional[str] = None,
    ) -> "OceanSimUWCamera":
        from isaacsim.oceansim.sensors.UW_Camera import UW_Camera as _UWCamera

        res = tuple(resolution or cls.resolution)
        sensor = _UWCamera(
            prim_path=prim_path,
            name=name,
            resolution=list(res),
            translation=translation,
            orientation=orientation,
        )
        try:
            cam_geom = UsdGeom.Camera(get_stage().GetPrimAtPath(prim_path))
            horizontal_aperture = float(cam_geom.GetHorizontalApertureAttr().Get())
            vertical_aperture = horizontal_aperture * (float(res[1]) / max(float(res[0]), 1.0))
            cam_geom.GetVerticalApertureAttr().Set(vertical_aperture)
        except Exception:
            pass
        if focal_length is not None:
            try:
                sensor.set_focal_length(float(focal_length))
            except Exception:
                pass
            try:
                horizontal_aperture = float(sensor.get_horizontal_aperture())
                vertical_aperture = horizontal_aperture * (float(res[1]) / max(float(res[0]), 1.0))
                sensor.set_vertical_aperture(float(vertical_aperture))
            except Exception:
                pass
        if clipping_range is not None:
            try:
                sensor.set_clipping_range(float(clipping_range[0]), float(clipping_range[1]))
            except Exception:
                pass
        return cls(
            prim_path=prim_path,
            sensor=sensor,
            water_profile_path=water_profile_path,
        )

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._sensor.initialize(
            viewport=False,
            UW_yaml_path=self._water_profile_path,
        )
        self._initialized = True

    def enable_rgb_rendering(self):
        self._rgb_enabled = True
        self._ensure_initialized()

    def enable_depth_rendering(self):
        self._depth_enabled = True
        self._ensure_initialized()

    def disable_rendering(self):
        if not self._initialized:
            return
        try:
            self._sensor.close()
        except Exception:
            pass
        self._initialized = False

    def _render_underwater_rgb(self, raw_rgba, depth) -> Optional[np.ndarray]:
        # The underwater camera is simulated as a post-process over optical RGB
        # plus depth, not as a separate 3D sensor. This keeps it lightweight and
        # makes the output compatible with the same RGB/depth recording pipeline.
        try:
            import warp as wp
            from isaacsim.oceansim.utils.UWrenderer_utils import UW_render
        except Exception:
            raw_np = _to_numpy_array(raw_rgba)
            if raw_np is None:
                return None
            return np.asarray(raw_np[..., :3], dtype=np.uint8)

        if raw_rgba is None or depth is None:
            return None

        try:
            uw_image = wp.zeros_like(raw_rgba)
            wp.launch(
                dim=np.flip(self._sensor.get_resolution()),
                kernel=UW_render,
                inputs=[
                    raw_rgba,
                    depth,
                    self._sensor._backscatter_value,
                    self._sensor._atten_coeff,
                    self._sensor._backscatter_coeff,
                ],
                outputs=[uw_image],
            )
            uw_np = _to_numpy_array(uw_image)
            if uw_np is None:
                return None
            return np.asarray(uw_np[..., :3], dtype=np.uint8)
        except Exception:
            raw_np = _to_numpy_array(raw_rgba)
            if raw_np is None:
                return None
            return np.asarray(raw_np[..., :3], dtype=np.uint8)

    def update_state(self):
        if self._initialized:
            raw_rgba = None
            depth = None
            try:
                raw_rgba = self._sensor._rgba_annot.get_data()
            except Exception:
                pass
            try:
                depth = self._sensor._depth_annot.get_data()
            except Exception:
                pass

            if self._rgb_enabled and raw_rgba is not None:
                raw_np = _to_numpy_array(raw_rgba)
                if raw_np is not None:
                    self.raw_rgb_image.set_value(np.asarray(raw_np[..., :3], dtype=np.uint8))
                rgb = self._render_underwater_rgb(raw_rgba, depth)
                if rgb is not None:
                    self.rgb_image.set_value(rgb)

            if self._depth_enabled and depth is not None:
                depth_np = _to_numpy_array(depth)
                if depth_np is not None:
                    self.depth_image.set_value(np.asarray(depth_np))

        try:
            position, orientation = self._xform_prim.get_world_pose()
            self.position.set_value(position)
            self.orientation.set_value(orientation)
        except Exception:
            self.position.set_value(None)
            self.orientation.set_value(None)

        super().update_state()


class OceanSimStereoUWCamera(Sensor):
    """Par estéreo de câmeras subaquáticas, no mesmo padrão das câmeras estéreo do projeto."""

    def __init__(self, left: OceanSimUWCamera, right: OceanSimUWCamera):
        self.left = left
        self.right = right

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        resolution: Tuple[int, int] | None = None,
        left_translation=None,
        right_translation=None,
        orientation=None,
        focal_length: Optional[float] = None,
        clipping_range: Tuple[float, float] | None = None,
        water_profile_path: Optional[str] = None,
    ) -> "OceanSimStereoUWCamera":
        left = OceanSimUWCamera.build(
            os.path.join(prim_path, "left_camera"),
            name="UW_Camera_Left",
            resolution=resolution,
            translation=left_translation,
            orientation=orientation,
            focal_length=focal_length,
            clipping_range=clipping_range,
            water_profile_path=water_profile_path,
        )
        right = OceanSimUWCamera.build(
            os.path.join(prim_path, "right_camera"),
            name="UW_Camera_Right",
            resolution=resolution,
            translation=right_translation,
            orientation=orientation,
            focal_length=focal_length,
            clipping_range=clipping_range,
            water_profile_path=water_profile_path,
        )
        return cls(left, right)


class OceanSimImagingSonar(Sensor):
    """Wrapper do ImagingSonarSensor com buffers compatíveis com preview e gravação."""

    def __init__(self, prim_path: str, sensor):
        self._prim_path = prim_path
        self._xform_prim = XFormPrim(self._prim_path)
        self._sensor = sensor
        self._initialized = False
        self._rgb_enabled = False
        self._pointcloud_enabled = False
        self._gau_noise_param = 0.05
        self._ray_noise_param = 0.05
        self._attenuation = 0.3

        # rgb_image stores the processed acoustic image produced by the sonar
        # model. pointcloud stores the raw scan pointcloud used internally by the
        # sonar backend before polar binning/noise are applied.
        self.rgb_image = Buffer(tags=["rgb"])
        self.pointcloud = Buffer(tags=["pointcloud"])
        self.position = Buffer()
        self.orientation = Buffer()
        self.status = Buffer("idle")
        self._sync_sensor_render_params()

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        translation=None,
        orientation=None,
        min_range: float = 0.2,
        max_range: float = 3.0,
        range_res: float = 0.005,
        angular_res: float = 0.25,
        hori_res: int = 4000,
    ) -> "OceanSimImagingSonar":
        from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor as _ImagingSonarSensor

        sensor = _ImagingSonarSensor(
            prim_path=prim_path,
            translation=translation,
            orientation=orientation,
            min_range=min_range,
            max_range=max_range,
            range_res=range_res,
            angular_res=angular_res,
            hori_res=hori_res,
        )
        try:
            cam_geom = UsdGeom.Camera(get_stage().GetPrimAtPath(prim_path))
            cam_geom.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
        except Exception:
            pass
        return cls(prim_path=prim_path, sensor=sensor)

    def _ensure_initialized(self):
        if self._initialized:
            return
        # include_unlabelled=True lets the sonar "see" meshes even when they do
        # not carry semantic labels. Semantic reflectivity still improves realism.
        self._sensor.sonar_initialize(
            viewport=False,
            include_unlabelled=True,
        )
        self._initialized = True
        self.status.set_value("sonar_ready")

    def enable_rgb_rendering(self):
        self._rgb_enabled = True
        self._ensure_initialized()

    def disable_rendering(self):
        self._rgb_enabled = False

    def set_pointcloud_enabled(self, enabled: bool):
        self._pointcloud_enabled = bool(enabled)
        if self._pointcloud_enabled:
            self._ensure_initialized()
        else:
            self.pointcloud.set_value(None)

    def _sync_sensor_render_params(self):
        if self._sensor is None:
            return
        try:
            self._sensor.gau_noise_param = float(self._gau_noise_param)
            self._sensor.ray_noise_param = float(self._ray_noise_param)
            self._sensor.attenuation = float(self._attenuation)
        except Exception:
            pass

    def set_render_model_params(
        self,
        *,
        gau_noise_param: Optional[float] = None,
        ray_noise_param: Optional[float] = None,
        attenuation: Optional[float] = None,
    ) -> None:
        if gau_noise_param is not None:
            self._gau_noise_param = float(gau_noise_param)
        if ray_noise_param is not None:
            self._ray_noise_param = float(ray_noise_param)
        if attenuation is not None:
            self._attenuation = float(attenuation)
        self._sync_sensor_render_params()

    def update_state(self):
        if self._initialized:
            try:
                # make_sonar_data() computes the acoustic response map from raw
                # pointcloud, normals and semantic reflectivity.
                self._sync_sensor_render_params()
                self._sensor.make_sonar_data(
                    attenuation=float(self._attenuation),
                    gau_noise_param=float(self._gau_noise_param),
                    ray_noise_param=float(self._ray_noise_param),
                )
                self.status.set_value("sonar_frame")
            except Exception as exc:
                self.status.set_value(f"sonar_error:{type(exc).__name__}")

            if self._rgb_enabled:
                try:
                    # The backend returns a rendered sonar image (polar acoustic
                    # intensity map), which we expose as an RGB-like preview.
                    sonar_np = _to_numpy_array(self._sensor.make_sonar_image())
                    if sonar_np is not None:
                        self.rgb_image.set_value(np.asarray(sonar_np[..., :3], dtype=np.uint8))
                except Exception:
                    pass

            if self._pointcloud_enabled:
                try:
                    pcl = self._sensor.scan_data.get("pcl")
                except Exception:
                    pcl = None
                # This is the raw scan pointcloud captured by the sonar camera
                # annotator, not the final binned sonar image/map.
                pcl_np = _sanitize_pointcloud_array(_to_numpy_array(pcl))
                self.pointcloud.set_value(pcl_np)

        try:
            position, orientation = self._xform_prim.get_world_pose()
            self.position.set_value(position)
            self.orientation.set_value(orientation)
        except Exception:
            self.position.set_value(None)
            self.orientation.set_value(None)

        super().update_state()

    # ------------------------------------------------------------------
    # Preview rendering helpers
    # ------------------------------------------------------------------

    def render_polar_preview(self, size: int = 400) -> Optional[np.ndarray]:
        """Converte a imagem de sonar (grade r×azi) para uma imagem Cartesiana (setor fan).

        O sonar OceanSim armazena os dados num grid 2D (N_range × N_azi) onde
        cada célula [i, j] representa (range_i, azimute_j). Exibir esse grid
        diretamente como imagem retangular produz distorção porque o espaço polar
        não é linear em X,Y.

        Este método projeta cada pixel do grid polar nas coordenadas Cartesianas
        correspondentes, gerando uma imagem em visão de cima (bird's-eye) no
        formato RGBA numpy compatível com ``omni.ui.ByteImageProvider``.

        Args:
            size: Lado em pixels da imagem de saída quadrada (padrão 400).

        Returns:
            np.ndarray RGBA (size, size, 4) uint8, ou None se não houver dados.
        """
        raw = self.rgb_image.get_value()
        if raw is None:
            return None
        try:
            src = np.asarray(raw, dtype=np.uint8)
            if src.ndim == 3 and src.shape[2] >= 3:
                gray = src[..., 0]  # sonar é grayscale nos 3 canais iguais
            elif src.ndim == 2:
                gray = src
            else:
                return None

            n_range, n_azi = gray.shape
            # Reconstrução dos vetores r e azi a partir dos parâmetros do sensor
            try:
                min_r = float(self._sensor.min_range)
                max_r = float(self._sensor.max_range)
                range_res = float(self._sensor.range_res)
                hori_fov = float(self._sensor.hori_fov)
                ang_res = float(self._sensor.angular_res)
            except Exception:
                # Fallback genérico se o sensor não estiver disponível
                min_r, max_r, range_res = 0.2, 10.0, (10.0 - 0.2) / max(n_range, 1)
                hori_fov, ang_res = 130.0, hori_fov / max(n_azi, 1)

            # Grade de ranges em metros
            r_vals = np.linspace(min_r, max_r, n_range, dtype=np.float32)
            # Azimutes em radianos (convenção do sonar: centrado em 90° = frente)
            azi_deg_min = 90.0 - hori_fov / 2.0
            azi_deg_max = 90.0 + hori_fov / 2.0
            azi_vals = np.deg2rad(
                np.linspace(azi_deg_min, azi_deg_max, n_azi, dtype=np.float32)
            )

            # Converte cada ponto polar → Cartesiano (x=frente, y=lateral)
            # Convenção: azi=90° → frente (+X), azi<90° → direita, azi>90° → esquerda
            r_grid, azi_grid = np.meshgrid(r_vals, azi_vals, indexing='ij')
            x_cart = r_grid * np.cos(azi_grid)   # profundidade (frente)
            y_cart = r_grid * np.sin(azi_grid)   # lateral

            # Mapeia para pixels na imagem de saída
            # X (frente) mapeado para V (linha), Y (lateral) para U (coluna)
            x_min, x_max = 0.0, float(max_r)
            y_min = -float(max_r) * np.sin(np.deg2rad(hori_fov / 2.0))
            y_max = float(max_r) * np.sin(np.deg2rad(hori_fov / 2.0))

            col_f = (y_cart - y_min) / max(y_max - y_min, 1e-6) * (size - 1)
            row_f = (1.0 - (x_cart - x_min) / max(x_max - x_min, 1e-6)) * (size - 1)
            col_i = np.clip(col_f.astype(np.int32), 0, size - 1)
            row_i = np.clip(row_f.astype(np.int32), 0, size - 1)

            out = np.zeros((size, size, 4), dtype=np.uint8)
            out[:, :, 3] = 255
            out_intensity = np.zeros((size, size), dtype=np.uint8)
            intensity = gray.ravel()
            colored = _apply_sonar_acoustic_colormap(gray).reshape(-1, 3)
            rows = row_i.ravel()
            cols = col_i.ravel()
            # Pintar mais claro sobre mais escuro para preservar retornos fortes
            current = out_intensity[rows, cols]
            mask = intensity > current
            out_intensity[rows[mask], cols[mask]] = intensity[mask]
            out[rows[mask], cols[mask], :3] = colored[mask]

            try:
                import cv2

                overlay_rgb = np.zeros((size, size, 3), dtype=np.uint8)
                overlay_alpha = np.zeros((size, size), dtype=np.uint8)
                overlay_color = (255, 255, 255)
                overlay_alpha_value = 100
                fan_angles_deg = np.linspace(azi_deg_min, azi_deg_max, 256, dtype=np.float32)

                def _to_cv_points(radius_m: float, angles_deg: np.ndarray) -> np.ndarray:
                    angles_rad = np.deg2rad(angles_deg)
                    x_vals = radius_m * np.cos(angles_rad)
                    y_vals = radius_m * np.sin(angles_rad)
                    cols_local = np.clip(
                        ((y_vals - y_min) / max(y_max - y_min, 1e-6) * (size - 1)).round().astype(np.int32),
                        0,
                        size - 1,
                    )
                    rows_local = np.clip(
                        ((1.0 - (x_vals - x_min) / max(x_max - x_min, 1e-6)) * (size - 1)).round().astype(np.int32),
                        0,
                        size - 1,
                    )
                    return np.stack([cols_local, rows_local], axis=1).reshape(-1, 1, 2)

                max_range_mark = float(max_r)
                default_marks = [2.0, 4.0, 6.0, 8.0, 10.0]
                arc_ranges = [mark for mark in default_marks if mark < max_range_mark]
                if not arc_ranges and max_range_mark > 0.0:
                    arc_ranges = list(np.linspace(max_range_mark * 0.2, max_range_mark, 5, dtype=np.float32))
                elif max_range_mark > 0.0 and max_range_mark not in arc_ranges:
                    arc_ranges.append(max_range_mark)

                for radius_m in arc_ranges:
                    pts = _to_cv_points(float(radius_m), fan_angles_deg)
                    if pts.shape[0] < 2:
                        continue
                    cv2.polylines(overlay_rgb, [pts], False, overlay_color, 1, lineType=cv2.LINE_AA)
                    cv2.polylines(overlay_alpha, [pts], False, overlay_alpha_value, 1, lineType=cv2.LINE_AA)
                    label_idx = int(np.argmin(pts[:, 0, 1]))
                    label_x = int(np.clip(pts[label_idx, 0, 0] + 4, 0, size - 24))
                    label_y = int(np.clip(pts[label_idx, 0, 1] - 4, 10, size - 4))
                    label = f"{int(round(float(radius_m)))}m"
                    cv2.putText(
                        overlay_rgb,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        overlay_color,
                        1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        overlay_alpha,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        overlay_alpha_value,
                        1,
                        lineType=cv2.LINE_AA,
                    )

                center_angle_deg = 90.0
                for offset_deg in (15.0, 30.0, 45.0, 60.0):
                    for sign in (-1.0, 1.0):
                        angle_deg = center_angle_deg + sign * offset_deg
                        if angle_deg < azi_deg_min or angle_deg > azi_deg_max:
                            continue
                        angle_rad = math.radians(angle_deg)
                        start_x = 0.0
                        start_y = 0.0
                        end_x = float(max_r) * math.cos(angle_rad)
                        end_y = float(max_r) * math.sin(angle_rad)
                        start_xy = (
                            int(np.clip(round((start_y - y_min) / max(y_max - y_min, 1e-6) * (size - 1)), 0, size - 1)),
                            int(np.clip(round((1.0 - (start_x - x_min) / max(x_max - x_min, 1e-6)) * (size - 1)), 0, size - 1)),
                        )
                        end_xy = (
                            int(np.clip(round((end_y - y_min) / max(y_max - y_min, 1e-6) * (size - 1)), 0, size - 1)),
                            int(np.clip(round((1.0 - (end_x - x_min) / max(x_max - x_min, 1e-6)) * (size - 1)), 0, size - 1)),
                        )
                        cv2.line(overlay_rgb, start_xy, end_xy, overlay_color, 1, lineType=cv2.LINE_AA)
                        cv2.line(
                            overlay_alpha,
                            start_xy,
                            end_xy,
                            overlay_alpha_value,
                            1,
                            lineType=cv2.LINE_AA,
                        )

                out = _blend_overlay_rgba(out, overlay_rgb, overlay_alpha)
            except Exception:
                pass
            return out
        except Exception:
            return None

    def render_planar_preview(self, width: int = 400, height: int = 300) -> Optional[np.ndarray]:
        """Retorna a imagem polar retangular (r×azi) redimensionada como RGBA.

        Esta é a representação direta do buffer do sonar — útil para depuração
        e para ver o conteúdo bruto da saída do ``make_sonar_image()``.

        Args:
            width: Largura da imagem de saída em pixels.
            height: Altura da imagem de saída em pixels.

        Returns:
            np.ndarray RGBA (height, width, 4) uint8, ou None se não houver dados.
        """
        raw = self.rgb_image.get_value()
        if raw is None:
            return None
        try:
            import cv2
            src = np.asarray(raw, dtype=np.uint8)
            if src.ndim == 3:
                gray = src[..., 0]
            else:
                gray = src
            resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
            colored = _apply_sonar_acoustic_colormap(resized)
            out = np.concatenate(
                [colored, np.full((height, width, 1), 255, dtype=np.uint8)],
                axis=-1,
            )
            return out
        except Exception:
            return None


class OceanSimDVL(Sensor):
    """Wrapper do DVL do OceanSim com leituras em buffers comuns."""

    def __init__(self, prim_path: str, sensor):
        self._prim_path = prim_path
        self._sensor = sensor

        # All DVL outputs remain untagged on purpose so they are captured in
        # ``state/common`` together with the robot pose/state instead of going
        # through the heavier image/pointcloud writers.
        self.linear_velocity = Buffer()
        self.beam_depth = Buffer()
        self.beam_hit = Buffer()
        self.dropout = Buffer(False)
        self.position = Buffer()
        self.orientation = Buffer()
        self.status = Buffer("idle")

    @classmethod
    def build(
        cls,
        rigid_body_path: str,
        *,
        name: str = "DVL",
        translation=None,
        orientation=None,
        max_range: float = 10.0,
        add_debug_lines: bool = False,
    ) -> "OceanSimDVL":
        from isaacsim.oceansim.sensors.DVLsensor import DVLsensor as _DVLsensor

        sensor = _DVLsensor(name=name, max_range=max_range)
        sensor.attachDVL(
            rigid_body_path=rigid_body_path,
            translation=translation,
            orientation=orientation,
        )
        if add_debug_lines:
            try:
                sensor.add_debug_lines()
            except Exception:
                pass
        return cls(
            prim_path=os.path.join(rigid_body_path, name),
            sensor=sensor,
        )

    def update_state(self):
        beam_hit = None
        try:
            beam_hit = np.asarray(self._sensor.get_beam_hit(), dtype=bool)
            self.beam_hit.set_value(beam_hit)
        except Exception:
            self.beam_hit.set_value(None)

        # O DVL do OceanSim faz log de dropout dentro de get_linear_vel()/get_depth().
        # Se chamarmos ambos a cada frame, o warning aparece varias vezes. Primeiro
        # checamos os hits e, se estiver em dropout, publicamos buffers vazios/seguros.
        in_dropout = bool(beam_hit is not None and np.count_nonzero(~beam_hit) >= 2)
        self.dropout.set_value(in_dropout)
        if in_dropout:
            self.linear_velocity.set_value(np.zeros(3, dtype=np.float32))
            self.beam_depth.set_value(np.full(4, np.nan, dtype=np.float32))
            self.status.set_value("dropout")
        else:
            try:
                self.linear_velocity.set_value(np.asarray(self._sensor.get_linear_vel(), dtype=np.float32))
            except Exception:
                self.linear_velocity.set_value(None)

            try:
                self.beam_depth.set_value(np.asarray(self._sensor.get_depth(), dtype=np.float32))
            except Exception:
                self.beam_depth.set_value(None)

            if beam_hit is None:
                self.status.set_value("unknown")
            else:
                self.status.set_value("tracking")

        if beam_hit is None:
            self.linear_velocity.set_value(self.linear_velocity.get_value())
            self.beam_depth.set_value(self.beam_depth.get_value())

        try:
            position, orientation = self._sensor.get_baseSensor().get_world_pose()
            self.position.set_value(position)
            self.orientation.set_value(orientation)
        except Exception:
            self.position.set_value(None)
            self.orientation.set_value(None)

        super().update_state()


class OceanSimBarometer(Sensor):
    """Wrapper do barômetro/pressure sensor do OceanSim."""

    def __init__(self, prim_path: str, sensor):
        self._prim_path = prim_path
        self._sensor = sensor
        self._water_surface_z = None
        try:
            self._water_surface_z = float(getattr(sensor, "_water_surface_z"))
        except Exception:
            self._water_surface_z = None

        # Barometer outputs are kept in ``state/common`` because they are
        # scalar navigation measurements rather than rendered products.
        self.pressure = Buffer()
        self.depth = Buffer()
        self.position = Buffer()
        self.orientation = Buffer()
        self.status = Buffer("idle")

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        translation=None,
        orientation=None,
        water_surface_z: float = 0.0,
    ) -> "OceanSimBarometer":
        from isaacsim.oceansim.sensors.BarometerSensor import BarometerSensor as _BarometerSensor

        sensor = _BarometerSensor(
            prim_path=prim_path,
            translation=translation,
            orientation=orientation,
            water_surface_z=water_surface_z,
        )
        return cls(prim_path=prim_path, sensor=sensor)

    def update_state(self):
        try:
            self.pressure.set_value(float(self._sensor.get_pressure()))
        except Exception:
            self.pressure.set_value(None)

        try:
            position, orientation = self._sensor.get_world_pose()
            self.position.set_value(position)
            self.orientation.set_value(orientation)
        except Exception:
            self.position.set_value(None)
            self.orientation.set_value(None)

        position = self.position.get_value()
        if position is not None and self._water_surface_z is not None:
            try:
                depth = max(0.0, float(self._water_surface_z) - float(np.asarray(position, dtype=np.float32)[2]))
                self.depth.set_value(depth)
                self.status.set_value("submerged" if depth > 0.0 else "surface_or_air")
            except Exception:
                self.depth.set_value(None)
                self.status.set_value("unknown")
        else:
            self.depth.set_value(None)
            self.status.set_value("unknown")

        super().update_state()
