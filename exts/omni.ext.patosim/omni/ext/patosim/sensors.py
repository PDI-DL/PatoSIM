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
        for value in payload.values():
            extracted = _extract_pointcloud_array(value)
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


def _normalize_asset_path(asset_path: str) -> str:
    text = str(asset_path or "").strip()
    if "://" in text:
        return text
    return os.path.normpath(text)


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

# tenho que editar os sensores de camera usando esse modulo como base 
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

        # Try RTX Lidar first
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

        # Fallback: replicator annotator
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


# =========================================================
# CÂMERAS BEV (padrão HawkCamera; sem atalhos próprios)
# =========================================================

class BevTopDownCamera(Sensor):
    """
    Câmera ortográfica (vista superior métrica) – ideal para GT BEV.
    build(): cria/configura o prim ortográfico (apertures em mm) e posiciona em +Z.
    attach(): envolve o prim em uma Camera (sua classe).
    """
    resolution: Tuple[int, int] = (1024, 1024)

    def __init__(self, cam: Camera):
        self.cam = cam

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        height_m: float = 5.0,
        view_width_m: float = 14.0,
        view_height_m: float = 14.0,
        resolution: Tuple[int, int] = (1024, 1024),
        near: float = 0.05,
        far: float = 2000.0,
    ) -> "BevTopDownCamera":
        """
        - horizontal/verticalAperture em **milímetros** (USD).
        - Para cobrir exatamente `view_width_m` x `view_height_m` em mundo, use mm = metros * 1000.
        - Em ortográfica, a câmera olha por padrão para -Z; subir em +Z basta.
        """
        cam_prim = _define_camera_prim(prim_path)
        cam_prim.CreateProjectionAttr(UsdGeom.Tokens.orthographic)

        # ✅ Conversão correta: metros -> milímetros
        cam_prim.CreateHorizontalApertureAttr(view_width_m * 1000.0)
        cam_prim.CreateVerticalApertureAttr(view_height_m * 1000.0)
        cam_prim.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))

        # Posição: acima do plano (olhando -Z por padrão)
        _xform_translate(prim_path, (0.0, 0.0, float(height_m)))

        return cls.attach(prim_path, resolution)

    @classmethod
    def attach(cls, prim_path: str, resolution: Tuple[int, int] = None) -> "BevTopDownCamera":
        res = resolution if resolution is not None else cls.resolution
        return BevTopDownCamera(Camera(prim_path, res))


class BevFrontDownCamera(Sensor):
    """
    Câmera perspectiva à frente e inclinada para baixo.
    - FOV horizontal definido por hfov_deg.
    - verticalAperture calculada pelo aspect ratio da resolução (robusto para não-16:9).
    """
    resolution: Tuple[int, int] = (1280, 720)

    def __init__(self, cam: Camera):
        self.cam = cam

    @classmethod
    def build(
        cls,
        prim_path: str,
        *,
        forward_m: float = 0.8,
        height_m: float = 1.8,
        pitch_down_deg: float = 55.0,
        hfov_deg: float = 70.0,
        resolution: Tuple[int, int] = (1280, 720),
        near: float = 0.05,
        far: float = 1000.0,
        filmback_mm: float = 36.0,  # horizontalAperture "clássica" de 36mm
    ) -> "BevFrontDownCamera":
        """
        Constrói uma câmera perspectiva com HFOV alvo:
        - focal = 0.5 * horiz_ap_mm / tan(HFOV/2)
        - verticalAperture = horiz_ap_mm / aspect (para respeitar a resolução escolhida)
        """
        cam_prim = _define_camera_prim(prim_path)
        cam_prim.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        cam_prim.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))

        # Óptica a partir do HFOV
        horiz_ap_mm = float(filmback_mm)
        focal_mm = 0.5 * horiz_ap_mm / math.tan(math.radians(hfov_deg) * 0.5)

        # Usa o aspect da resolução (robusto para qualquer res)
        res_x, res_y = resolution
        aspect = (res_x / res_y) if (res_x and res_y) else (16.0 / 9.0)
        vert_ap_mm = horiz_ap_mm / aspect

        cam_prim.CreateHorizontalApertureAttr(horiz_ap_mm)
        cam_prim.CreateVerticalApertureAttr(vert_ap_mm)
        cam_prim.CreateFocalLengthAttr(focal_mm)

        # Pose: desloca à frente e inclina "olhando para baixo"
        _xform_translate(prim_path, (float(forward_m), 0.0, float(height_m)))
        qw, qx, qy, qz = _quat_from_euler_xyz(0.0, -float(pitch_down_deg), 0.0)
        _xform_orient_quat(prim_path, (qw, qx, qy, qz))

        return cls.attach(prim_path, resolution)

    @classmethod
    def attach(cls, prim_path: str, resolution: Tuple[int, int] = None) -> "BevFrontDownCamera":
        res = resolution if resolution is not None else cls.resolution
        return BevFrontDownCamera(Camera(prim_path, res))
