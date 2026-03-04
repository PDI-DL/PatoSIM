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


import asyncio
import numpy as np
import os
import datetime
import tempfile
import glob
import math
from collections import OrderedDict

import omni.ext
import omni.ui as ui

from omni.ext.mobility_gen.utils.global_utils import save_stage
from omni.ext.mobility_gen.writer import Writer
from omni.ext.mobility_gen.inputs import GamepadDriver, KeyboardDriver
from omni.ext.mobility_gen.scenarios import SCENARIOS, Scenario
from omni.ext.mobility_gen.utils.global_utils import get_world
from omni.ext.mobility_gen.utils.global_utils import get_stage
from omni.ext.mobility_gen.utils.path_utils import PathHelper
from pxr import UsdGeom, Usd, Gf
from omni.ext.mobility_gen.robots import ROBOTS
from omni.ext.mobility_gen.config import Config
from omni.ext.mobility_gen.build import build_scenario_from_config

dev_scene_path = "/home/pdi_4/Documents/Documentos/bevlog-isaac/mundo_pallets.usd"  # --- DEV ---

if "MOBILITY_GEN_DATA" in os.environ:
    DATA_DIR = os.environ['MOBILITY_GEN_DATA']
else:
    DATA_DIR = os.path.expanduser("~/MobilityGenData")

RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
SCENARIOS_DIR = os.path.join(DATA_DIR, "scenarios")


class MobilityGenExtension(omni.ext.IExt):
    """
    MobilityGenExtension: Extension UI and runtime hooks.

    Responsibilities:
      - provide a small UI to build a scenario from a USD world, robot and scenario type
      - manage recording and pointcloud writer lifecycle
      - expose quick sensor controls (enable/disable cameras/lidar)
      - coordinate planning requests (Plan & Start Auto) via robot or scenario
    """

    def on_startup(self, ext_id):

        # Input drivers
        self.keyboard = KeyboardDriver.connect()
        self.gamepad = GamepadDriver.connect()

        # Runtime state
        self.scenario: Scenario = None
        self.config: Config = None

        self.count = 0
        self._physics_callback_name = f"scenario_physics_{id(self)}"
        self._physics_callback_registered = False
        self._physics_callback_world = None

        self.scenario_path: str | None = None
        self.cached_stage_path: str | None = None

        self.writer: Writer | None = None
        self.step: int = 0
        self.is_recording: bool = False
        self.recording_enabled: bool = False
        self.recording_time: float = 0.

        # Image provider for occupancy map preview
        self._occupancy_map_image_provider = omni.ui.ByteImageProvider()
        self._sensor_preview_image_provider = omni.ui.ByteImageProvider()
        self._lidar_preview_image_provider = omni.ui.ByteImageProvider()
        self._sensor_preview_camera_names = []
        self._sensor_preview_selected_index = 0
        self._sensor_preview_latest_rgb = {}
        self._sensor_preview_latest_camera_map = OrderedDict()
        self._sensor_preview_latest_pointcloud = {}
        self._sensor_pose_text = "Sensor poses: waiting for scenario..."
        self._sensor_pose_label = None
        self._sensor_preview_enabled = True
        self._preview_update_interval_frames = 8
        self._preview_frame_counter = 0
        self._lidar_preview_enabled = False
        self._lidar_preview_auto_range = True
        self._lidar_preview_manual_range_m = 8
        self._lidar_preview_point_size = 2
        self._lidar_preview_flip_x = False
        self._lidar_preview_flip_y = False
        self._lidar_preview_swap_xy = False
        self._lidar_preview_history_frames = 2
        self._lidar_preview_history_max_points_per_frame = 600
        self._lidar_preview_history = []
        self._lidar_preview_stats_text = "LiDAR stats: waiting for pointcloud..."
        self._lidar_preview_stats_label = None

        # Visualization window for occupancy map
        self._visualize_window = omni.ui.Window("MobilityGen - Occupancy Map", width=300, height=300)
        try:
            self._visualize_window.visible = True
        except Exception:
            pass
        with self._visualize_window.frame:
            self._occ_map_frame = ui.Frame()
            self._occ_map_frame.set_build_fn(self.build_occ_map_frame)

        # Low-resolution live preview window for RGB camera sensors.
        self._sensor_preview_target_width = 288
        self._sensor_preview_target_height = 162
        self._lidar_preview_target_size = 180
        self._sensor_preview_window = omni.ui.Window("MobilityGen - Sensor Preview", width=500, height=560)
        try:
            self._sensor_preview_window.visible = self._sensor_preview_enabled
        except Exception:
            pass
        with self._sensor_preview_window.frame:
            self._sensor_preview_frame = ui.Frame()
            self._sensor_preview_frame.set_build_fn(self._build_sensor_preview_frame)

        self._lidar_preview_window = omni.ui.Window("MobilityGen - LiDAR Preview", width=280, height=360)
        try:
            self._lidar_preview_window.visible = False
        except Exception:
            pass
        with self._lidar_preview_window.frame:
            self._lidar_preview_frame = ui.Frame()
            self._lidar_preview_frame.set_build_fn(self._build_lidar_preview_frame)
        try:
            blank = np.zeros(
                (self._sensor_preview_target_height, self._sensor_preview_target_width, 4),
                dtype=np.uint8,
            )
            self._sensor_preview_image_provider.set_bytes_data(
                list(blank.tobytes()),
                [self._sensor_preview_target_width, self._sensor_preview_target_height],
            )
            blank_lidar = np.zeros(
                (self._lidar_preview_target_size, self._lidar_preview_target_size, 4),
                dtype=np.uint8,
            )
            blank_lidar[:, :, 3] = 255
            self._lidar_preview_image_provider.set_bytes_data(
                list(blank_lidar.tobytes()),
                [self._lidar_preview_target_size, self._lidar_preview_target_size],
            )
        except Exception:
            pass
        self._lidar_preview_toggle_model = ui.SimpleBoolModel(self._lidar_preview_enabled)
        self._sensor_preview_toggle_model = ui.SimpleBoolModel(self._sensor_preview_enabled)
        self._lidar_preview_auto_range_model = ui.SimpleBoolModel(self._lidar_preview_auto_range)
        self._lidar_preview_range_model = ui.SimpleIntModel(self._lidar_preview_manual_range_m)
        self._lidar_preview_point_size_model = ui.SimpleIntModel(self._lidar_preview_point_size)
        self._lidar_preview_history_frames_model = ui.SimpleIntModel(self._lidar_preview_history_frames)
        self._lidar_preview_flip_x_model = ui.SimpleBoolModel(self._lidar_preview_flip_x)
        self._lidar_preview_flip_y_model = ui.SimpleBoolModel(self._lidar_preview_flip_y)
        self._lidar_preview_swap_xy_model = ui.SimpleBoolModel(self._lidar_preview_swap_xy)

        # discover available USD worlds in the working directory and DATA_DIR
        try:
            self._available_worlds = self._scan_worlds()
        except Exception:
            self._available_worlds = []

        self._teleop_window = omni.ui.Window("MobilityGen", width=300, height=300)

        with self._teleop_window.frame:
            with ui.VStack():
                with ui.VStack():
                    with ui.HStack():
                        ui.Label("USD Path / URL")
                        self.scene_usd_field_string_model = ui.SimpleStringModel()
                        self.scene_usd_field = ui.StringField(model=self.scene_usd_field_string_model, height=25)

                    with ui.HStack():
                        ui.Label("Scenario Type")
                        self.scenario_combo_box = ui.ComboBox(0, *SCENARIOS.names())

                    with ui.HStack():
                        ui.Label("Robot Type")
                        self.robot_combo_box = ui.ComboBox(0, *ROBOTS.names())

                    with ui.HStack(height=24):
                        ui.Label("Camera Preview")
                        ui.CheckBox(model=self._sensor_preview_toggle_model, width=22)
                        try:
                            self._sensor_preview_toggle_model.add_value_changed_fn(
                                self._on_sensor_preview_toggle_changed
                            )
                        except Exception:
                            pass
                        ui.Spacer(width=12)
                        ui.Label("LiDAR Preview")
                        ui.CheckBox(model=self._lidar_preview_toggle_model, width=22)
                        try:
                            self._lidar_preview_toggle_model.add_value_changed_fn(
                                self._on_lidar_preview_toggle_changed
                            )
                        except Exception:
                            pass
                
                    # -- Build button --
                    ui.Button("Build", clicked_fn=self.build_scenario)

                    # -- Quick parameter models --
                    # fallback internal format/index for environments where the
                    # pointcloud format ComboBox is not present in the UI
                    self._pc_format_items = ["npy", "ply", "pcd"]
                    self._pc_format_index = 0

                    # -- Quick Params frame: organized controls after Build --
                    with ui.Frame():
                        ui.Label("Quick Params")
                        with ui.HStack(height=40):
                            # Left column: fixed capture policy (always on)
                            with ui.VStack(width=220, spacing=4):
                                ui.Label("Record PointClouds: Always On")
                                ui.Label("Annotate BoundingBoxes: Always On")

                            # Right column: format and interval selectors
                            with ui.VStack(width=220, spacing=4):
                                with ui.HStack():
                                    ui.Label("Format")
                                    try:
                                        items = self._pc_format_items
                                        self._pc_format_combo = ui.ComboBox(self._pc_format_index, *items)
                                        try:
                                            self._pc_format_combo.model.add_value_changed_fn(lambda: setattr(self, '_pc_format_index', self._pc_format_combo.model.get_item_value_model().get_value_as_int()))
                                        except Exception:
                                            pass
                                    except Exception:
                                        # fallback label if combo fails
                                        ui.Label(self._pc_format_items[self._pc_format_index] if hasattr(self, '_pc_format_items') else "npy")

                                with ui.HStack():
                                    ui.Label("Interval (frames)")
                                    ui.Label("1 (always)")

                with ui.VStack():
                    self.recording_count_label = ui.Label("")
                    self.recording_dir_label = ui.Label(f"Output directory: {RECORDINGS_DIR}")
                    self.recording_name_label = ui.Label("")
                    self.recording_step_label = ui.Label("")

                    ui.Button("Reset", clicked_fn=self.reset)
                    with ui.HStack():
                        ui.Button("Start Recording", clicked_fn=self.enable_recording)
                        ui.Button("Stop Recording", clicked_fn=self.disable_recording)

        self.update_recording_count()
        self.clear_recording()

    def build_occ_map_frame(self):
        if self.scenario is not None:
            with ui.VStack():
                image_widget = ui.ImageWithProvider(
                    self._occupancy_map_image_provider
                )

    def draw_occ_map(self):
        if self.scenario is not None:
            image = self.scenario.occupancy_map.ros_image().copy().convert("RGBA")
            data = list(image.tobytes())
            self._occupancy_map_image_provider.set_bytes_data(data, [image.width, image.height])
            self._occ_map_frame.rebuild()

    def _build_sensor_preview_frame(self):
        with ui.VStack(spacing=4, height=0):
            with ui.HStack(height=26):
                ui.Label("Camera Preview")
                if len(self._sensor_preview_camera_names) > 0:
                    idx = min(self._sensor_preview_selected_index, len(self._sensor_preview_camera_names) - 1)
                    combo = ui.ComboBox(idx, *self._sensor_preview_camera_names)
                    self._sensor_preview_combo = combo
                    try:
                        combo.model.get_item_value_model().add_value_changed_fn(
                            self._on_sensor_preview_camera_changed
                        )
                    except Exception:
                        pass
                else:
                    self._sensor_preview_combo = None
                    ui.Label("(waiting for RGB data)")
            with ui.HStack(height=20):
                ui.Label("Resolution")
                ui.Label(f"{self._sensor_preview_target_width}x{self._sensor_preview_target_height}")
            with ui.HStack(height=self._sensor_preview_target_height + 10):
                ui.Spacer(width=6)
                ui.ImageWithProvider(
                    self._sensor_preview_image_provider,
                    width=self._sensor_preview_target_width,
                    height=self._sensor_preview_target_height,
                )
                ui.Spacer(width=6)
            ui.Spacer(height=6)
            ui.Label("Sensor Poses Relative To Robot")
            self._sensor_pose_label = ui.Label(self._sensor_pose_text)

    def _build_lidar_preview_frame(self):
        with ui.VStack(spacing=6, height=0):
            with ui.HStack(height=24):
                ui.Label("LiDAR Preview (Top-Down)")
                ui.Spacer(width=8)
                ui.Label(f"{self._lidar_preview_target_size}x{self._lidar_preview_target_size}")
            with ui.HStack(height=22):
                ui.Label("Auto Range")
                ui.CheckBox(model=self._lidar_preview_auto_range_model, width=22)
                try:
                    self._lidar_preview_auto_range_model.add_value_changed_fn(
                        self._on_lidar_preview_params_changed
                    )
                except Exception:
                    pass
                ui.Spacer(width=8)
                ui.Label("Range (m)")
                ui.IntField(model=self._lidar_preview_range_model, height=20, width=56)
                try:
                    self._lidar_preview_range_model.add_value_changed_fn(
                        self._on_lidar_preview_params_changed
                    )
                except Exception:
                    pass
            with ui.HStack(height=22):
                ui.Label("Point Size")
                ui.IntField(model=self._lidar_preview_point_size_model, height=20, width=56)
                try:
                    self._lidar_preview_point_size_model.add_value_changed_fn(
                        self._on_lidar_preview_params_changed
                    )
                except Exception:
                    pass
                ui.Spacer(width=8)
                ui.Label("History")
                ui.IntField(model=self._lidar_preview_history_frames_model, height=20, width=56)
                try:
                    self._lidar_preview_history_frames_model.add_value_changed_fn(
                        self._on_lidar_preview_params_changed
                    )
                except Exception:
                    pass
            with ui.HStack(height=22):
                ui.Label("Swap XY")
                ui.CheckBox(model=self._lidar_preview_swap_xy_model, width=22)
                try:
                    self._lidar_preview_swap_xy_model.add_value_changed_fn(
                        self._on_lidar_preview_params_changed
                    )
                except Exception:
                    pass
            with ui.HStack(height=22):
                ui.Label("Flip X")
                ui.CheckBox(model=self._lidar_preview_flip_x_model, width=22)
                try:
                    self._lidar_preview_flip_x_model.add_value_changed_fn(
                        self._on_lidar_preview_params_changed
                    )
                except Exception:
                    pass
                ui.Spacer(width=8)
                ui.Label("Flip Y")
                ui.CheckBox(model=self._lidar_preview_flip_y_model, width=22)
                try:
                    self._lidar_preview_flip_y_model.add_value_changed_fn(
                        self._on_lidar_preview_params_changed
                    )
                except Exception:
                    pass
            with ui.HStack(height=self._lidar_preview_target_size + 8):
                ui.Spacer(width=6)
                ui.ImageWithProvider(
                    self._lidar_preview_image_provider,
                    width=self._lidar_preview_target_size,
                    height=self._lidar_preview_target_size,
                )
                ui.Spacer(width=6)
            self._lidar_preview_stats_label = ui.Label(self._lidar_preview_stats_text)

    def _toggle_lidar_preview(self):
        self._set_lidar_preview_enabled(
            not bool(getattr(self, "_lidar_preview_enabled", False))
        )

    def _set_sensor_preview_enabled(self, enabled: bool):
        enabled = bool(enabled)
        self._sensor_preview_enabled = enabled
        try:
            self._sensor_preview_window.visible = enabled
        except Exception:
            pass
        try:
            model = getattr(self, "_sensor_preview_toggle_model", None)
            if model is not None and bool(model.as_bool) != enabled:
                model.set_value(enabled)
        except Exception:
            pass
        if enabled:
            try:
                scenario = getattr(self, "scenario", None)
                if scenario is not None:
                    scenario.enable_rgb_rendering()
            except Exception:
                pass
        else:
            try:
                blank = np.zeros(
                    (self._sensor_preview_target_height, self._sensor_preview_target_width, 4),
                    dtype=np.uint8,
                )
                self._sensor_preview_image_provider.set_bytes_data(
                    list(blank.tobytes()),
                    [self._sensor_preview_target_width, self._sensor_preview_target_height],
                )
            except Exception:
                pass

    def _set_lidar_preview_enabled(self, enabled: bool):
        enabled = bool(enabled)
        self._lidar_preview_enabled = enabled
        if not enabled:
            self._lidar_preview_history = []
        try:
            self._lidar_preview_window.visible = enabled
        except Exception:
            pass
        try:
            model = getattr(self, "_lidar_preview_toggle_model", None)
            if model is not None and bool(model.as_bool) != enabled:
                model.set_value(enabled)
        except Exception:
            pass
        if not enabled:
            try:
                blank_lidar = np.zeros(
                    (self._lidar_preview_target_size, self._lidar_preview_target_size, 4),
                    dtype=np.uint8,
                )
                blank_lidar[:, :, 3] = 255
                self._lidar_preview_image_provider.set_bytes_data(
                    list(blank_lidar.tobytes()),
                    [self._lidar_preview_target_size, self._lidar_preview_target_size],
                )
            except Exception:
                pass
        try:
            scenario = getattr(self, "scenario", None)
            if scenario is not None:
                scenario.set_pointcloud_enabled(enabled or (self.writer is not None))
        except Exception:
            pass

    def _on_lidar_preview_toggle_changed(self, model):
        try:
            enabled = bool(model.as_bool)
        except Exception:
            try:
                enabled = bool(model.get_value_as_bool())
            except Exception:
                enabled = False
        self._set_lidar_preview_enabled(enabled)

    def _on_sensor_preview_toggle_changed(self, model):
        try:
            enabled = bool(model.as_bool)
        except Exception:
            try:
                enabled = bool(model.get_value_as_bool())
            except Exception:
                enabled = False
        self._set_sensor_preview_enabled(enabled)

    def _on_lidar_preview_params_changed(self, *_args):
        try:
            self._lidar_preview_auto_range = bool(self._lidar_preview_auto_range_model.as_bool)
        except Exception:
            pass
        try:
            self._lidar_preview_manual_range_m = max(
                1, int(self._lidar_preview_range_model.as_int)
            )
        except Exception:
            pass
        try:
            self._lidar_preview_point_size = max(
                1, int(self._lidar_preview_point_size_model.as_int)
            )
        except Exception:
            pass
        try:
            self._lidar_preview_history_frames = max(
                1, int(self._lidar_preview_history_frames_model.as_int)
            )
            self._lidar_preview_history = self._lidar_preview_history[-self._lidar_preview_history_frames:]
        except Exception:
            pass
        try:
            self._lidar_preview_flip_x = bool(self._lidar_preview_flip_x_model.as_bool)
        except Exception:
            pass
        try:
            self._lidar_preview_flip_y = bool(self._lidar_preview_flip_y_model.as_bool)
        except Exception:
            pass
        try:
            self._lidar_preview_swap_xy = bool(self._lidar_preview_swap_xy_model.as_bool)
        except Exception:
            pass
        try:
            self._refresh_lidar_preview(self._sensor_preview_latest_pointcloud, update_history=False)
        except Exception:
            pass

    def _on_sensor_preview_camera_changed(self, *_args):
        try:
            if not hasattr(self, "_sensor_preview_combo"):
                return
            combo_model = self._sensor_preview_combo.model
            self._sensor_preview_selected_index = int(combo_model.get_item_value_model().get_value_as_int())
        except Exception:
            pass
        try:
            self._refresh_sensor_preview(self._sensor_preview_latest_camera_map)
        except Exception:
            pass

    @staticmethod
    def _resize_rgb_for_preview(image: np.ndarray, max_width: int = 320, max_height: int = 200):
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim != 3:
            return None

        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] >= 4:
            arr = arr[:, :, :4]
        elif arr.shape[2] == 2:
            arr = np.stack([arr[:, :, 0], arr[:, :, 1], arr[:, :, 0]], axis=2)

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        height, width = arr.shape[:2]
        if height <= 0 or width <= 0:
            return None

        scale = min(float(max_width) / float(width), float(max_height) / float(height), 1.0)
        if scale < 1.0:
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            x_idx = np.linspace(0, width - 1, new_w).astype(np.int32)
            y_idx = np.linspace(0, height - 1, new_h).astype(np.int32)
            arr = arr[np.ix_(y_idx, x_idx)]

        if arr.shape[2] == 3:
            alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=2)
        return arr

    def _compose_stereo_preview(self, left_image, right_image):
        """Compose left/right stereo images into a single side-by-side preview."""
        if left_image is None and right_image is None:
            return None

        target_w = int(self._sensor_preview_target_width)
        target_h = int(self._sensor_preview_target_height)
        half_w = max(1, target_w // 2)

        left_rgba = self._resize_rgb_for_preview(
            left_image, max_width=half_w - 4, max_height=target_h
        )
        right_rgba = self._resize_rgb_for_preview(
            right_image, max_width=half_w - 4, max_height=target_h
        )

        if left_rgba is None and right_rgba is None:
            return None

        canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255

        def _paste(img, x0, x1):
            if img is None:
                return
            ih, iw = img.shape[:2]
            if ih <= 0 or iw <= 0:
                return
            dst_w = max(1, x1 - x0)
            px = x0 + max(0, (dst_w - iw) // 2)
            py = max(0, (target_h - ih) // 2)
            px2 = min(target_w, px + iw)
            py2 = min(target_h, py + ih)
            canvas[py:py2, px:px2, :] = img[: py2 - py, : px2 - px, :]

        split = half_w
        _paste(left_rgba, 0, split)
        _paste(right_rgba, split, target_w)
        canvas[:, max(0, split - 1):min(target_w, split + 1), :3] = 80
        return canvas

    def _build_sensor_preview_camera_map(self, rgb_state: dict):
        """Group raw RGB buffers into user-facing camera previews."""
        camera_map = OrderedDict()
        if not isinstance(rgb_state, dict):
            return camera_map

        # Prefer explicit robot sensor handles first (stable even when the stage
        # has extra camera prims with similar names).
        try:
            scenario = getattr(self, "scenario", None)
            robot = getattr(scenario, "robot", None)
        except Exception:
            robot = None

        def _read_rgb(sensor_obj):
            if sensor_obj is None:
                return None
            try:
                buf = getattr(sensor_obj, "rgb", None)
                if buf is None:
                    return None
                if hasattr(buf, "get_value"):
                    return buf.get_value()
                return buf
            except Exception:
                return None

        def _preview_score_local(image) -> float:
            try:
                arr = np.asarray(image)
                if arr.ndim != 3:
                    return -1.0
                h, w = arr.shape[:2]
                if h <= 0 or w <= 0:
                    return -1.0
                step_h = max(1, h // 32)
                step_w = max(1, w // 32)
                sample = arr[::step_h, ::step_w, :3]
                return float(np.mean(sample))
            except Exception:
                return -1.0

        def _best_visible(*images):
            best = None
            best_score = -1.0
            for image in images:
                if image is None:
                    continue
                score = _preview_score_local(image)
                if score > best_score:
                    best_score = score
                    best = image
            return best

        if robot is not None:
            front = getattr(robot, "front_stereo", None)
            if front is None:
                front = getattr(robot, "front_camera", None)

            front_left = None
            front_right = None
            front_single = None
            try:
                if hasattr(front, "left"):
                    front_left = _read_rgb(front.left)
                if hasattr(front, "right"):
                    front_right = _read_rgb(front.right)
                front_single = _read_rgb(front)
            except Exception:
                pass
            front_img = _best_visible(front_left, front_right, front_single)
            front_stereo_pair = self._compose_stereo_preview(front_left, front_right)

            left_img = _read_rgb(getattr(robot, "fisheye_left", None))
            right_img = _read_rgb(getattr(robot, "fisheye_right", None))
            left_score = _preview_score_local(left_img)
            right_score = _preview_score_local(right_img)
            front_score = _preview_score_local(front_img)

            # Discard almost-black frames from direct handles so we can fallback
            # to rgb_state key matching below.
            if left_score < 2.0:
                left_img = None
            if right_score < 2.0:
                right_img = None
            if front_score < 2.0:
                front_img = None

            if left_img is not None:
                camera_map["fisheye_left"] = left_img
            if right_img is not None:
                camera_map["fisheye_right"] = right_img
            if front_stereo_pair is not None:
                camera_map["front_stereo"] = front_stereo_pair
            elif front_img is not None:
                camera_map["front_stereo"] = front_img
            if front_left is not None:
                camera_map["front_stereo_left"] = front_left
            if front_right is not None:
                camera_map["front_stereo_right"] = front_right

            if len(camera_map) > 0:
                return camera_map

        valid_items = [(k, v) for k, v in rgb_state.items() if v is not None]
        if len(valid_items) == 0:
            return camera_map

        def _preview_score(image) -> float:
            try:
                arr = np.asarray(image)
                if arr.ndim != 3:
                    return -1.0
                h, w = arr.shape[:2]
                if h <= 0 or w <= 0:
                    return -1.0
                # downsample for cheap scoring
                step_h = max(1, h // 32)
                step_w = max(1, w // 32)
                sample = arr[::step_h, ::step_w, :3]
                return float(np.mean(sample))
            except Exception:
                return -1.0

        def _find_camera(substrings: list[str], prefer_left: bool = False):
            matches = []
            for key, value in valid_items:
                lower = key.lower()
                if all(token in lower for token in substrings):
                    matches.append((key, value))
            if not matches:
                return None
            ordered = matches
            if prefer_left:
                left_first = []
                others = []
                for key, value in matches:
                    lower = key.lower()
                    if ".left." in lower or "camera_left" in lower or "left" in lower:
                        left_first.append((key, value))
                    else:
                        others.append((key, value))
                ordered = left_first + others

            # Prefer the candidate with visible content to avoid black preview when
            # duplicated streams exist and one of them is invalid/dark.
            best_value = None
            best_score = -1.0
            for _key, value in ordered:
                score = _preview_score(value)
                if score > best_score:
                    best_score = score
                    best_value = value
            return best_value

        def _first_non_none(*values):
            for value in values:
                if value is not None:
                    return value
            return None

        left_fisheye = _first_non_none(
            _find_camera(["fisheye", "left"]),
            _find_camera(["fisheye_left"]),
        )
        right_fisheye = _first_non_none(
            _find_camera(["fisheye", "right"]),
            _find_camera(["fisheye_right"]),
        )
        front_stereo = _first_non_none(
            _find_camera(["front_stereo"], prefer_left=True),
            _find_camera(["front_camera"], prefer_left=True),
            _find_camera(["stereo"], prefer_left=True),
        )

        if left_fisheye is not None:
            camera_map["fisheye_left"] = left_fisheye
        if right_fisheye is not None:
            camera_map["fisheye_right"] = right_fisheye
        if front_stereo is not None:
            camera_map["front_stereo"] = front_stereo

        # Fallback: if no canonical names were found, expose all RGB buffers.
        if len(camera_map) == 0:
            for key, value in sorted(valid_items, key=lambda kv: kv[0]):
                camera_map[key] = value

        return camera_map

    def _refresh_sensor_preview(self, camera_map: dict):
        if not isinstance(camera_map, dict):
            self._update_sensor_pose_preview_text()
            return

        # Read current ComboBox selection on every refresh to avoid stale index
        # when UI callbacks are skipped by the host.
        try:
            combo = getattr(self, "_sensor_preview_combo", None)
            if combo is not None:
                idx = int(combo.model.get_item_value_model().get_value_as_int())
                if idx >= 0:
                    self._sensor_preview_selected_index = idx
        except Exception:
            pass

        camera_names = list(camera_map.keys())
        if camera_names != self._sensor_preview_camera_names:
            self._sensor_preview_camera_names = camera_names
            if len(self._sensor_preview_camera_names) == 0:
                self._sensor_preview_selected_index = 0
            else:
                self._sensor_preview_selected_index = min(
                    self._sensor_preview_selected_index, len(self._sensor_preview_camera_names) - 1
                )
            try:
                self._sensor_preview_frame.rebuild()
            except Exception:
                pass

        if len(self._sensor_preview_camera_names) == 0:
            blank = np.zeros(
                (self._sensor_preview_target_height, self._sensor_preview_target_width, 4),
                dtype=np.uint8,
            )
            self._sensor_preview_image_provider.set_bytes_data(
                list(blank.tobytes()),
                [self._sensor_preview_target_width, self._sensor_preview_target_height],
            )
            self._update_sensor_pose_preview_text()
            return

        self._sensor_preview_selected_index = min(
            self._sensor_preview_selected_index,
            len(self._sensor_preview_camera_names) - 1,
        )
        selected_name = self._sensor_preview_camera_names[self._sensor_preview_selected_index]
        selected_image = camera_map.get(selected_name)
        preview_rgba = self._resize_rgb_for_preview(
            selected_image,
            max_width=self._sensor_preview_target_width,
            max_height=self._sensor_preview_target_height,
        )
        if preview_rgba is None:
            self._update_sensor_pose_preview_text()
            return
        self._sensor_preview_image_provider.set_bytes_data(
            list(preview_rgba.tobytes()),
            [int(preview_rgba.shape[1]), int(preview_rgba.shape[0])],
        )
        self._update_sensor_pose_preview_text()

    def _get_lidar_backend_status(self):
        try:
            scenario = getattr(self, "scenario", None)
            robot = getattr(scenario, "robot", None)
            lidar = getattr(robot, "lidar", None)
            status = getattr(getattr(lidar, "status", None), "value", None)
            return str(status) if status else "unknown"
        except Exception:
            return "unknown"

    def _select_lidar_preview_points(self, pointcloud_state: dict):
        if not isinstance(pointcloud_state, dict) or len(pointcloud_state) == 0:
            return None
        chosen = None
        for key, value in pointcloud_state.items():
            if value is None:
                continue
            if "lidar" in str(key).lower():
                chosen = value
                break
        if chosen is None:
            for _key, value in pointcloud_state.items():
                if value is not None:
                    chosen = value
                    break
        if chosen is None:
            return None
        try:
            pts = np.asarray(chosen)
        except Exception:
            return None
        if pts.ndim != 2 or pts.shape[0] <= 0 or pts.shape[1] < 2:
            return None
        pts = pts[:, :3] if pts.shape[1] >= 3 else pts[:, :2]
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.shape[0] <= 0:
            return None
        return np.asarray(pts, dtype=np.float32)

    def _format_lidar_preview_stats(
        self,
        *,
        status: str,
        current_points: int = 0,
        shown_points: int = 0,
        lidar_range: float | None = None,
        shape=None,
        x=None,
        y=None,
        z=None,
        history_frames: int = 0,
        note: str | None = None,
    ):
        lines = [
            "LiDAR stats",
            f"status: {status}",
        ]
        if note:
            lines.append(note)
        else:
            range_text = f"{lidar_range:.2f}m" if lidar_range is not None else "-"
            lines.append(
                f"current: {int(current_points)}  shown: {int(shown_points)}  history: {int(history_frames)}"
            )
            lines.append(f"range: {range_text}  shape: {shape}")
            if x is not None and y is not None and z is not None:
                lines.append(f"x: [{x[0]:.2f}, {x[1]:.2f}]")
                lines.append(f"y: [{y[0]:.2f}, {y[1]:.2f}]")
                lines.append(f"z: [{z[0]:.2f}, {z[1]:.2f}]")
        self._lidar_preview_stats_text = "\n".join(lines)

    def _build_lidar_preview_rgba(self, pointcloud_state: dict):
        backend_status = self._get_lidar_backend_status()

        size = int(getattr(self, "_lidar_preview_target_size", 220))
        canvas = np.zeros((size, size, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255
        canvas[:, :, :3] = 26

        # Draw axis guides and robot center.
        mid = size // 2
        canvas[mid, :, :3] = 55
        canvas[:, mid, :3] = 55
        robot_half = 3
        canvas[max(0, mid - robot_half):min(size, mid + robot_half + 1),
               max(0, mid - robot_half):min(size, mid + robot_half + 1), :3] = [220, 180, 60]
        self._format_lidar_preview_stats(status=backend_status, note="no pointcloud data")

        current_pts = self._select_lidar_preview_points(pointcloud_state)
        if current_pts is None:
            return canvas

        try:
            history_sets = [pts for pts in getattr(self, "_lidar_preview_history", []) if pts is not None and len(pts) > 0]
            draw_sets = history_sets + [current_pts]
            combined = np.concatenate(draw_sets, axis=0) if len(draw_sets) > 1 else current_pts

            x = combined[:, 0]
            y = combined[:, 1]
            z = combined[:, 2] if combined.shape[1] >= 3 else np.zeros_like(x)
            if bool(getattr(self, "_lidar_preview_swap_xy", False)):
                x, y = y, x
            if bool(getattr(self, "_lidar_preview_flip_x", False)):
                x = -x
            if bool(getattr(self, "_lidar_preview_flip_y", False)):
                y = -y
            d = np.sqrt(x * x + y * y)

            # Robust dynamic range (meters), clamped for stable visualization.
            if bool(getattr(self, "_lidar_preview_auto_range", True)):
                r95 = float(np.percentile(d, 95)) if d.size > 0 else 8.0
                lidar_range = float(np.clip(r95, 4.0, 20.0))
            else:
                lidar_range = float(max(1.0, int(getattr(self, "_lidar_preview_manual_range_m", 8))))
            scale = (size - 1) / (2.0 * lidar_range)

            # Top-down: +x forward (up in image), +y left (left in image).
            u = np.round(mid - y * scale).astype(np.int32)
            v = np.round(mid - x * scale).astype(np.int32)
            inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
            if not np.any(inside):
                self._format_lidar_preview_stats(
                    status=backend_status,
                    current_points=current_pts.shape[0],
                    shown_points=0,
                    lidar_range=lidar_range,
                    shape=tuple(combined.shape),
                    x=(float(np.min(x)), float(np.max(x))),
                    y=(float(np.min(y)), float(np.max(y))),
                    z=(float(np.min(z)), float(np.max(z))),
                    history_frames=len(history_sets),
                    note=f"0/{combined.shape[0]} points inside view",
                )
                return canvas

            offset = 0
            point_size = int(max(1, getattr(self, "_lidar_preview_point_size", 2)))
            half = max(0, point_size // 2)
            shown_points = 0
            for set_index, pts_set in enumerate(draw_sets):
                count = pts_set.shape[0]
                set_x = pts_set[:, 0]
                set_y = pts_set[:, 1]
                if bool(getattr(self, "_lidar_preview_swap_xy", False)):
                    set_x, set_y = set_y, set_x
                if bool(getattr(self, "_lidar_preview_flip_x", False)):
                    set_x = -set_x
                if bool(getattr(self, "_lidar_preview_flip_y", False)):
                    set_y = -set_y
                set_d = np.sqrt(set_x * set_x + set_y * set_y)
                set_u = np.round(mid - set_y * scale).astype(np.int32)
                set_v = np.round(mid - set_x * scale).astype(np.int32)
                set_inside = (set_u >= 0) & (set_u < size) & (set_v >= 0) & (set_v < size)
                if not np.any(set_inside):
                    offset += count
                    continue
                set_u = set_u[set_inside]
                set_v = set_v[set_inside]
                set_d = set_d[set_inside]
                shown_points += set_u.shape[0]
                if set_index < len(history_sets):
                    green = np.full(set_u.shape[0], 90, dtype=np.uint8)
                    blue = np.full(set_u.shape[0], 150, dtype=np.uint8)
                    red = np.full(set_u.shape[0], 45, dtype=np.uint8)
                else:
                    norm = np.clip(set_d / max(lidar_range, 1e-6), 0.0, 1.0)
                    green = np.clip((1.0 - norm) * 255.0, 50.0, 255.0).astype(np.uint8)
                    blue = np.full(set_u.shape[0], 255, dtype=np.uint8)
                    red = np.full(set_u.shape[0], 80, dtype=np.uint8)
                for px, py, p_r, p_g, p_b in zip(set_u, set_v, red, green, blue):
                    x0 = max(0, px - half)
                    x1 = min(size, px + half + 1)
                    y0 = max(0, py - half)
                    y1 = min(size, py + half + 1)
                    canvas[y0:y1, x0:x1, 0] = p_r
                    canvas[y0:y1, x0:x1, 1] = p_g
                    canvas[y0:y1, x0:x1, 2] = p_b
                offset += count
            self._format_lidar_preview_stats(
                status=backend_status,
                current_points=current_pts.shape[0],
                shown_points=shown_points,
                lidar_range=lidar_range,
                shape=tuple(combined.shape),
                x=(float(np.min(x)), float(np.max(x))),
                y=(float(np.min(y)), float(np.max(y))),
                z=(float(np.min(z)), float(np.max(z))),
                history_frames=len(history_sets),
            )
            return canvas
        except Exception:
            self._format_lidar_preview_stats(status=backend_status, note="preview rendering failed")
            return canvas

    def _refresh_lidar_preview(self, pointcloud_state: dict, update_history: bool = True):
        try:
            if update_history:
                pts = self._select_lidar_preview_points(pointcloud_state)
                if pts is not None and pts.shape[0] > 0:
                    max_points = int(max(200, getattr(self, "_lidar_preview_history_max_points_per_frame", 1500)))
                    if pts.shape[0] > max_points:
                        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int32)
                        pts = pts[idx]
                    self._lidar_preview_history.append(np.asarray(pts, dtype=np.float32))
                    keep = int(max(1, getattr(self, "_lidar_preview_history_frames", 4)))
                    self._lidar_preview_history = self._lidar_preview_history[-keep:]
            rgba = self._build_lidar_preview_rgba(pointcloud_state)
            self._lidar_preview_image_provider.set_bytes_data(
                list(rgba.tobytes()),
                [int(rgba.shape[1]), int(rgba.shape[0])],
            )
            if self._lidar_preview_stats_label is not None:
                self._lidar_preview_stats_label.text = self._lidar_preview_stats_text
        except Exception:
            pass

    def _resolve_sensor_mount_paths_for_preview(self, robot) -> OrderedDict:
        paths = OrderedDict()
        if robot is None:
            return paths

        def _sensor_prim(sensor_obj):
            if sensor_obj is None:
                return None
            path = getattr(sensor_obj, "_prim_path", None)
            if isinstance(path, str) and path:
                return path
            return None

        # Front stereo base path.
        front = getattr(robot, "front_stereo", None)
        if front is None:
            front = getattr(robot, "front_camera", None)
        front_path = _sensor_prim(front)
        if front_path is None and front is not None:
            left = getattr(front, "left", None)
            right = getattr(front, "right", None)
            left_path = _sensor_prim(left)
            right_path = _sensor_prim(right)
            if isinstance(left_path, str) and "/left/camera_left" in left_path:
                front_path = left_path.rsplit("/left/camera_left", 1)[0]
            elif isinstance(right_path, str) and "/right/camera_right" in right_path:
                front_path = right_path.rsplit("/right/camera_right", 1)[0]
            elif isinstance(left_path, str):
                front_path = os.path.dirname(os.path.dirname(left_path))
            elif isinstance(right_path, str):
                front_path = os.path.dirname(os.path.dirname(right_path))
        if isinstance(front_path, str) and front_path:
            paths["front_stereo"] = front_path

        left_path = _sensor_prim(getattr(robot, "fisheye_left", None))
        if isinstance(left_path, str) and left_path:
            if left_path.endswith("/camera"):
                left_path = left_path.rsplit("/camera", 1)[0]
            paths["fisheye_left"] = left_path

        right_path = _sensor_prim(getattr(robot, "fisheye_right", None))
        if isinstance(right_path, str) and right_path:
            if right_path.endswith("/camera"):
                right_path = right_path.rsplit("/camera", 1)[0]
            paths["fisheye_right"] = right_path

        lidar_path = _sensor_prim(getattr(robot, "lidar", None))
        if isinstance(lidar_path, str) and lidar_path:
            paths["lidar"] = lidar_path

        return paths

    @staticmethod
    def _quat_wxyz_to_euler_xyz_deg(quat_wxyz):
        if quat_wxyz is None or len(quat_wxyz) != 4:
            return None
        qw, qx, qy, qz = [float(v) for v in quat_wxyz]

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll_x = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch_y = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch_y = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_z = math.atan2(siny_cosp, cosy_cosp)

        return (
            math.degrees(roll_x),
            math.degrees(pitch_y),
            math.degrees(yaw_z),
        )

    @staticmethod
    def _fmt_xyz(values) -> str:
        if values is None or len(values) != 3:
            return "n/a"
        return f"({float(values[0]):+.2f}, {float(values[1]):+.2f}, {float(values[2]):+.2f})"

    def _get_sensor_config_pose(self, robot, sensor_name: str):
        mapping = {
            "front_stereo": ("front_camera_translation", "front_camera_rotation"),
            "fisheye_left": ("fisheye_left_translation", "fisheye_left_rotation"),
            "fisheye_right": ("fisheye_right_translation", "fisheye_right_rotation"),
            "lidar": ("lidar_translation", "lidar_rotation"),
        }
        translation_attr, rotation_attr = mapping.get(sensor_name, (None, None))
        config_translation = getattr(robot, translation_attr, None) if translation_attr else None
        config_rotation = getattr(robot, rotation_attr, None) if rotation_attr else None
        return config_translation, config_rotation, translation_attr, rotation_attr

    @staticmethod
    def _fmt_xyz_precise(values) -> str:
        if values is None or len(values) != 3:
            return "n/a"
        return f"({float(values[0]):.3f}, {float(values[1]):.3f}, {float(values[2]):.3f})"

    def _read_local_sensor_pose_from_prim(self, prim):
        if prim is None or not prim.IsValid():
            return None, None

        local_translation = None
        local_quat_wxyz = None
        rotate_xyz = None

        try:
            xf = UsdGeom.Xformable(prim)
            for op in xf.GetOrderedXformOps():
                op_type = op.GetOpType()
                try:
                    value = op.Get()
                except Exception:
                    value = None

                if op_type == UsdGeom.XformOp.TypeTranslate and value is not None:
                    local_translation = (float(value[0]), float(value[1]), float(value[2]))
                elif op_type == UsdGeom.XformOp.TypeOrient and value is not None:
                    imag = value.GetImaginary()
                    local_quat_wxyz = (
                        float(value.GetReal()),
                        float(imag[0]),
                        float(imag[1]),
                        float(imag[2]),
                    )
                elif op_type == UsdGeom.XformOp.TypeRotateXYZ and value is not None:
                    rotate_xyz = (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return local_translation, None

        if rotate_xyz is not None:
            return local_translation, rotate_xyz
        return local_translation, self._quat_wxyz_to_euler_xyz_deg(local_quat_wxyz)

    def _update_sensor_pose_preview_text(self):
        scenario = getattr(self, "scenario", None)
        robot = getattr(scenario, "robot", None) if scenario is not None else None
        if robot is None:
            text = "Sensor poses: no active scenario."
            self._sensor_pose_text = text
            try:
                if self._sensor_pose_label is not None:
                    self._sensor_pose_label.text = text
            except Exception:
                pass
            return

        stage = get_stage()
        if stage is None:
            return

        robot_path = getattr(robot, "prim_path", "/World/robot")
        robot_prim = stage.GetPrimAtPath(robot_path)
        if robot_prim is None or not robot_prim.IsValid():
            text = f"Sensor poses: robot prim not found ({robot_path})"
            self._sensor_pose_text = text
            try:
                if self._sensor_pose_label is not None:
                    self._sensor_pose_label.text = text
            except Exception:
                pass
            return

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        try:
            robot_world = xform_cache.GetLocalToWorldTransform(robot_prim)
            robot_world_inv = robot_world.GetInverse()
        except Exception:
            return

        lines = [f"Robot: {robot_path}"]
        sensor_paths = self._resolve_sensor_mount_paths_for_preview(robot)
        if len(sensor_paths) == 0:
            lines.append("No sensors found on robot object.")
        else:
            for sensor_name, prim_path in sensor_paths.items():
                prim = stage.GetPrimAtPath(prim_path)
                if prim is None or not prim.IsValid():
                    lines.append(f"{sensor_name}: prim not found ({prim_path})")
                    continue
                try:
                    config_translation, config_rotation, translation_attr, rotation_attr = self._get_sensor_config_pose(robot, sensor_name)
                    local_translation, local_rotation = self._read_local_sensor_pose_from_prim(prim)
                    sensor_world = xform_cache.GetLocalToWorldTransform(prim)
                    rel = robot_world_inv * sensor_world
                    rel_t = rel.ExtractTranslation()
                    world_t = sensor_world.ExtractTranslation()
                    lines.append(f"{sensor_name}:")
                    lines.append(f"  prim: {prim_path}")
                    lines.append(f"  config.translation: {self._fmt_xyz(config_translation)}")
                    lines.append(f"  config.rotation_xyz_deg: {self._fmt_xyz(config_rotation)}")
                    lines.append(
                        f"  stage.local_translation: "
                        f"{self._fmt_xyz(local_translation)}"
                    )
                    lines.append(
                        f"  stage.local_rotation_xyz_deg: "
                        f"{self._fmt_xyz(local_rotation)}"
                    )
                    lines.append(
                        f"  robot_relative_translation: "
                        f"({rel_t[0]:+.2f}, {rel_t[1]:+.2f}, {rel_t[2]:+.2f})"
                    )
                    lines.append(
                        f"  world_translation: "
                        f"({world_t[0]:+.2f}, {world_t[1]:+.2f}, {world_t[2]:+.2f})"
                    )
                    if rotation_attr is not None:
                        lines.append(
                            f"  script.rotation_hint: "
                            f"{rotation_attr} = {self._fmt_xyz_precise(local_rotation)}"
                        )
                    if translation_attr is not None:
                        lines.append(
                            f"  script.translation_hint: "
                            f"{translation_attr} = {self._fmt_xyz_precise(local_translation)}"
                        )
                except Exception:
                    lines.append(f"{sensor_name}: failed to compute transform ({prim_path})")

        text = "\n".join(lines)
        self._sensor_pose_text = text
        try:
            if self._sensor_pose_label is not None:
                self._sensor_pose_label.text = text
        except Exception:
            pass


    def update_recording_count(self):
        num_recordings = len(glob.glob(os.path.join(RECORDINGS_DIR, "*")))
        self.recording_count_label.text = f"Number of recordings: {num_recordings}"

    # ------- UI -> Config helper -------
    def create_config(self):
        # Minimal, clear create_config: read UI values and return Config
        try:
            scenario_type = list(SCENARIOS.names())[self.scenario_combo_box.model.get_item_value_model().get_value_as_int()]
        except Exception:
            scenario_type = list(SCENARIOS.names())[0]
        try:
            robot_type = list(ROBOTS.names())[self.robot_combo_box.model.get_item_value_model().get_value_as_int()]
        except Exception:
            robot_type = list(ROBOTS.names())[0]

        scene_path = self.scene_usd_field_string_model.as_string
        scene_path = dev_scene_path if  scene_path == "" else scene_path
        

        
        config = Config(
            scenario_type=scenario_type,
            robot_type=robot_type,
            scene_usd=scene_path,
        )
        return config
    
    def scenario_type(self):
        index = self.scenario_combo_box.model.get_item_value_model().get_value_as_int()
        return SCENARIOS.get_index(index)
    
    def on_shutdown(self):
        # Defensive shutdown: some Kit shutdown sequences remove the world
        # before extensions are asked to shutdown, so `get_world()` may
        # return None. Guard all disconnect/remove calls to avoid raising
        # during extension shutdown.
        try:
            if hasattr(self, 'keyboard') and self.keyboard is not None:
                try:
                    self.keyboard.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(self, 'gamepad') and self.gamepad is not None:
                try:
                    self.gamepad.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            world = get_world()
            if world is not None:
                self._detach_physics_callback(world)
        except Exception:
            pass

    def start_new_recording(self):
        self._enable_recording_modalities()
        recording_name = datetime.datetime.now().isoformat()
        recording_path = os.path.join(RECORDINGS_DIR, recording_name)
        writer = Writer(recording_path)
        writer.write_config(self.config)
        writer.write_occupancy_map(self.scenario.occupancy_map)
        writer.copy_stage(self.cached_stage_path)
        self.step = 0
        self.recording_time = 0.
        self.recording_name_label.text = f"Current recording name: {recording_name}"
        self.recording_step_label.text = f"Current recording duration: {self.recording_time:.2f}s"
        self.writer = writer
        self.update_recording_count()
    
    def clear_recording(self):
        self.writer = None
        self.recording_name_label.text = "Current recording name: "
        self.recording_step_label.text = "Current recording duration: "

    def clear_scenario(self):
        # Stop simulation before mutating/replacing stages to avoid native
        # PhysX/StageUpdate crashes during rapid rebuilds.
        try:
            world = get_world()
            if world is not None:
                try:
                    world.stop()
                except Exception:
                    try:
                        world.pause()
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self._detach_physics_callback(get_world())
        except Exception:
            pass
        self.scenario = None
        self.cached_stage_path = None

    def _remove_physics_callback_safe(self, world, name: str) -> bool:
        """Try one compatible API shape without double-removing."""
        if world is None:
            return False
        try:
            world.remove_physics_callback(name)
            return True
        except TypeError:
            # Older API requires callback function argument.
            pass
        except Exception as exc:
            if "doesn't exist" in str(exc):
                return False
        try:
            world.remove_physics_callback(name, self.on_physics)
            return True
        except Exception:
            return False

    def _detach_physics_callback(self, world):
        """Remove physics callback using both old/new Isaac APIs."""
        target_world = self._physics_callback_world if self._physics_callback_world is not None else world
        if target_world is None:
            return
        if not getattr(self, "_physics_callback_registered", False):
            return
        name = getattr(self, "_physics_callback_name", "scenario_physics")
        self._remove_physics_callback_safe(target_world, name)
        self._physics_callback_registered = False
        self._physics_callback_world = None

    def _attach_physics_callback(self, world) -> bool:
        """Register the scenario physics callback on the current world instance."""
        if world is None:
            return False
        self._detach_physics_callback(world)
        try:
            world.add_physics_callback(self._physics_callback_name, self.on_physics)
            self._physics_callback_registered = True
            self._physics_callback_world = world
            return True
        except Exception as exc:
            self._physics_callback_registered = False
            self._physics_callback_world = None
            print(
                f"[MobilityGenExtension] failed to register physics callback "
                f"'{self._physics_callback_name}': {exc}"
            )
            return False

    @staticmethod
    def _to_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, dict):
            return {str(k): MobilityGenExtension._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [MobilityGenExtension._to_jsonable(v) for v in value]
        return value

    def _extract_semantic_state_from_common(self, state_dict: dict) -> dict:
        semantic_state = {}
        for key, value in state_dict.items():
            if not isinstance(key, str):
                continue
            if key.endswith("segmentation_info") or key.endswith("instance_id_segmentation_info"):
                semantic_state[key] = self._to_jsonable(value)
        return semantic_state

    def _enable_recording_modalities(self):
        if self.scenario is None:
            return
        try:
            self.scenario.enable_rgb_rendering()
        except Exception:
            pass
        try:
            self.scenario.enable_segmentation_rendering()
        except Exception:
            pass
        try:
            self.scenario.enable_instance_id_segmentation_rendering()
        except Exception:
            pass
        try:
            self.scenario.enable_depth_rendering()
        except Exception:
            pass
        try:
            self.scenario.enable_normals_rendering()
        except Exception:
            pass

    def enable_recording(self):
        if not self.recording_enabled:
            if self.scenario is not None:
                self._enable_recording_modalities()
                self.start_new_recording()
            self.recording_enabled = True

    def disable_recording(self):
        self.recording_enabled = False
        self.clear_recording()

    def reset(self):
        self.writer = None
        scenario = getattr(self, "scenario", None)
        if scenario is None:
            return
        scenario.reset()
        if self.recording_enabled:
            self.start_new_recording()

    def on_physics(self, step_size: int):
        scenario = getattr(self, "scenario", None)
        if scenario is None:
            return

        if scenario is not None:
            sensor_preview_enabled = bool(getattr(self, "_sensor_preview_enabled", True))
            lidar_preview_enabled = bool(getattr(self, "_lidar_preview_enabled", False))
            need_pointcloud_state = lidar_preview_enabled or (self.writer is not None)
            try:
                scenario.set_pointcloud_enabled(need_pointcloud_state)
            except Exception:
                pass

            is_alive = scenario.step(step_size)

            if not is_alive:
                self.reset()

            self._preview_frame_counter += 1
            preview_interval = max(1, int(getattr(self, "_preview_update_interval_frames", 2)))
            should_update_preview = (self._preview_frame_counter % preview_interval) == 0

            rgb_state_for_preview = {}
            pointcloud_state_for_preview = None
            need_rgb_state = ((should_update_preview and sensor_preview_enabled) or (self.writer is not None))
            if need_rgb_state:
                try:
                    rgb_state_for_preview = scenario.state_dict_rgb()
                except Exception:
                    rgb_state_for_preview = {}

            if should_update_preview and lidar_preview_enabled:
                try:
                    pointcloud_state_for_preview = scenario.state_dict_pointcloud_preview()
                    if isinstance(pointcloud_state_for_preview, dict) and len(pointcloud_state_for_preview) > 0:
                        self._sensor_preview_latest_pointcloud = pointcloud_state_for_preview
                except Exception:
                    pointcloud_state_for_preview = None

            if should_update_preview and sensor_preview_enabled:
                if isinstance(rgb_state_for_preview, dict) and len(rgb_state_for_preview) > 0:
                    self._sensor_preview_latest_rgb = rgb_state_for_preview
                camera_input = (
                    rgb_state_for_preview
                    if isinstance(rgb_state_for_preview, dict) and len(rgb_state_for_preview) > 0
                    else self._sensor_preview_latest_rgb
                )
                self._sensor_preview_latest_camera_map = self._build_sensor_preview_camera_map(camera_input)
                try:
                    self._refresh_sensor_preview(self._sensor_preview_latest_camera_map)
                except Exception:
                    pass
                if lidar_preview_enabled:
                    try:
                        lidar_input = (
                            pointcloud_state_for_preview
                            if isinstance(pointcloud_state_for_preview, dict) and len(pointcloud_state_for_preview) > 0
                            else self._sensor_preview_latest_pointcloud
                        )
                        self._refresh_lidar_preview(lidar_input)
                    except Exception:
                        pass
            elif should_update_preview:
                if lidar_preview_enabled:
                    try:
                        lidar_input = (
                            pointcloud_state_for_preview
                            if isinstance(pointcloud_state_for_preview, dict) and len(pointcloud_state_for_preview) > 0
                            else self._sensor_preview_latest_pointcloud
                        )
                        self._refresh_lidar_preview(lidar_input)
                    except Exception:
                        pass
            else:
                try:
                    if sensor_preview_enabled:
                        self._update_sensor_pose_preview_text()
                except Exception:
                    pass
            
            if self.writer is not None:
                state_dict_common = scenario.state_dict_common()
                self.writer.write_state_dict_common(state_dict_common, step=self.step)

                # Always persist rendered camera modalities while recording.
                try:
                    self.writer.write_state_dict_rgb(rgb_state_for_preview, step=self.step)
                except Exception:
                    pass
                try:
                    self.writer.write_state_dict_segmentation(scenario.state_dict_segmentation(), step=self.step)
                except Exception:
                    pass
                try:
                    self.writer.write_state_dict_instance_id_segmentation(
                        scenario.state_dict_instance_id_segmentation(), step=self.step
                    )
                except Exception:
                    pass
                try:
                    self.writer.write_state_dict_depth(scenario.state_dict_depth(), step=self.step)
                except Exception:
                    pass
                try:
                    self.writer.write_state_dict_normals(scenario.state_dict_normals(), step=self.step)
                except Exception:
                    pass

                # Always persist pointclouds using the configured format/interval.
                interval = 1
                if (self.step % interval) == 0:
                    if isinstance(pointcloud_state_for_preview, dict):
                        state_pc = pointcloud_state_for_preview
                    else:
                        state_pc = scenario.state_dict_pointcloud()
                    fmt = "npy"
                    try:
                        fmt = self._pc_format_items[self._pc_format_index]
                    except Exception:
                        fmt = "npy"
                    try:
                        self.writer.write_state_dict_pointcloud(state_pc, step=self.step, save_format=fmt)
                    except Exception:
                        pass

                    # Persist per-sensor metadata (pose) next to the pointcloud
                    try:
                        metadata = {}
                        modules = scenario.named_modules()
                        for full_name, arr_value in state_pc.items():
                            if "." in full_name:
                                module_name = full_name.rsplit(".", 1)[0]
                            else:
                                module_name = full_name
                            module = modules.get(module_name, None)
                            if module is None:
                                continue
                            pos = None
                            ori = None
                            try:
                                if hasattr(module, "position") and module.position.get_value() is not None:
                                    pos = module.position.get_value()
                            except Exception:
                                pos = None
                            try:
                                if hasattr(module, "orientation") and module.orientation.get_value() is not None:
                                    ori = module.orientation.get_value()
                            except Exception:
                                ori = None

                            if (pos is None or ori is None) and hasattr(module, "_xform_prim"):
                                try:
                                    p, o = module._xform_prim.get_world_pose()
                                    if pos is None:
                                        pos = p
                                    if ori is None:
                                        ori = o
                                except Exception:
                                    pass

                            fields = None
                            try:
                                if arr_value is not None:
                                    a = np.asarray(arr_value)
                                    if a.ndim == 2:
                                        ncol = a.shape[1]
                                        if ncol == 3:
                                            fields = ["x", "y", "z"]
                                        elif ncol == 4:
                                            fields = ["x", "y", "z", "intensity"]
                                        elif ncol == 6:
                                            fields = ["x", "y", "z", "r", "g", "b"]
                                        elif ncol == 7:
                                            fields = ["x", "y", "z", "r", "g", "b", "intensity"]
                                        else:
                                            fields = ["x", "y", "z"]
                            except Exception:
                                fields = None

                            if pos is not None or ori is not None or fields is not None:
                                metadata[module_name] = {
                                    "position": None if pos is None else [float(x) for x in list(pos)],
                                    "orientation": None if ori is None else [float(x) for x in list(ori)],
                                    "prim_path": getattr(module, "_prim_path", None),
                                    "fields": fields,
                                }
                        if metadata:
                            self.writer.write_pointcloud_metadata(metadata, step=self.step)
                    except Exception:
                        pass

                # Always write bounding-box annotations and semantic payload.
                try:
                    annotations = self._gather_annotations(self.step)
                    payload = dict(annotations)
                    payload["step"] = int(self.step)
                    payload["semantic"] = self._extract_semantic_state_from_common(state_dict_common)
                    if not isinstance(payload.get("bboxes2d"), list):
                        payload["bboxes2d"] = []
                    if not isinstance(payload.get("bboxes3d"), list):
                        payload["bboxes3d"] = []
                    if not isinstance(payload.get("classes"), list):
                        payload["classes"] = []
                    if not isinstance(payload.get("semantic"), dict):
                        payload["semantic"] = {}
                    self.writer.write_annotations(payload, step=self.step)
                except Exception:
                    pass
                self.step += 1
                self.recording_time += step_size
                if self.step % 15 == 0:
                    self.recording_step_label.text = f"Current recording duration: {self.recording_time:.2f}s"

                # handle recording-for-N-frames feature (used by 'Record 30 frames & Play')
                try:
                    cnt = getattr(self, '_record_30_remaining', None)
                    if cnt is not None:
                        if cnt > 0:
                            self._record_30_remaining = cnt - 1
                            if self._record_30_remaining <= 0:
                                try:
                                    self.disable_recording()
                                except Exception:
                                    pass
                                try:
                                    self._open_pc_player()
                                except Exception:
                                    pass
                                self._record_30_remaining = None
                except Exception:
                    pass

    @staticmethod
    def _semantic_type_from_attr_prefix(prefix: str) -> str:
        if "_" in prefix:
            return prefix.split("_", 1)[0]
        return prefix

    def _extract_semantic_labels_from_prim(self, prim) -> dict:
        semantic_labels = {}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if not name.startswith("semantic:") or not name.endswith(":params:semanticData"):
                continue
            raw_value = attr.Get()
            if raw_value is None:
                continue
            prefix = name[len("semantic:") : -len(":params:semanticData")]
            type_attr = prim.GetAttribute(f"semantic:{prefix}:params:semanticType")
            semantic_type = None
            if type_attr is not None:
                semantic_type = type_attr.Get()
            if semantic_type is None:
                semantic_type = self._semantic_type_from_attr_prefix(prefix)
            semantic_labels[str(semantic_type)] = str(raw_value)
        return semantic_labels

    def _gather_annotations(self, step: int) -> dict:
        """Collect 3D and 2D bounding boxes for prims in the stage.

        3D boxes are returned as 8-corner lists in world coordinates. 2D
        boxes are projected into the first camera found on the stage and
        reported as [xmin, ymin, xmax, ymax] in pixel coordinates. Classes
        are defaulted to the prim name if no explicit metadata is present.
        This is a best-effort helper intended for synthetic data generation.
        """
        annotations = {"bboxes2d": [], "bboxes3d": [], "classes": []}
        stage = get_stage()
        if stage is None:
            return annotations

        # Build bbox cache and xform cache
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # Find a camera to project to (first Camera prim)
        camera_prim = None
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Camera":
                camera_prim = prim
                break

        # Determine image size and camera intrinsics if camera found
        image_w, image_h = 640, 480
        cam_params = None
        if camera_prim is not None:
            cam = UsdGeom.Camera(camera_prim)
            try:
                focal = cam.GetFocalLengthAttr().Get()
                h_ap = cam.GetHorizontalApertureAttr().Get()
                v_ap = cam.GetVerticalApertureAttr().Get()
                # assume default image size; derive focal length in pixels
                fx = focal * image_w / h_ap if (h_ap and image_w) else 1.0
                fy = focal * image_h / v_ap if (v_ap and image_h) else fx
                cx = image_w / 2.0
                cy = image_h / 2.0
                cam_params = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
            except Exception:
                cam_params = None

        # Iterate prims and compute bounds
        for prim in stage.Traverse():
            if prim.IsPseudoRoot() or not prim.IsActive():
                continue
            try:
                bound = bbox_cache.ComputeWorldBound(prim)
                if bound.IsEmpty():
                    continue
                rng = bound.GetRange()
                min_pt = rng.GetMin()
                max_pt = rng.GetMax()
                # 8 corners
                corners = []
                for x in [min_pt[0], max_pt[0]]:
                    for y in [min_pt[1], max_pt[1]]:
                        for z in [min_pt[2], max_pt[2]]:
                            corners.append([float(x), float(y), float(z)])

                semantic_labels = self._extract_semantic_labels_from_prim(prim)
                class_name = semantic_labels.get("class", prim.GetName())
                # 3D annotation entry
                annotations["bboxes3d"].append({
                    "prim_path": prim.GetPath().pathString,
                    "class": class_name,
                    "semantic_labels": semantic_labels,
                    "corners": corners,
                })

                # 2D projection (first camera only)
                if cam_params is not None and camera_prim is not None:
                    # compute world-to-camera matrix
                    cam_world = xform_cache.GetLocalToWorldTransform(camera_prim)
                    try:
                        cam_mat = cam_world.GetInverse()
                    except Exception:
                        cam_mat = None
                    if cam_mat is not None:
                        # project corners
                        xs = []
                        ys = []
                        for c in corners:
                            # Gf.Matrix4d multiply with Gf.Vec4d
                            wc = Gf.Vec4d(c[0], c[1], c[2], 1.0)
                            cc = cam_mat * wc
                            # camera coordinates: cc[0], cc[1], cc[2]
                            if cc[2] <= 0.0:
                                # behind camera: skip point
                                continue
                            x_pix = (cam_params["fx"] * (cc[0] / cc[2])) + cam_params["cx"]
                            y_pix = (cam_params["fy"] * (cc[1] / cc[2])) + cam_params["cy"]
                            xs.append(float(x_pix))
                            ys.append(float(y_pix))
                        if len(xs) > 0 and len(ys) > 0:
                            xmin = max(0.0, min(xs))
                            ymin = max(0.0, min(ys))
                            xmax = min(image_w - 1.0, max(xs))
                            ymax = min(image_h - 1.0, max(ys))
                            annotations["bboxes2d"].append({
                                "prim_path": prim.GetPath().pathString,
                                "class": class_name,
                                "semantic_labels": semantic_labels,
                                "bbox": [xmin, ymin, xmax, ymax],
                            })
            except Exception:
                continue

        annotations["classes"] = sorted(
            {entry.get("class", "") for entry in annotations["bboxes3d"] if entry.get("class", "")}
        )
        return annotations

    def build_scenario(self):
        """Build the scenario async from UI-selected Config.

        This launches an async task which constructs the scenario (loads the
        USD stage, spawns the robot and sensors), resets the world, registers
        the physics callback and optionally starts recording.
        """

        async def _build_scenario_async():
            try:
                self.clear_recording()
                self.clear_scenario()
                # Let Kit process stop/remove events before creating a new stage.
                try:
                    await asyncio.sleep(0)
                except Exception:
                    pass
                try:
                    KeyboardDriver.disconnect()
                    self.keyboard = KeyboardDriver.connect()
                except Exception:
                    pass
                try:
                    GamepadDriver.disconnect()
                    self.gamepad = GamepadDriver.connect()
                except Exception:
                    pass

                config = self.create_config()

                self.config = config
                self.scenario = await build_scenario_from_config(config)
                try:
                    if bool(getattr(self, "_sensor_preview_enabled", True)):
                        self.scenario.enable_rgb_rendering()
                except Exception:
                    pass

                self.draw_occ_map()
                try:
                    self._visualize_window.visible = True
                except Exception:
                    pass
                try:
                    self._sensor_preview_window.visible = bool(getattr(self, "_sensor_preview_enabled", True))
                except Exception:
                    pass
                try:
                    self._lidar_preview_window.visible = bool(getattr(self, "_lidar_preview_enabled", False))
                except Exception:
                    pass
                
                world = get_world()
                self._detach_physics_callback(world)
                await world.reset_async()
                world = get_world()

                self.scenario.reset()
                if not self._attach_physics_callback(world):
                    raise RuntimeError("failed to attach scenario physics callback")
                try:
                    world.play()
                except Exception:
                    pass
                # sync UI with the newly created robot/sensors
                try:
                    self._sync_ui_with_robot()
                except Exception:
                    pass

                # cache stage
                self.cached_stage_path = os.path.join(tempfile.mkdtemp(), "stage.usd")
                save_stage(self.cached_stage_path)

                if self.recording_enabled:
                    self.start_new_recording()
            except Exception as exc:
                import traceback

                print(f"[MobilityGenExtension] build_scenario failed: {exc}")
                traceback.print_exc()
                try:
                    self.clear_recording()
                    self.clear_scenario()
                except Exception:
                    pass

        asyncio.ensure_future(_build_scenario_async())

    def _open_pc_player(self):
        # open a simple player window for pointcloud replay
        try:
            from omni.ext.mobility_gen.pointcloud_player import PointCloudPlayer
            # Ask user for recording path? Use last recording dir if present
            if self.writer is None:
                # pick most recent recording
                import glob, os
                recs = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*")))
                if len(recs) == 0:
                    return
                path = recs[-1]
            else:
                path = self.writer.path
            # instantiate player and store on self
            player = PointCloudPlayer(path)
            self._pc_player = player

            # Build a small UI window for player controls
            self._pc_player_window = ui.Window("PointCloud Player", width=400, height=300)
            with self._pc_player_window.frame:
                with ui.VStack():
                    with ui.HStack():
                        ui.Button("Play", clicked_fn=lambda: player.play(fps=float(self._pc_fps_model.get_value())))
                        ui.Button("Pause", clicked_fn=lambda: player.pause())
                        ui.Button("Prev", clicked_fn=lambda: player.step(forward=False))
                        ui.Button("Next", clicked_fn=lambda: player.step(forward=True))
                    with ui.HStack():
                        ui.Label("Frame")
                        self._pc_index_model = ui.SimpleIntModel(0)
                        ui.IntField(model=self._pc_index_model, height=25)
                        ui.Button("Go", clicked_fn=lambda: player.goto(int(self._pc_index_model.get_value())))
                    with ui.HStack():
                        ui.Label("FPS")
                        self._pc_fps_model = ui.SimpleIntModel(10)
                        ui.IntField(model=self._pc_fps_model, height=25, enabled=True)
                        ui.Button("Set FPS", clicked_fn=lambda: player.set_fps(float(self._pc_fps_model.get_value())))
                    ui.Label("Sensors (toggle visibility)")
                    # per-sensor toggles
                    for s in player.sensors:
                        m = ui.SimpleBoolModel(True)
                        # bind change to player visibility
                        def make_toggle(sensor_name, model):
                            def _on_change():
                                player.set_visibility(sensor_name, model.get_value())
                            return _on_change
                        cb = ui.CheckBox(model=m)
                        cb.model.add_value_changed_fn(make_toggle(s, m))
                        ui.Label(s)
            # start paused by default
            # player.play(fps=10.0)
        except Exception as e:
            print("Failed to start pointcloud player:", e)

    def _save_stage_now(self):
        try:
            if self.cached_stage_path is None:
                p = os.path.join(tempfile.mkdtemp(), "stage.usd")
            else:
                p = self.cached_stage_path
            save_stage(p)
            print(f"Stage saved to: {p}")
        except Exception as e:
            print("Failed to save stage:", e)

    def _toggle_occ_map(self):
        try:
            # try both APIs: set_visible or .visible attribute
            if hasattr(self._visualize_window, 'visible'):
                try:
                    self._visualize_window.visible = not self._visualize_window.visible
                    return
                except Exception:
                    pass
            if hasattr(self._visualize_window, 'set_visible'):
                try:
                    self._visualize_window.set_visible(not self._visualize_window.get_visible())
                    return
                except Exception:
                    pass
        except Exception:
            pass

    def _show_help(self):
        try:
            print("MobilityGen extension help:\n - Build Scenario: build the chosen scene and robot.\n - Plan & Start Auto: plan a path and start autonomous following using the scenario if available.\n - Record PointClouds: enable automatic pointcloud capture during physics stepping.\n")
        except Exception:
            pass

    def _scan_worlds(self):
        """Search common locations for USD/world files to populate the Worlds combo.

        Returns a list of absolute paths (may be empty).
        """
        exts = ('.usd', '.usda', '.usdc')
        roots = [os.getcwd()]
        # allow user-provided worlds dir (UI) to be used for scanning
        try:
            try:
                # prefer the UI-provided value if present
                v = None
                if hasattr(self, 'worlds_dir_field') and self.worlds_dir_field is not None:
                    try:
                        v = self.worlds_dir_field.model.get_value()
                    except Exception:
                        try:
                            v = self.worlds_dir_field.model.as_string
                        except Exception:
                            v = None
                if v:
                    v = os.path.expanduser(v)
                    if os.path.isdir(v):
                        roots.append(v)
            except Exception:
                pass
            if DATA_DIR and os.path.isdir(DATA_DIR):
                roots.append(DATA_DIR)
        except Exception:
            pass

        found = []
        for r in roots:
            try:
                for root, dirs, files in os.walk(r):
                    for f in files:
                        if f.lower().endswith(exts):
                            found.append(os.path.join(root, f))
            except Exception:
                continue

        # dedupe and sort
        uniq = []
        seen = set()
        for p in found:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return sorted(uniq)[:64]

    def _refresh_worlds(self):
        try:
            self._available_worlds = self._scan_worlds()
            items = self._available_worlds if len(self._available_worlds) > 0 else ["(none found)"]
            # rebuild combo model if possible
            try:
                # attempt to reset items by creating a new ComboBox model value
                self.worlds_combo_box = ui.ComboBox(0, *items)
            except Exception:
                pass
            print(f"Refreshed worlds, found {len(self._available_worlds)} files")
        except Exception as e:
            print("Failed to refresh worlds:", e)

    def _apply_selected_world(self):
        try:
            if not hasattr(self, 'worlds_combo_box'):
                return
            idx = self.worlds_combo_box.model.get_item_value_model().get_value_as_int()
            items = self._available_worlds if len(self._available_worlds) > 0 else ["(none found)"]
            if idx < 0 or idx >= len(items):
                return
            sel = items[idx]
            if sel and sel != "(none found)":
                self.scene_usd_field.model.set_value(sel)
        except Exception as e:
            print("Failed to apply selected world:", e)

    # --- sensor quick helpers ---
    def _enable_all_cameras(self):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            for name in ('front_stereo', 'fisheye_left', 'fisheye_right'):
                try:
                    mod = getattr(robot, name, None)
                    if mod is None:
                        continue
                    if hasattr(mod, 'enable_rgb_rendering'):
                        mod.enable_rgb_rendering()
                    else:
                        if hasattr(mod, 'left') and hasattr(mod.left, 'enable_rgb_rendering'):
                            mod.left.enable_rgb_rendering()
                        if hasattr(mod, 'right') and hasattr(mod.right, 'enable_rgb_rendering'):
                            mod.right.enable_rgb_rendering()
                except Exception:
                    pass
            print('Enabled all cameras (best-effort)')
        except Exception as e:
            print('Error enabling cameras:', e)

    def _disable_all_cameras(self):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            for name in ('front_stereo', 'fisheye_left', 'fisheye_right'):
                try:
                    mod = getattr(robot, name, None)
                    if mod is None:
                        continue
                    if hasattr(mod, 'disable_rendering'):
                        mod.disable_rendering()
                    else:
                        if hasattr(mod, 'left') and hasattr(mod.left, 'disable_rendering'):
                            mod.left.disable_rendering()
                        if hasattr(mod, 'right') and hasattr(mod.right, 'disable_rendering'):
                            mod.right.disable_rendering()
                except Exception:
                    pass
            print('Disabled all cameras (best-effort)')
        except Exception as e:
            print('Error disabling cameras:', e)

    def _enable_all_lidar(self):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            try:
                mod = getattr(robot, 'lidar', None)
                if mod is not None and hasattr(mod, 'enable_lidar'):
                    mod.enable_lidar()
            except Exception:
                pass
            print('Enabled lidar (best-effort)')
        except Exception as e:
            print('Error enabling lidar:', e)

    def _disable_all_lidar(self):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            try:
                mod = getattr(robot, 'lidar', None)
                if mod is not None and hasattr(mod, 'disable_lidar'):
                    mod.disable_lidar()
            except Exception:
                pass
            print('Disabled lidar (best-effort)')
        except Exception as e:
            print('Error disabling lidar:', e)

    # ------------------ Robot / sensor UI handlers ------------------
    def _apply_control_mode(self):
        """Apply the control mode chosen in the UI to the active robot.

        This attempts to call a `set_control_mode` method on the robot and
        falls back to setting a `control_mode` attribute if present.
        """
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            idx = self.control_mode_combo.model.get_item_value_model().get_value_as_int()
            mode = "manual" if idx == 0 else "auto"
            try:
                robot.set_control_mode(mode)
            except Exception:
                # fallback: try setting attribute
                try:
                    robot.control_mode = mode
                except Exception:
                    pass
        except Exception:
            pass

    def _plan_path_and_set_auto(self):
        """Plan a path using the robot or scenario and enable autonomous mode.

        Behavior:
          - If the active scenario exposes `target_path` (a Buffer-like), we
            prefer to set that so the scenario's built-in follower handles
            control. We also set a PathHelper for the scenario.
          - Otherwise, ask the robot to plan a path and set the robot into
            'auto' control mode.
        """
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            occupancy_map = getattr(self.scenario, 'occupancy_map', None)
            if occupancy_map is None:
                return

            # Prefer driving the scenario (if it exposes a target_path Buffer)
            # so that the scenario's path-following logic is reused. Otherwise
            # fall back to the robot convenience planner.
            try:
                # If the scenario has a target_path Buffer, set it and also
                # update its PathHelper so the follower can use it.
                if hasattr(self.scenario, 'target_path'):
                    try:
                        path_list = robot.plan_path_from_occupancy_map(occupancy_map)
                    except Exception:
                        # If robot planner fails, try letting the scenario generate
                        # its own random target if it exposes that method.
                        if hasattr(self.scenario, '_set_random_target_path'):
                            try:
                                self.scenario._set_random_target_path()
                                # ensure UI reflects auto mode
                                try:
                                    robot.set_control_mode('auto')
                                except Exception:
                                    try:
                                        robot.control_mode = 'auto'
                                    except Exception:
                                        pass
                                return
                            except Exception as e:
                                raise e
                        raise

                    import numpy as _np
                    arr = _np.asarray(path_list, dtype=_np.float32)
                    try:
                        # Buffer-like API
                        self.scenario.target_path.set_value(arr)
                    except Exception:
                        try:
                            self.scenario.target_path = arr
                        except Exception:
                            pass
                    try:
                        # update helper used by the scenario
                        self.scenario._helper = PathHelper(arr)
                    except Exception:
                        pass
                    try:
                        robot.set_control_mode('auto')
                    except Exception:
                        try:
                            robot.control_mode = 'auto'
                        except Exception:
                            pass
                    return

                # fallback: ask robot to plan and set its own auto-path
                path = robot.plan_path_from_occupancy_map(occupancy_map)
                try:
                    robot.set_control_mode('auto')
                except Exception:
                    try:
                        robot.control_mode = 'auto'
                    except Exception:
                        pass
            except Exception as e:
                print('Path planning failed:', e)
        except Exception:
            pass

    def _toggle_sensor(self, sensor_name: str, enabled: bool):
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return
            module = getattr(robot, sensor_name, None)
            if module is None:
                return
            # cameras: Camera instances expose enable_rgb_rendering()/disable_rendering()
            if enabled:
                try:
                    if hasattr(module, 'enable_rgb_rendering'):
                        module.enable_rgb_rendering()
                    else:
                        # stereo wrapper may have left/right
                        if hasattr(module, 'left') and hasattr(module.left, 'enable_rgb_rendering'):
                            module.left.enable_rgb_rendering()
                        if hasattr(module, 'right') and hasattr(module.right, 'enable_rgb_rendering'):
                            module.right.enable_rgb_rendering()
                except Exception:
                    pass
                try:
                    if hasattr(module, 'enable_lidar'):
                        module.enable_lidar()
                except Exception:
                    pass
            else:
                try:
                    if hasattr(module, 'disable_rendering'):
                        module.disable_rendering()
                    else:
                        if hasattr(module, 'left') and hasattr(module.left, 'disable_rendering'):
                            module.left.disable_rendering()
                        if hasattr(module, 'right') and hasattr(module.right, 'disable_rendering'):
                            module.right.disable_rendering()
                except Exception:
                    pass
                try:
                    if hasattr(module, 'disable_lidar'):
                        module.disable_lidar()
                except Exception:
                    pass
        except Exception:
            pass

    def _sync_ui_with_robot(self):
        """Read scenario.robot and update UI models to reflect current sensor/control state."""
        try:
            if self.scenario is None:
                return
            robot = getattr(self.scenario, 'robot', None)
            if robot is None:
                return

            # control mode
            try:
                mode = getattr(robot, 'control_mode', None)
                if mode is None and hasattr(robot, 'control_mode'):
                    mode = robot.control_mode
                if mode is not None:
                    idx = 0 if mode == 'manual' else 1
                    try:
                        self.control_mode_combo.model.set_value(idx)
                    except Exception:
                        pass
            except Exception:
                pass

            # sensors: set checkbox models if sensors exist
            try:
                self._sensor_front_model.set_value(bool(getattr(robot, 'front_stereo', None) is not None))
            except Exception:
                pass
            try:
                self._sensor_left_model.set_value(bool(getattr(robot, 'fisheye_left', None) is not None))
            except Exception:
                pass
            try:
                self._sensor_right_model.set_value(bool(getattr(robot, 'fisheye_right', None) is not None))
            except Exception:
                pass
            try:
                self._sensor_lidar_model.set_value(bool(getattr(robot, 'lidar', None) is not None))
            except Exception:
                pass
        except Exception:
            pass

    def _record_30_and_play(self):
        """Start recording for 30 frames and then open the PC player automatically."""
        try:
            if self.scenario is None:
                return
            # ensure writer exists and recording enabled
            if not self.recording_enabled:
                self.start_new_recording()
                self.recording_enabled = True
            # set countdown
            self._record_30_remaining = 30
        except Exception as e:
            print('Failed to start 30-frame recording:', e)
