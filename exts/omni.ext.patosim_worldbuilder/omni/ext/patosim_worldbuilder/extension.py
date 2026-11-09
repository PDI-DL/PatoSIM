# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""PatoSim World Builder: a lightweight, standalone extension for preparing
and validating underwater worlds -- scene loading, asset import, physics
collisions and sonar reflectance semantic labeling -- WITHOUT spawning a
robot. Meant to be run before a data-recording scenario built with the main
omni.ext.patosim extension.

Reuses the pure scene-building helpers already in omni.ext.patosim.build
(new: build_world_only/check_scene_semantics/check_scene_physics; existing:
_insert_dataset_object_into_scene/_enable_scene_collisions/
_apply_sonar_reflectivity_to_scene/list_dataset_object_assets) instead of
duplicating that logic. See docs/plano_upgrade_simulacao_subaquatica.md and
docs/relatorio_implementacao_upgrade_underwater_v2.md in the repo root.
"""

import asyncio
import os
import sys
import traceback

import carb
import omni.ext
import omni.ui as ui

# Make the sibling omni.ext.patosim package importable WITHOUT enabling it
# as a Kit extension (that would risk Kit also instantiating its heavy
# robot/recording IExt class). omni.ext is a real Kit-provided package, not
# a namespace package, so a plain sys.path insert is not enough for
# `import omni.ext.patosim` to resolve -- the "patosim" subpackage must be
# added to the ALREADY-IMPORTED omni.ext package's own __path__ explicitly.
# This exact two-step bootstrap is identical to (and confirmed working
# against real Isaac Sim the same way as) scripts/replay_implementation.py's
# bootstrap_repo_paths() and tests/integration/run_gpu_integration_checks.py.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PATOSIM_EXT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "omni.ext.patosim"))
_PATOSIM_OMNI_EXT_ROOT = os.path.join(_PATOSIM_EXT_ROOT, "omni", "ext")
if os.path.isdir(_PATOSIM_EXT_ROOT) and _PATOSIM_EXT_ROOT not in sys.path:
    sys.path.insert(0, _PATOSIM_EXT_ROOT)

os.environ.setdefault("PATOSIM_IMPORT_MODE", "lite")

if os.path.isdir(_PATOSIM_OMNI_EXT_ROOT) and _PATOSIM_OMNI_EXT_ROOT not in omni.ext.__path__:
    omni.ext.__path__.append(_PATOSIM_OMNI_EXT_ROOT)

from omni.ext.patosim.build import (  # noqa: E402
    build_world_only,
    check_scene_semantics,
    check_scene_physics,
    list_dataset_object_assets,
    _insert_dataset_object_into_scene,
    _enable_scene_collisions,
    _apply_sonar_reflectivity_to_scene,
)
from omni.ext.patosim.config import Config  # noqa: E402

SCENE_ROOT_PATH = "/World/scene"


class PatoSimWorldBuilderExtension(omni.ext.IExt):
    """Standalone world-building/validation tool.

    Responsibilities:
      - load a scene USD with no robot
      - import dataset objects (same asset folders as the main extension)
      - check/apply sonar reflectance semantic labels
      - check/apply physics collisions
    """

    def on_startup(self, ext_id):
        self._scene_loaded = False
        self._status_text = "No world loaded."
        self._semantics_report_text = "-"
        self._physics_report_text = "-"

        self._scene_usd_model = ui.SimpleStringModel("")
        self._default_reflectivity_model = ui.SimpleFloatModel(1.0)

        self._dataset_object_assets = []
        self._dataset_object_selected_index = 0
        self._dataset_object_position_models = [ui.SimpleFloatModel(0.0) for _ in range(3)]
        self._dataset_object_rotation_models = [ui.SimpleFloatModel(0.0) for _ in range(3)]
        self._dataset_object_scale_model = ui.SimpleFloatModel(1.0)
        self._dataset_object_reflectivity_model = ui.SimpleFloatModel(1.5)

        self._window = ui.Window("PatoSim - World Builder", width=480, height=680)
        with self._window.frame:
            self._build_ui()

    def on_shutdown(self):
        self._window = None

    # ---------- layout ----------

    def _build_ui(self):
        with ui.ScrollingFrame():
            with ui.VStack(spacing=10, height=0):
                ui.Label(
                    "Load and validate a world BEFORE recording data -- no robot is "
                    "spawned here. Fix missing scale/reflectance/physics first, then "
                    "switch to the main PatoSim extension to build a recording scenario.",
                    word_wrap=True,
                )
                self._build_scene_section()
                self._build_dataset_object_section()
                self._build_semantics_section()
                self._build_physics_section()
                ui.Separator(height=4)
                self._status_label = ui.Label(self._status_text, word_wrap=True)

    def _set_status(self, text: str):
        self._status_text = text
        try:
            self._status_label.text = text
        except Exception:
            pass

    # ---------- world loading ----------

    def _build_scene_section(self):
        ui.Label("World / Scene", style={"font_size": 16})
        with ui.HStack(height=24):
            ui.Label("USD Path / URL", width=120)
            ui.StringField(model=self._scene_usd_model, height=24)
        with ui.HStack(height=28):
            ui.Button("Load World (no robot)", clicked_fn=self._on_load_world)

    def _on_load_world(self):
        scene_usd = str(self._scene_usd_model.as_string or "").strip()

        async def _load_async():
            try:
                await build_world_only(scene_usd)
                self._scene_loaded = True
                self._set_status(f"World loaded: {scene_usd}")
                self._refresh_dataset_object_assets()
            except Exception as exc:
                self._scene_loaded = False
                self._set_status(f"Failed to load world: {exc}")
                traceback.print_exc()

        asyncio.ensure_future(_load_async())

    # ---------- dataset object import ----------

    def _build_dataset_object_section(self):
        ui.Label("Import Dataset Object", style={"font_size": 16})
        with ui.HStack(height=24):
            ui.Button("Refresh Asset List", clicked_fn=self._refresh_dataset_object_assets)

        self._dataset_object_frame = ui.Frame()
        self._dataset_object_frame.set_build_fn(self._build_dataset_object_combo)

        with ui.HStack(height=24):
            ui.Label("Position XYZ (m)", width=120)
            for m in self._dataset_object_position_models:
                ui.FloatField(model=m, width=70)
        with ui.HStack(height=24):
            ui.Label("Rotation XYZ (deg)", width=120)
            for m in self._dataset_object_rotation_models:
                ui.FloatField(model=m, width=70)
        with ui.HStack(height=24):
            ui.Label("Scale", width=120)
            ui.FloatField(model=self._dataset_object_scale_model, width=70)
            ui.Label("Reflectivity", width=90)
            ui.FloatField(model=self._dataset_object_reflectivity_model, width=70)
        with ui.HStack(height=28):
            ui.Button("Insert Object Into World", clicked_fn=self._on_insert_object)

    def _build_dataset_object_combo(self):
        with ui.HStack(height=26):
            ui.Label("Object Asset", width=120)
            self._dataset_object_combo = ui.ComboBox(
                int(self._dataset_object_selected_index), *self._dataset_object_labels(), width=260
            )

            def _on_selection_changed(*_args):
                try:
                    self._dataset_object_selected_index = int(
                        self._dataset_object_combo.model.get_item_value_model().get_value_as_int()
                    )
                except Exception:
                    self._dataset_object_selected_index = 0

            self._dataset_object_combo.model.get_item_value_model().add_value_changed_fn(_on_selection_changed)

    def _dataset_object_labels(self):
        if not self._dataset_object_assets:
            return ["(none found -- load a world, then Refresh)"]
        return [entry.get("label", entry.get("path", "")) for entry in self._dataset_object_assets]

    def _refresh_dataset_object_assets(self):
        try:
            self._dataset_object_assets = list_dataset_object_assets()
        except Exception as exc:
            carb.log_warn(f"[PatoSimWorldBuilder] failed to list dataset objects: {exc}")
            self._dataset_object_assets = []
        self._dataset_object_selected_index = 0
        try:
            self._dataset_object_frame.rebuild()
        except Exception:
            pass

    def _on_insert_object(self):
        if not self._scene_loaded:
            self._set_status("Load a world first.")
            return
        assets = self._dataset_object_assets
        if not assets:
            self._set_status("No dataset object asset selected/available.")
            return
        idx = max(0, min(self._dataset_object_selected_index, len(assets) - 1))
        asset_path = assets[idx].get("path", "")
        position = tuple(float(m.as_float) for m in self._dataset_object_position_models)
        rotation = tuple(float(m.as_float) for m in self._dataset_object_rotation_models)
        scale = float(self._dataset_object_scale_model.as_float)
        reflectivity = float(self._dataset_object_reflectivity_model.as_float)
        # _insert_dataset_object_into_scene only reads a few fields off the
        # config object (position/scale/rotation/operating depth) -- a
        # throwaway Config with just those set is enough, no scenario/robot
        # is ever built from it.
        scratch_config = Config(
            scenario_type="",
            robot_type="",
            scene_usd="",
            dataset_object_position=position,
            dataset_object_scale=scale,
            dataset_object_rotation_euler_deg=rotation,
        )
        try:
            prim_path = _insert_dataset_object_into_scene(
                SCENE_ROOT_PATH, asset_path, reflectivity, config=scratch_config
            )
            self._set_status(f"Inserted object at {prim_path}")
        except Exception as exc:
            self._set_status(f"Failed to insert object: {exc}")
            traceback.print_exc()

    # ---------- semantic classes / sonar reflectance ----------

    def _build_semantics_section(self):
        ui.Label("Semantic Classes / Sonar Reflectance", style={"font_size": 16})
        with ui.HStack(height=24):
            ui.Label("Default reflectivity", width=140)
            ui.FloatField(model=self._default_reflectivity_model, width=70)
        with ui.HStack(height=28):
            ui.Button("Check Semantic Classes", clicked_fn=self._on_check_semantics)
            ui.Button("Apply Default Reflectivity", clicked_fn=self._on_apply_semantics)
        self._semantics_report_label = ui.Label(self._semantics_report_text, word_wrap=True)

    def _on_check_semantics(self):
        if not self._scene_loaded:
            self._set_status("Load a world first.")
            return
        try:
            report = check_scene_semantics(SCENE_ROOT_PATH)
            text = (
                f"Meshes: {report['total_meshes']} | with class: {report['with_class']} | "
                f"with reflectivity: {report['with_reflectivity']} | missing: {report['missing_count']}"
            )
        except Exception as exc:
            text = f"Check failed: {exc}"
            traceback.print_exc()
        self._semantics_report_text = text
        try:
            self._semantics_report_label.text = text
        except Exception:
            pass

    def _on_apply_semantics(self):
        if not self._scene_loaded:
            self._set_status("Load a world first.")
            return
        try:
            n = _apply_sonar_reflectivity_to_scene(
                SCENE_ROOT_PATH, default_reflectivity=float(self._default_reflectivity_model.as_float)
            )
            self._set_status(f"Applied reflectivity semantics to {n} mesh(es).")
            self._on_check_semantics()
        except Exception as exc:
            self._set_status(f"Failed to apply reflectivity: {exc}")
            traceback.print_exc()

    # ---------- physics / collisions ----------

    def _build_physics_section(self):
        ui.Label("Physics / Collisions", style={"font_size": 16})
        with ui.HStack(height=28):
            ui.Button("Check Physics", clicked_fn=self._on_check_physics)
            ui.Button("Apply Physics (Collisions)", clicked_fn=self._on_apply_physics)
        self._physics_report_label = ui.Label(self._physics_report_text, word_wrap=True)

    def _on_check_physics(self):
        if not self._scene_loaded:
            self._set_status("Load a world first.")
            return
        try:
            report = check_scene_physics(SCENE_ROOT_PATH)
            text = (
                f"Geometry prims: {report['total_geoms']} | with collision: "
                f"{report['with_collision']} | missing: {report['missing_count']}"
            )
        except Exception as exc:
            text = f"Check failed: {exc}"
            traceback.print_exc()
        self._physics_report_text = text
        try:
            self._physics_report_label.text = text
        except Exception:
            pass

    def _on_apply_physics(self):
        if not self._scene_loaded:
            self._set_status("Load a world first.")
            return
        try:
            n = _enable_scene_collisions(SCENE_ROOT_PATH)
            self._set_status(f"Applied collision to {n} prim(s).")
            self._on_check_physics()
        except Exception as exc:
            self._set_status(f"Failed to apply physics: {exc}")
            traceback.print_exc()
