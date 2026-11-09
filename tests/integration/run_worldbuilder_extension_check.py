# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""Verifies, against the real Kit extension manager, that enabling
omni.ext.patosim_worldbuilder does NOT also enable the heavy
omni.ext.patosim extension (the risk flagged as unverified in
docs/relatorio_implementacao_upgrade_underwater_v2.md section 3, item 6).

Run with: ./app/python.sh tests/integration/run_worldbuilder_extension_check.py
"""
import json
import os
import sys

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_worldbuilder_ext_results.json")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(launch_config={"headless": True})
    results = []

    def record(name, ok, detail=""):
        results.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}", flush=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    try:
        import omni.kit.app

        repo_root = "/home/pdi_4/Documents/Documentos/00_mestrado_gustavo/MOD_patosim"
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_manager.add_path(os.path.join(repo_root, "exts"))

        for _ in range(60):
            simulation_app.update()

        all_exts = list(ext_manager.get_extensions())
        patosim_exts = [
            e for e in all_exts
            if "patosim" in str(e.get("id", e.get("name", ""))).lower()
        ]
        record(
            "extensions_discovered",
            len(patosim_exts) > 0,
            f"found {len(patosim_exts)} patosim-related extension(s): "
            f"{[e.get('id', e.get('name')) for e in patosim_exts]}",
        )

        wb_id = None
        main_id = None
        for e in all_exts:
            eid = str(e.get("id", e.get("name", "")))
            if eid.startswith("omni.ext.patosim_worldbuilder"):
                wb_id = eid
            elif eid.startswith("omni.ext.patosim-") or eid == "omni.ext.patosim":
                main_id = eid

        record("worldbuilder_ext_id_found", wb_id is not None, f"wb_id={wb_id}")
        record("main_ext_id_found_in_registry", main_id is not None, f"main_id={main_id} (just discovery, not enabling it)")

        if wb_id is None:
            record("worldbuilder_enable_attempt", False, "cannot enable: extension id not found in registry")
        else:
            ok = ext_manager.set_extension_enabled_immediate(wb_id, True)
            for _ in range(30):
                simulation_app.update()
            enabled_now = ext_manager.is_extension_enabled(wb_id)
            record("worldbuilder_enabled_successfully", bool(enabled_now), f"set_extension_enabled_immediate returned {ok}, is_extension_enabled={enabled_now}")

            main_enabled = ext_manager.is_extension_enabled(main_id) if main_id else False
            record(
                "main_extension_not_auto_enabled",
                not main_enabled,
                f"omni.ext.patosim is_extension_enabled={main_enabled} "
                f"(expected False -- worldbuilder must not pull in the robot/recording UI)",
            )

            # Check that the window got created (on_startup ran) without raising.
            try:
                import omni.ui as ui

                windows = [w for w in ui.Workspace.get_windows() if "World Builder" in w.title]
                record(
                    "worldbuilder_window_created",
                    len(windows) > 0,
                    f"found window(s): {[w.title for w in windows]}" if windows else "no 'World Builder' window found",
                )
            except Exception as exc:
                record("worldbuilder_window_created", False, str(exc))

    except Exception as exc:
        import traceback
        record("worldbuilder_extension_check_crashed", False, f"{exc}\n{traceback.format_exc()}")
    finally:
        n_pass = sum(1 for r in results if r["ok"])
        print(f"\n==== SUMMARY: {n_pass}/{len(results)} checks passed ====", flush=True)
        sys.stdout.flush()
        simulation_app.close()


if __name__ == "__main__":
    main()
