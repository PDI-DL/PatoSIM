# SPDX-FileCopyrightText: Copyright (c) 2026 MOD_patosim contributors
# SPDX-License-Identifier: Apache-2.0
"""PatoSim e BEVSIM (os dois derivados do MobilityGen) não podem pegar código nem env um do outro.

Medidos em 18/11/2026: env conda com o mesmo nome (mobgen), extensão omni.ext.mobility_gen em legacy_exts, pacote
mobility_gen_path_planner com o mesmo nome, scripts de lançamento que dependiam da pasta atual e link_app.sh com o
Isaac de outra máquina como padrão.
"""

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_conda_env_has_its_own_name():
    text = (REPO / "environment.yml").read_text()
    name = re.search(r"^name:\s*(\S+)", text, re.M).group(1)
    assert name == "pato_sim", name
    for line in text.splitlines():  # versões fixas (o arquivo antigo não fixava nada)
        if line.strip().startswith("- ") and not line.strip().startswith(("- pip:", "- conda-forge", "- nodefaults")):
            assert re.search(r"==?\d", line), line


def test_no_bevsim_extension_id_in_repo():
    ids = [p.parent.parent.name for p in REPO.glob("**/config/extension.toml") if ".git" not in p.parts]
    assert "omni.ext.mobility_gen" not in ids, ids


def test_planner_guard_prefers_this_repository(tmp_path, monkeypatch):
    """Com o pacote do BEVSIM (mesmo nome) já carregado, use_own_planner troca pelo deste repositório e avisa.
    Os dois pacotes são falsos: o teste não depende da extensão C++ compilada."""
    sys.path.insert(0, str(REPO / "exts" / "omni.ext.patosim" / "omni" / "ext" / "patosim"))
    try:
        guard = importlib.import_module("planner_guard")
    finally:
        sys.path.pop(0)
    pkgs = {}
    for who in ("bevsim", "patosim"):
        folder = tmp_path / who / "path_planner"
        (folder / "mobility_gen_path_planner").mkdir(parents=True)
        (folder / "mobility_gen_path_planner" / "__init__.py").write_text(f"ORIGEM = {who!r}\n")
        pkgs[who] = folder
    saved = {k: v for k, v in sys.modules.items() if k.startswith("mobility_gen_path_planner")}
    for k in saved:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.syspath_prepend(str(pkgs["bevsim"]))
    import mobility_gen_path_planner as loaded  # a cópia "do outro simulador"
    assert loaded.ORIGEM == "bevsim"
    module, origin, warning = guard.use_own_planner(str(pkgs["patosim"]))
    try:
        assert module is not None and module.ORIGEM == "patosim", origin
        assert Path(origin).resolve().is_relative_to(pkgs["patosim"].resolve()), origin
        assert "trocado" in warning and str(pkgs["bevsim"]) in warning
        assert sys.path[0] == str(pkgs["patosim"])
    finally:
        for k in [k for k in sys.modules if k.startswith("mobility_gen_path_planner")]:
            del sys.modules[k]
        sys.modules.update(saved)
        while str(pkgs["patosim"]) in sys.path:
            sys.path.remove(str(pkgs["patosim"]))


@pytest.mark.parametrize("script,ext", [("scripts/launch_patosim.sh", "omni.ext.patosim"),
                                        ("scripts/launch_world_builder.sh", "omni.ext.patosim_worldbuilder")])
def test_launch_scripts_use_this_repository_from_any_folder(tmp_path, script, ext):
    out = subprocess.run(["bash", str(REPO / script), "--no-window"], cwd=str(tmp_path), capture_output=True,
                         text=True, env={**os.environ, "PATOSIM_DRY_RUN": "1"}, timeout=30)
    if not (REPO / "app" / "isaac-sim.sh").exists():
        assert out.returncode == 1 and "não está linkado" in out.stderr
        return
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith(str(REPO / "app" / "isaac-sim.sh"))
    assert f"--ext-folder {REPO / 'exts'}" in out.stdout and f"--enable {ext}" in out.stdout


def test_link_app_has_no_machine_default():
    env = {k: v for k, v in os.environ.items() if k != "ISAAC_SIM_PATH"}
    out = subprocess.run(["bash", str(REPO / "link_app.sh")], capture_output=True, text=True, env=env, timeout=30)
    assert out.returncode == 1 and "--path" in out.stderr
    assert "/home/bevlog" not in (REPO / "link_app.sh").read_text().split("Até 18/11/2026")[0]
