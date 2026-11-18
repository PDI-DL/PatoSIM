"""
Garante que o mobility_gen_path_planner importado é o deste repositório (sem Isaac: testável fora dele).

O PatoSim e o BEVSIM têm um pacote com esse mesmo nome e código diferente. Num Isaac usado pelos dois, um
`pip install -e path_planner` de um deles (ou um import anterior no mesmo processo) faria o outro usar o código errado
sem aviso. use_own_planner() põe o path_planner do repositório na frente do sys.path, tira da memória uma cópia de
outro lugar e devolve (módulo, origem, aviso).
"""

from __future__ import annotations

import importlib
import os
import sys

PACKAGE = "mobility_gen_path_planner"


def _inside(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(folder)]) == os.path.realpath(folder)
    except ValueError:
        return False


def use_own_planner(planner_dir: str):
    """planner_dir: <repo>/path_planner. Devolve (módulo ou None, pasta de onde veio, aviso ou "")."""
    warning = ""
    if os.path.isdir(planner_dir):
        while planner_dir in sys.path:
            sys.path.remove(planner_dir)
        sys.path.insert(0, planner_dir)
        loaded = sys.modules.get(PACKAGE)
        origin = os.path.dirname(getattr(loaded, "__file__", "") or "") if loaded else ""
        if loaded is not None and not _inside(origin, planner_dir):
            warning = (f"{PACKAGE} já estava carregado de {origin} (outro repositório ou instalação no Isaac); "
                       f"trocado pelo de {planner_dir}")
            for name in [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".")]:
                del sys.modules[name]
            importlib.invalidate_caches()
    try:
        module = importlib.import_module(PACKAGE)
    except Exception as exc:  # sem o pacote: o cenário usa o fallback do robô
        return None, "", f"{PACKAGE} indisponível ({exc})"
    origin = os.path.dirname(getattr(module, "__file__", "") or "")
    if os.path.isdir(planner_dir) and not _inside(origin, planner_dir):
        warning = f"{PACKAGE} veio de {origin}, não de {planner_dir}"
    return module, origin, warning
