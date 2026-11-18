#!/usr/bin/env bash
set -euo pipefail

# Launches the standalone World Builder extension: scene loading, asset
# import, physics/collision setup and sonar reflectance semantic labeling,
# with NO robot spawned. Meant to validate a world before switching to
# launch_patosim.sh to build an actual data-recording scenario against the
# same scene. See exts/omni.ext.patosim_worldbuilder/docs/README.md.

SAFE_START="${PATOSIM_SAFE_START:-${MOBILITY_SAFE_START:-1}}"
CLEAR_CACHE="${PATOSIM_CLEAR_CACHE:-${MOBILITY_CLEAR_CACHE:-0}}"
CLEAR_DATA="${PATOSIM_CLEAR_DATA:-${MOBILITY_CLEAR_DATA:-0}}"

SAFE_ARGS=()
if [[ "${SAFE_START}" == "1" ]]; then
    SAFE_ARGS+=(--reset-user)
fi
if [[ "${CLEAR_CACHE}" == "1" ]]; then
    SAFE_ARGS+=(--clear-cache)
fi
if [[ "${CLEAR_DATA}" == "1" ]]; then
    SAFE_ARGS+=(--clear-data)
fi

# Sempre o Isaac e a extensão DESTE repositório, de qualquer pasta atual (antes: ./app e "--ext-folder exts"
# relativos à pasta atual; de dentro de outro clone, como o do BEVSIM, abria o Isaac e a extensão daquele clone).
REPO="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [[ ! -x "$REPO/app/isaac-sim.sh" ]]; then
    echo "Isaac Sim não está linkado em $REPO/app (rode ./link_app.sh --path <isaac-sim-5.1>)." >&2
    exit 1
fi
CMD=("$REPO/app/isaac-sim.sh"
    --ext-folder "$REPO/exts"
    --enable omni.ext.patosim_worldbuilder
    "${SAFE_ARGS[@]}"
    "$@")
if [[ "${PATOSIM_DRY_RUN:-0}" == "1" ]]; then   # testes: mostra o comando sem abrir o Isaac
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi
exec "${CMD[@]}"
