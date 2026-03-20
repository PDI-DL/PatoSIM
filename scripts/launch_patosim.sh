#!/usr/bin/env bash
set -euo pipefail

SAFE_ARGS=()
if [[ "${MOBILITY_SAFE_START:-1}" == "1" ]]; then
    SAFE_ARGS+=(--reset-user)
fi
if [[ "${MOBILITY_CLEAR_CACHE:-0}" == "1" ]]; then
    SAFE_ARGS+=(--clear-cache)
fi
if [[ "${MOBILITY_CLEAR_DATA:-0}" == "1" ]]; then
    SAFE_ARGS+=(--clear-data)
fi

./app/isaac-sim.sh \
    --ext-folder exts \
    --enable omni.ext.patosim \
    "${SAFE_ARGS[@]}" \
    "$@"
