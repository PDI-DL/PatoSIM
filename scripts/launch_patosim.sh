#!/usr/bin/env bash
set -euo pipefail

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

./app/isaac-sim.sh \
    --ext-folder exts \
    --enable omni.ext.patosim \
    "${SAFE_ARGS[@]}" \
    "$@"
