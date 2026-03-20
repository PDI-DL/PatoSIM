#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path_to_oceansim_assets>"
    exit 1
fi

"${REPO_DIR}/app/python.sh" \
    "${REPO_DIR}/exts/omni.ext.patosim/config/register_asset_path.py" \
    "$1"
