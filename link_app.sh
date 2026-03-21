#!/bin/bash

set -e

DEFAULT_ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/mnt/external/isaac/isaac-sim-5.1}"
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
cd "$SCRIPT_DIR"

if [[ $# -eq 0 ]]; then
    exec "tools/packman/python.sh" tools/scripts/link_app.py --path "${DEFAULT_ISAAC_SIM_PATH}"
fi

exec "tools/packman/python.sh" tools/scripts/link_app.py "$@"
