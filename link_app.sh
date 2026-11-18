#!/bin/bash

set -e

# Sem argumentos usa $ISAAC_SIM_PATH. Até 18/11/2026 o padrão era /home/bevlog/isaac-sim, uma pasta de outra máquina
# (na bevlog, o app/ do PatoSim ficou apontando para ela depois que ela saiu de lá).
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
cd "$SCRIPT_DIR"

if [[ $# -eq 0 ]]; then
    if [[ -z "${ISAAC_SIM_PATH:-}" ]]; then
        echo "Uso: ./link_app.sh --path <pasta do Isaac Sim 5.1>   (ou defina ISAAC_SIM_PATH)" >&2
        echo "De preferência um Isaac só do PatoSim: o BEVSIM usa a mesma base e divide extsUser e pacotes." >&2
        exit 1
    fi
    exec "tools/packman/python.sh" tools/scripts/link_app.py --path "${ISAAC_SIM_PATH}"
fi

exec "tools/packman/python.sh" tools/scripts/link_app.py "$@"
