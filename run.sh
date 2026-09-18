#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${G1_SIM_PYTHON:-}" ]]; then
  sim_python="$G1_SIM_PYTHON"
elif [[ -x "$HOME/miniconda3/envs/lerobot-act/bin/python" ]]; then
  sim_python="$HOME/miniconda3/envs/lerobot-act/bin/python"
else
  sim_python=python
fi
if [[ ! -f assets/robot/g1_inspire.xml ]]; then
  "$sim_python" prepare_assets.py
fi
exec "$sim_python" -m g1_pipette_sim "$@"
