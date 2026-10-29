#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "${LIDAR_PPO_PYTHON:-$PWD/.venv-lidar-ppo/bin/python}" -m lidar_ppo "$@"
