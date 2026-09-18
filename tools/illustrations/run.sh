#!/usr/bin/env bash
# Reproduce the episode-64 presentation videos using saved accepted captures.
set -euo pipefail
cd "$(dirname "$0")/../.."
sim_python="${G1_SIM_PYTHON:-python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
kind="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
base=outputs/hil194/augmentation600/candidates/000025
study=outputs/hil194/episode64-illustration-study/candidates
case "$kind" in
  lab) exec "$sim_python" tools/illustrations/lab_environments.py "$@" ;;
  rack) exec "$sim_python" tools/illustrations/rack_tip_placements.py --candidate 25 --seed 64 --swap-seconds .3 --name episode64-rack-tip-placements "$@" ;;
  joint) exec "$sim_python" tools/illustrations/capture_comparison.py --base "$base" --capture "$study/000000" --name episode64-joint-perturbation --title 'Joint Perturbation' "$@" ;;
  elbow) exec "$sim_python" tools/illustrations/capture_comparison.py --base "$base" --capture "$study/000602" --name episode64-elbow-correction-closeup --title 'Joint Perturbation' --zoom --distance .1 --start 5.7 --repeat 2 "$@" ;;
  help|-h|--help) echo 'Usage: tools/illustrations/run.sh {lab|rack|joint|elbow} [generator options]'; echo 'Example: tools/illustrations/run.sh lab --swap-seconds 0.5';;
  *) echo "Unknown video: $kind" >&2; exit 2 ;;
esac
