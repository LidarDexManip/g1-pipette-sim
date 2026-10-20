#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
dataset_root="$PWD/.cache/huggingface/hub/datasets--jren313--g1-pipette-2view-teleop0925-5task-eerel/snapshots/c7adfc179e5e15ade41b06803542001a24959dda"
if [[ ! -f "$dataset_root/meta/info.json" ]]; then
  echo 'Download the dataset first with: python tools/download_lidar_dataset.py' >&2
  exit 1
fi
exec ./run.sh --environment lidar --dataset "$dataset_root" "$@"
