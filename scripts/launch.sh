#!/usr/bin/env bash
# scripts/launch.sh
# Usage:
#   ./scripts/launch.sh           # single run
#   ./scripts/launch.sh sweep     # start a sweep

CONFIG=configs/default.yaml

if [ "$1" == "sweep" ]; then
  echo "Starting WandB sweep with ${CONFIG}"
  SWEEP_ID=$(wandb sweep configs/sweep.yaml)
  wandb agent ${SWEEP_ID}
else
  echo "Running single experiment with ${CONFIG}"
  python src/train.py --config ${CONFIG}
fi
