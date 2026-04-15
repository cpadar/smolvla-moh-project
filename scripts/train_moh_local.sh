#!/bin/bash
# Local training script for MOH v4 — for smoke testing only (batch_size=64 OOMs on 12GB).
# Full training should be submitted on SOL via scripts/slurm/train_moh_full.sh.
# Usage:
#   bash scripts/train_moh_local.sh --test   # 3-step smoke test

set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."
echo "Working dir: $(pwd)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Patch LeRobot factory to register smolvla_moh
python scripts/slurm/patch_factory.py

if [[ "$1" == "--test" ]]; then
  echo "Running 3-step smoke test at batch_size=64..."
  STEPS=3
  SAVE_FREQ=10
  OUTPUT_DIR=/home/ceshank/smolvla-moh-project/results/moh_local_test
  WANDB=false
  # Remove stale test dir so lerobot-train can create it fresh
  rm -rf "$OUTPUT_DIR"
else
  echo "Full training should be run on SOL — use scripts/slurm/train_moh_full.sh"
  exit 1
fi

mkdir -p logs

cd ~/lerobot && lerobot-train \
  --policy.type=smolvla_moh \
  --dataset.repo_id=ceshank01/stack-pyramid-v1-v5 \
  --batch_size=64 \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --output_dir="$OUTPUT_DIR" \
  --job_name=smolvla_moh_v4_local \
  --policy.device=cuda \
  --wandb.enable=$WANDB \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.base_camera": "observation.images.camera1", "observation.images.hand_camera": "observation.images.camera2"}'

echo "Done!"
