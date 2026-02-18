#!/bin/bash
#SBATCH --job-name=smolvla_base_train
#SBATCH --output=results/logs/smolvla_base_%j.out
#SBATCH --error=results/logs/smolvla_base_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=public

set -a
source .env
set +a

mkdir -p results/logs results/checkpoints/smolvla_base

singularity exec --nv \
  --bind $(pwd):/workspace \
  --bind ${MS_ASSET_DIR}:${MS_ASSET_DIR} \
  --bind ~/.cache/huggingface:/root/.cache/huggingface \
  smolvla-moh.sif \
  bash -c "
    cd /workspace &&
    python scripts/train/train_base.py \
      --config configs/train_base.yaml
  "
