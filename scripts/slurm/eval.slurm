#!/bin/bash
#SBATCH --job-name=smolvla_eval
#SBATCH --output=results/logs/eval_%j.out
#SBATCH --error=results/logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -a
source .env
set +a

mkdir -p results/logs results/eval

singularity exec --nv \
  --bind $(pwd):/workspace \
  --bind ${MS_ASSET_DIR}:${MS_ASSET_DIR} \
  --bind ~/.cache/huggingface:/root/.cache/huggingface \
  smolvla-moh.sif \
  bash -c "
    cd /workspace &&
    python scripts/eval/evaluate.py \
      --config configs/eval.yaml \
      $@
  "
