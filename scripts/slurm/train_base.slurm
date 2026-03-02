#!/bin/bash
#SBATCH --job-name=smolvla_base_train
#SBATCH --output=results/logs/smolvla_base_%j.out
#SBATCH --error=results/logs/smolvla_base_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_eee515598spring2026

source ~/.bashrc
conda activate smolvla-moh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p results/logs results/checkpoints/smolvla_base

cd ~/smolvla-moh-project

python scripts/train/train_base.py \
  --config configs/train_base.yaml
