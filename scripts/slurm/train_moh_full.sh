#!/bin/bash
#SBATCH --job-name=smolvla_moh_v1
#SBATCH --output=logs/moh_v1_%j.out
#SBATCH --error=logs/moh_v1_%j.err
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_eee515598spring2026
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00

echo "Starting MoH full training run..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

source ~/.bashrc
conda activate smolvla-moh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd ~/smolvla-moh-project
git pull
mkdir -p logs

# Patch LeRobot factory
python scripts/slurm/patch_factory.py

# Full training run
cd ~/lerobot && lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.type=smolvla_moh \
  --dataset.repo_id=ceshank01/stack-pyramid-v1-v4 \
  --batch_size=16 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=~/smolvla-moh-project/results/moh_v1 \
  --job_name=smolvla_moh_v1 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.base_camera": "observation.images.camera1", "observation.images.hand_camera": "observation.images.camera2"}'

echo "Training complete!"