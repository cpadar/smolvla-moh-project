#!/bin/bash
#SBATCH --job-name=smolvla_moh_test
#SBATCH --output=logs/moh_test_%j.out
#SBATCH --error=logs/moh_test_%j.err
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --account=class_eee515598spring2026
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

echo "Starting MoH test run..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Setup environment
source ~/.bashrc
conda activate smolvla-moh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd ~/smolvla-moh-project
git pull

# Patch LeRobot factory
python scripts/slurm/patch_factory.py

# Test run - 3 steps only
cd ~/lerobot && lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.type=smolvla_moh \
  --dataset.repo_id=ceshank01/stack-pyramid-v1-v4 \
  --batch_size=16 \
  --steps=3 \
  --save_freq=3 \
  --output_dir=~/smolvla-moh-project/results/moh_test \
  --job_name=smolvla_moh_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.base_camera": "observation.images.camera1", "observation.images.hand_camera": "observation.images.camera2"}'

echo "Test run complete!"