#!/bin/bash
# Replay StackPyramid-v1 demonstrations to generate RGB+D observations
# Run this on SOL or a GPU machine — will fail on Mac (no Vulkan GPU)
#
# Usage: bash scripts/data/replay_trajectories.sh
#
# Output: ~/maniskill_demos/StackPyramid-v1/motionplanning/
#         trajectory.rgbd.pd_joint_pos.physx_cuda.h5

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/maniskill_demos/StackPyramid-v1/motionplanning/trajectory.h5 \
  --use-env-states \
  -c pd_joint_pos \
  -o rgbd \
  --save-traj \
  --num-procs 8 \
  -b physx_cuda \
  --shader default