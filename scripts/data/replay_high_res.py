"""
Replay trajectories at 512x512 resolution.
Usage: python scripts/data/replay_high_res.py
"""
import sys
sys.path.insert(0, '.')

import gymnasium as gym
import mani_skill.envs
from mani_skill.trajectory.replay_trajectory import main, parse_args

# Monkey-patch gym.make to inject sensor_configs
original_make = gym.make

def patched_make(env_id, **kwargs):
    if 'StackPyramid' in str(env_id):
        kwargs['sensor_configs'] = dict(width=512, height=512)
    return original_make(env_id, **kwargs)

gym.make = patched_make

if __name__ == "__main__":
    sys.argv = [
        'replay_trajectory.py',
        '--traj-path', '/home/ceshank/maniskill_demos/StackPyramid-v1/motionplanning/trajectory.h5',
        '--use-env-states',
        '-c', 'pd_joint_pos',
        '-o', 'rgbd',
        '--save-traj',
        '-b', 'gpu',
        '-n', '4',
    ]
    main(parse_args())