"""
convert_to_lerobot.py

Converts ManiSkill3 StackPyramid-v1 replayed HDF5 trajectories into
LeRobot v3 dataset format for smolVLA fine-tuning.

IMPORTANT: Run replay_trajectories.sh on a GPU machine FIRST to generate
the replayed HDF5 with observations. This script expects that file as input.

Usage:
    python scripts/data/convert_to_lerobot.py \
        --input ~/maniskill_demos/StackPyramid-v1/motionplanning/trajectory.rgbd.pd_joint_pos.physx_cuda.h5 \
        --output ~/lerobot_datasets/stack_pyramid_v1 \
        --repo-id YOUR_HF_USERNAME/stack-pyramid-v1 \
        --push-to-hub

Requirements:
    - pip install h5py
    - LeRobot installed from source (~/lerobot)
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ── Constants ──────────────────────────────────────────────────────────────────

# StackPyramid-v1 with Panda arm: 7 joints + 1 gripper = 8 DOF
ACTION_DIM = 8

# Camera keys in the replayed HDF5 — confirm these after replay
# ManiSkill3 RGBD mode typically produces these keys
CAMERA_KEYS = [
    "obs/sensor_data/base_camera/rgb",   # third-person view
    "obs/sensor_data/hand_camera/rgb",   # wrist camera
]

# LeRobot feature names for each camera
LEROBOT_CAMERA_NAMES = {
    "obs/sensor_data/base_camera/rgb": "observation.images.base_camera",
    "obs/sensor_data/hand_camera/rgb":  "observation.images.hand_camera",
}

# Task description — used by smolVLA's language conditioning
TASK_DESCRIPTION = "Pick up  a red cube, place it next to the green cube, then stack the blue cube on top of the red and green cube to form a pyramid."

# Frames per second of the ManiSkill3 simulation
FPS = 20


# ── Helpers ────────────────────────────────────────────────────────────────────

# def get_robot_state(traj: h5py.Group, step: int) -> np.ndarray:
#     """
#     Extract robot joint state at a given timestep.
#     Uses the panda_wristcam articulation state (31 values).
#     We take the first 8 values which correspond to joint positions.
#     """
#     return traj["env_states/articulations/panda_wristcam"][step, :ACTION_DIM]
def get_robot_state(traj: h5py.Group, step: int) -> np.ndarray:
    """Extract robot joint positions (qpos) at a given timestep."""
    return traj["obs/agent/qpos"][step]

def get_action(traj: h5py.Group, step: int) -> np.ndarray:
    """Extract action (joint position command) at a given timestep."""
    return traj["actions"][step]


def get_camera_image(traj: h5py.Group, camera_key: str, step: int) -> np.ndarray:
    """
    Extract RGB image from a camera at a given timestep.
    Returns HWC uint8 numpy array.
    """
    img = traj[camera_key][step]  # shape: (H, W, 3) or (H, W, 4) for RGBD
    if img.shape[-1] == 4:
        img = img[..., :3]  # drop depth channel, keep RGB only
    return img.astype(np.uint8)


def check_cameras_available(traj: h5py.Group) -> list[str]:
    """
    Check which camera keys actually exist in the replayed HDF5.
    The replay script may use different key names than expected.
    Prints available keys to help debug if cameras are missing.
    """
    available = []
    for key in CAMERA_KEYS:
        if key in traj:
            available.append(key)
        else:
            print(f"  WARNING: Expected camera key '{key}' not found in trajectory.")
            print(f"  Available obs keys:")
            if "obs" in traj:
                traj["obs"].visititems(lambda name, _: print(f"    obs/{name}"))
    return available


# ── Main conversion ────────────────────────────────────────────────────────────

def convert(
    input_path: str,
    output_path: str,
    repo_id: str,
    num_episodes: int | None = None,
    push_to_hub: bool = False,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Opening HDF5 file: {input_path}")
    f = h5py.File(input_path, "r")

    # Get all trajectory keys
    traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
    if num_episodes is not None:
        traj_keys = traj_keys[:num_episodes]

    print(f"Found {len(traj_keys)} trajectories to convert.")

    # Check cameras on first trajectory
    print("Checking camera availability on first trajectory...")
    available_cameras = check_cameras_available(f[traj_keys[0]])
    if not available_cameras:
        raise RuntimeError(
            "No camera data found. Make sure you ran replay_trajectories.sh "
            "with -o rgbd BEFORE running this script."
        )
    print(f"  Found cameras: {available_cameras}")

    # Build LeRobot feature spec
    # This tells LeRobot what each column in the dataset contains
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": [f"joint_{i}" for i in range(9)],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [f"joint_{i}" for i in range(ACTION_DIM)],
        },
    }
    # Add camera features
    for cam_key in available_cameras:
        lerobot_name = LEROBOT_CAMERA_NAMES[cam_key]
        # Get image shape from first frame of first trajectory
        sample_img = get_camera_image(f[traj_keys[0]], cam_key, 0)
        h, w, c = sample_img.shape
        features[lerobot_name] = {
            "dtype": "video",
            "shape": (c, h, w),  # LeRobot uses CHW format
            "names": ["channel", "height", "width"],
        }

    # Create the LeRobot dataset
    print(f"Creating LeRobot dataset at: {output_path}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=output_path,
        features=features,
    )

    # Convert each trajectory
    for ep_idx, traj_key in enumerate(traj_keys):
        traj = f[traj_key]
        n_steps = len(traj["actions"])

        print(f"  Converting episode {ep_idx + 1}/{len(traj_keys)}: "
              f"{traj_key} ({n_steps} steps)")

        for step in range(n_steps):
            frame = {
                "observation.state": get_robot_state(traj, step).astype(np.float32),
                "action": get_action(traj, step).astype(np.float32),
            }
            # Add camera images
            for cam_key in available_cameras:
                lerobot_name = LEROBOT_CAMERA_NAMES[cam_key]
                img = get_camera_image(traj, cam_key, step)
                # Convert HWC -> CHW for LeRobot
                frame[lerobot_name] = Image.fromarray(img)

            frame["task"] = "Pick up the red cube, place it next to the green cube, then stack the blue cube on top of the red and green cube to form a pyramid."
            dataset.add_frame(frame)

        # Save this episode with the task description
        dataset.save_episode()

    # Finalize — closes parquet writers, writes metadata
    print("Finalizing dataset...")
    dataset.finalize()

    print(f"Dataset saved to: {output_path}")
    print(f"Total episodes: {len(traj_keys)}")

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub as: {repo_id}")
        dataset.push_to_hub()
        print("Done!")

    f.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ManiSkill3 replayed HDF5 to LeRobot v3 format"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the replayed HDF5 file (output of replay_trajectories.sh)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Local directory to save the LeRobot dataset",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID, e.g. your-username/stack-pyramid-v1",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to convert (default: all). Use a small number for testing.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the finished dataset to HuggingFace Hub",
    )
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        repo_id=args.repo_id,
        num_episodes=args.num_episodes,
        push_to_hub=args.push_to_hub,
    )