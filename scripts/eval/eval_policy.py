# scripts/eval/eval_policy.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import logging
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from models.smolvla_moh.moh_policy import SmolVLAMoHPolicy
from lerobot.processor.converters import transition_to_policy_action, policy_action_to_transition

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TASK_DESCRIPTION = "Pick up the red cube, place it next to the green cube, then stack the blue cube on top of the red and green cube to form a pyramid."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True, choices=["base", "moh"])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="results/eval_videos")
    return parser.parse_args()


def evaluate(args):
    print(f"Starting evaluation: {args.num_episodes} episodes", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Evaluating on: {device}")

    # Load policy
    if args.model_type == "moh":
        policy = SmolVLAMoHPolicy.from_pretrained(args.model)
    else:
        policy = SmolVLAPolicy.from_pretrained(args.model)
    policy = policy.to(device)
    policy.eval()

    # # Load saved preprocessor from checkpoint (has correct normalization stats)
    # preprocessor, _ = make_pre_post_processors(
    #     policy_cfg=policy.config,
    #     pretrained_path=args.model,
    #     preprocessor_overrides={"device_processor": {"device": str(device)}},
    # )
    # Override pretrained_path to load OUR preprocessor, not the base model's
    policy.config.pretrained_path = args.model
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Create environment
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(
        "StackPyramid-v1",
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode=render_mode,
        num_envs=1,
    )

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    successes = []
    episode_lengths = []

    for episode in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        success = False
        frames = []

        try:
            policy.reset()
        except AttributeError:
            pass

        while not done and step < 300:
            # Only re-query model every n_action_steps
            if step % 10 == 0:
                raw_batch = {
                    "observation.state": obs["agent"]["qpos"].squeeze(0).unsqueeze(0).cpu(),
                    "observation.images.base_camera": obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).permute(2, 0, 1).unsqueeze(0).float().cpu() / 255.0,
                    "observation.images.hand_camera": obs["sensor_data"]["hand_camera"]["rgb"].squeeze(0).permute(2, 0, 1).unsqueeze(0).float().cpu() / 255.0,
                    "task": TASK_DESCRIPTION,
                }
                processed_batch = preprocessor(raw_batch)
                with torch.no_grad():
                    action_chunk = policy.predict_action_chunk(processed_batch)
                # Unnormalize the entire chunk
                policy_action = transition_to_policy_action({"action": action_chunk})
                unnorm = postprocessor(policy_action)
                unnorm_chunk = policy_action_to_transition(unnorm)["action"]
                chunk_idx = 0

            # Execute next action from current chunk
            action_np = unnorm_chunk[:, chunk_idx, :].cpu().numpy().squeeze()
            chunk_idx += 1

            obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated.any()) or bool(truncated.any())

            if bool(torch.tensor(info.get("success", False)).any()):
                success = True
                done = True

            if args.save_video:
                frames.append(obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).cpu().numpy())

            step += 1

        # while not done and step < 300:
        #     # Build raw batch using camera1/camera2 names to match training
        #     raw_batch = {
        #         "observation.state": obs["agent"]["qpos"].squeeze(0).unsqueeze(0).cpu(),
        #         "observation.images.base_camera": obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).permute(2, 0, 1).unsqueeze(0).float().cpu() / 255.0,
        #         "observation.images.hand_camera": obs["sensor_data"]["hand_camera"]["rgb"].squeeze(0).permute(2, 0, 1).unsqueeze(0).float().cpu() / 255.0,
        #         "task": TASK_DESCRIPTION,
        #     }

        #     # Apply preprocessor (handles normalization and tokenization)
        #     processed_batch = preprocessor(raw_batch)

        #     with torch.no_grad():
        #                     action = policy.select_action(processed_batch)

        #     # Apply postprocessor to unnormalize action
        #     from lerobot.processor.converters import transition_to_policy_action, policy_action_to_transition
        #     policy_action = transition_to_policy_action({"action": action})
        #     unnorm_policy_action = postprocessor(policy_action)
        #     unnorm_transition = policy_action_to_transition(unnorm_policy_action)
        #     action_np = unnorm_transition["action"].cpu().numpy().squeeze()
        #     obs, reward, terminated, truncated, info = env.step(action_np)
        #     done = bool(terminated.any()) or bool(truncated.any())

        #     if bool(torch.tensor(info.get("success", False)).any()):
        #         success = True
        #         done = True

        #     if args.save_video:
        #         frames.append(obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).cpu().numpy())

        #     step += 1

        successes.append(success)
        episode_lengths.append(step)

        msg = f"Episode {episode+1}/{args.num_episodes} | Success: {success} | Steps: {step} | Running SR: {np.mean(successes):.2%}"
        print(msg, flush=True)
        log.info(msg)

        if args.save_video and frames:
            import imageio
            video_path = os.path.join(
                args.video_dir,
                f"{args.model_type}_ep{episode:03d}_{'success' if success else 'fail'}.mp4"
            )
            imageio.mimsave(video_path, frames, fps=20)

    env.close()

    success_rate = np.mean(successes)
    avg_length = np.mean(episode_lengths)

    print("=" * 50, flush=True)
    print(f"RESULTS for {args.model_type.upper()} model:", flush=True)
    print(f"Success Rate: {success_rate:.2%} ({sum(successes)}/{args.num_episodes})", flush=True)
    print(f"Avg Episode Length: {avg_length:.1f} steps", flush=True)
    print("=" * 50, flush=True)

    return success_rate


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)