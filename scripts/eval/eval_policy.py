# scripts/eval/eval_policy.py
# Evaluate trained SmolVLA models on StackPyramid-v1
# Measures task success rate over N episodes

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import logging
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from transformers import AutoTokenizer
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from models.smolvla_moh.moh_policy import SmolVLAMoHPolicy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TASK_DESCRIPTION = "Pick up the red cube, place it next to the green cube, then stack the blue cube on top of the red and green cube to form a pyramid."

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace repo id or local path to checkpoint")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["base", "moh"],
                        help="Type of model: base or moh")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment visually")
    parser.add_argument("--save-video", action="store_true",
                        help="Save evaluation videos")
    parser.add_argument("--video-dir", type=str, default="results/eval_videos",
                        help="Directory to save videos")
    return parser.parse_args()


def load_policy(model_path, model_type, device):
    """Load the appropriate policy from checkpoint."""
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.images.base_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "observation.images.hand_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }

    if model_type == "moh":
        policy = SmolVLAMoHPolicy.from_pretrained(model_path)
    else:
        policy = SmolVLAPolicy.from_pretrained(model_path)

    policy.config.input_features = input_features
    policy.config.output_features = output_features
    policy = policy.to(device)
    policy.eval()
    return policy


def obs_to_batch(obs, tokenizer, device):
    """Convert ManiSkill3 observation to model batch format."""
    # Get robot state
    state = torch.tensor(obs["agent"]["qpos"], dtype=torch.float32).squeeze(0).unsqueeze(0)

    # Get camera images - convert from HWC to CHW and normalize
    base_img = torch.tensor(
            obs["sensor_data"]["base_camera"]["rgb"], dtype=torch.float32
        ).squeeze(0).permute(2, 0, 1).unsqueeze(0) / 255.0

    hand_img = torch.tensor(
        obs["sensor_data"]["hand_camera"]["rgb"], dtype=torch.float32
    ).squeeze(0).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Resize images to 128x128
    import torch.nn.functional as F
    base_img = F.interpolate(base_img, size=(128, 128), mode="bilinear", align_corners=False)
    hand_img = F.interpolate(hand_img, size=(128, 128), mode="bilinear", align_corners=False)

    # Tokenize task
    tokens = tokenizer(
        [TASK_DESCRIPTION],
        return_tensors="pt",
        padding="max_length",
        max_length=48,
        truncation=True,
    )

    batch = {
        "observation.state": state.to(device),
        "observation.images.base_camera": base_img.to(device),
        "observation.images.hand_camera": hand_img.to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }
    return batch


def evaluate(args):
    print(f"Starting evaluation: {args.num_episodes} episodes", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Evaluating on: {device}")
    log.info(f"Model: {args.model} ({args.model_type})")
    log.info(f"Episodes: {args.num_episodes}")

    # Load policy
    policy = load_policy(args.model, args.model_type, device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

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

    # Run evaluation
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
            batch = obs_to_batch(obs, tokenizer, device)

            with torch.no_grad():
                action = policy.select_action(batch)

            action_np = action.cpu().numpy().squeeze()
           
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated.any()) or bool(truncated.any())

            if bool(torch.tensor(info.get("success", False)).any()):
                success = True
                done = True

            if args.save_video:
                frames.append(obs["sensor_data"]["base_camera"]["rgb"].squeeze(0).cpu().numpy())

            step += 1

        successes.append(success)
        episode_lengths.append(step)

        msg = f"Episode {episode+1}/{args.num_episodes} | Success: {success} | Steps: {step} | Running SR: {np.mean(successes):.2%}"
        log.info(msg)
        print(msg, flush=True)

        # Save video if requested
        if args.save_video and frames:
            import imageio
            video_path = os.path.join(
                args.video_dir,
                f"{args.model_type}_ep{episode:03d}_{'success' if success else 'fail'}.mp4"
            )
            imageio.mimsave(video_path, frames, fps=20)

    env.close()

    # Final results
    success_rate = np.mean(successes)
    avg_length = np.mean(episode_lengths)

    log.info("=" * 50)
    log.info(f"RESULTS for {args.model_type.upper()} model:")
    log.info(f"Success Rate: {success_rate:.2%} ({sum(successes)}/{args.num_episodes})")
    log.info(f"Avg Episode Length: {avg_length:.1f} steps")
    log.info("=" * 50)

    return success_rate


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)