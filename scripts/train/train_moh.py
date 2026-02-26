# scripts/train/train_moh.py
# Training script for SmolVLA + Mixture of Horizons on StackPyramid-v1

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import logging
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature

from models.smolvla_moh.moh_config import SmolVLAMoHConfig
from models.smolvla_moh.moh_policy import SmolVLAMoHPolicy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TASK_DESCRIPTION = "Stack three cubes into a pyramid on the table."

DELTA_TIMESTAMPS = {
    "observation.state": [0],
    "observation.images.base_camera": [0],
    "observation.images.hand_camera": [0],
    "action": [i / 20 for i in range(50)],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def train(cfg):
    torch.manual_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device}")

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Load dataset with action chunks
    log.info(f"Loading dataset: {cfg.dataset.repo_id}")
    dataset = LeRobotDataset(cfg.dataset.repo_id, delta_timestamps=DELTA_TIMESTAMPS)
    log.info(f"Dataset: {len(dataset)} frames, {dataset.num_episodes} episodes")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Build MoH config and load pretrained weights
    moh_config = SmolVLAMoHConfig(
        horizons=list(cfg.moh.horizons),
        weighting=cfg.moh.weighting,
        moh_enabled=cfg.moh.enabled,
    )
    moh_config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.images.base_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "observation.images.hand_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    moh_config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }

    log.info(f"Loading pretrained SmolVLA from: {cfg.policy.path}")
    policy = SmolVLAMoHPolicy.from_pretrained(cfg.policy.path, config=moh_config)
    policy = policy.to(device)

    # Build tokenizer for language inputs
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    # Optimizer
    optimizer = moh_config.get_optimizer_preset().build(policy.parameters())
    scheduler = moh_config.get_scheduler_preset().build(optimizer, num_training_steps=cfg.training.steps)

    log.info("Starting training...")
    step = 0
    policy.train()

    while step < cfg.training.steps:
        for batch in dataloader:
            if step >= cfg.training.steps:
                break

            # Tokenize task description
            tokens = tokenizer(
                [TASK_DESCRIPTION] * len(batch["action"]),
                return_tensors="pt",
                padding="max_length",
                max_length=48,
                truncation=True,
            )
            batch["observation.language.tokens"] = tokens["input_ids"].to(device)
            batch["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(device)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, loss_info = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), moh_config.optimizer_grad_clip_norm
            )
            optimizer.step()
            scheduler.step()

            if step % cfg.training.log_freq == 0:
                log.info(f"Step {step}/{cfg.training.steps} | Loss: {loss.item():.4f}")
                if cfg.wandb.enable:
                    wandb.log({"train/loss": loss.item(), "step": step})
                    for k, v in loss_info.items():
                        if isinstance(v, torch.Tensor):
                            wandb.log({f"train/{k}": v.item(), "step": step})

            if step % cfg.training.save_freq == 0 and step > 0:
                ckpt_dir = output_dir / f"checkpoint_{step:06d}"
                policy.save_pretrained(ckpt_dir)
                log.info(f"Saved checkpoint: {ckpt_dir}")

            step += 1

    final_dir = output_dir / "checkpoint_final"
    policy.save_pretrained(final_dir)
    log.info(f"Done. Final checkpoint: {final_dir}")

    if cfg.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)