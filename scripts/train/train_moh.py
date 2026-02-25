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
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint

from models.smolvla_moh.moh_config import SmolVLAMoHConfig
from models.smolvla_moh.moh_policy import SmolVLAMoHPolicy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train config yaml")
    return parser.parse_args()


def train(cfg):
    # Setup
    torch.manual_seed(cfg.experiment.seed)
    device = torch.device(cfg.policy.device if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")

    # Setup output directories
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Load dataset
    log.info(f"Loading dataset: {cfg.dataset.repo_id}")
    dataset = LeRobotDataset(cfg.dataset.repo_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    log.info(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")

    # Build MoH config
    moh_config = SmolVLAMoHConfig(
        horizons=list(cfg.moh.horizons),
        weighting=cfg.moh.weighting,
        moh_enabled=cfg.moh.enabled,
    )

    # Load pretrained SmolVLA weights and wrap with MoH
    log.info(f"Loading pretrained SmolVLA from: {cfg.policy.path}")
    policy = SmolVLAMoHPolicy.from_pretrained(cfg.policy.path, config=moh_config)
    policy = policy.to(device)

    # Optimizer — include horizon_weights in optimization
    optimizer = moh_config.get_optimizer_preset().build(policy.parameters())
    scheduler = moh_config.get_scheduler_preset().build(optimizer)

    # Training loop
    log.info("Starting training...")
    step = 0
    policy.train()

    while step < cfg.training.steps:
        for batch in dataloader:
            if step >= cfg.training.steps:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            loss, loss_info = policy.forward(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), moh_config.optimizer_grad_clip_norm)
            optimizer.step()
            scheduler.step()

            # Logging
            if step % cfg.training.log_freq == 0:
                log.info(f"Step {step}/{cfg.training.steps} | Loss: {loss.item():.4f}")
                if cfg.wandb.enable:
                    wandb.log({"train/loss": loss.item(), "train/step": step})
                    for k, v in loss_info.items():
                        wandb.log({f"train/{k}": v, "train/step": step})

            # Save checkpoint
            if step % cfg.training.save_freq == 0 and step > 0:
                ckpt_dir = output_dir / f"checkpoint_{step:06d}"
                policy.save_pretrained(ckpt_dir)
                log.info(f"Saved checkpoint to {ckpt_dir}")

            step += 1

    # Save final checkpoint
    final_dir = output_dir / "checkpoint_final"
    policy.save_pretrained(final_dir)
    log.info(f"Training complete. Final checkpoint saved to {final_dir}")

    if cfg.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)