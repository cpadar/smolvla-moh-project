"""
Test MoH forward pass end to end.
Usage: python scripts/test_moh_forward.py
"""
import torch
import sys
sys.path.insert(0, '/home/ceshank/smolvla-moh-project')

from models.smolvla_moh.moh_config import SmolVLAMoHConfig
from models.smolvla_moh.moh_policy import SmolVLAMoHPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from torch.utils.data import DataLoader

device = torch.device('cuda')

DATASET_REPO = 'ceshank01/stack-pyramid-v1-v3'
TASK = "Pick up  a red cube, place it next to the green cube, then stack the blue cube on top of the red and green cube to form a pyramid."
BATCH_SIZE = 2

print('Loading dataset...')
dataset = LeRobotDataset(
    DATASET_REPO,
    delta_timestamps={
        'observation.images.base_camera': [0],
        'observation.images.hand_camera': [0],
        'observation.state': [0],
        'action': [i / 20 for i in range(50)],
    }
)
print(f'Dataset loaded: {len(dataset)} frames')

input_features = {
    'observation.images.camera1': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
    'observation.images.camera2': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
    'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(9,)),
}
output_features = {
    'action': PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
}

print('Loading policy...')
cfg = SmolVLAMoHConfig(pretrained_path='lerobot/smolvla_base')
cfg.input_features = input_features
cfg.output_features = output_features

policy = SmolVLAMoHPolicy.from_pretrained('lerobot/smolvla_base', config=cfg)
policy = policy.to(device).train()
print('Policy loaded')


preprocessor, _ = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path='lerobot/smolvla_base',
    preprocessor_overrides={'device_processor': {'device': 'cuda'}},
)

print('Loading batch...')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
batch = next(iter(dataloader))

batch['observation.images.camera1'] = batch.pop('observation.images.base_camera')
batch['observation.images.camera2'] = batch.pop('observation.images.hand_camera')
batch['task'] = [TASK] * BATCH_SIZE

batch = preprocessor(batch)
batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

print('Action shape:', batch['action'].shape)
print('State shape:', batch['observation.state'].shape)

print('Running forward pass...')
loss, loss_dict = policy.forward(batch)
print('Loss:', loss.item())
print('Loss dict:', loss_dict)
print('SUCCESS - MoH forward pass works!')