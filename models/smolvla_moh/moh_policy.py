# models/smolvla_moh/moh_policy.py
# SmolVLA with Mixture of Horizons action chunking

from networkx import config
import torch
import torch.nn as nn
from torch import Tensor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from models.smolvla_moh.moh_config import SmolVLAMoHConfig


class HorizonWeights(nn.Module):
    """Learnable weights for fusing multiple horizon predictions."""
    
    def __init__(self, num_horizons: int):
        super().__init__()
        # Initialize equal weights for all horizons
        self.weights = nn.Parameter(torch.ones(num_horizons) / num_horizons)
    
    def forward(self) -> Tensor:
        # Softmax ensures weights sum to 1 and are all positive
        return torch.softmax(self.weights, dim=0)


class SmolVLAMoHPolicy(SmolVLAPolicy):
    """
    SmolVLA with Mixture of Horizons (MoH) action chunking.
    
    Instead of predicting a single action chunk of fixed length,
    MoH predicts action chunks at multiple horizons (e.g. 5, 15, 50 steps)
    and fuses them with learned weights.
    
    Short horizons: precise short-term actions
    Long horizons: coherent long-term planning
    MoH: inherits strengths of both
    """
    
    def __init__(self, config):
        super().__init__(config)
    
        # Handle case where base SmolVLAConfig is passed instead of SmolVLAMoHConfig
        if not hasattr(config, 'moh_enabled'):
            config.moh_enabled = True
            config.horizons = [5, 15, 50]
            config.weighting = "learned"
    
        self.moh_config = config
    
        if config.moh_enabled and config.weighting == "learned":
            self.horizon_weights = HorizonWeights(len(config.horizons))
        else:
            self.horizon_weights = None
    
    def _get_action_chunk(self, batch, noise=None, **kwargs) -> Tensor:
        """
        Override SmolVLA's action chunk generation with MoH fusion.
        
        For each horizon h in self.moh_config.horizons:
            1. Sample actions with chunk_size=h
            2. Pad to max_horizon length
        Then fuse all horizon predictions with learned or uniform weights.
        """
        if not self.moh_config.moh_enabled:
            # Fall back to standard SmolVLA behavior
            return super()._get_action_chunk(batch, noise, **kwargs)
        
        # Prepare inputs (same as parent class)
        for k in batch:
            if k in self._queues and k != "action":
                batch[k] = torch.stack(list(self._queues[k]), dim=1)
        
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch["observation.language_tokens"]
        lang_masks = batch["observation.language_attention_mask"]
        
        max_horizon = max(self.moh_config.horizons)
        horizon_actions = []
        
        # Sample actions at each horizon
        for horizon in self.moh_config.horizons:
            # Temporarily override chunk size
            original_chunk_size = self.config.chunk_size
            self.config.chunk_size = horizon
            
            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )
            
            # Restore chunk size
            self.config.chunk_size = original_chunk_size
            
            # Unpad to original action dim
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            
            # Pad shorter horizons to max_horizon length by repeating last action
            if actions.shape[1] < max_horizon:
                pad_length = max_horizon - actions.shape[1]
                last_action = actions[:, -1:, :].expand(-1, pad_length, -1)
                actions = torch.cat([actions, last_action], dim=1)
            
            horizon_actions.append(actions)
        
        # Stack: (num_horizons, batch, max_horizon, action_dim)
        horizon_actions = torch.stack(horizon_actions, dim=0)
        
        # Get fusion weights
        if self.horizon_weights is not None:
            weights = self.horizon_weights()  # (num_horizons,)
        else:
            # Uniform weights
            num_horizons = len(self.moh_config.horizons)
            weights = torch.ones(num_horizons, device=horizon_actions.device) / num_horizons
        
        # Weighted sum: (batch, max_horizon, action_dim)
        weights = weights.view(-1, 1, 1, 1)
        fused_actions = (weights * horizon_actions).sum(dim=0)
        
        return fused_actions