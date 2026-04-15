# # models/smolvla_moh/moh_policy.py
# # SmolVLA with Mixture of Horizons (MoH) - proper implementation
# # Based on "Mixture of Horizons in Action Chunking" (Jing et al., 2025)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
# from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
# from models.smolvla_moh.moh_config import SmolVLAMoHConfig


# class SmolVLAMoHModel(VLAFlowMatching):
#     """
#     Extends SmolVLAModel with MoH gating network and multi-horizon
#     forward/inference passes.
#     """

#     def __init__(self, config: SmolVLAMoHConfig):
#         super().__init__(config)
#         self.horizons = config.horizons
#         expert_hidden_size = self.vlm_with_expert.expert_hidden_size

#         # Gating network: outputs per-timestep weight for each horizon
#         # One scalar per timestep, reshaped later
#         self.gate_out_proj = nn.Linear(expert_hidden_size, 1)

#         # Learnable noise for gate during training (improves exploration)
#         self.gate_noise_layer = nn.Linear(expert_hidden_size, 1)
#         self.softplus = nn.Softplus()

#     def cv_squared(self, x):
#         """Coefficient of variation squared - used for load balancing loss."""
#         eps = 1e-10
#         if x.shape[0] == 1:
#             return torch.tensor(0.0, device=x.device, dtype=x.dtype)
#         return x.float().var() / (x.float().mean() ** 2 + eps)

#     def forward(self, images, img_masks, lang_tokens, lang_masks, state,
#                 actions, noise=None, time=None) -> Tensor:
#         """
#         MoH training forward pass with 3-part loss:
#         L = L_mix + lambda_ind * L_ind + lambda_bal * L_bal
#         """
#         num_horizons = len(self.horizons)
#         max_horizon = self.horizons[-1]
#         batch_size = actions.shape[0]

#         if noise is None:
#             noise = self.sample_noise(actions.shape, actions.device)
#         if time is None:
#             time = self.sample_time(batch_size, actions.device)

#         # Expand time for all horizons: (num_horizons, batch_size)
#         time_expanded = time.unsqueeze(0).expand(num_horizons, -1)

#         # Compute noisy actions for all horizons simultaneously
#         # x_t: (num_horizons, batch, max_horizon, action_dim)
#         x_t = (time_expanded[:, :, None, None] * noise.unsqueeze(0) +
#                (1 - time_expanded[:, :, None, None]) * actions.unsqueeze(0))
#         u_t = noise - actions  # target velocity

#         # STAGE 1: Compute prefix KV cache once (expensive VLM pass)
#         prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
#             images, img_masks, lang_tokens, lang_masks, state=state
#         )
#         prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
#         prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
#         _, prefix_past_key_values = self.vlm_with_expert.forward(
#             attention_mask=prefix_att_2d_masks,
#             position_ids=prefix_position_ids,
#             past_key_values=None,
#             inputs_embeds=[prefix_embs, None],
#             use_cache=True,
#             fill_kv_cache=True,
#         )

#         # STAGE 2: Run suffix pass for each horizon separately
#         # reusing the same prefix KV cache
#         all_suffix_outs = []
        
#         for h_idx, h in enumerate(self.horizons):
#             # Get this horizon's noisy actions: (B, max_h, D)
#             x_t_h = x_t[h_idx]  # (B, max_h, D)
#             time_h = time_expanded[h_idx]  # (B,)

#             suffix_embs_h, suffix_pad_masks_h, suffix_att_masks_h = self.embed_suffix(
#                 x_t_h, time_h
#             )

#             suffix_len = suffix_pad_masks_h.shape[1]
#             prefix_len = prefix_pad_masks.shape[1]
#             prefix_pad_2d_masks_h = prefix_pad_masks[:, None, :].expand(
#                 batch_size, suffix_len, prefix_len
#             )
#             suffix_att_2d_masks_h = make_att_2d_masks(suffix_pad_masks_h, suffix_att_masks_h)
#             full_att_2d_masks_h = torch.cat([prefix_pad_2d_masks_h, suffix_att_2d_masks_h], dim=2)

#             prefix_offsets_h = torch.sum(prefix_pad_masks, dim=-1)[:, None]
#             position_ids_h = prefix_offsets_h + torch.cumsum(suffix_pad_masks_h, dim=1) - 1

#             outputs_embeds_h, _ = self.vlm_with_expert.forward(
#                 attention_mask=full_att_2d_masks_h,
#                 position_ids=position_ids_h,
#                 past_key_values=prefix_past_key_values,
#                 inputs_embeds=[None, suffix_embs_h],
#                 use_cache=False,
#                 fill_kv_cache=False,
#             )
#             suffix_out_h = outputs_embeds_h[1].to(torch.float32)
#             all_suffix_outs.append(suffix_out_h)

#         # Stack: (num_h, B, max_h, hidden)
#         # For gate and loss computation
#         suffix_out_stacked = torch.stack(all_suffix_outs, dim=0)  # (num_h, B, max_h, hidden)


#         # # STAGE 2: Batched suffix pass for all horizons
#         # # Repeat prefix masks and KV cache for each horizon
#         # batched_prefix_pad_masks = prefix_pad_masks.repeat_interleave(num_horizons, dim=0)
#         # batched_past_key_values = self._repeat_past_key_values(
#         #     prefix_past_key_values, num_horizons
#         # )

#         # # # Reshape x_t: (num_h, B, L, D) -> (B*num_h, L, D)
#         # # batched_x_t = x_t.permute(1, 0, 2, 3).reshape(
#         # #     batch_size * num_horizons, max_horizon, -1
#         # # )
#         # # Reshape x_t: (num_h, B, L, D) -> (B*num_h, L, D)
#         # print('DEBUG x_t shape:', x_t.shape)
#         # print('DEBUG actions shape:', actions.shape)
#         # print('DEBUG batch_size:', batch_size)
#         # print('DEBUG num_horizons:', num_horizons)
#         # print('DEBUG max_horizon:', max_horizon)
#         # batched_x_t = x_t.permute(1, 0, 2, 3).reshape(
#         #     batch_size * num_horizons, max_horizon, -1
#         # )


#         # # time: (num_h, B) -> (B*num_h,)
#         # batched_time = time_expanded.permute(1, 0).reshape(batch_size * num_horizons)

#         # # Action padding mask: shorter horizons are padded
#         # # Shape: (num_h, max_h) -> True where valid
#         # action_pad_mask = (torch.arange(max_horizon, device=actions.device)[None, :] < torch.tensor(self.horizons, device=actions.device)[:, None])
        
#         # # (num_h, B, max_h) -> (B*num_h, max_h)
#         # action_pad_mask = action_pad_mask.unsqueeze(1).expand(-1, batch_size, -1)
#         # batched_action_pad_mask = action_pad_mask.permute(1, 0, 2).reshape(
#         #     batch_size * num_horizons, max_horizon
#         # )

#         # # Embed suffix for all horizons at once
#         # suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
#         #     batched_x_t, batched_time
#         # )

    
#         # # Build attention masks - same as denoise_step in baseline
#         # suffix_len = suffix_pad_masks.shape[1]
#         # prefix_len = batched_prefix_pad_masks.shape[1]
#         # prefix_pad_2d_masks = batched_prefix_pad_masks[:, None, :].expand(
#         #     batch_size * num_horizons, suffix_len, prefix_len
#         # )
#         # suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
#         # full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

#         # prefix_offsets = torch.sum(batched_prefix_pad_masks, dim=-1)[:, None]
#         # position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

#         # print('DEBUG full_att_2d_masks shape:', full_att_2d_masks.shape)
#         # print('DEBUG full_att_2d_masks dtype:', full_att_2d_masks.dtype)
#         # print('DEBUG suffix_embs shape:', suffix_embs.shape)
#         # print('DEBUG position_ids shape:', position_ids.shape)
#         # print('DEBUG batched_past_key_values keys:', list(batched_past_key_values.keys()))
#         # print('DEBUG first layer key_states shape:', batched_past_key_values[0]['key_states'].shape)

#         # outputs_embeds, _ = self.vlm_with_expert.forward(
#         #     attention_mask=full_att_2d_masks,
#         #     position_ids=position_ids,
#         #     past_key_values=batched_past_key_values,
#         #     inputs_embeds=[None, suffix_embs],
#         #     use_cache=False,
#         #     fill_kv_cache=False,
#         # )
#         # suffix_out = outputs_embeds[1].to(torch.float32)

#         # Project to action space
#         v_t_batched = self.action_out_proj(suffix_out)
#         # (B*num_h, max_h, D) -> (B, num_h, max_h, D) -> (num_h, B, max_h, D)
#         all_v_t_preds = v_t_batched.view(
#             batch_size, num_horizons, max_horizon, -1
#         ).permute(1, 0, 2, 3)

#         # LOSS 1: Individual loss per horizon head
#         all_head_losses = []
#         for i, h in enumerate(self.horizons):
#             v_t_head = all_v_t_preds[i, :, :h, :self.config.max_action_dim]
#             target = u_t[:, :h, :]
#             all_head_losses.append(F.mse_loss(v_t_head, target))
#         individual_loss = torch.sum(torch.stack(all_head_losses))

#         # Compute gate logits
#         # suffix_out: (B*num_h, max_h, hidden)
#         gate_logits = self.gate_out_proj(suffix_out)
#         gate_logits = gate_logits[:, :max_horizon, :]  # (B*num_h, max_h, 1)

#         # Add learnable noise for exploration
#         noise_epsilon = 1e-2
#         raw_noise_stddev = self.gate_noise_layer(suffix_out)
#         raw_noise_stddev = raw_noise_stddev[:, :max_horizon, :]
#         noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
#         gate_logits = gate_logits + (torch.randn_like(gate_logits) * noise_stddev)

#         # Reshape: (B*num_h, max_h, 1) -> (B, max_h, num_h)
#         gate_logits = gate_logits.reshape(
#             batch_size, num_horizons, max_horizon
#         ).permute(0, 2, 1)

#         # Mask out invalid horizons (steps beyond each horizon's length)
#         valid_heads_mask = torch.tensor(
#             [[step < h for h in self.horizons] for step in range(max_horizon)],
#             device=actions.device, dtype=torch.bool
#         ).unsqueeze(0)  # (1, max_h, num_h)
#         masked_gate_logits = torch.where(
#             valid_heads_mask, gate_logits,
#             torch.finfo(gate_logits.dtype).min
#         )
#         gate_weights = F.softmax(masked_gate_logits, dim=-1)  # (B, max_h, num_h)

#         # LOSS 2: Auxiliary/mixture loss on fused predictions
#         # all_v_t_preds: (num_h, B, max_h, D) -> (B, num_h, max_h, D)
#         all_v_t_preds_t = all_v_t_preds.permute(1, 0, 2, 3)
#         # gate_weights: (B, max_h, num_h) -> (B, num_h, max_h, 1)
#         v_t_combined = (
#             gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v_t_preds_t
#         ).sum(dim=1)  # (B, max_h, D)
#         auxiliary_loss = F.mse_loss(
#             v_t_combined[:, :, :self.config.max_action_dim], u_t
#         )

#         # LOSS 3: Load balancing loss
#         loss_components = []
#         boundaries = sorted(list(set([0] + self.horizons)))
#         for i in range(len(boundaries) - 1):
#             start_step, end_step = boundaries[i], boundaries[i + 1]
#             active_expert_indices = [
#                 idx for idx, h in enumerate(self.horizons) if h > start_step
#             ]
#             if len(active_expert_indices) > 1:
#                 segment_gate_weights = gate_weights[:, start_step:end_step, :]
#                 active_expert_weights = segment_gate_weights[:, :, active_expert_indices]
#                 avg_expert_prob = active_expert_weights.mean(dim=(0, 1))
#                 loss_components.append(self.cv_squared(avg_expert_prob))

#         load_balancing_loss = torch.mean(torch.stack(loss_components))

#         # Combined loss: L = L_mix + 1.0 * L_ind + 0.001 * L_bal
#         total_loss = auxiliary_loss + 1.0 * individual_loss + 0.001 * load_balancing_loss

#         # Return in same format as parent (per-sample losses)
#         # For compatibility with LeRobot training loop
#         losses = F.mse_loss(v_t_combined[:, :, :self.config.max_action_dim],
#                             u_t, reduction="none")
#         return losses

#     def sample_actions(self, images, img_masks, lang_tokens, lang_masks,
#                        state, noise=None, **kwargs) -> Tensor:
#         """
#         MoH inference: batched multi-horizon denoising with gated fusion.
#         """
#         num_horizons = len(self.horizons)
#         max_horizon = self.horizons[-1]
#         bsize = state.shape[0]
#         device = state.device

#         if noise is None:
#             actions_shape = (bsize, max_horizon, self.config.max_action_dim)
#             noise = self.sample_noise(actions_shape, device)

#         # STAGE 1: Prefix KV cache (once)
#         prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
#             images, img_masks, lang_tokens, lang_masks, state=state
#         )
#         prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
#         prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

#         _, past_key_values = self.vlm_with_expert.forward(
#             attention_mask=prefix_att_2d_masks,
#             position_ids=prefix_position_ids,
#             past_key_values=None,
#             inputs_embeds=[prefix_embs, None],
#             use_cache=True,
#             fill_kv_cache=True,
#         )

#         # Repeat prefix for all horizons
#         batched_prefix_pad_masks = prefix_pad_masks.repeat_interleave(num_horizons, dim=0)
#         batched_past_key_values = self._repeat_past_key_values(past_key_values, num_horizons)

#         # STAGE 2: Iterative denoising
#         num_steps = self.config.num_steps
#         dt = -1.0 / num_steps
#         x_t = noise  # (B, max_h, D)

#         for step in range(num_steps):
#             time = 1.0 + step * dt
#             time_tensor = torch.tensor(
#                 time, dtype=torch.float32, device=device
#             ).expand(bsize * num_horizons)

#             # Pad each horizon's x_t to max_horizon
#             padded_x_t_list = []
#             for h in self.horizons:
#                 padded = F.pad(x_t[:, :h, :], (0, 0, 0, max_horizon - h))
#                 padded_x_t_list.append(padded)
#             batched_x_t = torch.cat(padded_x_t_list, dim=0)  # (B*num_h, max_h, D)

#             suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
#                 batched_x_t, time_tensor
#             )

#             # Build attention masks - same as denoise_step in baseline
#             suffix_len = suffix_pad_masks.shape[1]
#             prefix_len = batched_prefix_pad_masks.shape[1]
#             prefix_pad_2d_masks = batched_prefix_pad_masks[:, None, :].expand(
#                 bsize * num_horizons, suffix_len, prefix_len
#             )
#             suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
#             full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)


#             prefix_offsets = torch.sum(batched_prefix_pad_masks, dim=-1)[:, None]
#             position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

#             outputs_embeds, _ = self.vlm_with_expert.forward(
#                 attention_mask=full_att_2d_masks,
#                 position_ids=position_ids,
#                 past_key_values=batched_past_key_values,
#                 inputs_embeds=[None, suffix_embs],
#                 use_cache=False,
#                 fill_kv_cache=False,
#             )
#             suffix_out = outputs_embeds[1].to(torch.float32)

#             # Gate weights
#             gate_logits = self.gate_out_proj(suffix_out)
#             gate_logits = gate_logits[:, :max_horizon, :]
#             gate_logits = gate_logits.reshape(
#                 bsize, num_horizons, max_horizon
#             ).permute(0, 2, 1)

#             valid_heads_mask = torch.tensor(
#                 [[step_i < h for h in self.horizons] for step_i in range(max_horizon)],
#                 device=device, dtype=torch.bool
#             ).unsqueeze(0)
#             masked_gate_logits = torch.where(
#                 valid_heads_mask, gate_logits,
#                 torch.finfo(gate_logits.dtype).min
#             )
#             gate_weights = F.softmax(masked_gate_logits, dim=-1)

#             # Fuse predictions
#             v_t_batched = self.action_out_proj(suffix_out)
#             v_t_actions = v_t_batched[:, :max_horizon, :]
#             # (B*num_h, max_h, D) -> (B, num_h, max_h, D)
#             all_v_t_preds = v_t_actions.view(
#                 num_horizons, bsize, max_horizon, -1
#             ).permute(1, 0, 2, 3)
#             # Fuse: (B, max_h, D)
#             v_t = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v_t_preds).sum(dim=1)

#             x_t = x_t + dt * v_t

#         return x_t
    
#     def _repeat_past_key_values(self, past_key_values, n_repeats):
#         """Repeat KV cache for batched horizon processing.
#         LeRobot SmolVLA uses dict format: {layer_idx: {'key_states': k, 'value_states': v}}
#         """
#         if past_key_values is None:
#             return None
#         repeated = {}
#         for layer_idx, layer_cache in past_key_values.items():
#             repeated[layer_idx] = {
#                 "key_states": layer_cache["key_states"].repeat_interleave(n_repeats, dim=0),
#                 "value_states": layer_cache["value_states"].repeat_interleave(n_repeats, dim=0),
#             }
#         return repeated


# class SmolVLAMoHPolicy(SmolVLAPolicy):
#     """SmolVLA policy with Mixture of Horizons action chunking."""

#     def __init__(self, config: SmolVLAMoHConfig):
#         super().__init__(config)
#         # Replace the standard model with MoH model
#         self.model = SmolVLAMoHModel(config)



# models/smolvla_moh/moh_policy.py
# SmolVLA with Mixture of Horizons (MoH) - proper implementation
# Based on "Mixture of Horizons in Action Chunking" (Jing et al., 2025)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from models.smolvla_moh.moh_config import SmolVLAMoHConfig


class SmolVLAMoHModel(VLAFlowMatching):
    """
    Extends VLAFlowMatching with MoH gating network and multi-horizon
    forward/inference passes.
    """

    def __init__(self, config: SmolVLAMoHConfig):
        super().__init__(config)
        self.horizons = config.horizons
        expert_hidden_size = self.vlm_with_expert.expert_hidden_size

        # Gating network: outputs per-timestep weight for each horizon
        self.gate_out_proj = nn.Linear(expert_hidden_size, 1)

        # Learnable noise for gate during training (improves exploration)
        self.gate_noise_layer = nn.Linear(expert_hidden_size, 1)
        self.softplus = nn.Softplus()

    def cv_squared(self, x):
        """Coefficient of variation squared - used for load balancing loss."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _suffix_pass(self, x_t_h, time_h, prefix_pad_masks, prefix_past_key_values):
        """Run one suffix forward pass for a single horizon, reusing prefix KV cache."""
        suffix_embs_h, suffix_pad_masks_h, suffix_att_masks_h = self.embed_suffix(x_t_h, time_h)

        suffix_len = suffix_pad_masks_h.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]

        prefix_pad_2d_masks_h = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks_h = make_att_2d_masks(suffix_pad_masks_h, suffix_att_masks_h)
        full_att_2d_masks_h = torch.cat([prefix_pad_2d_masks_h, suffix_att_2d_masks_h], dim=2)

        prefix_offsets_h = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids_h = prefix_offsets_h + torch.cumsum(suffix_pad_masks_h, dim=1) - 1

        outputs_embeds_h, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks_h,
            position_ids=position_ids_h,
            past_key_values=prefix_past_key_values,
            inputs_embeds=[None, suffix_embs_h],
            use_cache=False,
            fill_kv_cache=False,
        )
        return outputs_embeds_h[1].to(torch.float32)

    def forward(self, images, img_masks, lang_tokens, lang_masks, state,
                actions, noise=None, time=None) -> Tensor:
        """
        MoH training forward pass with 3-part loss:
        L = L_mix + lambda_ind * L_ind + lambda_bal * L_bal
        """
        num_horizons = len(self.horizons)
        max_horizon = self.horizons[-1]
        batch_size = actions.shape[0]

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(batch_size, actions.device)

        # Expand time for all horizons: (num_horizons, batch_size)
        time_expanded = time.unsqueeze(0).expand(num_horizons, -1)

        # x_t: (num_horizons, batch, max_horizon, action_dim)
        x_t = (time_expanded[:, :, None, None] * noise.unsqueeze(0) +
               (1 - time_expanded[:, :, None, None]) * actions.unsqueeze(0))
        u_t = noise - actions  # target velocity

        # STAGE 1: Compute prefix KV cache once
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, prefix_past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        # STAGE 2: Run suffix pass per horizon, reusing prefix KV cache
        all_suffix_outs = []
        for h_idx in range(num_horizons):
            x_t_h = x_t[h_idx]       # (B, max_h, D)
            time_h = time_expanded[h_idx]  # (B,)
            suffix_out_h = self._suffix_pass(
                x_t_h, time_h, prefix_pad_masks, prefix_past_key_values
            )
            all_suffix_outs.append(suffix_out_h)

        # Project each horizon to action space and compute individual losses
        all_head_losses = []
        all_v_t_preds_list = []
        for i, h in enumerate(self.horizons):
            v_t_h = self.action_out_proj(all_suffix_outs[i])  # (B, max_h, D)
            all_v_t_preds_list.append(v_t_h)
            v_t_head = v_t_h[:, :h, :self.config.max_action_dim]
            target = u_t[:, :h, :]
            all_head_losses.append(F.mse_loss(v_t_head, target))

        # LOSS 1: Individual loss
        individual_loss = torch.sum(torch.stack(all_head_losses))

        # Stack predictions: (B, num_h, max_h, D)
        all_v_t_preds_t = torch.stack(all_v_t_preds_list, dim=1)

        # Compute gate logits from stacked suffix outputs
        # suffix_out_stacked: (num_h, B, max_h, hidden) -> (B, num_h, max_h, hidden)
        # -> (B*num_h, max_h, hidden) for gate projection
        suffix_out_stacked = torch.stack(all_suffix_outs, dim=0)  # (num_h, B, max_h, hidden)
        suffix_out_for_gate = suffix_out_stacked.permute(1, 0, 2, 3).reshape(
            batch_size * num_horizons, max_horizon, -1
        )

        gate_logits = self.gate_out_proj(suffix_out_for_gate)
        gate_logits = gate_logits[:, :max_horizon, :]  # (B*num_h, max_h, 1)

        # Add learnable noise for exploration
        noise_epsilon = 1e-2
        raw_noise_stddev = self.gate_noise_layer(suffix_out_for_gate)
        raw_noise_stddev = raw_noise_stddev[:, :max_horizon, :]
        noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
        gate_logits = gate_logits + (torch.randn_like(gate_logits) * noise_stddev)

        # Reshape: (B*num_h, max_h, 1) -> (B, max_h, num_h)
        gate_logits = gate_logits.reshape(
            batch_size, num_horizons, max_horizon
        ).permute(0, 2, 1)

        # Mask invalid horizons
        valid_heads_mask = torch.tensor(
            [[step < h for h in self.horizons] for step in range(max_horizon)],
            device=actions.device, dtype=torch.bool
        ).unsqueeze(0)  # (1, max_h, num_h)
        masked_gate_logits = torch.where(
            valid_heads_mask, gate_logits,
            torch.finfo(gate_logits.dtype).min
        )
        gate_weights = F.softmax(masked_gate_logits, dim=-1)  # (B, max_h, num_h)

        # LOSS 2: Auxiliary/mixture loss on fused predictions
        # gate_weights: (B, max_h, num_h) -> (B, num_h, max_h, 1)
        v_t_combined = (
            gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v_t_preds_t
        ).sum(dim=1)  # (B, max_h, D)
        auxiliary_loss = F.mse_loss(
            v_t_combined[:, :, :self.config.max_action_dim], u_t
        )

        # LOSS 3: Load balancing loss
        loss_components = []
        boundaries = sorted(list(set([0] + self.horizons)))
        for i in range(len(boundaries) - 1):
            start_step, end_step = boundaries[i], boundaries[i + 1]
            active_expert_indices = [
                idx for idx, h in enumerate(self.horizons) if h > start_step
            ]
            if len(active_expert_indices) > 1:
                segment_gate_weights = gate_weights[:, start_step:end_step, :]
                active_expert_weights = segment_gate_weights[:, :, active_expert_indices]
                avg_expert_prob = active_expert_weights.mean(dim=(0, 1))
                loss_components.append(self.cv_squared(avg_expert_prob))

        load_balancing_loss = torch.mean(torch.stack(loss_components))

        # Combined loss: L = L_mix + 1.0 * L_ind + 0.001 * L_bal
        total_loss = auxiliary_loss + 1.0 * individual_loss + 0.001 * load_balancing_loss

        # Return per-sample losses scaled by total_loss so all three components
        # contribute gradients during backprop
        losses = F.mse_loss(
            v_t_combined[:, :, :self.config.max_action_dim],
            u_t, reduction="none"
        )
        loss_scale = total_loss / (auxiliary_loss.detach() + 1e-8)
        return losses * loss_scale

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks,
                       state, noise=None, **kwargs) -> Tensor:
        """
        MoH inference: multi-horizon denoising with gated fusion.
        """
        num_horizons = len(self.horizons)
        max_horizon = self.horizons[-1]
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, max_horizon, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # STAGE 1: Prefix KV cache (once)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        # STAGE 2: Iterative denoising
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise  # (B, max_h, D)

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(
                time_val, dtype=torch.float32, device=device
            ).expand(bsize)

            # Run suffix pass per horizon
            all_suffix_outs = []
            all_v_t_preds = []
            for h in self.horizons:
                x_t_h = x_t[:, :h, :]  # (B, h, D)
                # Pad to max_horizon
                x_t_h_padded = F.pad(x_t_h, (0, 0, 0, max_horizon - h))
                suffix_out_h = self._suffix_pass(
                    x_t_h_padded, time_tensor, prefix_pad_masks, past_key_values
                )
                all_suffix_outs.append(suffix_out_h)
                v_t_h = self.action_out_proj(suffix_out_h)
                all_v_t_preds.append(v_t_h)

            # Stack: (num_h, B, max_h, hidden)
            suffix_out_stacked = torch.stack(all_suffix_outs, dim=0)
            suffix_out_for_gate = suffix_out_stacked.permute(1, 0, 2, 3).reshape(
                bsize * num_horizons, max_horizon, -1
            )

            # Gate weights
            gate_logits = self.gate_out_proj(suffix_out_for_gate)
            gate_logits = gate_logits[:, :max_horizon, :]
            gate_logits = gate_logits.reshape(
                bsize, num_horizons, max_horizon
            ).permute(0, 2, 1)

            valid_heads_mask = torch.tensor(
                [[step_i < h for h in self.horizons] for step_i in range(max_horizon)],
                device=device, dtype=torch.bool
            ).unsqueeze(0)
            masked_gate_logits = torch.where(
                valid_heads_mask, gate_logits,
                torch.finfo(gate_logits.dtype).min
            )
            gate_weights = F.softmax(masked_gate_logits, dim=-1)  # (B, max_h, num_h)

            # Fuse predictions: (B, num_h, max_h, D)
            all_v_t_preds_t = torch.stack(all_v_t_preds, dim=1)
            v_t = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v_t_preds_t).sum(dim=1)

            x_t = x_t + dt * v_t

        return x_t


class SmolVLAMoHPolicy(SmolVLAPolicy):
    """SmolVLA policy with Mixture of Horizons action chunking."""

    def __init__(self, config: SmolVLAMoHConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Replace the standard model with MoH model
        self.model = SmolVLAMoHModel(config)