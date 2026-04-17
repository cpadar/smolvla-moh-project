# models/smolvla_moh/moh_policy.py
# SmolVLA with Mixture of Horizons (MoH)
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
    Extends VLAFlowMatching with MoH gating network.
    Gate layers are added on top of pretrained weights.
    """

    def __init__(self, config: SmolVLAMoHConfig):
        super().__init__(config)
        self.horizons = config.horizons
        expert_hidden_size = self.vlm_with_expert.expert_hidden_size

        # New learnable layers for MoH (initialized randomly, rest uses pretrained weights)
        self.gate_out_proj = nn.Linear(expert_hidden_size, 1)
        self.gate_noise_layer = nn.Linear(expert_hidden_size, 1)
        self.softplus = nn.Softplus()

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _suffix_pass(self, x_t_h, time_h, prefix_pad_masks, prefix_past_key_values):
        """Run one suffix forward pass for a single horizon, reusing prefix KV cache.
        Identical to baseline denoise_step but returns suffix_out instead of v_t.
        """
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
            use_cache=True,
            fill_kv_cache=False,
        )
        return outputs_embeds_h[1].to(torch.float32)

    def forward(self, images, img_masks, lang_tokens, lang_masks, state,
                actions, noise=None, time=None) -> Tensor:
        """
        MoH training forward pass.
        Returns per-sample losses compatible with LeRobot training loop.
        Total loss = L_mix + L_ind + 0.001 * L_bal backpropagated via total_loss.
        """
        num_horizons = len(self.horizons)
        max_horizon = self.horizons[-1]
        batch_size = actions.shape[0]

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(batch_size, actions.device)

        time_expanded = time.unsqueeze(0).expand(num_horizons, -1)

        # Noisy actions for all horizons: (num_h, B, max_h, D)
        x_t = (time_expanded[:, :, None, None] * noise.unsqueeze(0) +
               (1 - time_expanded[:, :, None, None]) * actions.unsqueeze(0))
        u_t = noise - actions  # flow matching target

        # STAGE 1: Prefix KV cache (one expensive VLM pass)
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

        # STAGE 2: Per-horizon suffix passes reusing prefix KV cache
        all_suffix_outs = []
        for h_idx in range(num_horizons):
            suffix_out_h = self._suffix_pass(
                x_t[h_idx], time_expanded[h_idx],
                prefix_pad_masks, prefix_past_key_values
            )
            all_suffix_outs.append(suffix_out_h)

        # Project to action space
        all_v_t_preds_list = []
        all_head_losses = []
        for i, h in enumerate(self.horizons):
            v_t_h = self.action_out_proj(all_suffix_outs[i])  # (B, max_h, D)
            all_v_t_preds_list.append(v_t_h)
            # Individual loss: only over valid horizon steps
            v_t_head = v_t_h[:, :h, :self.config.max_action_dim]
            target = u_t[:, :h, :]
            all_head_losses.append(F.mse_loss(v_t_head, target))

        # LOSS 1: Individual
        individual_loss = torch.sum(torch.stack(all_head_losses))

        # Gate logits: reshape suffix outs for gate projection
        suffix_out_stacked = torch.stack(all_suffix_outs, dim=0)  # (num_h, B, max_h, hidden)
        suffix_out_for_gate = suffix_out_stacked.permute(1, 0, 2, 3).reshape(
            batch_size * num_horizons, max_horizon, -1
        )

        gate_logits = self.gate_out_proj(suffix_out_for_gate)[:, :max_horizon, :]

        # Learnable noise for gate exploration
        noise_stddev = self.softplus(
            self.gate_noise_layer(suffix_out_for_gate)[:, :max_horizon, :]
        ) + 1e-2
        gate_logits = gate_logits + torch.randn_like(gate_logits) * noise_stddev

        # Reshape to (B, max_h, num_h)
        gate_logits = gate_logits.reshape(
            batch_size, num_horizons, max_horizon
        ).permute(0, 2, 1)

        # Mask steps beyond each horizon's length
        valid_heads_mask = torch.tensor(
            [[step < h for h in self.horizons] for step in range(max_horizon)],
            device=actions.device, dtype=torch.bool
        ).unsqueeze(0)
        gate_weights = F.softmax(
            torch.where(valid_heads_mask, gate_logits, torch.finfo(gate_logits.dtype).min),
            dim=-1
        )  # (B, max_h, num_h)

        # Fused prediction
        all_v_t_preds_t = torch.stack(all_v_t_preds_list, dim=1)  # (B, num_h, max_h, D)
        v_t_combined = (
            gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v_t_preds_t
        ).sum(dim=1)  # (B, max_h, D)

        # LOSS 2: Auxiliary (mixture)
        auxiliary_loss = F.mse_loss(
            v_t_combined[:, :, :self.config.max_action_dim], u_t
        )

        # LOSS 3: Load balancing
        loss_components = []
        boundaries = sorted(list(set([0] + self.horizons)))
        for i in range(len(boundaries) - 1):
            start_step = boundaries[i]
            active_indices = [idx for idx, h in enumerate(self.horizons) if h > start_step]
            if len(active_indices) > 1:
                seg_weights = gate_weights[:, start_step:boundaries[i+1], :]
                avg_prob = seg_weights[:, :, active_indices].mean(dim=(0, 1))
                loss_components.append(self.cv_squared(avg_prob))
        load_balancing_loss = torch.mean(torch.stack(loss_components))

        # Total loss — backprop through all three components
        total_loss = auxiliary_loss + 1.0 * individual_loss + 0.001 * load_balancing_loss

        # Return per-sample losses in format LeRobot expects: (B, T, D)
        # We scale so that total_loss gradients flow correctly
        per_sample = F.mse_loss(
            v_t_combined[:, :, :self.config.max_action_dim],
            u_t, reduction="none"
        )
        # Multiply by ratio so the full total_loss gradient is preserved
        scale = (total_loss / (auxiliary_loss.detach() + 1e-8)).detach()
        return per_sample * scale

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks,
                       state, noise=None, **kwargs) -> Tensor:
        """MoH inference with per-horizon denoising and gated fusion."""
        num_horizons = len(self.horizons)
        max_horizon = self.horizons[-1]
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            noise = self.sample_noise(
                (bsize, max_horizon, self.config.max_action_dim), device
            )

        # Prefix KV cache
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

        # Iterative denoising
        dt = -1.0 / self.config.num_steps
        x_t = noise

        for step in range(self.config.num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(
                time_val, dtype=torch.float32, device=device
            ).expand(bsize)

            all_suffix_outs = []
            all_v_t_preds = []
            for h in self.horizons:
                x_t_h_padded = F.pad(x_t[:, :h, :], (0, 0, 0, max_horizon - h))
                suffix_out_h = self._suffix_pass(
                    x_t_h_padded, time_tensor, prefix_pad_masks, past_key_values
                )
                all_suffix_outs.append(suffix_out_h)
                all_v_t_preds.append(self.action_out_proj(suffix_out_h))

            # Gate
            suffix_out_for_gate = torch.stack(all_suffix_outs, dim=0).permute(
                1, 0, 2, 3
            ).reshape(bsize * num_horizons, max_horizon, -1)

            gate_logits = self.gate_out_proj(suffix_out_for_gate)[:, :max_horizon, :]
            gate_logits = gate_logits.reshape(
                bsize, num_horizons, max_horizon
            ).permute(0, 2, 1)

            valid_heads_mask = torch.tensor(
                [[si < h for h in self.horizons] for si in range(max_horizon)],
                device=device, dtype=torch.bool
            ).unsqueeze(0)
            gate_weights = F.softmax(
                torch.where(valid_heads_mask, gate_logits,
                            torch.finfo(gate_logits.dtype).min),
                dim=-1
            )

            all_v_t_preds_t = torch.stack(all_v_t_preds, dim=1)  # (B, num_h, max_h, D)
            v_t = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v_t_preds_t).sum(dim=1)
            x_t = x_t + dt * v_t

        return x_t


class SmolVLAMoHPolicy(SmolVLAPolicy):
    """SmolVLA + Mixture of Horizons. Loads pretrained SmolVLA weights then
    adds gate layers on top."""

    def __init__(self, config: SmolVLAMoHConfig, **kwargs):
        # Call SmolVLAPolicy.__init__ which loads pretrained weights into self.model
        super().__init__(config, **kwargs)

        # Grab the pretrained weights from the already-loaded model
        pretrained_state = self.model.state_dict()

        # Replace with MoH model (same architecture + gate layers)
        self.model = SmolVLAMoHModel(config)

        # Load pretrained weights back in (gate layers will be missing = random init)
        missing, unexpected = self.model.load_state_dict(pretrained_state, strict=False)
        print(f"MoH model loaded. New layers (random init): {missing}")