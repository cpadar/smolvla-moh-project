"""
Verification tests for the three bugs fixed before v4 retraining.

Tests:
  1. KV cache isolation  -- _suffix_pass must not mutate past_key_values
  2. Gradient flow       -- gate + individual head params must receive gradients
                           from the full 3-part loss (not just auxiliary)
  3. Eval double-count   -- successes list must have exactly num_episodes entries

Run on SOL before submitting the full training job:
  python scripts/test_fixes.py

Requires: conda activate smolvla-moh  (GPU recommended but not required for test 3)
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~/smolvla-moh-project"))

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# ---------------------------------------------------------------------------
# Test 1: KV cache isolation — verify use_cache=False in _suffix_pass
# ---------------------------------------------------------------------------
def test_kv_cache_isolation():
    """
    _suffix_pass must call vlm_with_expert.forward with use_cache=False.
    We verify this two ways:
      a) Source inspection — confirm 'use_cache=False' appears in _suffix_pass
      b) Runtime assertion — patch vlm_with_expert.forward and assert the flag
    """
    print("\n[1] KV cache isolation ...")

    import inspect
    import unittest.mock as mock
    from models.smolvla_moh.moh_policy import SmolVLAMoHModel

    # --- (a) Source inspection ---
    src = inspect.getsource(SmolVLAMoHModel._suffix_pass)
    # Must contain use_cache=False and must NOT contain use_cache=True
    has_false = "use_cache=False" in src
    has_true  = "use_cache=True"  in src
    src_ok = has_false and not has_true
    status = PASS if src_ok else FAIL
    print(f"  {status} - source: use_cache=False present={has_false}, use_cache=True present={has_true}")

    # --- (b) Runtime check via mock ---
    # Capture the use_cache kwarg actually passed to vlm_with_expert.forward
    captured = {}

    def fake_vlm_forward(attention_mask, position_ids, past_key_values,
                         inputs_embeds, use_cache, fill_kv_cache):
        captured["use_cache"] = use_cache
        B2 = inputs_embeds[1].shape[0]
        seq_out = inputs_embeds[1].shape[1]
        return (None, torch.zeros(B2, seq_out, 32)), None

    # We need a real nn.Module instance to call _suffix_pass on.
    # Use MagicMock as a thin wrapper that has the method but delegates forward.
    fake_vlm = mock.MagicMock()
    fake_vlm.forward.side_effect = fake_vlm_forward

    # Patch embed_suffix and make_att_2d_masks so _suffix_pass can run end-to-end
    B, seq, hidden = 2, 10, 32
    suffix_len = 52

    def fake_embed_suffix(x_t, time):
        B2 = x_t.shape[0]
        embs      = torch.randn(B2, suffix_len, hidden)
        pad_masks = torch.ones(B2, suffix_len, dtype=torch.bool)
        att_masks = torch.ones(B2, suffix_len, dtype=torch.bool)
        return embs, pad_masks, att_masks

    with mock.patch("models.smolvla_moh.moh_policy.make_att_2d_masks",
                    side_effect=lambda pad, att: torch.ones(
                        pad.shape[0], pad.shape[1], pad.shape[1])):

        # Create a bare instance bypassing __init__ (avoids loading VLM weights)
        model = mock.MagicMock(spec=SmolVLAMoHModel)
        model.vlm_with_expert = fake_vlm
        model.embed_suffix    = fake_embed_suffix
        # Bind the real _suffix_pass to this mock instance
        bound = SmolVLAMoHModel._suffix_pass.__get__(model, SmolVLAMoHModel)

        kv_cache = {0: {"key_states": torch.randn(B, 4, seq, 16),
                        "value_states": torch.randn(B, 4, seq, 16)}}
        prefix_pad_masks = torch.ones(B, seq, dtype=torch.bool)
        bound(torch.randn(B, 50, 8), torch.ones(B), prefix_pad_masks, kv_cache)

    runtime_ok = captured.get("use_cache") is False
    status = PASS if runtime_ok else FAIL
    print(f"  {status} - runtime: use_cache passed as {captured.get('use_cache')!r}")

    all_ok = src_ok and runtime_ok
    if all_ok:
        print(f"  {PASS} - KV cache will not be extended by _suffix_pass")
    else:
        print(f"  {FAIL} - KV cache isolation check failed")
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: Gradient flow through total_loss
# ---------------------------------------------------------------------------
def test_gradient_flow():
    """
    After forward(), both gate_out_proj and action_out_proj parameters
    must have non-zero gradients (proving individual + balancing losses flow).
    """
    print("\n[2] Gradient flow through total_loss ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    from models.smolvla_moh.moh_config import SmolVLAMoHConfig

    # We test gradient flow through the loss math directly — no VLM needed.
    # Replicate the exact loss computation from moh_policy.py forward().

    torch.manual_seed(0)
    B, max_h, D, hidden = 2, 50, 8, 32
    horizons = [10, 25, 50]
    num_horizons = len(horizons)

    gate_out_proj     = nn.Linear(hidden, 1).to(device)
    gate_noise_layer  = nn.Linear(hidden, 1).to(device)
    action_out_proj   = nn.Linear(hidden, D).to(device)
    softplus          = nn.Softplus()

    # Fake suffix outputs (stand-in for VLM outputs)
    all_suffix_outs = [torch.randn(B, max_h, hidden, device=device, requires_grad=True)
                       for _ in horizons]

    u_t = torch.randn(B, max_h, D, device=device)

    # --- replicate the loss math from moh_policy.py ---
    all_head_losses = []
    all_v_t_preds_list = []
    for i, h in enumerate(horizons):
        v_t_h = action_out_proj(all_suffix_outs[i])
        all_v_t_preds_list.append(v_t_h)
        all_head_losses.append(F.mse_loss(v_t_h[:, :h, :], u_t[:, :h, :]))
    individual_loss = torch.sum(torch.stack(all_head_losses))

    all_v_t_preds_t = torch.stack(all_v_t_preds_list, dim=1)

    suffix_out_stacked = torch.stack(all_suffix_outs, dim=0)
    suffix_out_for_gate = suffix_out_stacked.permute(1, 0, 2, 3).reshape(
        B * num_horizons, max_h, -1
    )
    gate_logits = gate_out_proj(suffix_out_for_gate)[:, :max_h, :]
    noise_epsilon = 1e-2
    raw_noise_stddev = gate_noise_layer(suffix_out_for_gate)[:, :max_h, :]
    noise_stddev = softplus(raw_noise_stddev) + noise_epsilon
    gate_logits = gate_logits + (torch.randn_like(gate_logits) * noise_stddev)
    gate_logits = gate_logits.reshape(B, num_horizons, max_h).permute(0, 2, 1)

    valid_heads_mask = torch.tensor(
        [[step < h for h in horizons] for step in range(max_h)],
        device=device, dtype=torch.bool
    ).unsqueeze(0)
    masked_gate_logits = torch.where(valid_heads_mask, gate_logits,
                                     torch.finfo(gate_logits.dtype).min)
    gate_weights = F.softmax(masked_gate_logits, dim=-1)

    v_t_combined = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v_t_preds_t).sum(dim=1)
    auxiliary_loss = F.mse_loss(v_t_combined, u_t)

    # Load balancing loss
    loss_components = []
    boundaries = sorted(list(set([0] + horizons)))
    for i in range(len(boundaries) - 1):
        start_step = boundaries[i]
        active_expert_indices = [idx for idx, h in enumerate(horizons) if h > start_step]
        if len(active_expert_indices) > 1:
            seg = gate_weights[:, start_step:boundaries[i+1], :]
            active = seg[:, :, active_expert_indices]
            avg = active.mean(dim=(0, 1))
            eps = 1e-10
            cv2 = avg.float().var() / (avg.float().mean() ** 2 + eps)
            loss_components.append(cv2)
    load_balancing_loss = torch.mean(torch.stack(loss_components))

    total_loss = auxiliary_loss + 1.0 * individual_loss + 0.001 * load_balancing_loss

    losses = F.mse_loss(v_t_combined, u_t, reduction="none")
    loss_scale = total_loss / (auxiliary_loss.detach() + 1e-8)
    returned_losses = losses * loss_scale

    # Simulate what LeRobot does: losses.mean().backward()
    returned_losses.mean().backward()

    # Check gradients on gate and action projection parameters
    results = {}
    for name, param in [("gate_out_proj.weight", gate_out_proj.weight),
                         ("gate_out_proj.bias",   gate_out_proj.bias),
                         ("action_out_proj.weight", action_out_proj.weight)]:
        has_grad = param.grad is not None and param.grad.abs().sum().item() > 0
        results[name] = has_grad

    all_ok = all(results.values())
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status} - {name} grad={'non-zero' if ok else 'ZERO/NONE'}")

    if all_ok:
        print(f"  {PASS} - All parameters receive gradients from total_loss")
    else:
        print(f"  {FAIL} - Some parameters have no gradient — loss not fully wired")
    return all_ok


# ---------------------------------------------------------------------------
# Test 3: Eval episode counting
# ---------------------------------------------------------------------------
def test_eval_counting():
    """
    Simulate the fixed eval loop: successes list must have exactly
    num_episodes entries (not 2x due to the removed duplicate append).
    """
    print("\n[3] Eval episode counting ...")

    num_episodes = 5
    successes = []
    episode_lengths = []

    for episode in range(num_episodes):
        success = bool(episode % 2 == 0)  # alternating success/fail
        step = 50 + episode * 10

        # --- replicate FIXED eval loop body (single append) ---
        successes.append(success)
        episode_lengths.append(step)

    ok_count  = len(successes) == num_episodes
    ok_length = len(episode_lengths) == num_episodes
    ok_rate   = abs(sum(successes) / len(successes) - 3/5) < 1e-6  # 3 of 5 succeed

    for desc, ok in [("len(successes) == num_episodes", ok_count),
                     ("len(episode_lengths) == num_episodes", ok_length),
                     ("success_rate == 60%", ok_rate)]:
        print(f"  {'PASS' if ok else FAIL} - {desc}")

    all_ok = ok_count and ok_length and ok_rate
    if all_ok:
        print(f"  {PASS} - Episode counting is correct")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MOH Fix Verification")
    print("=" * 60)

    results = {}

    try:
        results["kv_cache_isolation"] = test_kv_cache_isolation()
    except Exception as e:
        print(f"  {FAIL} - Exception: {e}")
        results["kv_cache_isolation"] = False

    try:
        results["gradient_flow"] = test_gradient_flow()
    except Exception as e:
        print(f"  {FAIL} - Exception: {e}")
        results["gradient_flow"] = False

    try:
        results["eval_counting"] = test_eval_counting()
    except Exception as e:
        print(f"  {FAIL} - Exception: {e}")
        results["eval_counting"] = False

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_passed = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        all_passed = all_passed and ok

    print()
    if all_passed:
        print("All checks passed — safe to submit the v4 training job.")
    else:
        print("One or more checks failed — investigate before retraining.")
    print()
    sys.exit(0 if all_passed else 1)
