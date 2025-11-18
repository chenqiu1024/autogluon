"""
Utilities for hierarchical RL (LayerPolicy + RoutingPolicy).

These helpers keep the core GRPO implementation unchanged while providing
lightweight data structures and advantage computation for multi-policy setups.
"""

from typing import Dict, List, Tuple

import torch


def build_hierarchical_rollout(
    layer_logprobs: torch.Tensor,
    layer_logits: torch.Tensor,
    routing_rollout_data: Dict,
    layer_kl_loss: torch.Tensor | None = None,
) -> Dict[str, Dict]:
    """
    Package rollout data for hierarchical RL.

    Parameters
    ----------
    layer_logprobs
        Flattened log-probabilities for layer decisions, shape (N_layer,)
        where N_layer = batch_size * num_layers or similar.
    layer_logits
        Raw logits for layer activations, shape (B, L).
    routing_rollout_data
        Rollout dict for routing policy, containing:
        - 'logprobs': list of 1D tensors
        - 'info_dicts': list of info dicts (logits, kl_loss, etc.)
    layer_kl_loss
        Optional KL loss for layer policy (unused in basic setup).

    Returns
    -------
    dict
        {
            "layer": {
                "logprobs": [Tensor],
                "info_dicts": [info_dict],
            },
            "routing": routing_rollout_data,
        }
    """
    layer_info = {"logits": layer_logits}
    if layer_kl_loss is not None:
        layer_info["kl_loss"] = layer_kl_loss

    layer_rollout = {
        "logprobs": [layer_logprobs],
        "info_dicts": [layer_info],
    }

    return {
        "layer": layer_rollout,
        "routing": routing_rollout_data,
    }


def compute_shared_advantages(
    reward: float | torch.Tensor,
    routing_rollout_data: Dict,
    layer_rollout_size: int = 1,
    device: str = "cuda",
    normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute simple shared advantages for layer & routing policies.

    Parameters
    ----------
    reward
        Scalar reward for the whole batch / episode.
    routing_rollout_data
        Rollout data for routing policy, must contain 'logprobs' list.
    layer_rollout_size
        Number of logprob elements for LayerPolicy (e.g., B * L).
    device
        Torch device.
    normalize
        If True, normalize concatenated advantages to zero mean / unit std.

    Returns
    -------
    adv_layer, adv_routing
        1D tensors of shape (layer_rollout_size,) and (N_routing,) respectively.
    """
    if isinstance(reward, torch.Tensor):
        reward_value = float(reward.detach().item())
    else:
        reward_value = float(reward)

    # Routing advantages
    num_routing_logprobs = sum(lp.numel() for lp in routing_rollout_data["logprobs"])
    adv_routing = torch.ones(num_routing_logprobs, device=device) * reward_value

    # Layer advantages
    adv_layer = torch.ones(layer_rollout_size, device=device) * reward_value

    if normalize and (num_routing_logprobs + layer_rollout_size) > 1:
        all_adv = torch.cat([adv_layer, adv_routing], dim=0)
        mean = all_adv.mean()
        std = all_adv.std(unbiased=False) + 1e-8
        all_adv = (all_adv - mean) / std
        adv_layer = all_adv[:layer_rollout_size]
        adv_routing = all_adv[layer_rollout_size:]

    return adv_layer, adv_routing


