"""
Prompt Policy for interactive SAM prompting (Scheme A).

This is a placeholder for future implementation.
"""

import torch
import torch.nn as nn


class PromptPolicy(nn.Module):
    """
    Policy network for selecting interactive prompts for SAM.
    
    To be implemented in Scheme A (future work).
    
    Expected architecture:
    - Input: [image_features, current_mask, uncertainty_map, step_count]
    - Output: distribution over K candidate prompts (points/boxes)
    
    Training: GRPO with reward = ΔIoU - λ·click_cost
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("PromptPolicy is a placeholder for Scheme A (future work)")
    
    def forward(self, state):
        raise NotImplementedError("PromptPolicy is a placeholder for Scheme A (future work)")

