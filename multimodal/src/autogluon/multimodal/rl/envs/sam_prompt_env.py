"""
SAM Prompt Environment for interactive prompt policy RL (Scheme A).

This is a placeholder for future implementation.
"""


class SamPromptEnv:
    """
    Gym-like environment for training interactive prompt policies.
    
    To be implemented in Scheme A (future work).
    
    Expected interface:
    - reset(image, gt_mask) -> state
    - step(action) -> (state, reward, done, info)
    - reward = ΔIoU - λ·click_cost
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SamPromptEnv is a placeholder for Scheme A (future work)")

