"""
Layer-level policy for hierarchical RL in Conv-LoRA.

This policy decides which Conv-LoRA layers are active for a given input.
It is designed to be generic and reusable beyond SAM:

- Input: sample-level global features, shape (B, D)
- Output: per-layer logits or pattern logits
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LayerPolicy(nn.Module):
    """
    High-level policy deciding which Conv-LoRA layers are active.

    Typical usage in hierarchical RL:
    - Forward once per batch to get layer-wise activation logits.
    - Apply sigmoid and threshold / sample Bernoulli to obtain layer masks.
    """

    def __init__(
        self,
        num_layers: int,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_patterns: bool = False,
        num_patterns: int = 4,
    ):
        """
        Parameters
        ----------
        num_layers
            Number of Conv-LoRA layers to control.
        feature_dim
            Dimension of input global features (e.g., GAP of encoder).
        hidden_dim
            Hidden size for MLP.
        dropout
            Dropout probability.
        use_patterns
            If True, output pattern logits over ``num_patterns`` and map them
            to fixed layer patterns. If False, output per-layer logits directly.
        num_patterns
            Number of predefined patterns if ``use_patterns`` is True.
        """
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_patterns = use_patterns
        self.num_patterns = num_patterns

        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if self.use_patterns:
            # Pattern-level logits, later expanded to per-layer masks.
            self.head = nn.Linear(hidden_dim, num_patterns)
            # Default patterns can be overridden by the user at runtime.
            self.register_buffer(
                "patterns",
                self._init_default_patterns(num_layers, num_patterns),
            )
        else:
            # Direct per-layer logits.
            self.head = nn.Linear(hidden_dim, num_layers)

        self._init_weights()

    def _init_default_patterns(self, num_layers: int, num_patterns: int) -> torch.Tensor:
        """
        Create simple default layer patterns:

        - Pattern 0: all layers active
        - Pattern 1: middle third active
        - Pattern 2: lower half active
        - Pattern 3: upper half active

        If ``num_patterns`` > 4, remaining patterns are copies of pattern 0.
        """
        patterns = []
        # P0: all on
        patterns.append(torch.ones(num_layers))

        # P1: middle third
        start = num_layers // 3
        end = 2 * num_layers // 3
        mask = torch.zeros(num_layers)
        mask[start:end] = 1.0
        patterns.append(mask)

        # P2: lower half
        mask = torch.zeros(num_layers)
        mask[: num_layers // 2] = 1.0
        patterns.append(mask)

        # P3: upper half
        mask = torch.zeros(num_layers)
        mask[num_layers // 2 :] = 1.0
        patterns.append(mask)

        # Fill remaining with P0
        while len(patterns) < num_patterns:
            patterns.append(patterns[0].clone())

        return torch.stack(patterns, dim=0)  # (P, L)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        global_feats: torch.Tensor,
        return_layer_probs: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        global_feats
            Global features, shape (B, D).
        return_layer_probs
            When ``use_patterns`` is True, additionally return expected
            layer-wise activation probabilities.

        Returns
        -------
        outputs
            If ``use_patterns`` is False:
                layer_logits: Tensor of shape (B, num_layers).
            If ``use_patterns`` is True:
                - if ``return_layer_probs`` is False:
                    pattern_logits: Tensor of shape (B, num_patterns).
                - else:
                    (pattern_logits, layer_probs) where layer_probs is
                    Tensor of shape (B, num_layers).
        """
        x = self.backbone(global_feats)  # (B, hidden_dim)

        if self.use_patterns:
            pattern_logits = self.head(x)  # (B, P)
            if not return_layer_probs:
                return pattern_logits

            pattern_probs = torch.softmax(pattern_logits, dim=-1)  # (B, P)
            # patterns: (P, L) -> (1, P, L)
            patterns = self.patterns.unsqueeze(0)  # (1, P, L)
            # (B, P, 1)
            pattern_probs_expanded = pattern_probs.unsqueeze(-1)
            # (B, L)
            layer_probs = (pattern_probs_expanded * patterns).sum(dim=1)
            return pattern_logits, layer_probs

        layer_logits = self.head(x)  # (B, L)
        return layer_logits

    # --- Interfaces for GRPO adapter regularization & compatibility ---

    def get_adapter_params(self):
        """
        Get adapter parameters for L2 regularization.

        For hierarchical RL we treat the entire LayerPolicy as adapter.
        """
        return list(self.parameters())

    def freeze_base(self):
        """
        Provided for interface symmetry with other policies.

        LayerPolicy is fully trainable and has no separate frozen base.
        """
        pass


