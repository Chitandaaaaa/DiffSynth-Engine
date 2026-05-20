import torch
import torch.nn as nn


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm wrapper for QwenImage-style modulation.

    Performs: output = LayerNorm(x) * (1 + scale) + shift

    This is a pure PyTorch implementation with NO NPU code. When the model
    is compiled with MindieSDBackend, the FX graph pattern
    (layer_norm → unsqueeze → unsqueeze → add → mul → add) is automatically
    matched and replaced with the MindIE fused ``layernorm_scale_shift``
    kernel.
    """

    def __init__(self, layernorm: nn.LayerNorm):
        super().__init__()
        self.layernorm = layernorm

    def forward(self, hidden_states: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S, H] input tensor
            scale: [B, H] scale parameter
            shift: [B, H] shift parameter
        Returns:
            [B, S, H] modulated tensor
        """
        normed = self.layernorm(hidden_states)
        scale = scale.unsqueeze(1)   # [B, H] → [B, 1, H]
        shift = shift.unsqueeze(1)   # [B, H] → [B, 1, H]
        return normed * (1 + scale) + shift
