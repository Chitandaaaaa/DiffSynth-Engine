import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from diffsynth_engine.utils.import_utils import is_npu_available


class RMSNorm(nn.Module):
    """NPU-optimized RMSNorm wrapper with fallback to diffusers implementation."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.diffusers_norm = DiffusersRMSNorm(hidden_size, eps)

    def forward(self, hidden_states):
        if is_npu_available():
            import torch_npu

            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
        else:
            return self.diffusers_norm(hidden_states)

    @property
    def weight(self):
        """透传到 diffusers 的 weight 参数，供 npu_rms_norm 使用。"""
        return self.diffusers_norm.weight
