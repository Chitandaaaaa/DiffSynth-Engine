import torch
import pytest
from diffsynth_engine.layers.norm import RMSNorm


class TestRMSNorm:
    def test_rmsnorm_basic(self):
        """Verify RMSNorm output shape is correct."""
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_rmsnorm_equivalence(self):
        """Verify output is equivalent to diffusers RMSNorm when NPU not available."""
        from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

        hidden_size, eps = 64, 1e-6
        norm = RMSNorm(hidden_size, eps)
        ref_norm = DiffusersRMSNorm(hidden_size, eps)
        x = torch.randn(2, 16, 64)
        out_ours = norm(x)
        out_ref = ref_norm(x)
        assert torch.allclose(out_ours, out_ref, atol=1e-5)

    def test_rmsnorm_weight_property(self):
        """Verify weight property is accessible."""
        norm = RMSNorm(64)
        assert hasattr(norm, "weight")
        assert norm.weight is not None

    def test_rmsnorm_different_eps(self):
        """Verify different eps values work correctly."""
        x = torch.randn(2, 16, 64)
        for eps in [1e-6, 1e-5, 1e-4]:
            norm = RMSNorm(64, eps=eps)
            out = norm(x)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()
