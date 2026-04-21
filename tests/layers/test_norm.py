import torch
import pytest
from unittest.mock import patch, MagicMock
from diffsynth_engine.layers.norm import RMSNorm


class TestRMSNormNPU:
    """Test NPU path with mocked torch_npu."""

    @patch("diffsynth_engine.layers.norm.is_npu_available", return_value=True)
    @patch("diffsynth_engine.layers.norm.torch_npu")
    def test_rmsnorm_calls_npu_impl(self, mock_torch_npu, mock_is_npu):
        """Verify npu_rms_norm is called when NPU available."""
        mock_torch_npu.npu_rms_norm.return_value = (torch.randn(2, 16, 64), None)

        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)

        mock_torch_npu.npu_rms_norm.assert_called_once()
        assert out.shape == x.shape

    @patch("diffsynth_engine.layers.norm.is_npu_available", return_value=True)
    @patch("diffsynth_engine.layers.norm.torch_npu")
    def test_rmsnorm_different_eps(self, mock_torch_npu, mock_is_npu):
        """Verify different eps values work correctly with NPU."""
        mock_torch_npu.npu_rms_norm.return_value = (torch.randn(2, 16, 64), None)

        x = torch.randn(2, 16, 64)
        for eps in [1e-6, 1e-5, 1e-4]:
            norm = RMSNorm(64, eps=eps)
            out = norm(x)
            assert out.shape == x.shape


class TestRMSNormFallback:
    """Test fallback path when NPU not available."""

    @patch("diffsynth_engine.layers.norm.is_npu_available", return_value=False)
    @patch("diffsynth_engine.layers.norm.torch_npu", None)
    def test_rmsnorm_uses_diffusers_when_npu_unavailable(self, mock_is_npu):
        """Verify diffusers implementation is used when NPU unavailable."""
        # Mock diffusers internal torch_npu to prevent it from using NPU
        with patch("diffusers.models.normalization.torch_npu", None):
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
