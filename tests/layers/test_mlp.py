import torch
import pytest
from unittest.mock import patch, MagicMock
from diffsynth_engine.layers.mlp import FastGELUMLP


class TestFastGELUMLPNPU:
    """Test NPU path with mocked torch_npu."""

    @patch("diffsynth_engine.layers.mlp.is_npu_available", return_value=True)
    @patch("diffsynth_engine.layers.mlp.torch_npu")
    def test_fastgelu_calls_npu_impl(self, mock_torch_npu, mock_is_npu):
        """Verify npu_fast_gelu is called when NPU available."""
        mock_torch_npu.npu_fast_gelu.return_value = torch.randn(2, 16, 64)

        mlp = FastGELUMLP(dim=64)
        x = torch.randn(2, 16, 64)
        out = mlp(x)

        mock_torch_npu.npu_fast_gelu.assert_called_once()
        assert out.shape == (2, 16, 64)

    @patch("diffsynth_engine.layers.mlp.is_npu_available", return_value=True)
    @patch("diffsynth_engine.layers.mlp.torch_npu")
    def test_fastgelu_different_mult(self, mock_torch_npu, mock_is_npu):
        """Verify different mult values work with NPU."""
        mock_torch_npu.npu_fast_gelu.return_value = torch.randn(2, 16, 256)

        mlp_4 = FastGELUMLP(dim=64, mult=4)
        mlp_2 = FastGELUMLP(dim=64, mult=2)
        x = torch.randn(2, 16, 64)

        out_4 = mlp_4(x)
        out_2 = mlp_2(x)

        assert out_4.shape == (2, 16, 64)
        assert out_2.shape == (2, 16, 64)


class TestFastGELUMLPFallback:
    """Test fallback path when NPU not available."""

    @patch("diffsynth_engine.layers.mlp.is_npu_available", return_value=False)
    def test_fastgelu_uses_diffusers_when_npu_unavailable(self, mock_is_npu):
        """Verify diffusers implementation is used when NPU unavailable."""
        from diffusers.models.attention import FeedForward

        dim = 64
        mlp = FastGELUMLP(dim=dim, dim_out=dim)
        ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        x = torch.randn(2, 16, dim)

        out_ours = mlp(x)
        out_ref = ff(x)

        assert torch.allclose(out_ours, out_ref, atol=1e-5)


class TestFastGELUMLPBasics:
    """Basic tests that don't require NPU or fallback."""

    def test_mult_parameter(self):
        """Verify mult parameter creates correct inner dimension."""
        mlp_4 = FastGELUMLP(dim=64, mult=4)
        mlp_2 = FastGELUMLP(dim=64, mult=2)
        assert mlp_4.proj_in.out_features == 256
        assert mlp_2.proj_in.out_features == 128

    def test_dim_out_default(self):
        """Verify dim_out defaults to dim."""
        mlp = FastGELUMLP(dim=64)
        assert mlp.proj_out.out_features == 64
