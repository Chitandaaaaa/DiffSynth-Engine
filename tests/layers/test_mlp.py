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
        # npu_fast_gelu doesn't change shape, just applies activation
        mock_torch_npu.npu_fast_gelu.side_effect = lambda x: x

        mlp = FastGELUMLP(dim=64)
        x = torch.randn(2, 16, 64)
        out = mlp(x)

        mock_torch_npu.npu_fast_gelu.assert_called_once()
        assert out.shape == (2, 16, 64)

    @patch("diffsynth_engine.layers.mlp.is_npu_available", return_value=True)
    @patch("diffsynth_engine.layers.mlp.torch_npu")
    def test_fastgelu_different_mult(self, mock_torch_npu, mock_is_npu):
        """Verify different mult values work with NPU."""
        mock_torch_npu.npu_fast_gelu.side_effect = lambda x: x

        mlp_4 = FastGELUMLP(dim=64, mult=4)
        x = torch.randn(2, 16, 64)

        out_4 = mlp_4(x)

        assert out_4.shape == (2, 16, 64)


class TestFastGELUMLPFallback:
    """Test fallback path when NPU not available."""

    @patch("diffsynth_engine.layers.mlp.is_npu_available", return_value=False)
    def test_fastgelu_uses_diffusers_when_npu_unavailable(self, mock_is_npu):
        """Verify diffusers implementation is used when NPU unavailable."""
        from diffusers.models.attention import FeedForward

        dim = 64
        mlp = FastGELUMLP(dim=dim, dim_out=dim)
        ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Copy weights from ff to mlp to verify equivalence with same weights
        mlp.net[0].weight.data = ff.net[0].proj.weight.data.clone()
        mlp.net[0].bias.data = ff.net[0].proj.bias.data.clone()
        mlp.net[2].weight.data = ff.net[2].weight.data.clone()
        mlp.net[2].bias.data = ff.net[2].bias.data.clone()

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
        assert mlp_4.net[0].out_features == 256
        assert mlp_2.net[0].out_features == 128

    def test_dim_out_default(self):
        """Verify dim_out defaults to dim."""
        mlp = FastGELUMLP(dim=64)
        assert mlp.net[2].out_features == 64
