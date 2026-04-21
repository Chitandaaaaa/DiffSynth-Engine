import torch
import pytest
from diffsynth_engine.layers.mlp import FastGELUMLP
from diffsynth_engine.utils.import_utils import is_npu_available


class TestFastGELUMLP:
    def test_fastgelu_output_shape(self):
        """Verify FastGELUMLP output shape is correct on NPU."""
        if not is_npu_available():
            pytest.skip("NPU not available")

        mlp = FastGELUMLP(dim=64, dim_out=64)
        x = torch.randn(2, 16, 64).npu()
        out = mlp(x)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()

    def test_fastgelu_equivalence(self):
        """Verify output is equivalent to FeedForward when NPU not available."""
        if is_npu_available():
            pytest.skip("NPU available, test requires CPU fallback path")

        from diffusers.models.attention import FeedForward

        dim = 64
        mlp = FastGELUMLP(dim=dim, dim_out=dim)
        ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        x = torch.randn(2, 16, dim)
        out_ours = mlp(x)
        out_ref = ff(x)
        assert torch.allclose(out_ours, out_ref, atol=1e-5)

    def test_fastgelu_mult_parameter(self):
        """Verify mult parameter creates correct inner dimension."""
        mlp_4 = FastGELUMLP(dim=64, mult=4)
        mlp_2 = FastGELUMLP(dim=64, mult=2)
        assert mlp_4.proj_in.out_features == 256
        assert mlp_2.proj_in.out_features == 128

    def test_fastgelu_dim_out_default(self):
        """Verify dim_out defaults to dim."""
        mlp = FastGELUMLP(dim=64)
        assert mlp.proj_out.out_features == 64
