import torch
import pytest
from diffsynth_engine.models.qwen_image.transformer_qwenimage import apply_rotary_emb_qwen


class TestRoPE:
    def test_rope_use_real_false_output_shape(self):
        """Verify use_real=False branch produces correct output shape."""
        x = torch.randn(2, 16, 8, 64)  # [B, S, H, D]
        freqs_cis = torch.randn(16, 64, dtype=torch.complex64)
        out = apply_rotary_emb_qwen(x, freqs_cis, use_real=False)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_rope_use_real_true_output_shape(self):
        """Verify use_real=True branch produces correct output shape."""
        x = torch.randn(2, 16, 8, 64)  # [B, S, H, D]
        cos = torch.randn(16, 64)
        sin = torch.randn(16, 64)
        out = apply_rotary_emb_qwen(x, (cos, sin), use_real=True)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_rope_use_real_false_equivalence(self):
        """Verify use_real=False branch is mathematically equivalent to original."""
        x = torch.randn(2, 16, 8, 64)
        freqs_cis = torch.randn(16, 64, dtype=torch.complex64)

        # Original implementation
        def original_impl(x, freqs_cis):
            x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(1)
            x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
            return x_out.type_as(x)

        out_original = original_impl(x, freqs_cis)
        out_ours = apply_rotary_emb_qwen(x, freqs_cis, use_real=False)
        assert torch.allclose(out_original, out_ours, atol=1e-5)

    def test_rope_use_real_true_unbind_dim(self):
        """Verify both unbind_dim values work."""
        x = torch.randn(2, 16, 8, 64)
        cos = torch.randn(16, 64)
        sin = torch.randn(16, 64)

        # unbind_dim = -1
        out_m1 = apply_rotary_emb_qwen(x, (cos, sin), use_real=True, use_real_unbind_dim=-1)
        assert out_m1.shape == x.shape

        # unbind_dim = -2
        out_m2 = apply_rotary_emb_qwen(x, (cos, sin), use_real=True, use_real_unbind_dim=-2)
        assert out_m2.shape == x.shape

    def test_rope_invalid_unbind_dim(self):
        """Verify invalid unbind_dim raises ValueError."""
        x = torch.randn(2, 16, 8, 64)
        cos = torch.randn(16, 64)
        sin = torch.randn(16, 64)

        with pytest.raises(ValueError):
            apply_rotary_emb_qwen(x, (cos, sin), use_real=True, use_real_unbind_dim=0)
