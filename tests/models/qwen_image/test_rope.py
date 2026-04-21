import torch
import pytest
from unittest.mock import patch, MagicMock
from diffsynth_engine.models.qwen_image.transformer_qwenimage import apply_rotary_emb_qwen


class TestRoPENPU:
    """Test NPU path with mocked rotary_position_embedding."""

    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=True)
    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.rotary_position_embedding")
    def test_rope_use_real_false_calls_npu(self, mock_rotary, mock_is_npu):
        """Verify rotary_position_embedding is called for use_real=False."""
        mock_rotary.return_value = torch.randn(2, 16, 8, 64)

        x = torch.randn(2, 16, 8, 64)
        freqs_cis = torch.randn(16, 64, dtype=torch.complex64)
        out = apply_rotary_emb_qwen(x, freqs_cis, use_real=False)

        mock_rotary.assert_called_once()
        assert out.shape == x.shape

    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=True)
    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.rotary_position_embedding")
    def test_rope_use_real_true_calls_npu(self, mock_rotary, mock_is_npu):
        """Verify rotary_position_embedding is called for use_real=True."""
        mock_rotary.return_value = torch.randn(2, 16, 8, 64)

        x = torch.randn(2, 16, 8, 64)
        cos = torch.randn(16, 64)
        sin = torch.randn(16, 64)
        out = apply_rotary_emb_qwen(x, (cos, sin), use_real=True)

        mock_rotary.assert_called_once()
        assert out.shape == x.shape

    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=True)
    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.rotary_position_embedding")
    def test_rope_use_real_true_unbind_dim_m2(self, mock_rotary, mock_is_npu):
        """Verify unbind_dim=-2 works with NPU."""
        mock_rotary.return_value = torch.randn(2, 16, 8, 64)

        x = torch.randn(2, 16, 8, 64)
        cos = torch.randn(16, 64)
        sin = torch.randn(16, 64)
        out = apply_rotary_emb_qwen(x, (cos, sin), use_real=True, use_real_unbind_dim=-2)

        assert out.shape == x.shape


class TestRoPEFallback:
    """Test fallback path when NPU not available."""

    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=False)
    def test_rope_use_real_false_equivalence(self, mock_is_npu):
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

    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=False)
    def test_rope_invalid_unbind_dim(self, mock_is_npu):
        """Verify invalid unbind_dim raises ValueError."""
        x = torch.randn(2, 16, 8, 64)
        cos = torch.randn(16, 64)
        sin = torch.randn(16, 64)

        with pytest.raises(ValueError):
            apply_rotary_emb_qwen(x, (cos, sin), use_real=True, use_real_unbind_dim=0)
