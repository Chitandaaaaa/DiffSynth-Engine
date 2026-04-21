import torch
import pytest
from unittest.mock import patch, MagicMock
import sys
from diffsynth_engine.models.qwen_image.transformer_qwenimage import apply_rotary_emb_qwen


class TestRoPENPU:
    """Test NPU path with mocked rotary_position_embedding."""

    def _setup_mindiesd_mock(self):
        """Create mock mindiesd module hierarchy."""
        mock_rotary_fn = MagicMock()
        mock_rotary_fn.return_value = torch.randn(2, 16, 8, 64)
        mock_rope = MagicMock()
        mock_rope.rotary_position_embedding = mock_rotary_fn
        mock_layers = MagicMock()
        mock_layers.rope = mock_rope
        mock_mindiesd = MagicMock()
        mock_mindiesd.layers = mock_layers
        sys.modules["mindiesd"] = mock_mindiesd
        sys.modules["mindiesd.layers"] = mock_layers
        sys.modules["mindiesd.layers.rope"] = mock_rope
        return mock_rotary_fn

    def test_rope_use_real_false_calls_npu(self):
        """Verify rotary_position_embedding is called for use_real=False."""
        mock_rotary = self._setup_mindiesd_mock()

        with patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=True):
            x = torch.randn(2, 16, 8, 64)
            freqs_cis = torch.randn(16, 64, dtype=torch.complex64)
            out = apply_rotary_emb_qwen(x, freqs_cis, use_real=False)

            mock_rotary.assert_called_once()
            assert out.shape == x.shape

    def test_rope_use_real_true_calls_npu(self):
        """Verify rotary_position_embedding is called for use_real=True."""
        mock_rotary = self._setup_mindiesd_mock()

        with patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=True):
            x = torch.randn(2, 16, 8, 64)
            cos = torch.randn(16, 64)
            sin = torch.randn(16, 64)
            out = apply_rotary_emb_qwen(x, (cos, sin), use_real=True)

            mock_rotary.assert_called_once()
            assert out.shape == x.shape

    def test_rope_use_real_true_unbind_dim_m2(self):
        """Verify unbind_dim=-2 works with NPU."""
        self._setup_mindiesd_mock()

        with patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=True):
            x = torch.randn(2, 16, 8, 64)
            cos = torch.randn(16, 64)
            sin = torch.randn(16, 64)
            out = apply_rotary_emb_qwen(x, (cos, sin), use_real=True, use_real_unbind_dim=-2)

            assert out.shape == x.shape


class TestRoPEFallback:
    """Test fallback path when NPU not available."""

    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=False)
    def test_rope_use_real_false_equivalence(self, mock_is_npu):
        """Verify use_real=False branch uses correct rotation formula."""
        x = torch.randn(2, 16, 8, 64)
        freqs_cis = torch.randn(16, 64, dtype=torch.complex64)

        # Reference implementation matching the fallback
        def reference_impl(x, freqs_cis):
            freqs_real = torch.view_as_real(freqs_cis)
            cos = freqs_real[..., 0]
            sin = freqs_real[..., 1]
            cos_bc = cos[None, :, None, :]
            sin_bc = sin[None, :, None, :]
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
            x_out = (x.float() * cos_bc + x_rotated.float() * sin_bc).to(x.dtype)
            return x_out.type_as(x)

        out_ref = reference_impl(x, freqs_cis)
        out_ours = apply_rotary_emb_qwen(x, freqs_cis, use_real=False)
        assert torch.allclose(out_ref, out_ours, atol=1e-5)

    @patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=False)
    def test_rope_invalid_unbind_dim(self, mock_is_npu):
        """Verify invalid unbind_dim raises ValueError."""
        x = torch.randn(2, 16, 8, 64)
        cos = torch.randn(16, 64)
        sin = torch.randn(16, 64)

        with pytest.raises(ValueError):
            apply_rotary_emb_qwen(x, (cos, sin), use_real=True, use_real_unbind_dim=0)
