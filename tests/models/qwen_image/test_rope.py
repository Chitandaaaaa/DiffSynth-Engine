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

    def test_rope_use_real_false_no_npu_due_to_dim_mismatch(self):
        """Verify use_real=False uses fallback even when NPU available.

        The use_real=False path cannot use rotary_position_embedding because
        x has dim=128 but cos/sin have dim=64 (incompatible for NPU op).
        The fallback formula is used instead.
        """
        mock_rotary = self._setup_mindiesd_mock()

        with patch("diffsynth_engine.models.qwen_image.transformer_qwenimage.is_npu_available", return_value=True):
            x = torch.randn(2, 16, 8, 128)
            freqs_cis = torch.randn(16, 64, dtype=torch.complex64)
            out = apply_rotary_emb_qwen(x, freqs_cis, use_real=False)

            # rotary_position_embedding should NOT be called due to dim mismatch
            mock_rotary.assert_not_called()
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
        # x is [B,S,H,D] where D = 2 * freq_dim
        # freqs_cis is [S, freq_dim] complex = [S, D//2] complex
        x = torch.randn(2, 16, 8, 128)  # D=128, freq_dim=64
        freqs_cis = torch.randn(16, 64, dtype=torch.complex64)  # [S, D//2]

        # Reference implementation: complex multiplication approach
        def reference_impl(x, freqs_cis):
            # x: [B,S,H,D] → view as complex → [B,S,H,D//2] complex
            x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            # freqs_cis: [S,D//2] → unsqueeze to [1,S,D//2]
            x_out = torch.view_as_real(x_rotated * freqs_cis.unsqueeze(1)).flatten(3)
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
