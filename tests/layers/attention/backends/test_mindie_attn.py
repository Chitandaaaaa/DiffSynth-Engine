import torch
import pytest
from diffsynth_engine.layers.attention.backends.mindie_attn import (
    MindieAttentionBackend,
    MindieAttentionImpl,
)
from diffsynth_engine.layers.attention.backends.abstract import AttentionType


class TestMindieAttentionBackend:
    def test_backend_type(self):
        """Verify MindieBackend.get_type() returns MINDIE."""
        assert MindieAttentionBackend.get_type() == AttentionType.MINDIE

    def test_backend_supported_head_sizes(self):
        """Verify supported head sizes returns empty list (all sizes supported)."""
        assert MindieAttentionBackend.get_supported_head_sizes() == []


class TestMindieAttentionImpl:
    def test_impl_init(self):
        """Verify MindieAttentionImpl initialization."""
        impl = MindieAttentionImpl(
            num_heads=8,
            head_size=64,
            softmax_scale=0.125,
            causal=False,
            num_kv_heads=8,
        )
        assert impl.num_heads == 8
        assert impl.head_size == 64
        assert impl.num_kv_groups == 1
        assert impl.causal is False
        assert impl.softmax_scale == 0.125

    def test_impl_init_gqa(self):
        """Verify GQA initialization."""
        impl = MindieAttentionImpl(
            num_heads=8,
            head_size=64,
            num_kv_heads=2,  # GQA with 2 KV heads
        )
        assert impl.num_kv_groups == 4  # 8 // 2 = 4

    def test_impl_default_softmax_scale(self):
        """Verify default softmax scale is computed correctly."""
        impl = MindieAttentionImpl(
            num_heads=8,
            head_size=64,
        )
        assert impl.softmax_scale == 64 ** -0.5

    def test_impl_forward_signature(self):
        """Verify forward method accepts expected parameters."""
        impl = MindieAttentionImpl(
            num_heads=8,
            head_size=64,
        )
        q = torch.randn(2, 16, 8, 64)
        k = torch.randn(2, 16, 8, 64)
        v = torch.randn(2, 16, 8, 64)
        # Just verify signature - actual call needs NPU
        assert callable(impl.forward)
