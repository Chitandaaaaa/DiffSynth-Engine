# Adapted from https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from diffsynth_engine.distributed.comm import SeqAllToAll4D
from diffsynth_engine.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_group,
    get_ulysses_parallel_world_size,
    is_sp_group_initialized,
)
from diffsynth_engine.forward_context import ForwardContext, get_forward_context
from diffsynth_engine.layers.attention.ring import ring_flash_attention_forward
from diffsynth_engine.registry import get_attn_backend

# Auto FA head-split for power capping (no user-facing fa_split config).
# Each local FA kernel targets at most this many heads when divisible.
# Example: Qwen heads=24 → fa_split=4 (6 heads per chunk).
_PREFERRED_HEADS_PER_CHUNK = 6


def _auto_fa_split(local_heads: int, preferred: int = _PREFERRED_HEADS_PER_CHUNK) -> int:
    """Auto-derive how many head-chunks to split local FA into."""
    if local_heads <= preferred:
        return 1
    for heads_per_chunk in range(preferred, 0, -1):
        if local_heads % heads_per_chunk == 0:
            return local_heads // heads_per_chunk
    return 1


class LocalAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        attn_type: str | None = None,
        **extra_impl_args,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads

        attn_backend = get_attn_backend(attn_type)
        if not attn_backend.supports_head_size(head_size):
            raise ValueError(f"Attention backend {attn_type!r} does not support head size {head_size}.")

        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            **extra_impl_args,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]

        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        attn_kwargs = {"attn_metadata": attn_metadata}
        attn_kwargs.update(kwargs)

        output = self.attn_impl.forward(q, k, v, **attn_kwargs)
        return output


class USPAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        attn_type: str | None = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        **extra_impl_args,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

        attn_backend = get_attn_backend(attn_type)
        if not attn_backend.supports_head_size(head_size):
            raise ValueError(f"Attention backend {attn_type!r} does not support head size {head_size}.")

        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            **extra_impl_args,
        )

    def _split_qkv_by_head(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, parts: int):
        # q/k/v: [B, S, H, D]
        _, _, head_count, _ = q.shape
        if head_count % parts != 0:
            raise ValueError(f"num_heads ({head_count}) must be divisible by fa_split ({parts}).")
        chunk = head_count // parts
        return q.split(chunk, dim=2), k.split(chunk, dim=2), v.split(chunk, dim=2)

    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_kwargs: dict,
    ) -> torch.Tensor:
        local_heads = q.shape[2]
        parts = _auto_fa_split(local_heads)
        if parts <= 1:
            return self.attn_impl.forward(q, k, v, **attn_kwargs)
        q_chunks, k_chunks, v_chunks = self._split_qkv_by_head(q, k, v, parts)
        outs = []
        for qc, kc, vc in zip(q_chunks, k_chunks, v_chunks):
            outs.append(self.attn_impl.forward(qc, kc, vc, **attn_kwargs))
        return torch.cat(outs, dim=2)

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]

        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        attn_kwargs = {"attn_metadata": attn_metadata}
        attn_kwargs.update(kwargs)

        ulysses_parallel_world_size = get_ulysses_parallel_world_size() if is_sp_group_initialized() else 1
        ring_parallel_world_size = get_ring_parallel_world_size() if is_sp_group_initialized() else 1

        if ulysses_parallel_world_size > 1:
            q = SeqAllToAll4D.apply(get_sp_group().ulysses_group, q, self.scatter_idx, self.gather_idx)
            k = SeqAllToAll4D.apply(get_sp_group().ulysses_group, k, self.scatter_idx, self.gather_idx)
            v = SeqAllToAll4D.apply(get_sp_group().ulysses_group, v, self.scatter_idx, self.gather_idx)

        if ring_parallel_world_size > 1:
            # warning: attn_kwargs is not supported for ring flash attention
            output = ring_flash_attention_forward(q, k, v, self.attn_impl, **attn_kwargs)
        else:
            output = self._local_attention(q, k, v, attn_kwargs)

        if ulysses_parallel_world_size > 1:
            output = SeqAllToAll4D.apply(get_sp_group().ulysses_group, output, self.gather_idx, self.scatter_idx)
        return output
