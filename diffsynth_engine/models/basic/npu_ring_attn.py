"""NPU Ring attention: KV rotation + Ascend fusion attention with softmax stats.

First-principles Ring for Ascend: each step calls ``npu_fusion_attention`` and
merges partial results with softmax_max / softmax_sum (not v1 FA-style LSE).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

_MAX_TOKEN = 2147483647


class RingComm:
    """Ring P2P helper (same contract as yunchang / DiffSynth v1 RingComm)."""

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = torch.empty_like(to_send) if recv_tensor is None else recv_tensor
        self._ops.append(dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group))
        self._ops.append(dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group))
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


def npu_fusion_attn_with_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Ascend fusion attention and keep softmax_max / softmax_sum.

    Args:
        q, k, v: [B, S, N, D] (BSND), same layout as MindIE ``head_first=False``.
    Returns:
        out: [B, S, N, D]
        softmax_max, softmax_sum: FA workspace tensors from ``npu_fusion_attention``
    """
    import torch_npu

    if q.dim() != 4:
        raise ValueError(f"expected 4D BSND tensors, got q.dim()={q.dim()}")
    head_num = q.shape[2]
    if scale is None:
        scale = q.shape[-1] ** -0.5

    mask = None
    if attn_mask is not None:
        # Match MindIE fused_attn_score: invert bool mask for atten_mask semantics.
        mask = ~attn_mask.to(torch.bool) if attn_mask.dtype == torch.bool else attn_mask

    out, softmax_max, softmax_sum, *_ = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=head_num,
        input_layout="BSND",
        scale=scale,
        atten_mask=mask,
        pre_tockens=_MAX_TOKEN,
        next_tockens=_MAX_TOKEN,
    )
    return out, softmax_max, softmax_sum


def update_out_with_softmax_stats(
    prev_out: Optional[torch.Tensor],
    prev_max: Optional[torch.Tensor],
    prev_sum: Optional[torch.Tensor],
    cur_out: torch.Tensor,
    cur_max: torch.Tensor,
    cur_sum: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online softmax merge using Ascend FA softmax_max / softmax_sum.

    Layout follows yunchang ``update_npu_out`` for BSND attention outputs:
    out is [B, S, N, D]; max/sum are typically [B, N, S, 8] (last dim tiled).
    """
    if prev_out is None:
        return cur_out, cur_max, cur_sum

    attn_h = cur_out.shape[2]
    # [B, S, N, D] -> [S, B, N*D]
    cur_flat = cur_out.reshape(cur_out.shape[0], cur_out.shape[1], -1).permute(1, 0, 2).contiguous()
    prev_flat = prev_out.reshape(prev_out.shape[0], prev_out.shape[1], -1).permute(1, 0, 2).contiguous()
    origin_dtype = prev_flat.dtype

    softmax_max = torch.maximum(prev_max, cur_max)
    prev_scale = torch.exp(prev_max - softmax_max)
    cur_scale = torch.exp(cur_max - softmax_max)

    prev_sum_scaled = prev_sum * prev_scale
    cur_sum_scaled = cur_sum * cur_scale
    softmax_sum = prev_sum_scaled + cur_sum_scaled

    prev_out_scale = prev_sum_scaled / softmax_sum
    cur_out_scale = cur_sum_scaled / softmax_sum

    # max/sum last dim is often 8 (broadcast tile); take first channel and expand to D
    n = prev_out_scale.shape[1]
    d = prev_flat.shape[-1] // n
    prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    prev_out_scale = prev_out_scale.permute(2, 0, 1, 3).reshape(prev_flat.shape[0], prev_flat.shape[1], -1)
    cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    cur_out_scale = cur_out_scale.permute(2, 0, 1, 3).reshape(cur_flat.shape[0], cur_flat.shape[1], -1)

    merged = (prev_flat.float() * prev_out_scale + cur_flat.float() * cur_out_scale).to(origin_dtype)
    # [S, B, N*D] -> [B, S, N, D]
    b = cur_out.shape[0]
    s = cur_out.shape[1]
    merged = merged.permute(1, 0, 2).reshape(b, s, attn_h, d).contiguous()
    return merged, softmax_max, softmax_sum


def _ring_all_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    scale: Optional[float],
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Correctness fallback: gather full K/V on ring group, one local fusion attn."""
    world = dist.get_world_size(group)
    k_list = [torch.empty_like(k) for _ in range(world)]
    v_list = [torch.empty_like(v) for _ in range(world)]
    dist.all_gather(k_list, k.contiguous(), group=group)
    dist.all_gather(v_list, v.contiguous(), group=group)
    k_full = torch.cat(k_list, dim=1)
    v_full = torch.cat(v_list, dim=1)
    out, _, _ = npu_fusion_attn_with_stats(q, k_full, v_full, scale=scale, attn_mask=attn_mask)
    return out


def _ring_p2p_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    scale: Optional[float],
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    comm = RingComm(group)
    world_size = comm.world_size

    out = None
    softmax_max = None
    softmax_sum = None
    key = k.contiguous()
    value = v.contiguous()

    for step in range(world_size):
        if step + 1 < world_size:
            next_key = comm.send_recv(key)
            next_value = comm.send_recv(value)
            comm.commit()

        block_out, block_max, block_sum = npu_fusion_attn_with_stats(
            q, key, value, scale=scale, attn_mask=attn_mask
        )
        out, softmax_max, softmax_sum = update_out_with_softmax_stats(
            out, softmax_max, softmax_sum, block_out, block_max, block_sum
        )

        if step + 1 < world_size:
            comm.wait()
            key = next_key
            value = next_value

    return out


def npu_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Ring attention on NPU using fusion attention stats + KV exchange.

    Env ``NPU_RING_COMM``:
      - ``p2p`` (default): RingComm isend/irecv
      - ``all_gather``: gather full KV then one fusion attn (correctness fallback)
    """
    from diffsynth_engine.utils.process_group import get_sp_ring_group

    if group is None:
        group = get_sp_ring_group()
    if group is None:
        raise RuntimeError("ring attention requires an initialized SP ring process group")

    mode = os.environ.get("NPU_RING_COMM", "p2p").strip().lower()
    if mode == "all_gather":
        logger.info("NPU ring attention using all_gather fallback (NPU_RING_COMM=all_gather)")
        return _ring_all_gather_attention(q, k, v, group, scale, attn_mask)
    if mode != "p2p":
        raise ValueError(f"unsupported NPU_RING_COMM={mode!r}, expected 'p2p' or 'all_gather'")
    return _ring_p2p_attention(q, k, v, group, scale, attn_mask)
