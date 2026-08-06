"""NPU Ulysses AllToAll helpers: slim sync path + optional async out-A2A.

Env:
  USE_NPU_A2A_SLIM=1     — reuse recv buffers; tighter reshape (inference)
  USE_NPU_A2A_OVERLAP=1  — async out-A2A on a side stream (implies slim for that call)
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist

USE_NPU_A2A_SLIM = os.environ.get("USE_NPU_A2A_SLIM", "0") == "1"
USE_NPU_A2A_OVERLAP = os.environ.get("USE_NPU_A2A_OVERLAP", "0") == "1"

_recv_buffers: Dict[tuple, torch.Tensor] = {}
_comm_stream = None


def a2a_slim_enabled() -> bool:
    slim = os.environ.get("USE_NPU_A2A_SLIM", "0") == "1"
    overlap = os.environ.get("USE_NPU_A2A_OVERLAP", "0") == "1"
    return slim or overlap


def a2a_overlap_enabled() -> bool:
    return os.environ.get("USE_NPU_A2A_OVERLAP", "0") == "1" and not torch.is_grad_enabled()


def get_comm_stream():
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.npu.Stream()
    return _comm_stream


def _get_recv_buf(like: torch.Tensor) -> torch.Tensor:
    key = (like.device, like.dtype, tuple(like.shape))
    buf = _recv_buffers.get(key)
    if buf is None or buf.shape != like.shape or buf.dtype != like.dtype or buf.device != like.device:
        buf = torch.empty_like(like)
        _recv_buffers[key] = buf
    return buf


def _prepare_send_heads_to_seq(input_: torch.Tensor, group) -> Tuple[torch.Tensor, tuple]:
    """scatter_idx=2, gather_idx=1: (B, S/P, H, D) -> all2all layout + meta for post."""
    seq_world_size = dist.get_world_size(group)
    bs, shard_seqlen, hc, hs = input_.shape
    seqlen = shard_seqlen * seq_world_size
    shard_hc = hc // seq_world_size
    # (B,S/P,H,D) -> (P, S/P, B, H/P, D)
    send = (
        input_.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
        .transpose(0, 2)
        .contiguous()
    )
    return send, (bs, seqlen, shard_hc, hs, shard_seqlen, seq_world_size)


def _post_heads_to_seq(output: torch.Tensor, meta: tuple) -> torch.Tensor:
    bs, seqlen, shard_hc, hs, shard_seqlen, seq_world_size = meta
    # (P,S/P,B,H/P,D) -> (B,S,H/P,D) with one permute+reshape (avoids transpose+contiguous+reshape)
    return (
        output.permute(2, 0, 1, 3, 4)
        .reshape(bs, seqlen, shard_hc, hs)
        .contiguous()
    )


def _prepare_send_seq_to_heads(input_: torch.Tensor, group) -> Tuple[torch.Tensor, tuple]:
    """scatter_idx=1, gather_idx=2: (B, S, H/P, D) -> all2all layout + meta."""
    seq_world_size = dist.get_world_size(group)
    bs, seqlen, shard_hc, hs = input_.shape
    hc = shard_hc * seq_world_size
    shard_seqlen = seqlen // seq_world_size
    send = (
        input_.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
        .transpose(0, 3)
        .transpose(0, 1)
        .contiguous()
        .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
    )
    return send, (bs, shard_seqlen, hc, hs, shard_hc, seq_world_size)


def _post_seq_to_heads(output: torch.Tensor, meta: tuple) -> torch.Tensor:
    bs, shard_seqlen, hc, hs, shard_hc, seq_world_size = meta
    # (P, H/P, S/P, B, D) -> (B, S/P, H, D)
    return (
        output.permute(3, 2, 0, 1, 4)
        .reshape(bs, shard_seqlen, hc, hs)
        .contiguous()
    )


def all_to_all_4d_slim(
    input_: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    group=None,
) -> torch.Tensor:
    """Inference-oriented 4D all-to-all matching yunchang layout semantics."""
    seq_world_size = dist.get_world_size(group)
    if seq_world_size <= 1:
        return input_

    if scatter_idx == 2 and gather_idx == 1:
        send, meta = _prepare_send_heads_to_seq(input_, group)
        recv = _get_recv_buf(send) if a2a_slim_enabled() else torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return _post_heads_to_seq(recv, meta)

    if scatter_idx == 1 and gather_idx == 2:
        send, meta = _prepare_send_seq_to_heads(input_, group)
        recv = _get_recv_buf(send) if a2a_slim_enabled() else torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return _post_seq_to_heads(recv, meta)

    raise RuntimeError(f"unsupported scatter/gather idx: {scatter_idx}/{gather_idx}")


def all_to_all_4d_async(
    input_: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    group=None,
) -> Tuple[torch.Tensor, Callable[[], None]]:
    """Launch all-to-all (+ postprocess) on NPU comm stream; caller must wait_fn() before use."""
    seq_world_size = dist.get_world_size(group)
    if seq_world_size <= 1:
        return input_, (lambda: None)

    if scatter_idx == 2 and gather_idx == 1:
        prepare, post = _prepare_send_heads_to_seq, _post_heads_to_seq
    elif scatter_idx == 1 and gather_idx == 2:
        prepare, post = _prepare_send_seq_to_heads, _post_seq_to_heads
    else:
        raise RuntimeError(f"unsupported scatter/gather idx: {scatter_idx}/{gather_idx}")

    send, meta = prepare(input_, group)
    recv = _get_recv_buf(send)
    # Snapshot into a private buffer for async lifetime (pool buf may be reused on next call).
    recv_work = torch.empty_like(recv)

    default = torch.npu.current_stream()
    comm = get_comm_stream()
    ready = torch.npu.Event()
    ready.record(default)

    with torch.npu.stream(comm):
        ready.wait(comm)
        dist.all_to_all_single(recv_work, send, group=group)
        out = post(recv_work, meta)

    def wait_fn():
        torch.npu.current_stream().wait_stream(comm)

    return out, wait_fn


def seq_all_to_all_4d(
    group,
    input_: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    async_op: bool = False,
):
    """Dispatch: async / slim / yunchang SeqAllToAll4D."""
    if async_op and a2a_overlap_enabled() and input_.device.type == "npu":
        return all_to_all_4d_async(input_, scatter_idx, gather_idx, group=group)

    if a2a_slim_enabled() and input_.device.type == "npu" and not torch.is_grad_enabled():
        return all_to_all_4d_slim(input_, scatter_idx, gather_idx, group=group), None

    from yunchang.comm.all_to_all import SeqAllToAll4D

    return SeqAllToAll4D.apply(group, input_, scatter_idx, gather_idx), None
