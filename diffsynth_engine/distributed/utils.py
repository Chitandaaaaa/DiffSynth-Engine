import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from diffsynth_engine.distributed.parallel_state import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_device_process_group,
    get_sp_group,
    model_parallel_is_initialized,
)


def _pad_and_chunk(
    tensor: torch.Tensor, seq_dim: int, sp_world_size: int, sp_rank: int
) -> Tuple[torch.Tensor, int]:
    seq_len = tensor.size(seq_dim)
    pad_len = math.ceil(seq_len / sp_world_size) * sp_world_size - seq_len
    if pad_len > 0:
        padding = [0] * (2 * tensor.ndim)
        padding[-2 * seq_dim - 1] = pad_len
        tensor = F.pad(tensor, padding)
    chunks = torch.chunk(tensor, sp_world_size, dim=seq_dim)
    return chunks[sp_rank], seq_len


def _edit_split_local_lengths(output_len: int, cond_len: int, sp_world_size: int) -> Tuple[int, int]:
    out_padded = math.ceil(output_len / sp_world_size) * sp_world_size
    cond_padded = math.ceil(cond_len / sp_world_size) * sp_world_size
    return out_padded // sp_world_size, cond_padded // sp_world_size


def sequence_parallel_shard(tensors: List[torch.Tensor], seq_dims: List[int]) -> List[torch.Tensor]:
    if not model_parallel_is_initialized() or get_sequence_parallel_world_size() == 1:
        return tensors

    sp_world_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()

    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have the same number of elements"

    shard_tensors = []
    for tensor, seq_dim in zip(tensors, seq_dims):
        if tensor is None:
            shard_tensors.append(tensor)
            continue
        chunk, _ = _pad_and_chunk(tensor, seq_dim, sp_world_size, sp_rank)
        shard_tensors.append(chunk)
    return shard_tensors


def sequence_parallel_shard_edit_split(
    tensors: List[torch.Tensor],
    seq_dims: List[int],
    output_len: int,
) -> List[torch.Tensor]:
    """Shard output and cond segments separately, then concat [local_out | local_cond] per rank."""
    if not model_parallel_is_initialized() or get_sequence_parallel_world_size() == 1:
        return tensors

    sp_world_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()

    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have the same number of elements"

    shard_tensors = []
    for tensor, seq_dim in zip(tensors, seq_dims):
        if tensor is None:
            shard_tensors.append(tensor)
            continue

        total_len = tensor.size(seq_dim)
        if output_len <= 0 or output_len >= total_len:
            (fallback,) = sequence_parallel_shard([tensor], [seq_dim])
            shard_tensors.append(fallback)
            continue

        cond_len = total_len - output_len
        out_part = tensor.narrow(seq_dim, 0, output_len)
        cond_part = tensor.narrow(seq_dim, output_len, cond_len)
        out_shard, _ = _pad_and_chunk(out_part, seq_dim, sp_world_size, sp_rank)
        cond_shard, _ = _pad_and_chunk(cond_part, seq_dim, sp_world_size, sp_rank)
        shard_tensors.append(torch.cat([out_shard, cond_shard], dim=seq_dim))
    return shard_tensors


def sequence_parallel_unshard(
    tensors: List[torch.Tensor],
    seq_dims: List[int],
    seq_lens: List[int],
) -> List[torch.Tensor]:
    if not model_parallel_is_initialized() or get_sequence_parallel_world_size() == 1:
        return tensors

    sp_group = get_sp_group()

    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have the same number of elements"
    assert len(tensors) == len(seq_lens), "tensors and seq_lens must have the same number of elements"

    device_group = get_sp_device_process_group()
    unshard_tensors = []
    for tensor, seq_dim, seq_len in zip(tensors, seq_dims, seq_lens):
        unshard = sp_group.all_gather(tensor, dim=seq_dim, group=device_group)
        unshard = unshard.narrow(dim=seq_dim, start=0, length=seq_len)
        unshard_tensors.append(unshard)
    return unshard_tensors


def sequence_parallel_unshard_edit_output(
    tensors: List[torch.Tensor],
    seq_dims: List[int],
    output_len: int,
    total_image_seq_len: int,
) -> List[torch.Tensor]:
    """Gather only the output segment from [local_out | local_cond] shards."""
    if not model_parallel_is_initialized() or get_sequence_parallel_world_size() == 1:
        return tensors

    sp_group = get_sp_group()
    sp_world_size = get_sequence_parallel_world_size()
    device_group = get_sp_device_process_group()

    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have the same number of elements"

    cond_len = total_image_seq_len - output_len
    out_local_len, _cond_local_len = _edit_split_local_lengths(output_len, cond_len, sp_world_size)

    unshard_tensors = []
    for tensor, seq_dim in zip(tensors, seq_dims):
        out_local = tensor.narrow(seq_dim, 0, out_local_len)
        unshard = sp_group.all_gather(out_local, dim=seq_dim, group=device_group)
        unshard = unshard.narrow(dim=seq_dim, start=0, length=output_len)
        unshard_tensors.append(unshard)
    return unshard_tensors
