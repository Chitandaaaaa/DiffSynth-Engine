# SPDX-License-Identifier: Apache-2.0
"""Layout helpers for MindIE BSA on Qwen-Image / Edit.

BSA assumes joint QKV is [text | image] with image tokens a single grid:
``S_image = t * H * W``. Edit uses ``t = 1 + N_ref`` (target + references),
all packed to the same ``(H, W)`` — not hardcoded 64×64.

Ulysses shards text and image separately then concatenates per rank, so after
all-to-all the sequence is interleaved ``[txt_r | img_r] * ranks``. BSA needs
the logical ``[txt | img]`` order; we restore before the kernel and invert after.
"""

from __future__ import annotations

from typing import Sequence

import torch


def packed_hw(height: int, width: int, vae_scale_factor: int) -> tuple[int, int]:
    """Packed DiT grid from pixel size. 1024×1024 + vae_sf=8 → (64, 64)."""
    return height // vae_scale_factor // 2, width // vae_scale_factor // 2


def align_up(value: int, align: int) -> int:
    if align <= 1:
        return value
    return (value + align - 1) // align * align


def latent_shape_from_img_shapes(img_shapes: Sequence[Sequence[tuple[int, int, int]]]) -> tuple[int, int, int]:
    """Fold per-image ``(1, H, W)`` segments into one BSA ``(t, H, W)``.

    ``t`` is how many equal H×W blocks sit in the image stream (1 + N_ref for Edit),
    not "how many tensor types".
    """
    if not img_shapes or not img_shapes[0]:
        raise ValueError("img_shapes is empty; cannot derive BSA latent_shape")
    segments = img_shapes[0]
    hs = {seg[1] for seg in segments}
    ws = {seg[2] for seg in segments}
    if len(hs) != 1 or len(ws) != 1:
        raise ValueError(
            "BSA requires every image segment to share packed H×W "
            f"(t = 1+N_ref of identical grids). Got segments={list(segments)}"
        )
    h, w = segments[0][1], segments[0][2]
    t = sum(int(seg[0]) for seg in segments)
    return t, h, w


def interleaved_to_txt_img(x: torch.Tensor, txt_len: int, image_len: int, sp_size: int) -> torch.Tensor:
    """``[txt_r | img_r] * sp`` → ``[txt | img]``. Identity when ``sp_size == 1``."""
    if sp_size <= 1:
        return x
    txt_local = txt_len // sp_size
    img_local = image_len // sp_size
    chunk = txt_local + img_local
    if x.shape[1] != chunk * sp_size:
        raise ValueError(
            f"BSA layout mismatch: seq={x.shape[1]} != sp({sp_size}) * "
            f"(txt_local={txt_local} + img_local={img_local})"
        )
    parts = x.split(chunk, dim=1)
    txt = torch.cat([p[:, :txt_local] for p in parts], dim=1)
    img = torch.cat([p[:, txt_local:] for p in parts], dim=1)
    return torch.cat([txt, img], dim=1)


def txt_img_to_interleaved(x: torch.Tensor, txt_len: int, image_len: int, sp_size: int) -> torch.Tensor:
    """``[txt | img]`` → ``[txt_r | img_r] * sp``. Identity when ``sp_size == 1``."""
    if sp_size <= 1:
        return x
    txt_local = txt_len // sp_size
    img_local = image_len // sp_size
    txt = x[:, :txt_len]
    img = x[:, txt_len:]
    txt_parts = txt.split(txt_local, dim=1)
    img_parts = img.split(img_local, dim=1)
    return torch.cat([torch.cat([t, i], dim=1) for t, i in zip(txt_parts, img_parts)], dim=1)


def padded_txt_image_len(txt_len: int, image_len: int, sp_size: int) -> tuple[int, int]:
    return align_up(txt_len, sp_size), align_up(image_len, sp_size)
