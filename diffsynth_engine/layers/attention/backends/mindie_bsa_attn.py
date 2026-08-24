# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import torch

from diffsynth_engine.distributed.parallel_state import (
    get_ulysses_parallel_world_size,
    is_sp_group_initialized,
)
from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from diffsynth_engine.layers.attention.bsa_utils import (
    interleaved_to_txt_img,
    padded_txt_image_len,
    txt_img_to_interleaved,
)
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)

_BSA_IMPORTS = (
    ("mindiesd.layers.flash_attn.attention_forward", "bsa_sparse_attention_v3"),
    ("mindiesd.layers.flash_attn.bsa_sparse_attention", "bsa_sparse_attention_v3"),
    ("mindiesd.layers.sparse_attention", "bsa_sparse_attention_v3"),
    ("mindiesd", "bsa_sparse_attention_v3"),
)


def _load_bsa_fn() -> Callable[..., Any]:
    errors: list[str] = []
    for module_name, attr in _BSA_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, attr)
            if callable(fn):
                return fn
        except (ImportError, AttributeError) as e:
            errors.append(f"{module_name}.{attr}: {e}")
    raise ImportError(
        "bsa_sparse_attention_v3 not found in MindIE-SD. Tried:\n  " + "\n  ".join(errors)
    )


@dataclass
class MindieBsaAttentionMetadata(AttentionMetadata):
    latent_shape: tuple[int, int, int] | None = None
    txt_len: int = 0
    sparsity: float = 0.6
    inner_precise: int = 4
    protect_first_frame: bool = False
    cached_mask: torch.Tensor | None = None


class MindieBsaAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def build(
        self,
        latent_shape: tuple[int, int, int] | None = None,
        txt_len: int = 0,
        sparsity: float = 0.6,
        inner_precise: int = 4,
        protect_first_frame: bool = False,
        cached_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> MindieBsaAttentionMetadata:
        return MindieBsaAttentionMetadata(
            latent_shape=tuple(latent_shape) if latent_shape is not None else None,
            txt_len=int(txt_len),
            sparsity=float(sparsity),
            inner_precise=int(inner_precise),
            protect_first_frame=bool(protect_first_frame),
            cached_mask=cached_mask,
        )


class MindieBsaAttentionBackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        from diffsynth_engine.platforms import AscendPlatform

        if not AscendPlatform.supports("device"):
            raise RuntimeError("MindIE BSA attention requires an available Ascend NPU device.")
        if not AscendPlatform.supports("mindie"):
            raise RuntimeError("MindIE BSA attention requires MindIE-SD.")
        try:
            _load_bsa_fn()
        except ImportError as e:
            raise RuntimeError(str(e)) from e

    @staticmethod
    def get_type() -> str:
        return str(AttentionType.MINDIE_BSA)

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return MindieBsaAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return MindieBsaAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return MindieBsaAttentionMetadataBuilder

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []

    @classmethod
    def supports_ring_attention(cls) -> bool:
        return False


class MindieBsaAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        **extra_impl_args,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.softmax_scale = softmax_scale if softmax_scale is not None else head_size**-0.5
        self._bsa_fn = None

    def _bsa(self):
        if self._bsa_fn is None:
            self._bsa_fn = _load_bsa_fn()
        return self._bsa_fn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_metadata: MindieBsaAttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if attn_metadata is None or attn_metadata.latent_shape is None:
            raise RuntimeError(
                "mindie_bsa requires MindieBsaAttentionMetadata.latent_shape=(t, H, W). "
                "Build it from img_shapes in the Edit pipeline (t=1+N_ref, H/W from packed pixels)."
            )
        t, h, w = attn_metadata.latent_shape
        image_len = int(t) * int(h) * int(w)
        txt_len = int(attn_metadata.txt_len)
        sp_size = get_ulysses_parallel_world_size() if is_sp_group_initialized() else 1
        txt_pad, img_pad = padded_txt_image_len(txt_len, image_len, sp_size)
        if query.shape[1] != txt_pad + img_pad:
            raise ValueError(
                f"BSA seq mismatch: q_seq={query.shape[1]} != txt_len={txt_pad} + t*H*W={img_pad} "
                f"(latent_shape={attn_metadata.latent_shape}, sp={sp_size})"
            )

        q = interleaved_to_txt_img(query, txt_pad, img_pad, sp_size)
        k = interleaved_to_txt_img(key, txt_pad, img_pad, sp_size)
        v = interleaved_to_txt_img(value, txt_pad, img_pad, sp_size)

        out, new_mask = self._bsa()(
            q,
            k,
            v,
            latent_shape_q=attn_metadata.latent_shape,
            latent_shape_k=attn_metadata.latent_shape,
            txt_len=txt_pad,
            sparsity=attn_metadata.sparsity,
            input_layout="BSND",
            head_num=q.shape[2],
            num_key_value_heads=k.shape[2],
            scale=self.softmax_scale,
            inner_precise=attn_metadata.inner_precise,
            cached_mask=attn_metadata.cached_mask,
            protect_first_frame=attn_metadata.protect_first_frame,
        )
        attn_metadata.cached_mask = new_mask
        return txt_img_to_interleaved(out, txt_pad, img_pad, sp_size)
