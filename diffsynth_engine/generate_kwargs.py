"""Normalize generate() kwargs before pipeline invocation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from PIL import Image


def load_images_from_paths(
    dataset_dir: str, image_names: Sequence[str]
) -> Image.Image | list[Image.Image]:
    images = [
        Image.open(os.path.join(dataset_dir, name)).convert("RGB") for name in image_names
    ]
    return images[0] if len(images) == 1 else images


def resolve_generate_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert broadcast-friendly path fields into pipeline ``image`` input.

    Distributed generate broadcasts ``dataset_dir`` and ``image_names`` from rank 0;
    every rank calls this after broadcast so each loads PIL images locally.
    """
    if "image_names" not in kwargs:
        return kwargs

    if "dataset_dir" not in kwargs:
        raise ValueError("dataset_dir is required when image_names is set")

    resolved = dict(kwargs)
    dataset_dir = resolved.pop("dataset_dir")
    image_names = resolved.pop("image_names")
    resolved["image"] = load_images_from_paths(dataset_dir, image_names)
    return resolved
