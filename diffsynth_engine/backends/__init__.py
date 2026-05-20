"""NPU compilation backend utilities."""

import torch

from diffsynth_engine.utils.import_utils import is_npu_available


def maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    """Apply MindieSDBackend compile on NPU, no-op on other devices.

    When NPU is available, this wraps the model with
    ``torch.compile(model, backend=MindieSDBackend())``, which
    automatically fuses RMSNorm, GELU, AdaLayerNorm, RoPE, and
    MulAdd patterns into NPU-optimized kernels at compile time.

    On non-NPU devices, the model is returned unchanged.

    Usage::

        model = init_transformer()
        model = maybe_compile(model)
    """
    if not is_npu_available():
        return model

    try:
        from mindiesd.compilation import MindieSDBackend
    except ImportError:
        return model

    return torch.compile(model, backend=MindieSDBackend())
