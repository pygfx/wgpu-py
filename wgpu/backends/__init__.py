"""
The backend implementations of the wgpu API.
You need to select one by importing it, e.g. ``import wgpu.backends.rs``.
"""

import sys

from ..base import GPU as _base_GPU  # noqa


def _register_backend(cls):
    """Backends call this to activate themselves."""
    GPU = cls  # noqa: N806
    if not (
        hasattr(GPU, "request_adapter")
        and callable(GPU.request_adapter)
        and hasattr(GPU, "request_adapter_async")
        and callable(GPU.request_adapter_async)
    ):
        raise RuntimeError(
            "The registered WGPU backend object must have methods "
            + "'request_adapter' and 'request_adapter_async'"
        )

    # Set gpu object and reset request_adapter-functions
    root_namespace = sys.modules["wgpu"].__dict__

    if root_namespace["GPU"] is not _base_GPU:
        raise RuntimeError("WGPU backend can only be set once.")
    gpu = GPU()
    root_namespace["GPU"] = GPU
    root_namespace["request_adapter"] = gpu.request_adapter
    root_namespace["request_adapter_async"] = gpu.request_adapter_async
    root_namespace["wgsl_language_features"] = gpu.wgsl_language_features
    return cls


_register_backend(_base_GPU)
