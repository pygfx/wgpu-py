"""
The backend implementations of the wgpu API.
"""

import sys

from ..classes import GPU as _base_GPU  # noqa


def _register_backend(gpu):
    """Backends call this to activate themselves.
    It replaces ``wgpu.gpu`` with the ``gpu`` object from the backend.
    """

    root_namespace = sys.modules["wgpu"].__dict__
    needed_attributes = (
        "request_adapter",
        "request_adapter_async",
        "wgsl_language_features",
    )

    # Check
    for attr in needed_attributes:
        if not (hasattr(gpu, attr)):
            raise RuntimeError(
                "The registered WGPU backend object must have attributes "
                + ", ".join(f"'{a}'" for a in needed_attributes)
                + f". The '{attr}' is missing."
            )

    # Only allow registering a backend once
    if not isinstance(root_namespace["gpu"], _base_GPU):
        raise RuntimeError("WGPU backend can only be set once.")

    # Apply
    root_namespace["gpu"] = gpu
    return gpu
