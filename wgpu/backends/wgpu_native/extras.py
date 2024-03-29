import os

from ._api import structs, enums, Dict, logger


# NOTE: these functions represent backend-specific extra API.
# NOTE: changes to this module must be reflected in docs/backends.rst.
# We don't use Sphinx automodule because this way the doc build do not
# need to be able to load wgpu-native.


def enumerate_adapters():
    """Deprecated."""
    raise RuntimeError("Deprecated: use wgpu.gpu.enumerate_adapters() instead.")


def request_device_tracing(
    adapter,
    trace_path,
    *,
    label="",
    required_features: "list(enums.FeatureName)" = [],
    required_limits: "Dict[str, int]" = {},
    default_queue: "structs.QueueDescriptor" = {},
):
    """Write a trace of all commands to a file so it can be reproduced
    elsewhere. The trace is cross-platform!
    """
    if not os.path.isdir(trace_path):
        os.makedirs(trace_path, exist_ok=True)
    elif os.listdir(trace_path):
        logger.warning(f"Trace directory not empty: {trace_path}")
    return adapter._request_device(
        label, required_features, required_limits, default_queue, trace_path
    )
