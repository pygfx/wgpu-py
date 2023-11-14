import os

from ._api import ffi, libf, structs, enums, Dict, logger
from ._helpers import get_wgpu_instance


# NOTE: these functions represent backend-specific extra API.
# NOTE: changes to this module must be reflected in docs/backends.rst.
# We don't use Sphinx automodule because this way the doc build do not
# need to be able to load wgpu-native.


def enumerate_adapters():
    """Return a list of all available adapters."""
    # The first call is to get the number of adapters, and the second
    # call is to get the actual adapters. Note that the second arg (now
    # NULL) can be a `WGPUInstanceEnumerateAdapterOptions` to filter
    # by backend.

    adapter_count = libf.wgpuInstanceEnumerateAdapters(
        get_wgpu_instance(), ffi.NULL, ffi.NULL
    )

    adapters = ffi.new("WGPUAdapter[]", adapter_count)
    libf.wgpuInstanceEnumerateAdapters(get_wgpu_instance(), ffi.NULL, adapters)

    from . import gpu  # noqa

    return [gpu._create_adapter(adapter) for adapter in adapters]


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
