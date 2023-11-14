import os

from ._api import gpu, ffi, libf, structs, enums, Dict, logger
from ._helpers import get_wgpu_instance


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
