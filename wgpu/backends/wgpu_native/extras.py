import os

from ._api import GPUBindGroupLayout, structs, enums, Dict, logger
from typing import List

# NOTE: these functions represent backend-specific extra API.
# NOTE: changes to this module must be reflected in docs/backends.rst.
# We don't use Sphinx automodule because this way the doc build do not
# need to be able to load wgpu-native.


def enumerate_adapters():
    """Deprecated."""
    raise RuntimeError("Deprecated: use wgpu.gpu.enumerate_adapters() instead.")


def request_device_sync(
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


# Backwards compat for deprecated function
def request_device(*args, **kwargs):
    logger.warning(
        "WGPU: wgpu.backends.wgpu_native.request_device() is deprecated, use request_device_sync() instead."
    )
    return request_device_sync(*args, **kwargs)


def create_pipeline_layout(
    device,
    *,
    label="",
    bind_group_layouts: "List[GPUBindGroupLayout]",
    push_constant_layouts: "List[Dict]" = [],
):
    return device._create_pipeline_layout(
        label, bind_group_layouts, push_constant_layouts
    )


def set_push_constants(
    render_pass_encoder, visibility, offset, size_in_bytes, data, data_offset=0
):
    """
    Set push-constant data for subsequent draw calls.

    Writes the first size_in_bytes bytes of data to push-constant storage,
    starting at the specified offset. These bytes are visible to the pipeline
    stages indicated by the visibility argument.
    """

    # Actual implementation is hidden in _api.py
    render_pass_encoder._set_push_constants(
        visibility, offset, size_in_bytes, data, data_offset
    )


def multi_draw_indirect(render_pass_encoder, buffer, *, offset=0, count):
    """
    This is equvalent to
    for i in range(count):
        render_pass_encoder.draw(buffer, offset + i * 16)

    You must enable the featue "multi-draw-indirect" to use this function.
    """
    render_pass_encoder._multi_draw_indirect(buffer, offset, count)


def multi_draw_indexed_indirect(render_pass_encoder, buffer, *, offset=0, count):
    """
    This is equvalent to

    for i in range(count):
        render_pass_encoder.draw_indexed(buffer, offset + i * 20)

    You must enable the featue "multi-draw-indirect" to use this function.
    """
    render_pass_encoder._multi_draw_indexed_indirect(buffer, offset, count)
