import os
from typing import Sequence, Union

from . import (
    GPUAdapter,
    GPUDevice,
    GPUBuffer,
    GPUCommandEncoder,
    GPUComputePassEncoder,
    GPURenderPassEncoder,
    GPUPipelineLayout,
    GPUQuerySet,
)
from ._api import (
    GPUBindGroupLayout,
    enums,
    logger,
    structs,
    flags,
    new_struct_p,
    to_c_string_view,
    enum_str2int,
)
from ...enums import Enum
from ._helpers import get_wgpu_instance
from ..._coreutils import get_library_filename, ArrayLike
from ._ffi import lib, ffi
from ._mappings import native_flags


# NOTE: these functions represent backend-specific extra API.
# NOTE: changes to this module must be reflected in docs/backends.rst.
# We don't use Sphinx automodule because this way the doc build do not
# need to be able to load wgpu-native.


class PipelineStatisticName(Enum):  # wgpu native
    VertexShaderInvocations = "vertex-shader-invocations"
    ClipperInvocations = "clipper-invocations"
    ClipperPrimitivesOut = "clipper-primitives-out"
    FragmentShaderInvocations = "fragment-shader-invocations"
    ComputeShaderInvocations = "compute-shader-invocations"


def request_device_sync(
    adapter: GPUAdapter,
    trace_path: str,
    *,
    label: str = "",
    required_features: Sequence[enums.FeatureName] = (),
    required_limits: dict[str, int] | None = None,
    default_queue: structs.QueueDescriptorStruct | None = None,
) -> GPUDevice:
    """Write a trace of all commands to a file so it can be reproduced
    elsewhere. The trace is cross-platform!
    """
    required_limits = {} if required_limits is None else required_limits
    default_queue = {} if default_queue is None else default_queue
    if not os.path.isdir(trace_path):
        os.makedirs(trace_path, exist_ok=True)
    elif os.listdir(trace_path):
        logger.warning(f"Trace directory not empty: {trace_path}")
    promise = adapter._request_device_async(
        label, required_features, required_limits, default_queue, trace_path
    )
    return promise.sync_wait()


# Backwards compat for deprecated function
def request_device(*args, **kwargs):
    logger.warning(
        "WGPU: wgpu.backends.wgpu_native.request_device() is deprecated, use request_device_sync() instead."
    )
    return request_device_sync(*args, **kwargs)


def create_pipeline_layout(
    device: GPUDevice,
    *,
    label: str = "",
    bind_group_layouts: Sequence[GPUBindGroupLayout],
    push_constant_layouts: Sequence[dict] = (),
) -> GPUPipelineLayout:
    return device._create_pipeline_layout(
        label, bind_group_layouts, push_constant_layouts
    )


def set_push_constants(
    render_pass_encoder: GPURenderPassEncoder,
    visibility: flags.ShaderStageFlags,
    offset: int,
    size_in_bytes: int,
    data: ArrayLike,
    data_offset: int = 0,
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


def multi_draw_indirect(
    render_pass_encoder: GPURenderPassEncoder,
    buffer: GPUBuffer,
    *,
    offset: int = 0,
    count: int,
):
    """
    This is equivalent to
    for i in range(count):
        render_pass_encoder.draw(buffer, offset + i * 16)

    You must enable the feature "multi-draw-indirect" to use this function.
    """
    render_pass_encoder._multi_draw_indirect(buffer, offset, count)


def multi_draw_indexed_indirect(
    render_pass_encoder: GPURenderPassEncoder,
    buffer: GPUBuffer,
    *,
    offset: int = 0,
    count: int,
):
    """
    This is equivalent to

    for i in range(count):
        render_pass_encoder.draw_indexed(buffer, offset + i * 20)

    You must enable the feature "multi-draw-indirect" to use this function.
    """
    render_pass_encoder._multi_draw_indexed_indirect(buffer, offset, count)


def multi_draw_indirect_count(
    render_pass_encoder: GPURenderPassEncoder,
    buffer: GPUBuffer,
    *,
    offset: int = 0,
    count_buffer: GPUBuffer,
    count_buffer_offset: int = 0,
    max_count: int,
):
    """
    This is equivalent to:

    count = min(<u32 at offset count_buffer_offset of count_buffer>, max_count)
    for i in range(count):
        render_pass_encoder.draw(buffer, offset + i * 16)

    You must enable the feature "multi-draw-indirect-count" to use this function.
    """
    render_pass_encoder._multi_draw_indirect_count(
        buffer, offset, count_buffer, count_buffer_offset, max_count
    )


def multi_draw_indexed_indirect_count(
    render_pass_encoder: GPURenderPassEncoder,
    buffer: GPUBuffer,
    *,
    offset: int = 0,
    count_buffer: GPUBuffer,
    count_buffer_offset: int = 0,
    max_count: int,
):
    """
    This is equivalent to:

    count = min(<u32 at offset count_buffer_offset of count_buffer>, max_count)
    for i in range(count):
        render_pass_encoder.draw_indexed(buffer, offset + i * 20)

    You must enable the feature "multi-draw-indirect-count" to use this function.
    """
    render_pass_encoder._multi_draw_indexed_indirect_count(
        buffer, offset, count_buffer, count_buffer_offset, max_count
    )


def create_statistics_query_set(device, *, label="", count: int, statistics):
    """
    Create a query set that can collect the specified pipeline statistics.
    You must enable the feature "pipeline-statitistics_query" to collect pipeline
    statistics.
    """
    return device._create_statistics_query_set(label, count, statistics)


def begin_pipeline_statistics_query(
    encoder: GPURenderPassEncoder | GPUComputePassEncoder,
    query_set: GPUQuerySet,
    query_index: int,
):
    print(encoder, type(encoder))
    assert isinstance(encoder, (GPURenderPassEncoder, GPUComputePassEncoder))
    encoder._begin_pipeline_statistics_query(query_set, query_index)


def end_pipeline_statistics_query(
    encoder: GPURenderPassEncoder | GPUComputePassEncoder,
):
    assert isinstance(encoder, (GPURenderPassEncoder, GPUComputePassEncoder))
    encoder._end_pipeline_statistics_query()


def write_timestamp(
    encoder: GPURenderPassEncoder | GPUComputePassEncoder | GPUCommandEncoder,
    query_set: GPUQuerySet,
    query_index: int,
):
    assert isinstance(
        encoder, (GPURenderPassEncoder, GPUComputePassEncoder, GPUCommandEncoder)
    )
    encoder._write_timestamp(query_set, query_index)


def set_instance_extras(
    backends: Sequence[str] = ("All",),
    flags: Sequence[str] = ("Default",),
    dx12_compiler="fxc",
    gles3_minor_version="Atomic",
    fence_behavior="Normal",
    dxc_path: Union[os.PathLike, None] = None,
    dxc_max_shader_model: float = 6.5,
    budget_for_device_creation: Union[int, None] = None,
    budget_for_device_loss: Union[int, None] = None,
):
    """
    Sets the global instance with extras. Needs to be called before instance is created (in enumerate_adapters or request_adapter).
    Most of these options are for specific backends, and might not create an instance or crash when used in the wrong combinations.
    Args:
        backends: bitflags as list[str], which backends to enable on the instance level. Defaults to ``["All"]``.
        flags: bitflags as list[str], for debugging the instance and compiler. Defaults to ``["Default"]``.
        dx12_compiler: enum/str, either "Fxc", "Dxc" or "Undefined". Defaults to "Fxc" same as "Undefined". Dxc requires additional library files.
        gles3_minor_version: enum/int, 0, 1 or 2. Defaults to "Atomic" (handled by driver).
        fence_behavior: enum/int, "Normal" or "AutoFinish". Defaults to "Normal".
        dxc_path: Path to the dxcompiler.dll file, if not provided or `None`, will try to load from wgpu/resources.
        dxc_max_shader_model: float between 6.0 and 6.7, the maximum shader model to use with DXC. Defaults to 6.5.
        budget_for_device_creation: Optional[int], between 0 and 100, to specify memory budget threshold for when creating resources (buffer, textures...) will fail. Defaults to None.
        budget_for_device_loss: Optional[int], between 0 and 100, to specify memory budget threshold when the device will be lost. Defaults to None.
    """
    # TODO document and explain, add examples

    backend_bitflags = 0
    for backend in backends:
        # there will be KeyErrors and no fallback to warn the user.
        backend_bitflags |= native_flags["InstanceBackend." + backend]

    flag_bitflags = 0
    for flag in flags:
        flag_bitflags |= native_flags["InstanceFlag." + flag]

    c_dx12_compiler = enum_str2int["Dx12Compiler"].get(
        dx12_compiler.capitalize(), enum_str2int["Dx12Compiler"]["Undefined"]
    )
    # https://docs.rs/wgpu/latest/wgpu/enum.Dx12Compiler.html#variant.DynamicDxc #explains the idea, will improve in the future.
    if (
        c_dx12_compiler == enum_str2int["Dx12Compiler"]["Dxc"] and not dxc_path
    ):  # or os.path.exists(dxc_path)): # this check errors with None as default. but we can't have empty strings.
        # if dxc is specified but no paths are provided, there will be a panic about static-dxc, so maybe we check against that.
        try:
            dxc_path = get_library_filename("dxcompiler.dll")
        except RuntimeError as e:
            # here we couldn't load the libs from wgpu/resources... so we assume the user doesn't have them.
            # TODO: explain user to add DXC manually or provide a script/package it? (in the future)
            logger.warning(
                f"could not load .dll files for DXC from /resource: {e}.\n Please provide a path manually which can panic. Falling back to FXC"
            )
            c_dx12_compiler = enum_str2int["Dx12Compiler"]["Fxc"]

    # https://docs.rs/wgpu/latest/wgpu/enum.Gles3MinorVersion.html
    if gles3_minor_version[-1].isdigit():
        gles3_minor_version = (
            int(gles3_minor_version[-1]) + 1
        )  # hack as the last char easily maps to the enum.
    elif isinstance(gles3_minor_version, str):
        gles3_minor_version = 0  # likely means atomic

    # https://docs.rs/wgpu/latest/wgpu/enum.GlFenceBehavior.html
    fence_behavior_map = {
        "Normal": 0,  # WGPUGLFenceBehavior_Normal
        "AutoFinish": 1,  # WGPUGLFenceBehavior_AutoFinish
    }
    fence_behavior = fence_behavior_map.get(fence_behavior, 0)

    # hack as only version 6.0..6.7 are supported and enum mapping fits.
    c_max_shader_model = int((dxc_max_shader_model - 6.0) * 1.0)

    # https://docs.rs/wgpu/latest/wgpu/struct.MemoryBudgetThresholds.html
    c_budget_creation = (
        ffi.new("uint8_t *", budget_for_device_creation)
        if budget_for_device_creation is not None
        else ffi.NULL
    )
    c_budget_loss = (
        ffi.new("uint8_t *", budget_for_device_loss)
        if budget_for_device_loss is not None
        else ffi.NULL
    )

    # H: chain: WGPUChainedStruct, backends: WGPUInstanceBackend/int, flags: WGPUInstanceFlag/int, dx12ShaderCompiler: WGPUDx12Compiler, gles3MinorVersion: WGPUGles3MinorVersion, glFenceBehaviour: WGPUGLFenceBehaviour, dxcPath: WGPUStringView, dxcMaxShaderModel: WGPUDxcMaxShaderModel, const uint8_t* budgetForDeviceCreation, const uint8_t* budgetForDeviceLoss
    c_extras = new_struct_p(
        "WGPUInstanceExtras *",
        # not used: chain
        backends=backend_bitflags,
        flags=flag_bitflags,
        dx12ShaderCompiler=c_dx12_compiler,
        gles3MinorVersion=gles3_minor_version,
        glFenceBehaviour=fence_behavior,
        dxcPath=to_c_string_view(dxc_path),
        dxcMaxShaderModel=c_max_shader_model,
        budgetForDeviceCreation=c_budget_creation,
        budgetForDeviceLoss=c_budget_loss,
    )

    c_extras.chain.sType = lib.WGPUSType_InstanceExtras
    get_wgpu_instance(extras=c_extras)  # this sets a global
