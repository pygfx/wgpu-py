"""Create a Dawn device + buffer + compute pipeline via cffi ABI mode, for benchmarks."""

import os
import sys

from cffi import FFI

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.environ.get("CONDA_PREFIX", sys.prefix)
LIB = os.path.join(PREFIX, "lib", "libwebgpu_dawn.so")

ffi = FFI()
ffi.cdef(open(os.path.join(HERE, "dawn_clean.h")).read())
lib = ffi.dlopen(LIB)


def sv(s):
    b = s.encode()
    keep = ffi.new("char[]", b)
    v = ffi.new("WGPUStringView *")
    v.data = keep
    v.length = len(b)
    return v[0], keep


def setup():
    feats = ffi.new(
        "WGPUInstanceFeatureName[1]", [lib.WGPUInstanceFeatureName_TimedWaitAny]
    )
    idesc = ffi.new("WGPUInstanceDescriptor *")
    idesc.requiredFeatureCount = 1
    idesc.requiredFeatures = feats
    instance = lib.wgpuCreateInstance(idesc)
    assert instance != ffi.NULL
    result = {}

    @ffi.callback(
        "void(WGPURequestAdapterStatus, WGPUAdapter, WGPUStringView, void*, void*)"
    )
    def on_adapter(status, adapter, msg, u1, u2):
        result["adapter"] = adapter
        result["msg"] = ffi.string(msg.data, msg.length) if msg.data else b""

    opts = ffi.new("WGPURequestAdapterOptions *")
    backend = os.environ.get("BENCH_BACKEND", "vulkan")
    opts.backendType = {
        "vulkan": lib.WGPUBackendType_Vulkan,
        "null": lib.WGPUBackendType_Null,
    }[backend]
    cbi = ffi.new("WGPURequestAdapterCallbackInfo *")
    cbi.mode = lib.WGPUCallbackMode_WaitAnyOnly
    cbi.callback = on_adapter
    fut = lib.wgpuInstanceRequestAdapter(instance, opts, cbi[0])
    wi = ffi.new("WGPUFutureWaitInfo[1]")
    wi[0].future = fut
    lib.wgpuInstanceWaitAny(instance, 1, wi, 10**10)
    adapter = result["adapter"]
    assert adapter != ffi.NULL, result

    @ffi.callback(
        "void(WGPURequestDeviceStatus, WGPUDevice, WGPUStringView, void*, void*)"
    )
    def on_device(status, device, msg, u1, u2):
        result["device"] = device

    @ffi.callback(
        "void(WGPUDevice const *, WGPUErrorType, WGPUStringView, void*, void*)"
    )
    def on_error(dev, typ, msg, u1, u2):
        print("ERROR", typ, ffi.string(msg.data, msg.length))

    ddesc = ffi.new("WGPUDeviceDescriptor *")
    ddesc.uncapturedErrorCallbackInfo.callback = on_error
    dcbi = ffi.new("WGPURequestDeviceCallbackInfo *")
    dcbi.mode = lib.WGPUCallbackMode_WaitAnyOnly
    dcbi.callback = on_device
    wi[0].future = lib.wgpuAdapterRequestDevice(adapter, ddesc, dcbi[0])
    lib.wgpuInstanceWaitAny(instance, 1, wi, 10**10)
    device = result["device"]
    queue = lib.wgpuDeviceGetQueue(device)

    bdesc = ffi.new("WGPUBufferDescriptor *")
    bdesc.size = 1024
    bdesc.usage = lib.WGPUBufferUsage_Storage | lib.WGPUBufferUsage_CopySrc
    buffer = lib.wgpuDeviceCreateBuffer(device, bdesc)

    code = b"@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) i: vec3<u32>) { data[i.x] = data[i.x] + 1u; }"
    wgsl = ffi.new("WGPUShaderSourceWGSL *")
    wgsl.chain.sType = lib.WGPUSType_ShaderSourceWGSL
    keep_code = ffi.new("char[]", code)
    wgsl.code.data = keep_code
    wgsl.code.length = len(code)
    smdesc = ffi.new("WGPUShaderModuleDescriptor *")
    smdesc.nextInChain = ffi.cast("WGPUChainedStruct *", wgsl)
    sm = lib.wgpuDeviceCreateShaderModule(device, smdesc)
    cpdesc = ffi.new("WGPUComputePipelineDescriptor *")
    cpdesc.compute.module = sm
    ep, keep_ep = sv("main")
    cpdesc.compute.entryPoint = ep
    pipeline = lib.wgpuDeviceCreateComputePipeline(device, cpdesc)
    bgl = lib.wgpuComputePipelineGetBindGroupLayout(pipeline, 0)
    entry = ffi.new("WGPUBindGroupEntry[1]")
    entry[0].binding = 0
    entry[0].buffer = buffer
    entry[0].offset = 0
    entry[0].size = 1024
    bgdesc = ffi.new("WGPUBindGroupDescriptor *")
    bgdesc.layout = bgl
    bgdesc.entryCount = 1
    bgdesc.entries = entry
    bindgroup = lib.wgpuDeviceCreateBindGroup(device, bgdesc)
    keep = [on_adapter, on_device, on_error, feats, idesc, wgsl, keep_code, keep_ep]
    return dict(
        instance=instance,
        adapter=adapter,
        device=device,
        queue=queue,
        buffer=buffer,
        pipeline=pipeline,
        bindgroup=bindgroup,
        _keep=keep,
    )


def addr(cdata):
    return int(ffi.cast("uintptr_t", cdata))
