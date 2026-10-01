# cython: language_level=3, boundscheck=False, wraparound=False
from libc.stdint cimport uint32_t, uint64_t, uintptr_t
from libc.stdlib cimport malloc, free

cdef extern from "webgpu/webgpu.h":
    ctypedef void* WGPUBuffer
    ctypedef void* WGPUDevice
    ctypedef void* WGPUQueue
    ctypedef void* WGPUCommandEncoder
    ctypedef void* WGPUComputePassEncoder
    ctypedef void* WGPUComputePipeline
    ctypedef void* WGPUBindGroup
    ctypedef void* WGPUCommandBuffer
    ctypedef struct WGPUCommandEncoderDescriptor
    ctypedef struct WGPUComputePassDescriptor
    ctypedef struct WGPUCommandBufferDescriptor
    uint64_t wgpuBufferGetSize(WGPUBuffer)
    WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice, const WGPUCommandEncoderDescriptor*)
    WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder, const WGPUComputePassDescriptor*)
    void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder, WGPUComputePipeline)
    void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder, uint32_t, WGPUBindGroup, size_t, const uint32_t*)
    void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder, uint32_t, uint32_t, uint32_t)
    void wgpuComputePassEncoderEnd(WGPUComputePassEncoder)
    void wgpuComputePassEncoderRelease(WGPUComputePassEncoder)
    WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder, const WGPUCommandBufferDescriptor*)
    void wgpuCommandEncoderRelease(WGPUCommandEncoder)
    void wgpuCommandBufferRelease(WGPUCommandBuffer)
    void wgpuQueueSubmit(WGPUQueue, size_t, const WGPUCommandBuffer*)

cdef class Handle:
    cdef void* ptr
    def get_size(self):
        return wgpuBufferGetSize(self.ptr)
    def dispatch_workgroups(self, uint32_t x, uint32_t y=1, uint32_t z=1):
        wgpuComputePassEncoderDispatchWorkgroups(self.ptr, x, y, z)
    @staticmethod
    def from_address(uintptr_t a):
        cdef Handle h = Handle.__new__(Handle)
        h.ptr = <void*>a
        return h

cdef inline Handle _wrap(void* p):
    cdef Handle h = Handle.__new__(Handle)
    h.ptr = p
    return h

def buffer_get_size(Handle b):
    return wgpuBufferGetSize(b.ptr)

def device_create_command_encoder(Handle d):
    return _wrap(wgpuDeviceCreateCommandEncoder(d.ptr, NULL))

def command_encoder_begin_compute_pass(Handle e):
    return _wrap(wgpuCommandEncoderBeginComputePass(e.ptr, NULL))

def compute_pass_set_pipeline(Handle p, Handle pipe):
    wgpuComputePassEncoderSetPipeline(p.ptr, pipe.ptr)

def compute_pass_set_bind_group(Handle p, uint32_t index, Handle bg):
    wgpuComputePassEncoderSetBindGroup(p.ptr, index, bg.ptr, 0, NULL)

def compute_pass_dispatch(Handle p, uint32_t x, uint32_t y, uint32_t z):
    wgpuComputePassEncoderDispatchWorkgroups(p.ptr, x, y, z)

def compute_pass_end(Handle p):
    wgpuComputePassEncoderEnd(p.ptr)

def compute_pass_release(Handle p):
    wgpuComputePassEncoderRelease(p.ptr)

def command_encoder_finish(Handle e):
    return _wrap(wgpuCommandEncoderFinish(e.ptr, NULL))

def command_encoder_release(Handle e):
    wgpuCommandEncoderRelease(e.ptr)

def command_buffer_release(Handle c):
    wgpuCommandBufferRelease(c.ptr)

def queue_submit(Handle q, list cbs):
    cdef size_t n = len(cbs)
    cdef WGPUCommandBuffer arr[16]
    cdef size_t i
    for i in range(n):
        arr[i] = (<Handle>cbs[i]).ptr
    wgpuQueueSubmit(q.ptr, n, arr)

def c_encode_loop(Handle device, Handle queue, Handle pipe, Handle bg, int n):
    """Same encode/submit sequence entirely in C (the floor)."""
    cdef int i
    cdef void* e
    cdef void* p
    cdef void* cb
    for i in range(n):
        e = wgpuDeviceCreateCommandEncoder(device.ptr, NULL)
        p = wgpuCommandEncoderBeginComputePass(e, NULL)
        wgpuComputePassEncoderSetPipeline(p, pipe.ptr)
        wgpuComputePassEncoderSetBindGroup(p, 0, bg.ptr, 0, NULL)
        wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1)
        wgpuComputePassEncoderEnd(p)
        wgpuComputePassEncoderRelease(p)
        cb = wgpuCommandEncoderFinish(e, NULL)
        wgpuQueueSubmit(queue.ptr, 1, <WGPUCommandBuffer*>&cb)
        wgpuCommandBufferRelease(cb)
        wgpuCommandEncoderRelease(e)

def c_get_size_loop(Handle b, int n):
    cdef int i
    cdef uint64_t s = 0
    for i in range(n):
        s += wgpuBufferGetSize(b.ptr)
    return s
