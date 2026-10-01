# cython: language_level=3
from libc.stdint cimport uint64_t, uintptr_t
cdef extern from "webgpu/webgpu.h":
    ctypedef void* WGPUBuffer
    uint64_t wgpuBufferGetSize(WGPUBuffer)

class PyBase:  # stands in for wgpu._classes.GPUBuffer (pure Python)
    def get_size(self):
        raise NotImplementedError()

# (a) regular Python class compiled by Cython, handle stored as Python int
class BufA(PyBase):
    def __init__(self, addr):
        self._internal = addr
    def get_size(self):
        return wgpuBufferGetSize(<WGPUBuffer><uintptr_t>self._internal)

# (b) extension base type holding the C pointer, combined with the Python API class
cdef class _HandleBase:
    cdef void* _ptr
    def get_size(self):
        return wgpuBufferGetSize(<WGPUBuffer>self._ptr)

class BufB(_HandleBase, PyBase):
    def __init__(self, addr):
        (<_HandleBase>self)._ptr = <void*><uintptr_t>addr

# (c) Python class compiled by Cython, multiple inheritance with the extension base,
#     method body reaches the C pointer via an unchecked cast.
class BufC(PyBase, _HandleBase):
    def __init__(self, addr):
        (<_HandleBase>self)._ptr = <void*><uintptr_t>addr
    def get_size(self):
        return wgpuBufferGetSize(<WGPUBuffer>(<_HandleBase>self)._ptr)
