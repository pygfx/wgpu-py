// shared between nanobind and pybind11 variants; expects NS and MOD macros
#include <webgpu/webgpu.h>
#include <cstdint>
#include <vector>
struct Handle { void* ptr; };
static Handle wrap(void* p) { return Handle{p}; }
#define H(x) ((decltype(x.ptr))x.ptr)
static void define_all(MODTYPE& m) {
  CLASS<Handle>(m, "Handle")
    .def_static("from_address", [](uintptr_t a) { return Handle{(void*)a}; })
    .def("get_size", [](const Handle& b) { return wgpuBufferGetSize((WGPUBuffer)b.ptr); })
    .def("dispatch_workgroups", [](const Handle& p, uint32_t x, uint32_t y, uint32_t z) { wgpuComputePassEncoderDispatchWorkgroups((WGPUComputePassEncoder)p.ptr, x, y, z); }, MODTYPE_ARG("x"), MODTYPE_ARG("y")=1, MODTYPE_ARG("z")=1);
  m.def("buffer_get_size", [](const Handle& b) { return wgpuBufferGetSize((WGPUBuffer)b.ptr); });
  m.def("device_create_command_encoder", [](const Handle& d) { return wrap(wgpuDeviceCreateCommandEncoder((WGPUDevice)d.ptr, nullptr)); });
  m.def("command_encoder_begin_compute_pass", [](const Handle& e) { return wrap(wgpuCommandEncoderBeginComputePass((WGPUCommandEncoder)e.ptr, nullptr)); });
  m.def("compute_pass_set_pipeline", [](const Handle& p, const Handle& pipe) { wgpuComputePassEncoderSetPipeline((WGPUComputePassEncoder)p.ptr, (WGPUComputePipeline)pipe.ptr); });
  m.def("compute_pass_set_bind_group", [](const Handle& p, uint32_t i, const Handle& bg) { wgpuComputePassEncoderSetBindGroup((WGPUComputePassEncoder)p.ptr, i, (WGPUBindGroup)bg.ptr, 0, nullptr); });
  m.def("compute_pass_dispatch", [](const Handle& p, uint32_t x, uint32_t y, uint32_t z) { wgpuComputePassEncoderDispatchWorkgroups((WGPUComputePassEncoder)p.ptr, x, y, z); });
  m.def("compute_pass_end", [](const Handle& p) { wgpuComputePassEncoderEnd((WGPUComputePassEncoder)p.ptr); });
  m.def("compute_pass_release", [](const Handle& p) { wgpuComputePassEncoderRelease((WGPUComputePassEncoder)p.ptr); });
  m.def("command_encoder_finish", [](const Handle& e) { return wrap(wgpuCommandEncoderFinish((WGPUCommandEncoder)e.ptr, nullptr)); });
  m.def("command_encoder_release", [](const Handle& e) { wgpuCommandEncoderRelease((WGPUCommandEncoder)e.ptr); });
  m.def("command_buffer_release", [](const Handle& c) { wgpuCommandBufferRelease((WGPUCommandBuffer)c.ptr); });
  m.def("queue_submit", [](const Handle& q, const std::vector<Handle*>& cbs) {
    WGPUCommandBuffer arr[16]; size_t n = cbs.size();
    for (size_t i = 0; i < n; i++) arr[i] = (WGPUCommandBuffer)cbs[i]->ptr;
    wgpuQueueSubmit((WGPUQueue)q.ptr, n, arr); });
}
