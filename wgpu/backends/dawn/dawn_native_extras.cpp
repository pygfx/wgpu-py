// Native-only helpers for the Dawn backend, see dawn_native_extras.h.

#include <dawn/native/DawnNative.h>

#include <vector>

#include "dawn_native_extras.h"

extern "C" size_t wgpupy_dawn_enumerate_adapters(WGPUInstance instance,
                                                 const WGPURequestAdapterOptions* options,
                                                 WGPUAdapter* adapters,
                                                 size_t max_count) {
    // webgpu.h can only request "the best" adapter for some options, and Dawn's
    // forceFallbackAdapter only matches SwiftShader. Dawn's C++ API can list
    // them all, e.g. a GPU as well as lavapipe. A WGPUInstance is a Dawn
    // InstanceBase (the wrapper takes its own reference).
    dawn::native::Instance wrapper(reinterpret_cast<dawn::native::InstanceBase*>(instance));
    std::vector<dawn::native::Adapter> found = wrapper.EnumerateAdapters(options);
    if (adapters != nullptr) {
        for (size_t i = 0; i < found.size() && i < max_count; i++) {
            adapters[i] = found[i].Get();
            wgpuAdapterAddRef(adapters[i]);
        }
    }
    return found.size();
}
