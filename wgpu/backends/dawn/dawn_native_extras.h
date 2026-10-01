// Native-only helpers for the Dawn backend that are not in webgpu.h.
//
// These are implemented in dawn_native_extras.cpp, using Dawn's C++ API
// (dawn/native/DawnNative.h). That file is only compiled natively, and only if
// Dawn's C++ headers are available (see tools/build_dawn.py); otherwise these
// report that they are not available.

#ifndef WGPU_PY_DAWN_NATIVE_EXTRAS_H_
#define WGPU_PY_DAWN_NATIVE_EXTRAS_H_

#include <stddef.h>
#include <webgpu/webgpu.h>

#ifdef WGPU_PY_HAVE_DAWN_NATIVE_EXTRAS

#ifdef __cplusplus
extern "C" {
#endif

// Enumerate all adapters of the instance that match the options (which may be
// NULL). Returns the number of adapters. If ``adapters`` is not NULL, it is
// filled with (at most ``max_count``) new references to the adapters.
size_t wgpupy_dawn_enumerate_adapters(WGPUInstance instance,
                                      const WGPURequestAdapterOptions* options,
                                      WGPUAdapter* adapters,
                                      size_t max_count);

#ifdef __cplusplus
}
#endif

#else

static inline size_t wgpupy_dawn_enumerate_adapters(WGPUInstance instance,
                                                    const WGPURequestAdapterOptions* options,
                                                    WGPUAdapter* adapters,
                                                    size_t max_count) {
    (void)instance;
    (void)options;
    (void)adapters;
    (void)max_count;
    return (size_t)-1;  // not available
}

#endif  // WGPU_PY_HAVE_DAWN_NATIVE_EXTRAS

#endif  // WGPU_PY_DAWN_NATIVE_EXTRAS_H_
