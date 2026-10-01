// Install Emdawnwebgpu's JS glue into the Pyodide runtime (see emdawn_glue.cpp).
// Natively, there is nothing to install.

#ifndef WGPU_PY_DAWN_EMDAWN_GLUE_H_
#define WGPU_PY_DAWN_EMDAWN_GLUE_H_

#ifdef __EMSCRIPTEN__
#ifdef __cplusplus
extern "C"
#endif
int wgpu_dawn_install(const char* code);
#else
static inline int wgpu_dawn_install(const char* code) {
    (void)code;
    return 0;
}
#endif

#endif  // WGPU_PY_DAWN_EMDAWN_GLUE_H_
