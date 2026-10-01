// Emdawnwebgpu inside a Pyodide extension (an Emscripten *side module*).
//
// Only compiled when building the Dawn backend for Pyodide (tools/build_dawn.py).
//
// Emdawnwebgpu has a C++ half (webgpu.cpp, compiled right here) and a JS half
// (library_webgpu*.js). Side modules cannot carry Emscripten JS libraries, so
// tools/gen_emdawn_glue.py pre-processes the JS half into emdawn_glue.js. At
// import time, _api.pyx passes that code to wgpu_dawn_install(), which
// evaluates it inside Pyodide's module scope (EM_JS bodies are eval'ed there by
// Emscripten's dynamic linker) and registers its functions in wasmImports.
// This side module's imports of wgpu*/emwgpu* functions are lazy stubs that
// resolve through wasmImports on first call, so they then bind to the glue.

#include "webgpu.cpp"  // from emdawnwebgpu_pkg/webgpu/src

#include <emscripten/em_js.h>

#include "emdawn_cpp_funcs.h"  // generated: the C++ functions the JS calls into
#include "emdawn_glue.h"

// clang-format off
EM_JS(int, wgpu_dawn_install_js, (const char* code, const char* names, void* const* fptrs), {
  var factory = eval(UTF8ToString(code));
  var nameList = UTF8ToString(names).split(',').filter((s) => s);
  var cpp = {};
  for (var i = 0; i < nameList.length; i++) {
    cpp[nameList[i]] = getWasmTableEntry(HEAPU32[(fptrs >>> 2) + i]);
  }
  var lib = factory(cpp);
  var n = 0;
  for (var key in lib) {
    var existing = wasmImports[key];
    if (!existing || existing.stub) {
      wasmImports[key] = lib[key];
      n++;
    }
  }
  return n;
});
// clang-format on

extern "C" int wgpu_dawn_install(const char* code) {
    static void* const fptrs[] = {
#define X(n) reinterpret_cast<void*>(&n),
        EMDAWN_CPP_FUNCS(X)
#undef X
    };
    static const char names[] =
#define X(n) #n ","
        EMDAWN_CPP_FUNCS(X)
#undef X
        ;
    return wgpu_dawn_install_js(code, names, fptrs);
}
