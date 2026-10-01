# wgpu-dawn: one set of bindings for native Dawn and Pyodide

`wgpu_dawn` is a small compiled package with cffi bindings to `webgpu.h`. It is
used by `wgpu.backends.dawn`. The same binding source and the same backend code
target:

* **native**: [Dawn](https://dawn.googlesource.com/dawn) (`libwebgpu_dawn`), which
  runs on Vulkan, Metal, D3D12, ...;
* **Pyodide**: Emdawnwebgpu, Dawn's implementation of `webgpu.h` on top of the
  browser's WebGPU (`navigator.gpu`). The GPU work is done by the browser, so it is
  hardware accelerated wherever the browser's WebGPU is.

wgpu-py remains installable without it; the backend is used when selected
(`WGPUPY_BACKEND=dawn`), or by default in Pyodide when `wgpu_dawn` is installed.

## Build

Native (Dawn from conda, `mamba install -c mark.harfouche -c conda-forge dawn`,
plus a C++20 compiler, e.g. `cxx-compiler`):

```sh
DAWN_PREFIX=$CONDA_PREFIX pip install --no-build-isolation ./dawn_bindings
WGPUPY_BACKEND=dawn python dawn_bindings/tests/compute_async.py
```

Pyodide (needs `pyodide-build` and the Emscripten of the target Pyodide):

```sh
pip install pyodide-build==0.39.1
pyodide xbuildenv install 314.0.7 && pyodide xbuildenv install-emscripten
source <xbuildenv>/314.0.7/emsdk/emsdk_env.sh
cd dawn_bindings && pyodide build --exports pyinit
```

The build downloads the pinned Emdawnwebgpu package (or set `EMDAWNWEBGPU_PKG`
to an extracted `emdawnwebgpu_pkg`). `tests/` has runners for Pyodide in Node.js
(with the `webgpu` npm package) and in headless Chrome; see
`.github/workflows/dawn-pyodide.yml`. `demo/` is a static page that runs on the
browser's GPU.

## How it works

**Binding layer: cffi in API (out-of-line) mode.** `tools/gen_cdef.py`
generates the cdef from Emdawnwebgpu's `webgpu.h`. At the same Dawn release,
that header is a strict subset of Dawn's native `webgpu.h` (`gen_cdef.py diff`:
all 607 declarations are identical; native Dawn only adds enum members and
Dawn-specific functions/structs), so one cdef serves both targets. In API mode,
the C compiler checks the cdef against the real header of each target, and
constants (e.g. `WGPU_STRLEN`, which is 32-bit on wasm32) come from the target.
A few native-only declarations (window-system surfaces) are added for native
builds only.

Why cffi: the Python side is then the same `ffi`/`lib` interface that the
wgpu-native backend already uses, so `wgpu/backends/dawn/_api.py` is a modest
diff from `wgpu/backends/wgpu_native/_api.py`, and codegen validates it against
Dawn's header in the same way. cffi builds for Pyodide (it is a Pyodide
package), and its generated C uses the limited API. cffi's ABI mode is not an
option in Pyodide (the Emdawnwebgpu glue must be linked and initialized, see
below, and `ffi.callback` needs runtime code generation), but API mode is.
Natively, API mode is also faster than the current ABI mode (see the PR for
numbers); the remaining overhead is mostly in the Python layer of the backend.

**Callbacks** use cffi `extern "Python"` entry points, one per webgpu.h
callback type, which dispatch on `userdata1` to the Python function.

**Per-call overhead.** For cheap calls, most of the cost is in the Python layer
and in cffi's generic call wrapper, not in Dawn:

* `build_cffi.py` post-processes the C code that cffi generates, so that the
  wrappers of pass and bundle encoder functions (`wgpuRenderPassEncoder*`,
  `wgpuComputePassEncoder*`, `wgpuRenderBundleEncoder*`) do not release the GIL
  (and do not save/restore `errno`). These calls are cheap, never block, and
  never call back into Python. All other functions (waits, present, submit,
  object creation, ...) still release the GIL. (The Cython backend makes the
  same trade-off.)
* In `wgpu/backends/dawn/_api.py`, the hot pass encoder methods call `lib`
  directly (no error-checking wrapper, since Dawn defers encoder errors to
  `finish()`), avoid per-call allocations, and don't keep objects alive from
  Python (Dawn holds its own references).
* Other calls go through a thin wrapper that raises a validation error that
  Dawn reported during the call (via the uncaptured-error callback) as a
  Python exception at the call site.

**Adapter enumeration.** webgpu.h can only request the "best" adapter (so e.g.
lavapipe is never returned on a machine with a GPU). Natively, `wgpu_dawn`
includes a small C++ helper that uses Dawn's native API
(`dawn::native::Instance::EnumerateAdapters`), so `enumerate_adapters_sync()`
returns all adapters, for all backends.

**Emdawnwebgpu in a Pyodide extension.** Pyodide extensions are Emscripten
*side modules*, and side modules cannot contain Emscripten JS libraries, while
half of Emdawnwebgpu is one (`library_webgpu.js`, the other half is
`webgpu.cpp`). So:

1. At build time, `tools/gen_emdawn_glue.py` links a throw-away program with
   `--use-port=emdawnwebgpu` (same Emscripten and relevant settings as Pyodide's
   main module) and cuts the expanded JS library code out of the result
   (`emdawn_glue.js`, shipped in the wheel).
2. `webgpu.cpp` is compiled into the extension (`emdawn_glue.cpp`).
3. At import, `wgpu_dawn` passes the glue to an `EM_JS` function. Emscripten's
   dynamic linker evaluates `EM_JS` bodies inside the main module's scope, so the
   glue sees Pyodide's `HEAPU32`, `_malloc`, `wasmImports`, etc. It registers its
   functions in `wasmImports`; the extension's imports of `wgpu*` functions are
   lazy stubs that resolve through `wasmImports` on first call. The C++
   functions that the JS calls back into are passed as function pointers.

## Async model

* Native: callbacks use `WGPUCallbackMode_AllowProcessEvents`. A sync wait
  blocks in `wgpuInstanceWaitAny()` (with the GIL released) on the promise's
  future. `await` polls `wgpuInstanceProcessEvents()` with a backoff. For
  `promise.then()`, a lightweight thread schedules `process_events()` in the
  event loop (via `call_soon_threadsafe`) while promises are pending; Dawn is
  never called from that thread.
* Pyodide: callbacks use `WGPUCallbackMode_AllowSpontaneous`; they are called
  from the JS event loop when the browser's promise resolves, and resolve the
  `GPUPromise`. `await` is plain asyncio. The sync API (`*_sync`, `sync_wait`,
  `queue.read_buffer`, ...) uses JSPI via `pyodide.ffi.run_sync`, which needs a
  JSPI-capable runtime (Chrome 137+, Node 25+) and code entered through e.g.
  `pyodide.runPythonAsync`.

## Version constraints

* Emdawnwebgpu: Dawn `v20260929.200159`, matching the native `dawn` conda
  package. Its remote port needs Emscripten 4.0.10+ (we use the package zip).
* Pyodide 314.0.7 (Python 3.14, Emscripten 5.0.3); pyodide-build 0.39.1. The
  glue must be generated with the Emscripten of the target Pyodide, and must
  be rebuilt when either changes. Pyodide 0.29.x (Emscripten 4.0.9) is older
  than what the Emdawnwebgpu remote port requires.
* `pyodide build --exports pyinit` is needed, since the default export mode
  tries to export the `EM_JS` symbol as a function.

## Status

Verified on both targets with the scripts in `tests/` and the wgpu-py test
files that do not depend on wgpu-native specifics (compute, render, render to
texture, textures and samplers, buffer mapping; see the workflow), plus the
`<canvas>` in the browser. Not supported: wgpu-native specific extras (GLSL,
pipeline statistics, polygon mode, native features/limits, reports); SPIR-V in
the browser. In the browser, validation errors are usually reported
asynchronously (logged), not raised from the failing call. Natively they are
raised at the call site, and error scopes (`push_error_scope()`,
`pop_error_scope_async()`), compilation info and the device-lost promise are
supported.
