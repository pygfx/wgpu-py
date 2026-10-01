"""Compiled bindings to webgpu.h, as implemented by Dawn (native) or Emdawnwebgpu (Pyodide).

Exposes ``ffi`` and ``lib`` (cffi API mode). The binding layer is identical on
both targets: the cdef is generated from Emdawnwebgpu's webgpu.h, which is a
strict subset of Dawn's native webgpu.h at the same Dawn release.
"""

import sys

from ._wgpu_dawn import ffi, lib

__all__ = ["dawn_version", "ffi", "is_emscripten", "lib"]

dawn_version = "v20260929.200159"
is_emscripten = sys.platform == "emscripten"

if is_emscripten:
    # Inject Emdawnwebgpu's JS glue into the Pyodide runtime (see emdawn_glue.cpp).
    from importlib.resources import files as _files

    _code = _files(__name__).joinpath("emdawn_glue.js").read_bytes()
    _n = lib.wgpu_dawn_install(_code)
    del _code, _files
