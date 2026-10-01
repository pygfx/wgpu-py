"""cffi (API / out-of-line mode) build script for wgpu_dawn.

The same cdef (src/wgpu_dawn/webgpu_cdef.h, generated from webgpu.h) is
compiled for two targets:

* native: linked against Dawn's ``libwebgpu_dawn`` (e.g. from conda-forge /
  the ``dawn`` conda package). Set ``DAWN_PREFIX`` to the install prefix; it
  defaults to ``$CONDA_PREFIX`` / ``sys.prefix``.
* Pyodide (``pyodide build``): compiled with Emdawnwebgpu, Dawn's webgpu.h
  implementation on top of the browser's ``navigator.gpu``. The pinned
  Emdawnwebgpu release is downloaded unless ``EMDAWNWEBGPU_PKG`` points to an
  extracted ``emdawnwebgpu_pkg`` directory.
"""

import hashlib
import io
import os
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile

from cffi import FFI

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src", "wgpu_dawn")
BUILD = os.path.join(HERE, "build", "wgpu_dawn_gen")

# Keep in sync with the Dawn version used for native builds (the dawn conda package).
DAWN_TAG = "v20260929.200159"
EMDAWN_ZIP_SHA256 = "d697687e82b7377fc63f162fff0ef1cc5f025828aa4ff6ec4ac52566c949fe5f"
EMDAWN_URL = (
    f"https://github.com/google/dawn/releases/download/{DAWN_TAG}/"
    f"emdawnwebgpu_pkg-{DAWN_TAG}.zip"
)


def is_emscripten():
    plat = os.environ.get("_PYTHON_HOST_PLATFORM", "") or sysconfig.get_platform()
    return plat.startswith("emscripten") or plat.startswith("pyodide")


def get_emdawnwebgpu_pkg():
    pkg = os.environ.get("EMDAWNWEBGPU_PKG")
    if pkg:
        return pkg
    pkg = os.path.join(HERE, "build", "emdawnwebgpu_pkg")
    if not os.path.isdir(pkg):
        print(f"Downloading {EMDAWN_URL}")
        data = urllib.request.urlopen(EMDAWN_URL).read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != EMDAWN_ZIP_SHA256:
            raise RuntimeError(f"Unexpected sha256 {digest} for {EMDAWN_URL}")
        zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.join(HERE, "build"))
    return pkg


cdef = open(os.path.join(SRC, "webgpu_cdef.h")).read()
cdef += "\nint wgpu_dawn_install(const char* code);\n"
if not is_emscripten():
    cdef += open(os.path.join(SRC, "webgpu_native_extra_cdef.h")).read()

ffibuilder = FFI()
ffibuilder.cdef(cdef)

if is_emscripten():
    pkg = get_emdawnwebgpu_pkg()
    os.makedirs(BUILD, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            os.path.join(HERE, "tools", "gen_emdawn_glue.py"),
            pkg,
            os.path.join(SRC, "emdawn_glue.js"),
            os.path.join(BUILD, "emdawn_cpp_funcs.h"),
            os.environ.get("EMCC", "emcc"),
        ]
    )
    ffibuilder.set_source(
        "wgpu_dawn._wgpu_dawn",
        '#include <webgpu/webgpu.h>\nextern "C" int wgpu_dawn_install(const char* code);\n',
        source_extension=".cpp",
        sources=[os.path.relpath(os.path.join(SRC, "emdawn_glue.cpp"), os.getcwd())],
        include_dirs=[
            os.path.join(pkg, "webgpu", "include"),
            os.path.join(pkg, "webgpu", "src"),
            BUILD,
        ],
        extra_compile_args=["-std=c++20", "-O2", "-DNDEBUG"],
    )
else:
    prefix = (
        os.environ.get("DAWN_PREFIX") or os.environ.get("CONDA_PREFIX") or sys.prefix
    )
    if sys.platform.startswith("win"):
        prefix = os.path.join(prefix, "Library")
    libdir = os.path.join(prefix, "lib")
    kwargs = {}
    if not sys.platform.startswith("win"):
        kwargs["runtime_library_dirs"] = [libdir]
    ffibuilder.set_source(
        "wgpu_dawn._wgpu_dawn",
        "#include <webgpu/webgpu.h>\n"
        "static int wgpu_dawn_install(const char* code) { (void)code; return 0; }\n",
        include_dirs=[os.path.join(prefix, "include")],
        library_dirs=[libdir],
        libraries=["webgpu_dawn"],
        **kwargs,
    )

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
