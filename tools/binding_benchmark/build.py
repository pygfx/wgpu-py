"""
Build the binding variants for the Python -> Dawn call-overhead micro-benchmark.

This benchmark compares binding strategies for the *same* tiny slice of Dawn's
webgpu.h (wgpuBufferGetSize, and an encode+submit sequence of 11 calls):

* cffi ABI mode (dlopen; what the wgpu-native backend uses)
* cffi API / out-of-line mode (compiled, abi3)
* Cython 3 (full C-API, and Limited API / abi3)
* nanobind (stable ABI)
* pybind11

Requirements (conda): dawn cffi cython nanobind pybind11 cmake ninja c-compiler cxx-compiler

Usage::

    python tools/binding_benchmark/build.py
    BENCH_BACKEND=null python tools/binding_benchmark/bench.py
    BENCH_BACKEND=null python tools/binding_benchmark/bench_cls.py
"""

import os
import sys
import sysconfig
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PREFIX = os.environ.get("CONDA_PREFIX", sys.prefix)


def run(*cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=HERE)


def write_clean_header():
    sys.path.insert(0, ROOT)
    from codegen.hparser import _get_wgpu_header

    source = _get_wgpu_header(
        os.path.join(PREFIX, "include", "dawn", "webgpu.h"), output_filename=None
    )
    source = "\n".join(
        line
        for line in source.splitlines()
        if "emscripten_webgpu_get_device" not in line
    )
    with open(os.path.join(HERE, "dawn_clean.h"), "w") as f:
        f.write(source)


def build_cffi_api():
    from cffi import FFI

    ffi = FFI()
    with open(os.path.join(HERE, "dawn_clean.h")) as f:
        ffi.cdef(f.read())
    ffi.set_source(
        "_bench_cffi_api",
        "#include <webgpu/webgpu.h>",
        include_dirs=[PREFIX + "/include"],
        library_dirs=[PREFIX + "/lib"],
        runtime_library_dirs=[PREFIX + "/lib"],
        libraries=["webgpu_dawn"],
        extra_compile_args=["-O2", "-w"],
        py_limited_api=True,
    )
    ffi.compile(tmpdir=HERE, verbose=False)


def build_cython(name, limited):
    cc = os.environ.get("CC", "cc")
    include = sysconfig.get_paths()["include"]
    suffix = ".abi3.so" if limited else sysconfig.get_config_var("EXT_SUFFIX")
    flags = ["-DPy_LIMITED_API=0x030C0000", "-DCYTHON_LIMITED_API=1"] if limited else []
    run(sys.executable, "-m", "cython", "-3", f"{name}.pyx", "-o", f"{name}.c")
    run(
        cc, "-O2", "-shared", "-fPIC", *flags, f"-I{include}", f"-I{PREFIX}/include",
        f"{name}.c", "-o", f"{name}{suffix}", f"-L{PREFIX}/lib",
        f"-Wl,-rpath,{PREFIX}/lib", "-lwebgpu_dawn",
    )  # fmt: skip


def build_cython_variants():
    for base in ("_bench_cython", "_bench_cls"):
        with open(os.path.join(HERE, base + ".pyx")) as f:
            src = f.read()
        with open(os.path.join(HERE, base + "_ltd.pyx"), "w") as f:
            f.write(src.replace(base, base + "_ltd"))
        build_cython(base, False)
        build_cython(base + "_ltd", True)


def build_cmake():
    import pybind11

    run(
        "cmake", "-S", ".", "-B", "build", "-G", "Ninja",
        f"-DCMAKE_PREFIX_PATH={PREFIX}", f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
    )  # fmt: skip
    run("cmake", "--build", "build")
    for fname in os.listdir(os.path.join(HERE, "build")):
        if fname.endswith(".so"):
            os.replace(os.path.join(HERE, "build", fname), os.path.join(HERE, fname))


if __name__ == "__main__":
    write_clean_header()
    build_cffi_api()
    build_cython_variants()
    build_cmake()
