"""Micro-benchmark: per-call overhead of different Python bindings to Dawn.

Run ``build.py`` first. Use ``BENCH_BACKEND=null`` to use Dawn's Null backend,
so that only CPU-side costs are measured.
"""

import time
import os
import platform
import common
from common import lib as abi_lib, ffi as abi_ffi, addr

S = common.setup()
N1 = int(os.environ.get("N1", 1_000_000))
N2 = int(os.environ.get("N2", 20_000))
REPEAT = 7


def best(fn, n):
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn(n)
        ts.append(time.perf_counter() - t0)
    return min(ts) / n * 1e9  # ns per iteration


def tick():
    abi_lib.wgpuDeviceTick(S["device"])


results = {}


class PyWrapper:
    """Mimics wgpu-py: a Python class whose method calls a binding function."""

    __slots__ = ["_internal"]
    _fn = None

    def __init__(self, internal):
        self._internal = internal

    def get_size(self):
        return type(self)._fn(self._internal)


def pywrap_bench(fn, handle):
    cls = type("W", (PyWrapper,), {"_fn": staticmethod(fn)})
    w = cls(handle)

    def run(n):
        for _ in range(n):
            w.get_size()

    return best(run, N1)


# ---- cffi variants
def make_cffi(name, ffi, lib):
    def h(key):
        return ffi.cast(
            abi_ffi.typeof(S[key]).cname
            if False
            else "WGPU"
            + {
                "buffer": "Buffer",
                "device": "Device",
                "queue": "Queue",
                "pipeline": "ComputePipeline",
                "bindgroup": "BindGroup",
            }[key],
            addr(S[key]),
        )

    buf, dev, q, pipe, bg = (
        h("buffer"),
        h("device"),
        h("queue"),
        h("pipeline"),
        h("bindgroup"),
    )

    def get_size(n):
        f = lib.wgpuBufferGetSize
        for _ in range(n):
            f(buf)

    def get_size_attr(n):
        for _ in range(n):
            lib.wgpuBufferGetSize(buf)

    def encode(n):
        for i in range(n):
            e = lib.wgpuDeviceCreateCommandEncoder(dev, ffi.NULL)
            p = lib.wgpuCommandEncoderBeginComputePass(e, ffi.NULL)
            lib.wgpuComputePassEncoderSetPipeline(p, pipe)
            lib.wgpuComputePassEncoderSetBindGroup(p, 0, bg, 0, ffi.NULL)
            lib.wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1)
            lib.wgpuComputePassEncoderEnd(p)
            lib.wgpuComputePassEncoderRelease(p)
            cb = lib.wgpuCommandEncoderFinish(e, ffi.NULL)
            cbs = ffi.new("WGPUCommandBuffer[]", [cb])
            lib.wgpuQueueSubmit(q, 1, cbs)
            lib.wgpuCommandBufferRelease(cb)
            lib.wgpuCommandEncoderRelease(e)
            if i % 1000 == 999:
                tick()

    pw = pywrap_bench(lib.wgpuBufferGetSize, buf)
    results[name] = dict(
        get_size=best(get_size, N1),
        get_size_attr=best(get_size_attr, N1),
        method=pw,
        pywrap=pw,
        encode=best(encode, N2),
    )


make_cffi("cffi ABI (dlopen)", abi_ffi, abi_lib)
import _bench_cffi_api  # noqa: E402

make_cffi("cffi API (out-of-line, abi3)", _bench_cffi_api.ffi, _bench_cffi_api.lib)


# ---- compiled-wrapper variants (cython / nanobind / pybind11)
def make_compiled(name, m):
    wrap = m.Handle.from_address
    buf, dev, q, pipe, bg = (
        wrap(addr(S[k])) for k in ("buffer", "device", "queue", "pipeline", "bindgroup")
    )

    def get_size(n):
        f = m.buffer_get_size
        for _ in range(n):
            f(buf)

    def get_size_attr(n):
        for _ in range(n):
            m.buffer_get_size(buf)

    def encode(n):
        for i in range(n):
            e = m.device_create_command_encoder(dev)
            p = m.command_encoder_begin_compute_pass(e)
            m.compute_pass_set_pipeline(p, pipe)
            m.compute_pass_set_bind_group(p, 0, bg)
            m.compute_pass_dispatch(p, 1, 1, 1)
            m.compute_pass_end(p)
            m.compute_pass_release(p)
            cb = m.command_encoder_finish(e)
            m.queue_submit(q, [cb])
            m.command_buffer_release(cb)
            m.command_encoder_release(e)
            if i % 1000 == 999:
                tick()

    def meth(n):
        for _ in range(n):
            buf.get_size()

    r = dict(
        get_size=best(get_size, N1),
        get_size_attr=best(get_size_attr, N1),
        method=best(meth, N1),
        pywrap=pywrap_bench(m.buffer_get_size, buf),
        encode=best(encode, N2),
    )
    if hasattr(m, "c_encode_loop"):

        def c_enc(n):
            for _ in range(n // 1000):
                m.c_encode_loop(dev, q, pipe, bg, 1000)
                tick()

        def c_size(n):
            m.c_get_size_loop(buf, n)

        results["pure C (floor)"] = dict(
            get_size=best(c_size, N1),
            get_size_attr=float("nan"),
            method=float("nan"),
            pywrap=float("nan"),
            encode=best(c_enc, N2),
        )
    results[name] = r


import _bench_cython  # noqa: E402
import _bench_cython_ltd  # noqa: E402
import _bench_nanobind  # noqa: E402
import _bench_pybind11  # noqa: E402

make_compiled("Cython 3 (full CPython API)", _bench_cython)
make_compiled("Cython 3 (Limited API, abi3)", _bench_cython_ltd)
make_compiled("nanobind (stable ABI)", _bench_nanobind)
make_compiled("pybind11", _bench_pybind11)

print(
    f"Python {platform.python_version()}, backend={os.environ.get('BENCH_BACKEND', 'vulkan')}, N1={N1}, N2={N2}, best of {REPEAT}"
)
print(
    "| binding | GetSize, bound func (ns) | GetSize, mod.attr (ns) | buffer.get_size() method (ns) | Python wrapper class (ns) | encode+submit, 11 calls (us) |"
)
print("|---|---:|---:|---:|---:|---:|")
for k, v in results.items():
    print(
        f"| {k} | {v['get_size']:.1f} | {v['get_size_attr']:.1f} | {v['method']:.1f} | {v['pywrap']:.1f} | {v['encode'] / 1000:.2f} |"
    )
