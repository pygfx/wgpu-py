"""Micro-benchmark: how to structure compiled wgpu-py classes on top of Cython.

* BufA: a regular Python class compiled by Cython; the handle is a Python int attribute.
* BufB: a cdef (extension) base class that holds the pointer; method defined on it.
* BufC: a Python class (as in wgpu-py) that inherits from the public Python API class
  *and* the extension base class; the compiled method reaches the C pointer directly.
  This is the design used by the Dawn backend.
"""

import time

import common
import _bench_cls
import _bench_cls_ltd


def best(f, n=1_000_000):
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        f(n)
        times.append(time.perf_counter() - t0)
    return min(times) / n * 1e9


def main():
    setup = common.setup()
    address = common.addr(setup["buffer"])
    for mod in (_bench_cls, _bench_cls_ltd):
        for cls in (mod.BufA, mod.BufB, mod.BufC):
            b = cls(address)
            assert b.get_size() == 1024

            def run(n, b=b):
                for _ in range(n):
                    b.get_size()

            print(f"{mod.__name__:16} {cls.__name__}: {best(run):.1f} ns")


if __name__ == "__main__":
    main()
