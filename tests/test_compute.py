import random
import ctypes
from ctypes import c_int32, c_ubyte

from python_shader import python2shader, Array, i32
import wgpu.backend.rs  # noqa
from wgpu.utils import compute_with_buffers

from pytest import mark
from testutils import can_use_wgpu_lib, iters_equal


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_0_1():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), out: ("output", 0, Array(i32)),
    ):
        out[index] = index

    # Create some ints!
    out = compute_with_buffers({}, {0: c_int32 * 100}, compute_shader)
    assert isinstance(out, dict) and len(out) == 1
    assert isinstance(out[0], ctypes.Array)
    assert iters_equal(out[0], range(100))

    # Same, but specify in bytes
    out = compute_with_buffers({}, {0: c_ubyte * 80}, compute_shader, n=20)
    assert isinstance(out, dict) and len(out) == 1
    assert isinstance(out[0], ctypes.Array)
    out0 = (c_int32 * 20).from_buffer(out[0])  # cast (a view in np)
    assert iters_equal(out0, range(20))


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_1_3():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        in1: ("input", 0, Array(i32)),
        out1: ("output", 1, Array(i32)),
        out2: ("output", 2, Array(i32)),
    ):
        out1[index] = in1[index]
        out2[index] = index

    # Create an array of 100 random int32
    in1 = [int(random.uniform(0, 100)) for i in range(100)]
    in1 = (c_int32 * 100)(*in1)

    outspecs = {0: 100 * c_int32, 1: 100 * c_int32, 2: 100 * c_int32}
    out = compute_with_buffers({0: in1}, outspecs, compute_shader)
    assert isinstance(out, dict) and len(out) == 3
    assert isinstance(out[0], ctypes.Array)
    assert isinstance(out[1], ctypes.Array)
    assert isinstance(out[2], ctypes.Array)
    assert iters_equal(out[0], in1)  # because it's the same buffer
    assert iters_equal(out[1], in1)  # because the shader copied the data
    assert iters_equal(out[2], range(100))  # because this is the index


if __name__ == "__main__":
    test_compute_0_1()
    test_compute_1_3()
