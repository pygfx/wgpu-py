import random
import ctypes
from ctypes import c_int32, c_ubyte

from python_shader import python2shader, Array, i32
import wgpu.backends.rs  # noqa
from wgpu.utils import compute_with_buffers

from pytest import skip, raises
from testutils import can_use_wgpu_lib, iters_equal


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def test_compute_0_1():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32), out: ("buffer", 0, Array(i32)),
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


def test_compute_1_3():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        in1: ("buffer", 0, Array(i32)),
        out1: ("buffer", 1, Array(i32)),
        out2: ("buffer", 2, Array(i32)),
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


def test_compute_fails():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", i32),
        in1: ("buffer", 0, Array(i32)),
        out1: ("buffer", 1, Array(i32)),
    ):
        out1[index] = in1[index]

    in1 = [int(random.uniform(0, 100)) for i in range(100)]
    in1 = (c_int32 * 100)(*in1)

    # Baseline; this works
    out = compute_with_buffers(
        {0: in1}, {0: c_int32 * 100}, compute_shader, n=(100, 1, 1)
    )
    assert iters_equal(out[0], in1)

    with raises(TypeError):  # input_arrays is not a dict
        compute_with_buffers([in1], {0: c_int32 * 100}, compute_shader)
    with raises(TypeError):  # input_arrays key not int
        compute_with_buffers({"0": in1}, {0: c_int32 * 100}, compute_shader)
    with raises(TypeError):  # input_arrays value not ctypes array
        compute_with_buffers({0: list(in1)}, {0: c_int32 * 100}, compute_shader)

    with raises(TypeError):  # output_arrays is not a dict
        compute_with_buffers({0: in1}, [c_int32 * 100], compute_shader)
    with raises(TypeError):  # output_arrays key not int
        compute_with_buffers({0: in1}, {"0": c_int32 * 100}, compute_shader)
    with raises(TypeError):  # output_arrays value not a ctypes Array type
        compute_with_buffers({0: in1}, {0: "foobar"}, compute_shader)

    with raises(TypeError):  # invalid n
        compute_with_buffers({0: in1}, {0: c_int32 * 100}, compute_shader, n="100")
    with raises(ValueError):  # invalid n
        compute_with_buffers({0: in1}, {0: c_int32 * 100}, compute_shader, n=-1)

    with raises(TypeError):  # invalid shader
        compute_with_buffers({0: in1}, {0: c_int32 * 100}, "not a shader")


if __name__ == "__main__":
    test_compute_0_1()
    test_compute_1_3()
    test_compute_fails()
