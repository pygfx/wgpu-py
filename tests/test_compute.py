import numpy as np
from python_shader import python2shader
import wgpu.backend.rs  # noqa
from wgpu.utils import compute_with_buffers


def test_compute_0_1():
    @python2shader
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("out", 0, Array(i32))
        buffer.out[input.index] = input.index

    out = compute_with_buffers({}, {0: (100, np.int32)}, compute_shader)
    assert isinstance(out, dict) and len(out) == 1
    assert isinstance(out[0], np.ndarray)
    assert np.all(out[0] == np.arange(100))


def test_compute_1_3():
    @python2shader
    def compute_shader(input, buffer):
        input.define("index", "GlobalInvocationId", i32)
        buffer.define("in1", 0, Array(i32))
        buffer.define("out1", 1, Array(i32))
        buffer.define("out2", 2, Array(i32))

        buffer.out1[input.index] = buffer.in1[input.index]
        buffer.out2[input.index] = input.index

    in1 = np.random.uniform(0, 100, (100,)).astype(np.int32)

    outspecs = {0: (100, np.int32), 1: (100, np.int32), 2: (100, np.int32)}
    out = compute_with_buffers({0: in1}, outspecs, compute_shader)
    assert isinstance(out, dict) and len(out) == 3
    assert isinstance(out[0], np.ndarray)
    assert isinstance(out[1], np.ndarray)
    assert isinstance(out[2], np.ndarray)
    assert np.all(out[0] == in1)  # because it's the same buffer
    assert np.all(out[1] == in1)  # because the shader copied the data
    assert np.all(out[2] == np.arange(100))  # because this is the index


if __name__ == "__main__":
    test_compute_0_1()
    test_compute_1_3()
