import random
import ctypes
import base64
from ctypes import c_int32, c_ubyte
import sys

import wgpu.backends.rs  # noqa
from wgpu.utils import compute_with_buffers

from pytest import skip, mark, raises
from testutils import run_tests, can_use_wgpu_lib, is_ci, iters_equal


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


simple_compute_shader = """
    @group(0)
    @binding(0)
    var<storage,read_write> data2: array<i32>;

    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) index: vec3<u32>) {
        let i: u32 = index.x;
        data2[i] = i32(i);
    }
"""

# To generate compute_shader_spirv from a Python function
#
# from pyshader import python2shader, Array, i32, ivec3
#
# def simple_compute_shader_py(
#     index: ("input", "GlobalInvocationId", ivec3),
#     out: ("buffer", 0, Array(i32)),
# ):
#     out[index.x] = index.x
#
# print(base64.encodebytes(python2shader(simple_compute_shader_py).to_spirv()).decode())

simple_compute_shader_spirv = base64.decodebytes(
    """
AwIjBwADAQAAAAAAFgAAAAAAAAARAAIAAQAAAA4AAwAAAAAAAAAAAA8ABgAFAAAAAQAAAG1haW4A
AAAACAAAABAABgABAAAAEQAAAAEAAAABAAAAAQAAAAUABAABAAAAbWFpbgAAAAAFAAQACAAAAGlu
ZGV4AAAABQADAAwAAABvdXQABQADAA0AAAAwAAAARwAEAAgAAAALAAAAHAAAAEcABAAJAAAABgAA
AAQAAABIAAUACgAAAAAAAAAjAAAAAAAAAEcAAwAKAAAAAwAAAEcABAAMAAAAIgAAAAAAAABHAAQA
DAAAACEAAAAAAAAAEwACAAIAAAAhAAMAAwAAAAIAAAAVAAQABQAAACAAAAABAAAAFwAEAAYAAAAF
AAAAAwAAACAABAAHAAAAAQAAAAYAAAA7AAQABwAAAAgAAAABAAAAHQADAAkAAAAFAAAAHgADAAoA
AAAJAAAAIAAEAAsAAAACAAAACgAAADsABAALAAAADAAAAAIAAAArAAQABQAAAA0AAAAAAAAAIAAE
AA4AAAACAAAABQAAACAABAAQAAAAAQAAAAUAAAAgAAQAEwAAAAEAAAAFAAAANgAFAAIAAAABAAAA
AAAAAAMAAAD4AAIABAAAAEEABQAQAAAAEQAAAAgAAAANAAAAPQAEAAUAAAASAAAAEQAAAEEABgAO
AAAADwAAAAwAAAANAAAAEgAAAEEABQATAAAAFAAAAAgAAAANAAAAPQAEAAUAAAAVAAAAFAAAAD4A
AwAPAAAAFQAAAP0AAQA4AAEA
""".encode()
)


def test_compute_0_1_ctype():
    compute_shader = simple_compute_shader
    assert isinstance(compute_shader, str)

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


def test_compute_0_1_tuple():
    compute_shader = simple_compute_shader

    out = compute_with_buffers({}, {0: (100, "i")}, compute_shader)
    assert isinstance(out, dict) and len(out) == 1
    assert isinstance(out[0], memoryview)
    assert out[0].tolist() == list(range(100))


def test_compute_0_1_str():
    compute_shader = simple_compute_shader

    out = compute_with_buffers({}, {0: "100xi"}, compute_shader)
    assert isinstance(out, dict) and len(out) == 1
    assert isinstance(out[0], memoryview)
    assert out[0].tolist() == list(range(100))


def test_compute_0_1_int():
    compute_shader = simple_compute_shader

    out = compute_with_buffers({}, {0: 400}, compute_shader, n=100)
    assert isinstance(out, dict) and len(out) == 1
    assert isinstance(out[0], memoryview)
    assert out[0].cast("i").tolist() == list(range(100))


@mark.skipif(
    is_ci and sys.platform == "win32", reason="Cannot use SpirV shader on dx12"
)
def test_compute_0_1_spirv():
    compute_shader = simple_compute_shader_spirv
    assert isinstance(compute_shader, bytes)

    out = compute_with_buffers({}, {0: c_int32 * 100}, compute_shader)
    assert isinstance(out, dict) and len(out) == 1
    assert isinstance(out[0], ctypes.Array)
    assert iters_equal(out[0], range(100))


def test_compute_1_3():
    compute_shader = """

        @group(0)
        @binding(0)
        var<storage,read> data0: array<i32>;

        @group(0)
        @binding(1)
        var<storage,read_write> data1: array<i32>;

        @group(0)
        @binding(2)
        var<storage,read_write> data2: array<i32>;

        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {
            let i = i32(index.x);
            data1[i] = data0[i];
            data2[i] = i;
        }
    """

    # Create an array of 100 random int32
    in1 = [int(random.uniform(0, 100)) for i in range(100)]
    in1 = (c_int32 * 100)(*in1)

    outspecs = {1: 100 * c_int32, 2: 100 * c_int32}
    out = compute_with_buffers({0: in1}, outspecs, compute_shader)
    assert isinstance(out, dict) and len(out) == 2
    assert isinstance(out[1], ctypes.Array)
    assert isinstance(out[2], ctypes.Array)
    assert iters_equal(out[1], in1)  # because the shader copied the data
    assert iters_equal(out[2], range(100))  # because this is the index


def test_compute_in_is_out():
    compute_shader = """

        @group(0)
        @binding(0)
        var<storage,read_write> data0: array<i32>;

        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {
            let i = i32(index.x);
            data0[i] = data0[i] * 2;
        }
    """

    # Create an array of 100 random int32
    in1 = [int(random.uniform(0, 100)) for i in range(100)]
    expected_out = [i * 2 for i in in1]
    buf = (c_int32 * 100)(*in1)

    out = compute_with_buffers({0: buf}, {0: 100 * c_int32}, compute_shader)
    assert isinstance(out, dict) and len(out) == 1
    assert isinstance(out[0], ctypes.Array)
    assert out[0] is not buf  # a copy was made
    assert iters_equal(out[0], expected_out)


def test_compute_indirect():
    compute_shader = """
        @group(0)
        @binding(0)
        var<storage,read> data1: array<i32>;

        @group(0)
        @binding(1)
        var<storage,read_write> data2: array<i32>;

        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {
            let i = i32(index.x);
            data2[i] = data1[i] + 1;
        }
    """

    # Create an array of 100 random int32
    n = 100
    in1 = [int(random.uniform(0, 100)) for i in range(n)]
    in1 = (c_int32 * n)(*in1)

    # Create device and shader object
    device = wgpu.utils.get_default_device()
    cshader = device.create_shader_module(code=compute_shader)

    # Create input buffer and upload data to in
    buffer1 = device.create_buffer_with_data(data=in1, usage=wgpu.BufferUsage.STORAGE)

    # Create output buffer
    buffer2 = device.create_buffer(
        size=ctypes.sizeof(in1),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    # Create buffer to hold the dispatch parameters for the indirect call
    params = (ctypes.c_int32 * 3)(n - 2, 1, 1)  # note the minus 2!
    buffer3 = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.INDIRECT,
    )

    # Setup layout and bindings
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
    ]
    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
        },
    ]

    # Put everything together
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create and run the pipeline, fail - test check_struct
    with raises(ValueError):
        compute_pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": cshader, "entry_point": "main", "foo": 42},
        )

    # Create and run the pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 args not used
    compute_pass.dispatch_workgroups_indirect(buffer3, 0)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])

    # Read result
    out1 = in1.__class__.from_buffer(device.queue.read_buffer(buffer2))
    in2 = list(in1)[:]
    out2 = [i - 1 for i in out1]
    # The shader was applied to all but the last two elements
    assert in2[:-2] == out2[:-2]
    assert out2[-2:] == [-1, -1]


def test_compute_fails():
    compute_shader = """
        @group(0)
        @binding(0)
        var<storage,read> data1: array<i32>;

        @group(0)
        @binding(1)
        var<storage,read_write> data2: array<i32>;

        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {
            let i = i32(index.x);
            data2[i] = data1[i];
        }
    """

    in1 = [int(random.uniform(0, 100)) for i in range(100)]
    in1 = (c_int32 * 100)(*in1)

    # Baseline; this works
    out = compute_with_buffers(
        {0: in1}, {1: c_int32 * 100}, compute_shader, n=(100, 1, 1)
    )
    assert iters_equal(out[1], in1)

    with raises(TypeError):  # input_arrays is not a dict
        compute_with_buffers([in1], {1: c_int32 * 100}, compute_shader)
    with raises(TypeError):  # input_arrays key not int
        compute_with_buffers({"0": in1}, {1: c_int32 * 100}, compute_shader)
    with raises(TypeError):  # input_arrays value not ctypes array
        compute_with_buffers({0: list(in1)}, {1: c_int32 * 100}, compute_shader)

    with raises(TypeError):  # output_arrays is not a dict
        compute_with_buffers({0: in1}, [c_int32 * 100], compute_shader)
    with raises(TypeError):  # output_arrays key not int
        compute_with_buffers({0: in1}, {"1": c_int32 * 100}, compute_shader)
    with raises(TypeError):  # output_arrays value not a ctypes Array type
        compute_with_buffers({0: in1}, {1: "foobar"}, compute_shader)

    with raises(ValueError):  # output_arrays format invalid
        compute_with_buffers({0: in1}, {1: "10xfoo"}, compute_shader)
    with raises(ValueError):  # output_arrays shape invalid
        compute_with_buffers({0: in1}, {1: ("i",)}, compute_shader)
    with raises(ValueError):  # output_arrays shape invalid
        compute_with_buffers(
            {0: in1},
            {
                1: (
                    0,
                    "i",
                )
            },
            compute_shader,
        )
    with raises(ValueError):  # output_arrays shape invalid
        compute_with_buffers(
            {0: in1},
            {
                1: (
                    -1,
                    "i",
                )
            },
            compute_shader,
        )

    with raises(TypeError):  # invalid n
        compute_with_buffers({0: in1}, {1: c_int32 * 100}, compute_shader, n="100")
    with raises(ValueError):  # invalid n
        compute_with_buffers({0: in1}, {1: c_int32 * 100}, compute_shader, n=-1)

    with raises(TypeError):  # invalid shader
        compute_with_buffers({0: in1}, {1: c_int32 * 100}, {"not", "a", "shader"})


if __name__ == "__main__":
    run_tests(globals())
