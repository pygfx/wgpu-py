import wgpu.utils
import gc

from testutils import run_tests, can_use_wgpu_lib, is_pypy
from pytest import mark


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_query_set():
    if is_pypy:
        gc.collect()  # avoid a panic here when using pypy
    shader_source = """
    @group(0) @binding(0)
    var<storage,read> data1: array<f32>;

    @group(0) @binding(1)
    var<storage,read_write> data2: array<f32>;

    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) index: vec3<u32>) {
        let i: u32 = index.x;
        data2[i] = data1[i] / 2.0;
    }
    """

    n = 1024
    data1 = memoryview(bytearray(n * 4)).cast("f")

    for i in range(n):
        data1[i] = float(i)

    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    device = adapter.request_device(
        required_features=[wgpu.FeatureName.timestamp_query]
    )

    assert repr(device).startswith("<wgpu.backends.wgpu_native.GPUDevice ")

    cshader = device.create_shader_module(code=shader_source)

    # Create buffer objects, input buffer is mapped.
    buffer1 = device.create_buffer_with_data(data=data1, usage=wgpu.BufferUsage.STORAGE)
    buffer2 = device.create_buffer(
        size=data1.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
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
        {"binding": 0, "resource": {"buffer": buffer1}},
        {"binding": 1, "resource": {"buffer": buffer2}},
    ]

    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )

    query_count = 4
    query_type = wgpu.QueryType.timestamp
    query_label = "div_by_2"
    query_set = device.create_query_set(
        label=query_label, type=query_type, count=query_count
    )
    assert query_set.count == query_count
    assert query_set.type == query_type
    assert query_set._device == device._internal
    assert query_set.label == query_label

    query_buf_size = 8 * query_set.count
    query_usage = wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC
    query_buf = device.create_buffer(
        size=query_buf_size,
        usage=query_usage,
    )

    assert query_buf.size == query_buf_size
    assert query_buf.usage == query_usage

    command_encoder = device.create_command_encoder()

    compute_pass = command_encoder.begin_compute_pass(
        timestamp_writes={
            "query_set": query_set,
            "beginning_of_pass_write_index": 0,
            "end_of_pass_write_index": 1,
        }
    )
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n, 1, 1)  # x y z
    compute_pass.end()

    # Make sure the code works when you only write at the beginning
    compute_pass = command_encoder.begin_compute_pass(
        timestamp_writes={
            "query_set": query_set,
            "beginning_of_pass_write_index": 2,
        }
    )
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n, 1, 1)  # x y z
    compute_pass.end()

    # Make sure the code works when you only write at the end.
    compute_pass = command_encoder.begin_compute_pass(
        timestamp_writes={
            "query_set": query_set,
            "end_of_pass_write_index": 3,
        }
    )
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n, 1, 1)  # x y z
    compute_pass.end()

    command_encoder.resolve_query_set(
        query_set=query_set,
        first_query=0,
        query_count=4,
        destination=query_buf,
        destination_offset=0,
    )

    device.queue.submit([command_encoder.finish()])
    timestamps = device.queue.read_buffer(query_buf).cast("Q").tolist()
    assert len(timestamps) == 4
    assert 0 < timestamps[0] < timestamps[1] < timestamps[2] < timestamps[3]

    out = device.queue.read_buffer(buffer2).cast("f")
    result = out.tolist()

    # Perform the same division on the CPU.
    # We kept on dividing the same numbers repeatedly, so we only divided once.
    result_cpu = [a / 2.0 for a in data1]

    # Ensure results are the same
    assert result == result_cpu


if __name__ == "__main__":
    run_tests(globals())
