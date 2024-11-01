import asyncio

import pytest

import wgpu.utils
from tests.testutils import run_tests
from wgpu import MapMode, TextureFormat
from wgpu.backends.wgpu_native import WgpuAwaitable


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_awaitable_async(use_async, loop_scope="function"):
    count = 0

    def finalizer(i):
        return i * i

    def callback(i):
        awaitable.set_result(i)

    def poll_function():
        nonlocal count
        count += 1
        if count >= 3:
            callback(10)

    awaitable = WgpuAwaitable("test", callback, finalizer, poll_function)

    if use_async:
        result = await awaitable.wait_async()
    else:
        result = awaitable.wait_sync()
    assert result == 10 * 10


@pytest.mark.asyncio
async def test_asynchronous_get_device(loop_scope="function"):
    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
    device = await adapter.request_device_async()
    assert device is not None


@pytest.mark.asyncio
async def test_asynchronous_buffer_map(loop_scope="function"):
    device = wgpu.utils.get_default_device()

    data = b"1" * 10000
    buffer1 = device.create_buffer(size=len(data), usage="MAP_WRITE|COPY_SRC")
    buffer2 = device.create_buffer(size=len(data), usage="MAP_READ|COPY_DST")
    await buffer1.map_async(MapMode.WRITE)
    buffer1.write_mapped(data)
    buffer1.unmap()

    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(buffer1, 0, buffer2, 0, len(data))
    device.queue.submit([command_encoder.finish()])

    await buffer2.map_async(MapMode.READ)
    data2 = buffer2.read_mapped()
    buffer2.unmap()

    assert bytes(data2) == data


@pytest.mark.asyncio
async def test_asynchronous_make_pipeline(loop_scope="function"):
    device = wgpu.utils.get_default_device()

    shader_source = """
        @vertex
        fn vertex_main() -> @builtin(position) vec4f {
            return vec4f(0, 0, 0, 1.);
        }

        @compute @workgroup_size(1)
        fn compute_main() { }
    """

    shader = device.create_shader_module(code=shader_source)

    render_pipeline, compute_pipeline = await asyncio.gather(
        device.create_render_pipeline_async(
            layout="auto",
            vertex={
                "module": shader,
            },
            depth_stencil={"format": TextureFormat.rgba8unorm},
        ),
        device.create_compute_pipeline_async(layout="auto", compute={"module": shader}),
    )

    assert compute_pipeline is not None
    assert render_pipeline is not None

    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.dispatch_workgroups(10, 10)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    await device.queue.on_submitted_work_done_async()


if __name__ == "__main__":
    run_tests(globals())
