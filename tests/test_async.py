import anyio

from pytest import mark

import wgpu.utils
from testutils import can_use_wgpu_lib, run_tests
from wgpu import GPUDevice, MapMode, TextureFormat, GPUPromise


@mark.anyio
@mark.parametrize("use_async", [False, True])
async def test_awaitable_async(use_async):
    count = 0

    def handler(i):
        return i * i

    def callback(i):
        awaitable._wgpu_set_input(i)

    def poll_function():
        nonlocal count
        count += 1
        if count >= 3:
            callback(10)

    awaitable = GPUPromise(
        "test", None, handler, poller=poll_function, keepalive=callable
    )

    if use_async:
        result = await awaitable
    else:
        result = awaitable.sync_wait()
    assert result == 10 * 10


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.anyio
async def test_enumerate_adapters_async():
    adapters = await wgpu.gpu.enumerate_adapters_async()
    assert len(adapters) > 0
    for adapter in adapters:
        device = await adapter.request_device_async()
        assert isinstance(device, GPUDevice)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.anyio
async def test_request_device_async():
    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
    device = await adapter.request_device_async()
    assert device is not None


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.anyio
async def test_buffer_map_async():
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


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.anyio
async def make_pipeline_async():
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

    results = [None, None]
    async with anyio.create_task_group() as tg:
        # It's unfortunate anyio doesn't have async.gather. This code would just be
        # compute_pipeline, render_pipeline = asyncio.gather(.....)
        async def create_compute_pipeline():
            results[0] = await device.create_compute_pipeline_async(
                layout="auto", compute={"module": shader}
            )

        async def create_render_pipeline():
            results[1] = await device.create_render_pipeline_async(
                layout="auto",
                vertex={
                    "module": shader,
                },
                depth_stencil={"format": TextureFormat.rgba8unorm},
            )

        tg.start_soon(create_compute_pipeline)
        tg.start_soon(create_render_pipeline)

    compute_pipeline, render_pipeline = results
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
