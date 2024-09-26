"""
A few objects have a destroy method. We test its behavior here.
In practice it does not really affect the lifetime, so these tests look
a lot like the corresponding release tests :)
"""

import pytest
import testutils  # noqa
from testutils import can_use_wgpu_lib, create_and_release


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need wgpu lib", allow_module_level=True)


import wgpu

DEVICE = wgpu.utils.get_default_device()


@create_and_release
def test_destroy_device(n):
    yield {
        "expected_counts_after_create": {"Device": (n, n), "Queue": (n, n)},
    }

    adapter = DEVICE.adapter
    for i in range(n):
        d = adapter.request_device_sync()
        d.destroy()
        # NOTE: destroy is not yet implemented in wgpu-natice - this does not actually do anything yet
        yield d


@create_and_release
def test_destroy_query_set(n):
    yield {}
    for i in range(n):
        qs = DEVICE.create_query_set(type=wgpu.QueryType.occlusion, count=2)
        qs.destroy()
        # NOTE: destroy is not yet implemented in wgpu-natice - this does not actually do anything yet
        yield qs


@create_and_release
def test_destroy_buffer(n):
    yield {}
    for i in range(n):
        b = DEVICE.create_buffer(
            size=128, usage=wgpu.MapMode.READ | wgpu.BufferUsage.COPY_DST
        )
        b.destroy()
        b.destroy()  # fine to call multiple times

        # The buffer is now in a destroyed state (in wgpu-core). It still exists, its size and usage
        # can still be queries from wgpu-native, but it cannot be used.

        # Uncomment the following lines to see. These are commented because it makes wgpu-core create a command-buffer.
        # try:
        #     b.map_sync("READ")
        # except wgpu.GPUValidationError as err:
        #     error = err
        # assert "destroyed" in error.message.lower()

        yield b


@create_and_release
def test_destroy_texture(n):
    yield {}
    for i in range(n):
        t = DEVICE.create_texture(
            size=(16, 16, 16),
            usage=wgpu.TextureUsage.TEXTURE_BINDING,
            format="rgba8unorm",
        )
        t.destroy()

        # Uncomment the following lines to see. These are commented because the views are created at the native side, but we never store them, but we also don't release them.
        # try:
        #     t.create_view()
        # except wgpu.GPUValidationError as err:
        #     error = err
        # assert "destroyed" in error.message.lower()
        yield t


# %% The end


TEST_FUNCS = [
    ob
    for name, ob in list(globals().items())
    if name.startswith("test_") and callable(ob)
]

if __name__ == "__main__":
    # testutils.TEST_ITERS = 40  # Uncomment for a mem-usage test run

    test_destroy_device()
    test_destroy_query_set()
    test_destroy_buffer()
    test_destroy_texture()
