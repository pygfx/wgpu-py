"""
Test creation of GLFW canvas windows.
"""

import gc
import asyncio

import pytest
import testutils
from testutils import create_and_release, can_use_glfw, can_use_wgpu_lib
from test_gui_offscreen import make_draw_func_for_canvas


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need wgpu lib", allow_module_level=True)
if not can_use_glfw:
    pytest.skip("Need glfw for this test", allow_module_level=True)

loop = asyncio.get_event_loop_policy().get_event_loop()
if loop.is_running():
    pytest.skip("Asyncio loop is running", allow_module_level=True)


async def stub_event_loop():
    pass


@create_and_release
def test_release_canvas_context(n):
    # Test with GLFW canvases.

    # Note: in a draw, the textureview is obtained (thus creating a
    # Texture and a TextureView, but these are released in present(),
    # so we don't see them in the counts.

    from wgpu.gui.glfw import WgpuCanvas  # noqa

    yield {}

    for i in range(n):
        c = WgpuCanvas()
        c.request_draw(make_draw_func_for_canvas(c))
        loop.run_until_complete(stub_event_loop())
        yield c.get_context()

    # Need some shakes to get all canvas refs gone.
    # Note that the canvas objects are really deleted,
    # otherwise the CanvasContext objects would not be freed.
    del c
    loop.run_until_complete(stub_event_loop())
    gc.collect()
    loop.run_until_complete(stub_event_loop())


if __name__ == "__main__":
    # Set to true and run as script to do a memory stress test
    testutils.TEST_MEM_USAGE = False

    test_release_canvas_context()
