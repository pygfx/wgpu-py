"""
Test the force offscreen auto gui mechanism.
"""

import os
import gc
import weakref

import wgpu
from pytest import fixture, skip
from testutils import can_use_wgpu_lib, is_pypy


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


@fixture(autouse=True, scope="module")
def force_offscreen():
    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
    try:
        yield
    finally:
        del os.environ["WGPU_FORCE_OFFSCREEN"]


def test_canvas_class():
    """Check if we get an offscreen canvas when the WGPU_FORCE_OFFSCREEN
    environment variable is set."""
    from wgpu.gui.auto import WgpuCanvas
    from wgpu.gui.offscreen import WgpuManualOffscreenCanvas

    assert WgpuCanvas is WgpuManualOffscreenCanvas
    assert issubclass(WgpuCanvas, wgpu.gui.WgpuCanvasBase)
    assert issubclass(WgpuCanvas, wgpu.gui.WgpuAutoGui)


def test_event_loop():
    """Check that the event loop handles queued tasks and then returns."""
    # Note: if this test fails, it may run forever, so it's a good idea to have a timeout on the CI job or something

    from wgpu.gui.auto import run, call_later

    ran = False

    def check():
        nonlocal ran
        ran = True

    call_later(0, check)
    run()

    assert ran


def test_offscreen_canvas_del():
    from wgpu.gui.offscreen import WgpuCanvas

    canvas = WgpuCanvas()
    ref = weakref.ref(canvas)

    assert ref() is not None
    del canvas
    if is_pypy:
        gc.collect()
    assert ref() is None
