"""
Test the force offscreen auto gui mechanism.
"""

import os
import weakref

import wgpu.backends.rs  # noqa
from pytest import fixture, skip
from testutils import can_use_wgpu_lib


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
    """Check that the event loop handles queued tasks and then returns.
    TODO: This test may hang indefinitely in the failure case, can we
    prevent that?"""
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
    assert ref() is None
