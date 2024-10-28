"""
Test the offscreen canvas and some related mechanics.
"""

import os
import gc
import weakref

from testutils import is_pypy, run_tests


def test_offscreen_selection_using_env_var():
    from wgpu.gui.offscreen import WgpuManualOffscreenCanvas

    ori = os.environ.get("WGPU_FORCE_OFFSCREEN", "")
    os.environ["WGPU_FORCE_OFFSCREEN"] = "1"

    # We only need the func, but this triggers the auto-import
    from wgpu.gui.auto import select_backend

    try:
        if not os.getenv("CI"):
            for value in ["", "0", "false", "False", "wut"]:
                os.environ["WGPU_FORCE_OFFSCREEN"] = value
                module = select_backend()
                assert module.WgpuCanvas is not WgpuManualOffscreenCanvas

        for value in ["1", "true", "True"]:
            os.environ["WGPU_FORCE_OFFSCREEN"] = value
            module = select_backend()
            assert module.WgpuCanvas is WgpuManualOffscreenCanvas

    finally:
        os.environ["WGPU_FORCE_OFFSCREEN"] = ori


def test_offscreen_event_loop():
    """Check that the event loop handles queued tasks and then returns."""
    # Note: if this test fails, it may run forever, so it's a good idea to have a timeout on the CI job or something

    from wgpu.gui.offscreen import loop

    ran = False

    def check():
        nonlocal ran
        ran = True

    loop.call_later(0, check)
    loop.run()

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


if __name__ == "__main__":
    run_tests(globals())
