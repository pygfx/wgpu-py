"""
Test creation of Qt canvas windows.
"""

import gc
import weakref

import pytest
import testutils  # noqa
from testutils import create_and_release, can_use_pyside6, can_use_wgpu_lib
from test_gui_offscreen import make_draw_func_for_canvas


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need wgpu lib", allow_module_level=True)
if not can_use_pyside6:
    pytest.skip("Need pyside6 for this test", allow_module_level=True)


@create_and_release
def test_release_canvas_context(n):
    # Test with PySide canvases.

    # Note: in a draw, the textureview is obtained (thus creating a
    # Texture and a TextureView, but these are released in present(),
    # so we don't see them in the counts.

    import PySide6  # noqa
    from wgpu.gui.qt import WgpuCanvas  # noqa

    app = PySide6.QtWidgets.QApplication.instance()
    if app is None:
        app = PySide6.QtWidgets.QApplication([""])

    yield {}

    canvases = weakref.WeakSet()

    for i in range(n):
        c = WgpuCanvas()
        canvases.add(c)
        c.request_draw(make_draw_func_for_canvas(c))
        app.processEvents()
        yield c.get_context()

    # Need some shakes to get all canvas refs gone.
    del c
    gc.collect()
    app.processEvents()

    # Check that the canvas objects are really deleted
    assert not canvases


if __name__ == "__main__":
    # testutils.TEST_ITERS = 40  # Uncomment for a mem-usage test run

    test_release_canvas_context()
