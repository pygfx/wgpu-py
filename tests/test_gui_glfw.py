"""
Test the canvas, and parts of the rendering that involves a canvas,
like the swap chain.
"""

import os
import sys
import time
import weakref
import asyncio

import wgpu.backends.rs  # noqa
from pytest import skip
from testutils import run_tests, can_use_glfw, can_use_wgpu_lib
from renderutils import render_to_texture, render_to_screen  # noqa


if not can_use_glfw or not can_use_wgpu_lib:
    skip("Skipping tests that need a window or the wgpu lib", allow_module_level=True)


def setup_module():
    import glfw

    glfw.init()


def teardown_module():
    pass  # Do not glfw.terminate() because other tests may still need glfw


def test_is_autogui():
    from wgpu.gui.glfw import WgpuCanvas

    assert issubclass(WgpuCanvas, wgpu.gui.WgpuCanvasBase)
    assert issubclass(WgpuCanvas, wgpu.gui.WgpuAutoGui)


def test_glfw_canvas_basics():
    """Create a window and check some of its behavior. No wgpu calls here."""

    import glfw
    from wgpu.gui.glfw import WgpuCanvas

    canvas = WgpuCanvas()

    canvas.set_logical_size(300, 200)
    etime = time.time() + 0.1
    while time.time() < etime:
        glfw.poll_events()
    lsize = canvas.get_logical_size()
    assert isinstance(lsize, tuple) and len(lsize) == 2
    assert isinstance(lsize[0], float) and isinstance(lsize[1], float)
    assert lsize == (300.0, 200.0)

    assert len(canvas.get_physical_size()) == 2
    assert isinstance(canvas.get_pixel_ratio(), float)

    # Close
    assert not canvas.is_closed()
    if sys.platform.startswith("win"):  # On Linux we cant do this multiple times
        canvas.close()
        glfw.poll_events()
        assert canvas.is_closed()


def test_glfw_canvas_del():
    from wgpu.gui.glfw import WgpuCanvas, update_glfw_canvasses
    import glfw

    loop = asyncio.get_event_loop()

    async def miniloop():
        for i in range(10):
            glfw.poll_events()
            update_glfw_canvasses()
            await asyncio.sleep(0.01)

    canvas = WgpuCanvas()
    ref = weakref.ref(canvas)

    assert ref() is not None
    loop.run_until_complete(miniloop())
    assert ref() is not None
    del canvas
    loop.run_until_complete(miniloop())
    assert ref() is None


shader_source = """
@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4<f32> {
    var positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2<f32>(0.0, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.7));
    let p: vec2<f32> = positions[vertex_index];
    return vec4<f32>(p, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.0, 1.0);
}
"""


def test_glfw_canvas_render():
    """Render an orange square ... in a glfw window."""

    import glfw
    from wgpu.gui.glfw import update_glfw_canvasses, WgpuCanvas

    loop = asyncio.get_event_loop()

    canvas = WgpuCanvas(max_fps=9999)

    # wgpu.utils.get_default_device()
    adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
    device = adapter.request_device()
    draw_frame1 = _get_draw_function(device, canvas)

    frame_counter = 0

    def draw_frame2():
        nonlocal frame_counter
        frame_counter += 1
        draw_frame1()

    canvas.request_draw(draw_frame2)

    # Give it a few rounds to start up
    async def miniloop():
        for i in range(10):
            glfw.poll_events()
            update_glfw_canvasses()
            await asyncio.sleep(0.01)

    loop.run_until_complete(miniloop())
    # There should have been exactly one draw now
    assert frame_counter == 1

    # Ask for a lot of draws
    for i in range(5):
        canvas.request_draw()
    # Process evens for a while
    loop.run_until_complete(miniloop())
    # We should have had just one draw
    assert frame_counter == 2

    # Change the canvase size
    canvas.set_logical_size(300, 200)
    canvas.set_logical_size(400, 300)
    # We should have had just one draw
    loop.run_until_complete(miniloop())
    assert frame_counter == 3

    # canvas.close()
    glfw.poll_events()


def test_glfw_canvas_render_custom_canvas():
    """Render an orange square ... in a glfw window. But not using WgpuCanvas.
    This helps make sure that WgpuCanvasInterface is indeed the minimal
    required canvas API.
    """

    import glfw

    class CustomCanvas:  # implements wgpu.WgpuCanvasInterface
        def __init__(self):
            glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
            glfw.window_hint(glfw.RESIZABLE, True)
            self.window = glfw.create_window(300, 200, "canvas", None, None)
            self._present_context = None

        def get_window_id(self):
            if sys.platform.startswith("win"):
                return int(glfw.get_win32_window(self.window))
            elif sys.platform.startswith("darwin"):
                return int(glfw.get_cocoa_window(self.window))
            elif sys.platform.startswith("linux"):
                is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
                if is_wayland:
                    return int(glfw.get_wayland_window(self.window))
                else:
                    return int(glfw.get_x11_window(self.window))
            else:
                raise RuntimeError(f"Cannot get GLFW window id on {sys.platform}.")

        def get_display_id(self):
            return wgpu.WgpuCanvasInterface.get_display_id(self)

        def get_physical_size(self):
            psize = glfw.get_framebuffer_size(self.window)
            return int(psize[0]), int(psize[1])

        def get_context(self):
            if self._present_context is None:
                backend_module = sys.modules["wgpu"].GPU.__module__
                PC = sys.modules[backend_module].GPUCanvasContext  # noqa N806
                self._present_context = PC(self)
            return self._present_context

    canvas = CustomCanvas()

    adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
    device = adapter.request_device()
    draw_frame = _get_draw_function(device, canvas)

    for i in range(5):
        time.sleep(0.01)
        glfw.poll_events()
        draw_frame()
        canvas.get_context().present()  # WgpuCanvasBase normally automates this

    glfw.hide_window(canvas.window)


def _get_draw_function(device, canvas):
    # Bindings and layout
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    shader = device.create_shader_module(code=shader_source)

    present_context = canvas.get_context()
    render_texture_format = present_context.get_preferred_format(device.adapter)
    present_context.configure(device=device, format=render_texture_format)

    render_pipeline = device.create_render_pipeline(
        label="my-debug-pipeline",
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_strip,
            "strip_index_format": wgpu.IndexFormat.uint32,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil=None,
        multisample={
            "count": 1,
            "mask": 0xFFFFFFFF,
            "alpha_to_coverage_enabled": False,
        },
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [
                {
                    "format": render_texture_format,
                    "blend": {
                        "color": (
                            wgpu.BlendFactor.one,
                            wgpu.BlendFactor.zero,
                            wgpu.BlendOperation.add,
                        ),
                        "alpha": (
                            wgpu.BlendFactor.one,
                            wgpu.BlendFactor.zero,
                            wgpu.BlendOperation.add,
                        ),
                    },
                },
            ],
        },
    )

    def draw_frame():
        current_texture_view = present_context.get_current_texture()
        command_encoder = device.create_command_encoder()
        assert current_texture_view.size
        ca = {
            "view": current_texture_view,
            "resolve_target": None,
            "clear_value": (0, 0, 0, 0),
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
        }
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[ca],
        )

        render_pass.set_pipeline(render_pipeline)
        render_pass.draw(4, 1, 0, 0)
        render_pass.end()
        device.queue.submit([command_encoder.finish()])

    return draw_frame


if __name__ == "__main__":
    setup_module()
    run_tests(globals())
    teardown_module()
