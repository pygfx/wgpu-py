"""
Manual (not using the canvas abstraction) glfw integration example.

# run_example = false
"""
import sys
from pathlib import Path

import glfw

import wgpu
from wgpu.utils.device import get_default_device


sys.path.insert(0, str(Path(__file__).parent))

from triangle import shader_source


def get_surface_info_glfw(window):
    if sys.platform.startswith("win"):
        return {
            "platform": "windows",
            "window": int(glfw.get_win32_window(window)),
        }
    elif sys.platform.startswith("darwin"):
        return {
            "platform": "cocoa",
            "window": int(glfw.get_cocoa_window(window)),
        }
    elif sys.platform.startswith("linux"):
        if hasattr(glfw, "get_wayland_window"):
            return {
                "platform": "wayland",
                "window": int(glfw.get_wayland_window(window)),
                "display": int(glfw.get_wayland_display()),
            }
        return {
            "platform": "x11",
            "window": int(glfw.get_x11_window(window)),
            "display": int(glfw.get_x11_display()),
        }
    else:
        raise RuntimeError(f"Cannot get GLFW surface info on {sys.platform}.")


class GlfwCanvasMock:
    def __init__(self, window):
        self._window = window
    
    def get_surface_info(self):
        if sys.platform.startswith("win"):
            return {
                "platform": "windows",
                "window": int(glfw.get_win32_window(self._window)),
            }
        elif sys.platform.startswith("darwin"):
            return {
                "platform": "cocoa",
                "window": int(glfw.get_cocoa_window(self._window)),
            }
        elif sys.platform.startswith("linux"):
            if hasattr(glfw, "get_wayland_window"):
                return {
                    "platform": "wayland",
                    "window": int(glfw.get_wayland_window(self._window)),
                    "display": int(glfw.get_wayland_display()),
                }
            return {
                "platform": "x11",
                "window": int(glfw.get_x11_window(self._window)),
                "display": int(glfw.get_x11_display()),
            }
        else:
            raise RuntimeError(f"Cannot get GLFW surface info on {sys.platform}.")



def get_context(canvas, kind="webgpu"):
    assert kind == "webgpu"
    # TODO: why is this so complicated?
    # seems like GPUCanvasContext is tightly coupled to the Canvas abstraction
    # now I understand #430
    # I cannot continue past this point without completely re-implementing GlfwCanvas
    from wgpu.backends.auto import gpu
    CC = sys.modules[gpu.__module__].GPUCanvasContext
    return CC(canvas)


def setup_triangle(context, device):
    shader = device.create_shader_module(code=shader_source)

    # No bind group and layout, we should not create empty ones.
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    render_texture_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=render_texture_format)

    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil=None,
        multisample=None,
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
        current_texture = context.get_current_texture()
        command_encoder = device.create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(render_pipeline)
        # render_pass.set_bind_group(0, no_bind_group, [], 0, 1)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        device.queue.submit([command_encoder.finish()])

    return draw_frame


def main():
    glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)
    window = glfw.create_window(640, 480, "glfw window", None, None)

    canvas = GlfwCanvasMock(window)
    context = get_context(canvas)
    device = get_default_device()
    draw_frame = setup_triangle(context, device)

    while True:
        glfw.poll_events()
        draw_frame()
        if glfw.window_should_close(window):
            break
    glfw.terminate()


if __name__ == "__main__":
    main()
