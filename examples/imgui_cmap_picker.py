"""
Example showing how to use multiple imgui contexts to draw to multiple canvases

# run_example = false
"""

import ctypes

import numpy as np

import wgpu
from imgui_bundle import imgui
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils.imgui import ImguiRenderer

from cmap import Colormap

# Create a canvas to render to
canvas = WgpuCanvas(title="imgui", size=(512, 256))

# Create a wgpu device
adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
device = adapter.request_device()

imgui_renderer = ImguiRenderer(device, canvas)

texture_sampler = device.create_sampler(
    label="img-sampler",
    mag_filter=wgpu.FilterMode.linear,
    min_filter=wgpu.FilterMode.linear,
    mipmap_filter=wgpu.FilterMode.linear,
)


def create_texture_and_upload(data) -> int:
    texture = device.create_texture(
        size=(data.shape[1], data.shape[0], 4),
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        mip_level_count=1,
        sample_count=1
    )

    texture_view = texture.create_view()

    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        data,
        {"offset": 0, "bytes_per_row": data.shape[1] * 4},
        (data.shape[1], data.shape[0], 1),
    )

    id_texture = ctypes.c_int32(id(texture_view)).value
    imgui_renderer.backend._texture_views[id_texture] = texture_view

    return id_texture


cmaps = [
    "viridis",
    "plasma",
    "turbo",
    "spring",
    "winter",
    "bwr",
    "gnuplot2"
]
cmap_data = dict()


for name in cmaps:
    data = Colormap(name)(np.linspace(0, 1)) * 255
    tex_id = create_texture_and_upload(
        np.vstack([[data]] * 2).astype(np.uint8)
    )
    cmap_data[name] = tex_id


current_cmap = cmaps[0]


def update_gui():
    imgui.new_frame()
    global current_cmap

    imgui.set_next_window_size((175, 0), imgui.Cond_.appearing)
    imgui.set_next_window_pos((0, 20), imgui.Cond_.appearing)

    imgui.begin("window", None)

    for cmap_name, tex_id in cmap_data.items():
        clicked, enabled = imgui.menu_item(
            cmap_name,
            None,
            p_selected=current_cmap == cmap_name
        )
        imgui.same_line()
        imgui.image(tex_id, image_size=(50, 10))
        if enabled:
            current_cmap = cmap_name

    imgui.end()

    imgui.end_frame()
    imgui.render()

    return imgui.get_draw_data()


def draw_frame():
    imgui_renderer.render(update_gui())
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(draw_frame)
    run()
