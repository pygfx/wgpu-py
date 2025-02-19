"""
Imgui example that shows how to create a colormap picker menu

Uses the cmap library: https://github.com/tlambert03/cmap

pip install cmap

Instead of using the cmap library you can create 2D arrays in
the shape [2, 255, 3] that represent the LUT for the colormap
"""

# run_example = false

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
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()

imgui_renderer = ImguiRenderer(device, canvas)


def create_texture_and_upload(data: np.ndarray) -> int:
    # crates a GPUTexture and uploads it

    # create a GPUTexture
    texture = device.create_texture(
        size=(data.shape[1], data.shape[0], 4),
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        mip_level_count=1,
        sample_count=1,
    )

    # upload to the GPU
    device.queue.write_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        data,
        {"offset": 0, "bytes_per_row": data.shape[1] * 4},
        (data.shape[1], data.shape[0], 1),
    )

    # get a view
    texture_view = texture.create_view()

    # get the id so that imgui can display it
    id_texture = ctypes.c_int32(id(texture_view)).value
    # add texture view to the backend so that it can be retrieved for rendering
    imgui_renderer.backend._texture_views[id_texture] = texture_view

    return id_texture


# list of colormaps that we will display in the picker
cmaps = ["viridis", "plasma", "turbo", "spring", "winter", "bwr", "gnuplot2"]
cmap_data = dict()


# creates texture for each colormap, uploads to the GPU, stores ids so we can display them in imgui
for name in cmaps:
    # creates LUT
    data = Colormap(name)(np.linspace(0, 1)) * 255

    # vstack it so we have 2 rows to make a Texture, an array of shape [2, 255, 3], [rows, cols, RGB]
    tex_id = create_texture_and_upload(np.vstack([[data]] * 2).astype(np.uint8))

    # store the texture id
    cmap_data[name] = tex_id


current_cmap = cmaps[0]


def update_gui():
    imgui.new_frame()
    global current_cmap

    imgui.set_next_window_size((175, 0), imgui.Cond_.appearing)
    imgui.set_next_window_pos((0, 20), imgui.Cond_.appearing)

    imgui.begin("window", None)

    # make the cmap images display height similar to the text height so that it looks nice
    texture_height = (
        imgui_renderer.backend.io.font_global_scale * imgui.get_font().font_size
    ) - 2

    # add the items for the picker
    for cmap_name, tex_id in cmap_data.items():
        # text part of each item
        clicked, enabled = imgui.menu_item(
            cmap_name, "", p_selected=current_cmap == cmap_name
        )
        imgui.same_line()
        # the image part of each item, give it the texture id
        imgui.image(tex_id, image_size=(50, texture_height), border_col=(1, 1, 1, 1))
        if enabled:
            current_cmap = cmap_name

    imgui.end()

    imgui.end_frame()
    imgui.render()

    return imgui.get_draw_data()


imgui_renderer.set_gui(update_gui)


def draw_frame():
    imgui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(draw_frame)
    run()
