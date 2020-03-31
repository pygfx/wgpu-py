""" Utils to render to a texture or screen. Tuned to the tests, so quite some
assumptions here.
"""


import ctypes
import numpy as np

import wgpu.backends.rs  # noqa


def render_to_texture(
    device,
    vertex_shader,
    fragment_shader,
    pipeline_layout,
    bind_group,
    *,
    size,
    topology=wgpu.PrimitiveTopology.triangle_strip,
    ibo=None,
    vbos=None,
    vbo_views=None,
    color_attachement=None,
):

    # https://github.com/gfx-rs/wgpu-rs/blob/master/examples/capture/main.rs

    vbos = vbos or []
    vbo_views = vbo_views or []

    # Select texture format. The srgb norm maps to the srgb colorspace which
    # appears to be the default for render pipelines https://en.wikipedia.org/wiki/SRGB
    texture_format = wgpu.TextureFormat.rgba8unorm  # rgba8unorm or bgra8unorm_srgb

    # Create texture to render to
    nx, ny, bpp = size[0], size[1], 4
    nbytes = nx * ny * bpp
    texture = device.create_texture(
        size=(nx, ny, 1),
        dimension=wgpu.TextureDimension.d2,
        format=texture_format,
        usage=wgpu.TextureUsage.OUTPUT_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    current_texture_view = texture.create_default_view()

    # Also a buffer to read the data to CPU
    buffer = device.create_buffer_mapped(
        size=nbytes, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    vshader = device.create_shader_module(code=vertex_shader)
    fshader = device.create_shader_module(code=fragment_shader)

    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex_stage={"module": vshader, "entry_point": "main"},
        fragment_stage={"module": fshader, "entry_point": "main"},
        primitive_topology=topology,
        rasterization_state={
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
            "depth_bias": 0,
            "depth_bias_slope_scale": 0.0,
            "depth_bias_clamp": 0.0,
        },
        color_states=[
            {
                "format": texture_format,
                "alpha_blend": (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.zero,
                    wgpu.BlendOperation.add,
                ),
                "color_blend": (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.zero,
                    wgpu.BlendOperation.add,
                ),
                "write_mask": wgpu.ColorWrite.ALL,
            }
        ],
        depth_stencil_state=None,
        vertex_state={
            "index_format": wgpu.IndexFormat.uint32,
            "vertex_buffers": vbo_views,
        },
        sample_count=1,
        sample_mask=0xFFFFFFFF,
        alpha_to_coverage_enabled=False,
    )

    command_encoder = device.create_command_encoder()

    color_attachement = color_attachement or {
        "resolve_target": None,
        "load_value": (0, 0, 0, 0),  # LoadOp.load or color
        "store_op": wgpu.StoreOp.store,
    }
    color_attachement["attachment"] = current_texture_view
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[color_attachement], depth_stencil_attachment=None,
    )

    render_pass.set_pipeline(render_pipeline)
    render_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
    for slot, vbo in enumerate(vbos):
        render_pass.set_vertex_buffer(slot, vbo, 0)
    if ibo is None:
        render_pass.draw(4, 1, 0, 0)
    else:
        render_pass.set_index_buffer(ibo, 0)
        render_pass.draw_indexed(6, 1, 0, 0, 0)
    render_pass.end_pass()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {"buffer": buffer, "offset": 0, "row_pitch": bpp * nx, "image_height": 0},
        (nx, ny, 1),
    )
    device.default_queue.submit([command_encoder.finish()])

    # Read the current data of the output buffer - numpy is much easier to work with
    array_uint8 = buffer.map_read()  # slow, can also be done async
    data = (ctypes.c_uint8 * 4 * nx * ny).from_buffer(array_uint8)
    return np.frombuffer(data, dtype=np.uint8).reshape(size[0], size[1], 4)


def render_to_screen(
    device,
    vertex_shader,
    fragment_shader,
    pipeline_layout,
    bind_group,
    *,
    topology=wgpu.PrimitiveTopology.triangle_strip,
    ibo=None,
    vbos=None,
    vbo_views=None,
    color_attachement=None,
):
    """ Render to a window on screen, for debugging purposes.
    """
    import glfw
    from wgpu.gui.glfw import WgpuCanvas

    vbos = vbos or []
    vbo_views = vbo_views or []

    # Setup canvas
    glfw.init()
    canvas = WgpuCanvas(title="wgpu test render with GLFW")

    vshader = device.create_shader_module(code=vertex_shader)
    fshader = device.create_shader_module(code=fragment_shader)

    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex_stage={"module": vshader, "entry_point": "main"},
        fragment_stage={"module": fshader, "entry_point": "main"},
        primitive_topology=topology,
        rasterization_state={
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
            "depth_bias": 0,
            "depth_bias_slope_scale": 0.0,
            "depth_bias_clamp": 0.0,
        },
        color_states=[
            {
                "format": wgpu.TextureFormat.bgra8unorm_srgb,
                "alpha_blend": (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.zero,
                    wgpu.BlendOperation.add,
                ),
                "color_blend": (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.zero,
                    wgpu.BlendOperation.add,
                ),
                "write_mask": wgpu.ColorWrite.ALL,
            }
        ],
        depth_stencil_state=None,
        vertex_state={
            "index_format": wgpu.IndexFormat.uint32,
            "vertex_buffers": vbo_views,
        },
        sample_count=1,
        sample_mask=0xFFFFFFFF,
        alpha_to_coverage_enabled=False,
    )

    swap_chain = canvas.configure_swap_chain(
        device, wgpu.TextureFormat.bgra8unorm_srgb, wgpu.TextureUsage.OUTPUT_ATTACHMENT
    )

    def draw_frame():
        current_texture_view = swap_chain.get_current_texture_view()
        command_encoder = device.create_command_encoder()

        ca = color_attachement or {
            "resolve_target": None,
            "load_value": (0, 0, 0, 0),  # LoadOp.load or color
            "store_op": wgpu.StoreOp.store,
        }
        ca["attachment"] = current_texture_view
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[ca], depth_stencil_attachment=None,
        )

        render_pass.set_pipeline(render_pipeline)
        render_pass.set_bind_group(
            0, bind_group, [], 0, 999999
        )  # last 2 elements not used
        for slot, vbo in enumerate(vbos):
            render_pass.set_vertex_buffer(slot, vbo, 0)
        if ibo is None:
            render_pass.draw(4, 1, 0, 0)
        else:
            render_pass.set_index_buffer(ibo, 0)
            render_pass.draw_indexed(6, 1, 0, 0, 0)
        render_pass.end_pass()
        device.default_queue.submit([command_encoder.finish()])

    canvas.draw_frame = draw_frame

    # Enter main loop
    while not canvas.is_closed():
        glfw.poll_events()
    glfw.terminate()
