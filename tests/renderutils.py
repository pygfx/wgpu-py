""" Utils to render to a texture or screen. Tuned to the tests, so quite some
assumptions here.
"""


import ctypes
import numpy as np

import wgpu.backends.rs  # noqa


def upload_to_texture(device, texture, data, nx, ny, nz):

    nbytes = ctypes.sizeof(data)
    bpp = nbytes // (nx * ny * nz)

    # Create a buffer to get the data into the GPU
    buffer = device.create_buffer_mapped(size=nbytes, usage=wgpu.BufferUsage.COPY_SRC)

    # Upload to buffer
    ctypes.memmove(buffer.mapping, data, nbytes)
    buffer.unmap()

    # Copy to texture (rows_per_image must only be nonzero for 3D textures)
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {"buffer": buffer, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": 0},
        {"texture": texture, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    device.default_queue.submit([command_encoder.finish()])


def download_from_texture(device, texture, data_type, nx, ny, nz):
    nbytes = ctypes.sizeof(data_type)
    bpp = nbytes // (nx * ny * nz)

    # Create a buffer to get the data into the GPU
    buffer = device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.COPY_DST)

    # Copy to buffer
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {"buffer": buffer, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": 0},
        (nx, ny, nz),
    )
    device.default_queue.submit([command_encoder.finish()])

    # Download
    mapped_array = buffer.map_read()
    data = data_type.from_buffer(mapped_array)
    buffer.unmap()
    return data


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
    indirect_buffer=None,
    color_attachment=None,
    depth_stencil_state=None,
    depth_stencil_attachment=None,
    renderpass_callback=lambda *args: None,
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
    buffer = device.create_buffer(
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
        depth_stencil_state=depth_stencil_state,
        vertex_state={
            "index_format": wgpu.IndexFormat.uint32,
            "vertex_buffers": vbo_views,
        },
        sample_count=1,
        sample_mask=0xFFFFFFFF,
        alpha_to_coverage_enabled=False,
    )

    command_encoder = device.create_command_encoder()

    color_attachment = color_attachment or {
        "resolve_target": None,
        "load_value": (0, 0, 0, 0),  # LoadOp.load or color
        "store_op": wgpu.StoreOp.store,
    }
    color_attachment["attachment"] = current_texture_view
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[color_attachment],
        depth_stencil_attachment=depth_stencil_attachment,
        occlusion_query_set=None,
    )
    render_pass.push_debug_group("foo")

    render_pass.insert_debug_marker("setting pipeline")
    render_pass.set_pipeline(render_pipeline)
    render_pass.insert_debug_marker("setting bind group")
    render_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
    for slot, vbo in enumerate(vbos):
        render_pass.insert_debug_marker(f"setting vbo {slot}")
        render_pass.set_vertex_buffer(slot, vbo, 0, vbo.size)
    render_pass.insert_debug_marker(f"invoking callback")
    renderpass_callback(render_pass)
    render_pass.insert_debug_marker(f"draw!")
    if ibo is None:
        if indirect_buffer is None:
            render_pass.draw(4, 1, 0, 0)
        else:
            render_pass.draw_indirect(indirect_buffer, 0)
    else:
        render_pass.set_index_buffer(ibo, 0, ibo.size)
        if indirect_buffer is None:
            render_pass.draw_indexed(6, 1, 0, 0, 0)
        else:
            render_pass.draw_indexed_indirect(indirect_buffer, 0)
    render_pass.pop_debug_group()
    render_pass.end_pass()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {"buffer": buffer, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": 0},
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
    indirect_buffer=None,
    color_attachment=None,
    depth_stencil_state=None,
    depth_stencil_attachment=None,
    renderpass_callback=lambda *args: None,
):
    """ Render to a window on screen, for debugging purposes.
    """
    import glfw
    from wgpu.gui.glfw import WgpuCanvas, update_glfw_canvasses

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
        depth_stencil_state=depth_stencil_state,
        vertex_state={
            "index_format": wgpu.IndexFormat.uint32,
            "vertex_buffers": vbo_views,
        },
        sample_count=1,
        sample_mask=0xFFFFFFFF,
        alpha_to_coverage_enabled=False,
    )

    swap_chain = device.configure_swap_chain(
        canvas, wgpu.TextureFormat.bgra8unorm_srgb, wgpu.TextureUsage.OUTPUT_ATTACHMENT
    )

    def draw_frame():
        with swap_chain as current_texture_view:
            command_encoder = device.create_command_encoder()

            ca = color_attachment or {
                "resolve_target": None,
                "load_value": (0, 0, 0, 0),  # LoadOp.load or color
                "store_op": wgpu.StoreOp.store,
            }
            ca["attachment"] = current_texture_view
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[ca],
                depth_stencil_attachment=depth_stencil_attachment,
                occlusion_query_set=None,
            )
            render_pass.push_debug_group("foo")

            render_pass.insert_debug_marker("setting pipeline")
            render_pass.set_pipeline(render_pipeline)
            render_pass.insert_debug_marker("setting bind group")
            render_pass.set_bind_group(
                0, bind_group, [], 0, 999999
            )  # last 2 elements not used
            for slot, vbo in enumerate(vbos):
                render_pass.insert_debug_marker(f"setting vbo {slot}")
                render_pass.set_vertex_buffer(slot, vbo, 0, vbo.size)
            render_pass.insert_debug_marker("invoking callback")
            renderpass_callback(render_pass)
            render_pass.insert_debug_marker("draw!")
            if ibo is None:
                if indirect_buffer is None:
                    render_pass.draw(4, 1, 0, 0)
                else:
                    render_pass.draw_indirect(indirect_buffer, 0)
            else:
                render_pass.set_index_buffer(ibo, 0, ibo.size)
                if indirect_buffer is None:
                    render_pass.draw_indexed(6, 1, 0, 0, 0)
                else:
                    render_pass.draw_indexed_indirect(indirect_buffer, 0)
            render_pass.pop_debug_group()
            render_pass.end_pass()
            device.default_queue.submit([command_encoder.finish()])

    canvas.draw_frame = draw_frame

    # Enter main loop
    while update_glfw_canvasses():
        glfw.poll_events()
    glfw.terminate()
