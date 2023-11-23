""" Utils to render to a texture or screen. Tuned to the tests, so quite some
assumptions here.
"""


import ctypes
import numpy as np

import wgpu.backends.wgpu_native  # noqa


def upload_to_texture(device, texture, data, nx, ny, nz):
    nbytes = ctypes.sizeof(data)
    bpp = nbytes // (nx * ny * nz)

    # Create a buffer to get the data into the GPU
    buffer = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.COPY_SRC)

    # Copy to texture
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {
            "buffer": buffer,
            "offset": 0,
            "bytes_per_row": bpp * nx,
            "rows_per_image": ny,
        },
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    device.queue.submit([command_encoder.finish()])


def download_from_texture(device, texture, data_type, nx, ny, nz):
    nbytes = ctypes.sizeof(data_type)
    bpp = nbytes // (nx * ny * nz)

    # Create a buffer to get the data into the GPU
    buffer = device.create_buffer(size=nbytes, usage=wgpu.BufferUsage.COPY_DST)

    # Copy to buffer
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {
            "buffer": buffer,
            "offset": 0,
            "bytes_per_row": bpp * nx,
            "rows_per_image": ny,
        },
        (nx, ny, nz),
    )
    device.queue.submit([command_encoder.finish()])

    # Download
    return data_type.from_buffer(device.queue.read_buffer(buffer))


def render_to_texture(
    device,
    shader_source,
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
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    current_texture_view = texture.create_view()

    # Also a buffer to read the data to CPU
    buffer = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )

    shader = device.create_shader_module(code=shader_source)

    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": vbo_views,
        },
        primitive={
            "topology": topology,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil=depth_stencil_state,
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
                    "format": texture_format,
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
                    "write_mask": wgpu.ColorWrite.ALL,
                },
            ],
        },
    )

    if bind_group:
        # if the bind_group is provided, we can at least retrieve
        # the first bind group layout from the pipeline
        _ = render_pipeline.get_bind_group_layout(0)

    command_encoder = device.create_command_encoder()

    color_attachment = color_attachment or {
        "resolve_target": None,
        "clear_value": (0, 0, 0, 0),
        "load_op": wgpu.LoadOp.clear,
        "store_op": wgpu.StoreOp.store,
    }
    color_attachment["view"] = current_texture_view
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[color_attachment],
        depth_stencil_attachment=depth_stencil_attachment,
        occlusion_query_set=None,
    )
    render_pass.push_debug_group("foo")

    render_pass.insert_debug_marker("setting pipeline")
    render_pass.set_pipeline(render_pipeline)
    render_pass.insert_debug_marker("setting bind group")
    if bind_group:
        render_pass.set_bind_group(
            0, bind_group, [], 0, 999999
        )  # last 2 elements not used
    for slot, vbo in enumerate(vbos):
        render_pass.insert_debug_marker(f"setting vbo {slot}")
        render_pass.set_vertex_buffer(slot, vbo, 0, 0)
    render_pass.insert_debug_marker("invoking callback")
    renderpass_callback(render_pass)
    render_pass.insert_debug_marker("draw!")
    if ibo is None:
        if indirect_buffer is None:
            render_pass.draw(4, 1, 0, 0)
        else:
            render_pass.draw_indirect(indirect_buffer, 0)
    else:
        render_pass.set_index_buffer(ibo, wgpu.IndexFormat.uint32, 0, 0)
        if indirect_buffer is None:
            render_pass.draw_indexed(6, 1, 0, 0, 0)
        else:
            render_pass.draw_indexed_indirect(indirect_buffer, 0)
    render_pass.pop_debug_group()
    render_pass.end()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {
            "buffer": buffer,
            "offset": 0,
            "bytes_per_row": bpp * nx,
            "rows_per_image": ny,
        },
        (nx, ny, 1),
    )
    device.queue.submit([command_encoder.finish()])

    # Read the current data of the output buffer - numpy is much easier to work with
    mem = device.queue.read_buffer(buffer)
    data = (ctypes.c_uint8 * 4 * nx * ny).from_buffer(mem)
    return np.frombuffer(data, dtype=np.uint8).reshape(size[0], size[1], 4)


def render_to_screen(
    device,
    shader_source,
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
    """Render to a window on screen, for debugging purposes."""
    import glfw
    from wgpu.gui.glfw import WgpuCanvas, update_glfw_canvasses

    vbos = vbos or []
    vbo_views = vbo_views or []

    # Setup canvas
    glfw.init()
    canvas = WgpuCanvas(title="wgpu test render with GLFW")

    shader = device.create_shader_module(code=shader_source)

    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": vbo_views,
        },
        primitive={
            "topology": topology,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil=depth_stencil_state,
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
                    "format": wgpu.TextureFormat.bgra8unorm_srgb,
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
                    "write_mask": wgpu.ColorWrite.ALL,
                },
            ],
        },
    )

    present_context = canvas.get_context()
    present_context.configure(device=device, format=None)

    def draw_frame():
        current_texture_view = present_context.get_current_texture().create_view()
        command_encoder = device.create_command_encoder()

        ca = color_attachment or {
            "resolve_target": None,
            "clear_value": (0, 0, 0, 0),
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
        }
        ca["view"] = current_texture_view
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
            render_pass.set_index_buffer(ibo, wgpu.IndexFormat.uint32, 0, ibo.size)
            if indirect_buffer is None:
                render_pass.draw_indexed(6, 1, 0, 0, 0)
            else:
                render_pass.draw_indexed_indirect(indirect_buffer, 0)
        render_pass.pop_debug_group()
        render_pass.end()
        device.queue.submit([command_encoder.finish()])

    canvas.request_draw(draw_frame)

    # Enter main loop
    while update_glfw_canvasses():
        glfw.poll_events()
    glfw.terminate()
