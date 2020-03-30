"""
Test render pipeline by rendering to a texture.
"""

import ctypes
import numpy as np

import python_shader
from python_shader import python2shader, f32, vec2, vec4, i32
from python_shader import RES_INPUT, RES_OUTPUT
import wgpu.backends.rs  # noqa
from pytest import skip, mark
from testutils import can_use_wgpu_lib, get_default_device


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


@python2shader
def vertex_shader(
    index: (RES_INPUT, "VertexId", i32),
    pos: (RES_OUTPUT, "Position", vec4),
    tcoord: (RES_OUTPUT, 0, vec2),
):
    positions = [
        vec2(-0.5, -0.5),
        vec2(-0.5, +0.5),
        vec2(+0.5, -0.5),
        vec2(+0.5, +0.5),
    ]
    p = positions[index]
    pos = vec4(p, 0.0, 1.0)  # noqa
    tcoord = vec2(p + 0.5)  # noqa - map to 0..1


def test_render_orange_square():
    """ Render an orange square and check that there is an orange square.
    """

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.5, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(bindings=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, bindings=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args)
    a = render_to_texture(*render_args, size=(64, 64))

    # Check that the background is all zero
    bg = a.copy()
    bg[16:-16, 16:-16, :] = 0
    assert np.all(bg == 0)

    # Check the square
    sq = a[16:-16, 16:-16, :]
    assert np.all(sq[:, :, 0] == 255)  # red
    assert np.all(sq[:, :, 1] == 127)  # green
    assert np.all(sq[:, :, 2] == 0)  # blue
    assert np.all(sq[:, :, 3] == 255)  # alpha


def test_render_orange_dots():
    """ Render four orange dots and check that there are four orange square dots.
    """

    @python2shader
    def vertex_shader(
        index: (RES_INPUT, "VertexId", i32),
        pos: (RES_OUTPUT, "Position", vec4),
        pointsize: (RES_OUTPUT, "PointSize", f32),
        tcoord: (RES_OUTPUT, 0, vec2),
    ):
        positions = [
            vec2(-0.5, -0.5),
            vec2(-0.5, +0.5),
            vec2(+0.5, -0.5),
            vec2(+0.5, +0.5),
        ]
        p = positions[index]
        pos = vec4(p, 0.0, 1.0)  # noqa
        tcoord = vec2(p + 0.5)  # noqa - map to 0..1
        pointsize = 16.0  # noqa

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.5, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(bindings=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, bindings=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    top = wgpu.PrimitiveTopology.point_list
    # render_to_screen(*render_args, topology=top)
    a = render_to_texture(*render_args, size=(64, 64), topology=top)

    # Check that the background is all zero
    bg = a.copy()
    bg[8:24, 8:24, :] = 0
    bg[8:24, 40:56, :] = 0
    bg[40:56, 8:24, :] = 0
    bg[40:56, 40:56, :] = 0
    assert np.all(bg == 0)

    # Check the square
    for dot in (
        a[8:24, 8:24, :],
        a[8:24, 40:56, :],
        a[40:56, 8:24, :],
        a[40:56, 40:56, :],
    ):
        assert np.all(dot[:, :, 0] == 255)  # red
        assert np.all(dot[:, :, 1] == 127)  # green
        assert np.all(dot[:, :, 2] == 0)  # blue
        assert np.all(dot[:, :, 3] == 255)  # alpha


# %% rgba textures


def test_render_textured_square_rgba8unorm():
    """ Test a texture with format rgba8unorm.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d f32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        out_color = tex.sample(sampler, tcoord)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 50, 0, 255, 100, 100, 0, 255, 150, 150, 0, 255, 200, 200, 0, 255]
    texture_data = (ctypes.c_uint8 * (4 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rgba8unorm, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rgba8uint():
    """ Test a texture with format rgba8uint.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d i32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        out_color = vec4(tex.sample(sampler, tcoord)) / 255.0  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 50, 0, 255, 100, 100, 0, 255, 150, 150, 0, 255, 200, 200, 0, 255]
    texture_data = (ctypes.c_uint8 * (4 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rgba8uint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rgba16sint():
    """ Test a texture with format rgba16sint.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d i32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        out_color = vec4(tex.sample(sampler, tcoord)) / 255.0  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 50, 0, 255, 100, 100, 0, 255, 150, 150, 0, 255, 200, 200, 0, 255]
    texture_data = (ctypes.c_int16 * (4 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rgba16sint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rgba32float():
    """ Test a texture with format rgba32float.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d f32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        out_color = tex.sample(sampler, tcoord) / 255.0  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 50, 0, 255, 100, 100, 0, 255, 150, 150, 0, 255, 200, 200, 0, 255]
    texture_data = (ctypes.c_float * (4 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rgba32float, (nx, ny, nz), texture_data
    )


# %% rg textures


def test_render_textured_square_rg8unorm():
    """ Test a texture with format rg8unorm.
    The GPU considers blue to be 0 and alpha to be 1.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d f32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        out_color = tex.sample(sampler, tcoord)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 50, 100, 100, 150, 150, 200, 200]
    texture_data = (ctypes.c_ubyte * (2 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rg8unorm, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rg8uint():
    """ Test a texture with format rg8uint.
    The GPU considers blue to be 0 and alpha to be 1.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d i32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        val = vec2(tex.sample(sampler, tcoord).rg)
        out_color = vec4(val.rg / 255.0, 0, 1.0)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 50, 100, 100, 150, 150, 200, 200]
    texture_data = (ctypes.c_ubyte * (2 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rg8uint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rg16sint():
    """ Test a texture with format rg16sint.
    The GPU considers blue to be 0 and alpha to be 1.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d i32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        val = vec2(tex.sample(sampler, tcoord).rg)
        out_color = vec4(val.rg / 255.0, 0.0, 1.0)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 50, 100, 100, 150, 150, 200, 200]
    texture_data = (ctypes.c_int16 * (2 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rg16sint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rg32float():
    """ Test a texture with format rg32float.
    The GPU considers blue to be 0 and alpha to be 1.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d f32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        val = tex.sample(sampler, tcoord).rg
        out_color = vec4(val.rg / 255.0, 0.0, 1.0)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 50, 100, 100, 150, 150, 200, 200]
    texture_data = (ctypes.c_float * (2 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rg32float, (nx, ny, nz), texture_data
    )


# %% r textures


def test_render_textured_square_r8unorm():
    """ Test a texture with format r8unorm.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d f32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        val = tex.sample(sampler, tcoord).r
        out_color = vec4(val, val, 0.0, 1.0)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 100, 150, 200]
    texture_data = (ctypes.c_uint8 * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r8unorm, (nx, ny, nz), texture_data
    )


def test_render_textured_square_r8uint():
    """ Test a texture with format r8uint.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d i32"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        val = f32(tex.sample(sampler, tcoord).r)
        out_color = vec4(val / 255.0, val / 255.0, 0.0, 1.0)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 100, 150, 200]
    texture_data = (ctypes.c_uint8 * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r8uint, (nx, ny, nz), texture_data
    )


@mark.skip(reason="This test does not pass yet!")
def test_render_textured_square_r16sint():
    """ Test a texture with format r16sint. Because e.g. CT data.
    """

    # todo: WHY does this not work???

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d r16i"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        val = f32(tex.sample(sampler, tcoord).r)
        out_color = vec4(val / 255.0, val / 255.0, 0.0, 1.0)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 100, 150, 200]
    texture_data = (ctypes.c_int16 * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r16sint, (nx, ny, nz), texture_data
    )


@mark.skip(reason="This test does not pass yet!")
def test_render_textured_square_r32sint():
    """ Test a texture with format r32sint. Because e.g. CT data.
    """

    # todo: WHY does this not work???

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d r32i"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        val = f32(tex.sample(sampler, tcoord).r)
        out_color = vec4(val / 255.0, val / 255.0, 0.0, 1.0)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 100, 150, 200]
    texture_data = (ctypes.c_int32 * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r32sint, (nx, ny, nz), texture_data
    )


@mark.skip(reason="This test does not pass yet!")
def test_render_textured_square_r32float():
    """ Test a texture with format r32float.
    """
    # todo: WHY does this not work???

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d r32f"),
        sampler: ("sampler", 1, ""),
        tcoord: (RES_INPUT, 0, vec2),
        out_color: (RES_OUTPUT, 0, vec4),
    ):
        val = f32(tex.sample(sampler, tcoord).r)
        out_color = vec4(val / 255.0, val / 255.0, 0.0, 1.0)  # noqa

    # Create texture data
    nx, ny, nz = 2, 2, 1
    x = [50, 100, 150, 200]
    texture_data = (ctypes.c_float * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r32float, (nx, ny, nz), texture_data
    )


# %% utils


def render_textured_square(fragment_shader, texture_format, texture_size, texture_data):
    """ Render, and test the result. The resulting image must be a
    gradient on R and B, zeros on G and ones on A.
    """
    nx, ny, nz = texture_size

    device = get_default_device()

    python_shader.dev.validate(vertex_shader)
    python_shader.dev.validate(fragment_shader)

    # Create texture
    texture = device.create_texture(
        size=(nx, ny, nz),
        dimension=wgpu.TextureDimension.d2,
        format=texture_format,
        usage=wgpu.TextureUsage.SAMPLED | wgpu.TextureUsage.COPY_DST,
    )
    upload_to_texture(device, texture, texture_data, nx, ny, nz)

    sampler = device.create_sampler(mag_filter="linear", min_filter="linear")
    texture_view = texture.create_default_view()

    # Bindings and layout
    bindings = [
        {"binding": 0, "resource": texture_view},
        {"binding": 1, "resource": sampler},
    ]
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "type": wgpu.BindingType.sampled_texture,
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "type": wgpu.BindingType.sampler,
        },
    ]
    bind_group_layout = device.create_bind_group_layout(bindings=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, bindings=bindings)

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args)
    a = render_to_texture(*render_args, size=(64, 64))

    # print(a.max(), a[:,:,0].max())

    # Check that the background is all zero
    bg = a.copy()
    bg[16:-16, 16:-16, :] = 0
    assert np.all(bg == 0)

    # Check the square
    sq = a[16:-16, 16:-16, :]
    ref1 = [
        [50, 50, 50, 50, 50, 50, 50, 50, 52, 55, 58, 61, 64, 67, 70, 73],
        [77, 80, 83, 86, 89, 92, 95, 98, 100, 100, 100, 100, 100, 100, 100, 100],
    ]
    ref2 = [
        [50, 50, 50, 50, 50, 50, 50, 50, 53, 59, 66, 72, 78, 84, 91, 97],
        [103, 109, 116, 122, 128, 134, 141, 147],
        [150, 150, 150, 150, 150, 150, 150, 150],
    ]
    ref1, ref2 = sum(ref1, []), sum(ref2, [])

    assert np.equal(sq[0, :, 0], ref1).all()
    assert np.equal(sq[:, 0, 0], ref2).all()
    assert np.equal(sq[0, :, 1], ref1).all()
    assert np.equal(sq[:, 0, 1], ref2).all()
    assert np.all(sq[:, :, 2] == 0)  # blue
    assert np.all(sq[:, :, 3] == 255)  # alpha


def upload_to_texture(device, texture, data, nx, ny, nz):

    nbytes = ctypes.sizeof(data)
    bpp = nbytes // (nx * ny * nz)

    # Create a buffer to get the data into the GPU
    buffer = device.create_buffer_mapped(size=nbytes, usage=wgpu.BufferUsage.COPY_SRC)
    ctypes.memmove(buffer.mapping, data, nbytes)
    buffer.unmap()

    # Upload (image_height must only be nonzero for 3D textures)
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {"buffer": buffer, "offset": 0, "row_pitch": bpp * nx, "image_height": 0},
        {"texture": texture, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    device.default_queue.submit([command_encoder.finish()])


def render_to_texture(
    device,
    vertex_shader,
    fragment_shader,
    pipeline_layout,
    bind_group,
    *,
    size,
    topology=wgpu.PrimitiveTopology.triangle_strip,
):

    # https://github.com/gfx-rs/wgpu-rs/blob/master/examples/capture/main.rs

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
        vertex_state={"index_format": wgpu.IndexFormat.uint32, "vertex_buffers": []},
        sample_count=1,
        sample_mask=0xFFFFFFFF,
        alpha_to_coverage_enabled=False,
    )

    command_encoder = device.create_command_encoder()

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "attachment": current_texture_view,
                "resolve_target": None,
                "load_value": (0, 0, 0, 0),  # LoadOp.load or color
                "store_op": wgpu.StoreOp.store,
            }
        ],
        depth_stencil_attachment=None,
    )

    render_pass.set_pipeline(render_pipeline)
    render_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
    render_pass.draw(4, 1, 0, 0)
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
):
    """ Render to a window on screen, for debugging purposes.
    """
    import glfw
    from wgpu.gui.glfw import WgpuCanvas

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
        vertex_state={"index_format": wgpu.IndexFormat.uint32, "vertex_buffers": []},
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

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "attachment": current_texture_view,
                    "resolve_target": None,
                    "load_value": (0, 0, 0, 1),  # LoadOp.load or color
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment=None,
        )

        render_pass.set_pipeline(render_pipeline)
        render_pass.set_bind_group(
            0, bind_group, [], 0, 999999
        )  # last 2 elements not used
        render_pass.draw(4, 1, 0, 0)
        render_pass.end_pass()
        device.default_queue.submit([command_encoder.finish()])

    canvas.draw_frame = draw_frame

    # Enter main loop
    while not canvas.is_closed():
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    test_render_orange_square()
    test_render_orange_dots()

    test_render_textured_square_rgba8unorm()
    test_render_textured_square_rgba8uint()
    test_render_textured_square_rgba16sint()
    test_render_textured_square_rgba32float()

    test_render_textured_square_rg8unorm()
    test_render_textured_square_rg8uint()
    test_render_textured_square_rg16sint()
    test_render_textured_square_rg32float()

    test_render_textured_square_r8unorm()
    test_render_textured_square_r8uint()
    # test_render_textured_square_r16sint()  # fails, why?
    # test_render_textured_square_r32sint()  # fails, why?
    # test_render_textured_square_r32float()  # fails, why?
