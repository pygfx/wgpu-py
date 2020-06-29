"""
Test render pipeline by rendering to a texture.
"""

import ctypes
import numpy as np

import pyshader
from pyshader import python2shader, f32, vec2, vec4, i32
import wgpu.backends.rs  # noqa
from pytest import skip, raises
from testutils import run_tests, get_default_device
from testutils import can_use_wgpu_lib, can_use_vulkan_sdk
from renderutils import upload_to_texture, render_to_texture, render_to_screen  # noqa


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


@python2shader
def vertex_shader(
    index: ("input", "VertexId", i32),
    pos: ("output", "Position", vec4),
    tcoord: ("output", 0, vec2),
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


# %% rgba textures


def test_render_textured_square_rgba8unorm():
    """ Test a texture with format rgba8unorm.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d f32"),
        sampler: ("sampler", 1, ""),
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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


def test_render_textured_square_r16sint():
    """ Test a texture with format r16sint. Because e.g. CT data.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d r16i"),
        sampler: ("sampler", 1, ""),
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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


def test_render_textured_square_r32sint():
    """ Test a texture with format r32sint. Because e.g. CT data.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d r32i"),
        sampler: ("sampler", 1, ""),
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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


def test_render_textured_square_r32float():
    """ Test a texture with format r32float.
    """

    @python2shader
    def fragment_shader(
        tex: ("texture", 0, "2d r32f"),
        sampler: ("sampler", 1, ""),
        tcoord: ("input", 0, vec2),
        out_color: ("output", 0, vec4),
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


# %% Utils


def render_textured_square(fragment_shader, texture_format, texture_size, texture_data):
    """ Render, and test the result. The resulting image must be a
    gradient on R and B, zeros on G and ones on A.
    """
    nx, ny, nz = texture_size

    device = get_default_device()

    if can_use_vulkan_sdk:
        pyshader.dev.validate(vertex_shader)
        pyshader.dev.validate(fragment_shader)

    # Create texture
    texture = device.create_texture(
        size=(nx, ny, nz),
        dimension=wgpu.TextureDimension.d2,
        format=texture_format,
        usage=wgpu.TextureUsage.SAMPLED | wgpu.TextureUsage.COPY_DST,
    )
    upload_to_texture(device, texture, texture_data, nx, ny, nz)

    # texture_view = texture.create_view()
    # or:
    texture_view = texture.create_view(
        format=texture_format, dimension=wgpu.TextureDimension.d2,
    )
    # But not like these ...
    with raises(ValueError):
        texture_view = texture.create_view(dimension=wgpu.TextureDimension.d2,)
    with raises(ValueError):
        texture_view = texture.create_view(mip_level_count=1,)

    sampler = device.create_sampler(mag_filter="linear", min_filter="linear")

    # Determine texture component type from the format
    if texture_format.endswith(("norm", "float")):
        texture_component_type = wgpu.TextureComponentType.float
    elif "uint" in texture_format:
        texture_component_type = wgpu.TextureComponentType.uint
    else:
        texture_component_type = wgpu.TextureComponentType.sint

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
            "view_dimension": wgpu.TextureViewDimension.d2,
            "texture_component_type": texture_component_type,
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "type": wgpu.BindingType.sampler,
        },
    ]
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

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
        [150, 150, 150, 150, 150, 150, 150, 150, 152, 155, 158, 161],
        [164, 167, 170, 173, 177, 180, 183, 186, 189, 192],
        [195, 198, 200, 200, 200, 200, 200, 200, 200, 200],
    ]
    ref2 = [
        [150, 150, 150, 150, 150, 150, 150, 150, 147, 141, 134, 128],
        [122, 116, 109, 103, 97, 91, 84, 78, 72, 66],
        [59, 53, 50, 50, 50, 50, 50, 50, 50, 50],
    ]
    ref1, ref2 = sum(ref1, []), sum(ref2, [])

    assert np.allclose(sq[0, :, 0], ref1, atol=1)
    assert np.allclose(sq[:, 0, 0], ref2, atol=1)
    assert np.allclose(sq[0, :, 1], ref1, atol=1)
    assert np.allclose(sq[:, 0, 1], ref2, atol=1)
    assert np.all(sq[:, :, 2] == 0)  # blue
    assert np.all(sq[:, :, 3] == 255)  # alpha


if __name__ == "__main__":
    run_tests(globals())
