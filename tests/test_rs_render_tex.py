"""
Test render pipeline by rendering to a texture.
"""

import ctypes
import numpy as np
import sys

import wgpu.backends.rs  # noqa
from pytest import skip
from testutils import run_tests, get_default_device
from testutils import can_use_wgpu_lib, is_ci
from renderutils import upload_to_texture, render_to_texture, render_to_screen  # noqa


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)
elif is_ci and sys.platform == "win32":
    skip("These tests fail on dx12 for some reason", allow_module_level=True)


default_vertex_shader = """
struct VertexOutput {
    @location(0) texcoord : vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var positions: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
    );
    let p: vec2<f32> = positions[vertex_index];
    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.texcoord = p + 0.5;
    return out;
}
"""


def _create_data(v1, v2, v3, v4):
    assert len(v1) == len(v2)
    assert len(v1) == len(v3)
    assert len(v1) == len(v4)
    data = []
    for y in range(128):
        data.extend(list(v1) * 128)
        data.extend(list(v2) * 128)
    for y in range(128):
        data.extend(list(v3) * 128)
        data.extend(list(v4) * 128)
    return data


# %% rgba textures


def test_render_textured_square_rgba8unorm():
    """Test a texture with format rgba8unorm."""

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<f32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            let sample = textureSample(r_tex, r_sampler, in.texcoord);
            return sample;
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data(
        (50, 50, 0, 255), (100, 100, 0, 255), (150, 150, 0, 255), (200, 200, 0, 255)
    )
    texture_data = (ctypes.c_uint8 * (4 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rgba8unorm, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rgba8uint():
    """Test a texture with format rgba8uint."""

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<u32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            // let sample = textureSample(r_tex, r_sampler, in.texcoord);
            let texcoords_u = vec2<i32>(in.texcoord * vec2<f32>(textureDimensions(r_tex)));
            let sample = textureLoad(r_tex, texcoords_u, 0);
            return vec4<f32>(sample) / 255.0;
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data(
        (50, 50, 0, 255), (100, 100, 0, 255), (150, 150, 0, 255), (200, 200, 0, 255)
    )
    texture_data = (ctypes.c_uint8 * (4 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rgba8uint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rgba16sint():
    """Test a texture with format rgba16sint."""

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<i32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            // let sample = textureSample(r_tex, r_sampler, in.texcoord);
            let texcoords_u = vec2<i32>(in.texcoord * vec2<f32>(textureDimensions(r_tex)));
            let sample = textureLoad(r_tex, texcoords_u, 0);
            return vec4<f32>(sample) / 255.0;
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data(
        (50, 50, 0, 255), (100, 100, 0, 255), (150, 150, 0, 255), (200, 200, 0, 255)
    )
    texture_data = (ctypes.c_int16 * (4 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rgba16sint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rgba32float():
    """Test a texture with format rgba32float."""

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<f32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            let sample = textureSample(r_tex, r_sampler, in.texcoord);
            return sample / 255.0;
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data(
        (50, 50, 0, 255), (100, 100, 0, 255), (150, 150, 0, 255), (200, 200, 0, 255)
    )
    texture_data = (ctypes.c_float * (4 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rgba32float, (nx, ny, nz), texture_data
    )


# %% rg textures


def test_render_textured_square_rg8unorm():
    """Test a texture with format rg8unorm.
    The GPU considers blue to be 0 and alpha to be 1.
    """

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<f32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            let sample = textureSample(r_tex, r_sampler, in.texcoord);
            return sample;
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data((50, 50), (100, 100), (150, 150), (200, 200))
    texture_data = (ctypes.c_ubyte * (2 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rg8unorm, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rg8uint():
    """Test a texture with format rg8uint.
    The GPU considers blue to be 0 and alpha to be 1.
    """

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<u32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            // let sample = textureSample(r_tex, r_sampler, in.texcoord);
            let texcoords_u = vec2<i32>(in.texcoord * vec2<f32>(textureDimensions(r_tex)));
            let sample = textureLoad(r_tex, texcoords_u, 0);
            return vec4<f32>(f32(sample.r) / 255.0, f32(sample.g) / 255.0, 0.0, 1.0);
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data((50, 50), (100, 100), (150, 150), (200, 200))
    texture_data = (ctypes.c_ubyte * (2 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rg8uint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rg16sint():
    """Test a texture with format rg16sint.
    The GPU considers blue to be 0 and alpha to be 1.
    """

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<i32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            // let sample = textureSample(r_tex, r_sampler, in.texcoord);
            let texcoords_u = vec2<i32>(in.texcoord * vec2<f32>(textureDimensions(r_tex)));
            let sample = textureLoad(r_tex, texcoords_u, 0);
            return vec4<f32>(f32(sample.r) / 255.0, f32(sample.g) / 255.0, 0.0, 1.0);
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data((50, 50), (100, 100), (150, 150), (200, 200))
    texture_data = (ctypes.c_int16 * (2 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rg16sint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_rg32float():
    """Test a texture with format rg32float.
    The GPU considers blue to be 0 and alpha to be 1.
    """

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<f32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            let sample = textureSample(r_tex, r_sampler, in.texcoord);
            return vec4<f32>(sample.rg / 255.0, 0.0, 1.0);
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data((50, 50), (100, 100), (150, 150), (200, 200))
    texture_data = (ctypes.c_float * (2 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.rg32float, (nx, ny, nz), texture_data
    )


# %% r textures


def test_render_textured_square_r8unorm():
    """Test a texture with format r8unorm."""

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<f32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            let sample = textureSample(r_tex, r_sampler, in.texcoord);
            let val = sample.r;
            return vec4<f32>(val, val, 0.0, 1.0);
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data((50,), (100,), (150,), (200,))
    texture_data = (ctypes.c_uint8 * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r8unorm, (nx, ny, nz), texture_data
    )


def test_render_textured_square_r8uint():
    """Test a texture with format r8uint."""

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<u32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            let texcoords_u = vec2<i32>(in.texcoord * vec2<f32>(textureDimensions(r_tex)));
            let sample = textureLoad(r_tex, texcoords_u, 0);
            let val = f32(sample.r) / 255.0;
            return vec4<f32>(val, val, 0.0, 1.0);
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data((50,), (100,), (150,), (200,))
    texture_data = (ctypes.c_uint8 * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r8uint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_r16sint():
    """Test a texture with format r16sint. Because e.g. CT data."""

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<i32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            let texcoords_u = vec2<i32>(in.texcoord * vec2<f32>(textureDimensions(r_tex)));
            let sample = textureLoad(r_tex, texcoords_u, 0);
            let val = f32(sample.r) / 255.0;
            return vec4<f32>(val, val, 0.0, 1.0);
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data((50,), (100,), (150,), (200,))
    texture_data = (ctypes.c_int16 * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r16sint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_r32sint():
    """Test a texture with format r32sint. Because e.g. CT data."""

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<i32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            let texcoords_u = vec2<i32>(in.texcoord * vec2<f32>(textureDimensions(r_tex)));
            let sample = textureLoad(r_tex, texcoords_u, 0);
            let val = f32(sample.r) / 255.0;
            return vec4<f32>(val, val, 0.0, 1.0);
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data((50,), (100,), (150,), (200,))
    texture_data = (ctypes.c_int32 * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r32sint, (nx, ny, nz), texture_data
    )


def test_render_textured_square_r32float():
    """Test a texture with format r32float."""

    fragment_shader = """
        @group(0) @binding(0)
        var r_tex: texture_2d<f32>;
        @group(0) @binding(1)
        var r_sampler: sampler;

        @fragment
        fn fs_main(in: VertexOutput, ) -> @location(0) vec4<f32> {
            let sample = textureSample(r_tex, r_sampler, in.texcoord);
            let val = sample.r / 255.0;
            return vec4<f32>(val, val, 0.0, 1.0);
        }
    """

    # Create texture data
    nx, ny, nz = 256, 256, 1
    x = _create_data((50,), (100,), (150,), (200,))
    texture_data = (ctypes.c_float * (1 * nx * ny))(*x)

    # Render and validate
    render_textured_square(
        fragment_shader, wgpu.TextureFormat.r32float, (nx, ny, nz), texture_data
    )


# %% Utils


def render_textured_square(fragment_shader, texture_format, texture_size, texture_data):
    """Render, and test the result. The resulting image must be a
    gradient on R and B, zeros on G and ones on A.
    """
    nx, ny, nz = texture_size

    device = get_default_device()

    shader_source = default_vertex_shader + fragment_shader

    # Create texture
    texture = device.create_texture(
        size=(nx, ny, nz),
        dimension=wgpu.TextureDimension.d2,
        format=texture_format,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    upload_to_texture(device, texture, texture_data, nx, ny, nz)

    texture_view = texture.create_view()
    # or:
    texture_view = texture.create_view(
        format=texture_format,
        dimension=wgpu.TextureDimension.d2,
    )

    sampler = device.create_sampler(mag_filter="nearest", min_filter="nearest")

    # Determine texture component type from the format
    if texture_format.endswith(("norm", "float")):
        texture_sample_type = wgpu.TextureSampleType.float
    elif "uint" in texture_format:
        texture_sample_type = wgpu.TextureSampleType.uint
    else:
        texture_sample_type = wgpu.TextureSampleType.sint

    # Determine sampler type.
    # Note that integer texture types cannot even use a sampler.
    sampler_type = wgpu.SamplerBindingType.filtering
    # On Vanilla wgpu, float32 textures cannot use a filtering
    # (interpolating) texture, but we request a feature so that we can.
    # if "32float" in texture_format:
    #     sampler_type = wgpu.SamplerBindingType.non_filtering
    #     texture_sample_type = wgpu.TextureSampleType.unfilterable_float

    # Bindings and layout
    bindings = [
        {"binding": 0, "resource": texture_view},
        {"binding": 1, "resource": sampler},
    ]
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "texture": {
                "sample_type": texture_sample_type,
                "view_dimension": wgpu.TextureViewDimension.d2,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "sampler": {
                "type": sampler_type,
            },
        },
    ]
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
        [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150],
        [150, 150, 150, 200, 200, 200],
        [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
    ]
    ref2 = [
        [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150],
        [150, 150, 150, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
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
