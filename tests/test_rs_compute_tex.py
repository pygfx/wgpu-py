import random
import ctypes

import pyshader
from pyshader import python2shader, ivec3
import wgpu.backends.rs  # noqa
import numpy as np

from pytest import skip
from testutils import run_tests, get_default_device
from testutils import can_use_wgpu_lib, can_use_vulkan_sdk
from renderutils import render_to_texture, render_to_screen  # noqa


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)

# %% 1D


def test_compute_tex_1d_rgba8uint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "1d rgba8ui"),
        tex2: ("texture", 1, "1d rgba8ui"),
    ):
        color = tex1.read(index.x)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.x, color)

    # Generate data
    nx, ny, nz, nc = 64, 1, 1, 4
    data1 = (ctypes.c_uint8 * nc * nx)()
    for x in range(nx):
        for c in range(nc):
            data1[x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.rgba8uint,
        wgpu.TextureDimension.d1,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_1d_rgba16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "1d rgba16i"),
        tex2: ("texture", 1, "1d rgba16i"),
    ):
        color = tex1.read(index.x)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.x, color)

    # Generate data
    nx, ny, nz, nc = 128, 1, 1, 4
    data1 = (ctypes.c_int16 * nc * nx)()
    for x in range(nx):
        for c in range(nc):
            data1[x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.rgba16sint,
        wgpu.TextureDimension.d1,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_1d_r32sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "1d r32i"),
        tex2: ("texture", 1, "1d r32i"),
    ):
        color = tex1.read(index.x)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.x, color)

    # Generate data
    nx, ny, nz, nc = 256, 1, 1, 1
    data1 = (ctypes.c_int32 * nc * nx)()
    for x in range(nx):
        for c in range(nc):
            data1[x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.r32sint,
        wgpu.TextureDimension.d1,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_1d_r32float():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "1d r32f"),
        tex2: ("texture", 1, "1d r32f"),
    ):
        color = tex1.read(index.x)
        color = vec4(color.x + f32(index.x), color.y + 1.0, color.z * 2.0, color.a)
        tex2.write(index.x, color)

    # Generate data
    nx, ny, nz, nc = 256, 1, 1, 1
    data1 = (ctypes.c_float * nc * nx)()
    for x in range(nx):
        for c in range(nc):
            data1[x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.r32float,
        wgpu.TextureDimension.d1,
        (nx, ny, nz, nc),
        data1,
    )


# %% 2D


def test_compute_tex_2d_rgba8uint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "2d rgba8ui"),
        tex2: ("texture", 1, "2d rgba8ui"),
    ):
        color = tex1.read(index.xy)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        # tex2.write(index.xy, color)  # is syntactic sugar for:
        stdlib.write(tex2, index.xy, color)

    # Generate data
    nx, ny, nz, nc = 64, 8, 1, 4
    data1 = (ctypes.c_uint8 * nc * nx * ny)()
    for y in range(ny):
        for x in range(nx):
            for c in range(nc):
                data1[y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.rgba8uint,
        wgpu.TextureDimension.d2,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_2d_rgba16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "2d rgba16i"),
        tex2: ("texture", 1, "2d rgba16i"),
    ):
        color = tex1.read(index.xy)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.xy, color)

    # Generate data
    nx, ny, nz, nc = 128, 8, 1, 4
    data1 = (ctypes.c_int16 * nc * nx * ny)()
    for y in range(ny):
        for x in range(nx):
            for c in range(nc):
                data1[y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.rgba16sint,
        wgpu.TextureDimension.d2,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_2d_r32sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "2d r32i"),
        tex2: ("texture", 1, "2d r32i"),
    ):
        color = tex1.read(index.xy)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.xy, color)

    # Generate data
    nx, ny, nz, nc = 256, 8, 1, 1
    data1 = (ctypes.c_int32 * nc * nx * ny)()
    for y in range(ny):
        for x in range(nx):
            for c in range(nc):
                data1[y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.r32sint,
        wgpu.TextureDimension.d2,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_2d_r32float():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "2d r32f"),
        tex2: ("texture", 1, "2d r32f"),
    ):
        color = tex1.read(index.xy)
        color = vec4(color.x + f32(index.x), color.y + 1.0, color.z * 2.0, color.a)
        tex2.write(index.xy, color)

    # Generate data
    nx, ny, nz, nc = 256, 8, 1, 1
    data1 = (ctypes.c_float * nc * nx * ny)()
    for y in range(ny):
        for x in range(nx):
            for c in range(nc):
                data1[y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.r32float,
        wgpu.TextureDimension.d2,
        (nx, ny, nz, nc),
        data1,
    )


# %% 3D


def test_compute_tex_3d_rgba8uint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "3d rgba8ui"),
        tex2: ("texture", 1, "3d rgba8ui"),
    ):
        color = tex1.read(index.xyz)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.xyz, color)

    # Generate data
    nx, ny, nz, nc = 64, 8, 6, 4
    data1 = (ctypes.c_uint8 * nc * nx * ny * nz)()
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nc):
                    data1[z][y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.rgba8uint,
        wgpu.TextureDimension.d3,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_3d_rgba16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "3d rgba16i"),
        tex2: ("texture", 1, "3d rgba16i"),
    ):
        color = tex1.read(index.xyz)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.xyz, color)

    # Generate data
    nx, ny, nz, nc = 128, 8, 6, 4
    data1 = (ctypes.c_int16 * nc * nx * ny * nz)()
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nc):
                    data1[z][y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.rgba16sint,
        wgpu.TextureDimension.d3,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_3d_r32sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "3d r32i"),
        tex2: ("texture", 1, "3d r32i"),
    ):
        color = tex1.read(index.xyz)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.xyz, color)

    # Generate data
    nx, ny, nz, nc = 256, 8, 6, 1
    data1 = (ctypes.c_int32 * nc * nx * ny * nz)()
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nc):
                    data1[z][y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.r32sint,
        wgpu.TextureDimension.d3,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_3d_r32float():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "3d r32f"),
        tex2: ("texture", 1, "3d r32f"),
    ):
        color = tex1.read(index.xyz)
        color = vec4(color.x + f32(index.x), color.y + 1.0, color.z * 2.0, color.a)
        tex2.write(index.xyz, color)

    # Generate data
    nx, ny, nz, nc = 64, 8, 6, 1
    data1 = (ctypes.c_float * nc * nx * ny * nz)()
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nc):
                    data1[z][y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.r32float,
        wgpu.TextureDimension.d3,
        (nx, ny, nz, nc),
        data1,
    )


# %%


def _compute_texture(compute_shader, texture_format, texture_dim, texture_size, data1):
    """
    Apply a computation on a texture and validate the result. The shader should:
    * Add the x-coordinate to the red channel.
    * Add 1 to the green channel.
    * Multiply the blue channel by 2.
    * The alpha channel must remain equal.
    """

    nx, ny, nz, nc = texture_size
    nbytes = ctypes.sizeof(data1)
    bpp = nbytes // (nx * ny * nz)  # bytes per pixel

    if can_use_vulkan_sdk:
        pyshader.dev.validate(compute_shader)

    device = get_default_device()
    cshader = device.create_shader_module(code=compute_shader)

    # Create textures and views
    texture1 = device.create_texture(
        size=(nx, ny, nz),
        dimension=texture_dim,
        format=texture_format,
        usage=wgpu.TextureUsage.STORAGE | wgpu.TextureUsage.COPY_DST,
    )
    texture2 = device.create_texture(
        size=(nx, ny, nz),
        dimension=texture_dim,
        format=texture_format,
        usage=wgpu.TextureUsage.STORAGE | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view1 = texture1.create_view()
    texture_view2 = texture2.create_view()

    # Create buffer that we need to upload the data
    buffer_usage = wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    buffer = device.create_buffer_with_data(data=data1, usage=buffer_usage)
    assert buffer.usage == buffer_usage

    # Define bindings
    # One can see here why we need 2 textures: one is readonly, one writeonly
    bindings = [
        {"binding": 0, "resource": texture_view1},
        {"binding": 1, "resource": texture_view2},
    ]
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "storage_texture": {
                "access": wgpu.StorageTextureAccess.read_only,  # <-
                "format": texture_format,
                "view_dimension": texture_dim,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "storage_texture": {
                "access": wgpu.StorageTextureAccess.write_only,  # <-
                "format": texture_format,
                "view_dimension": texture_dim,
            },
        },
    ]
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create a pipeline and run it
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )
    assert compute_pipeline.get_bind_group_layout(0) is bind_group_layout
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {
            "buffer": buffer,
            "offset": 0,
            "bytes_per_row": bpp * nx,
            "rows_per_image": ny,
        },
        {"texture": texture1, "mip_level": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.push_debug_group("foo")
    compute_pass.insert_debug_marker("setting pipeline")
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.insert_debug_marker("setting bind group")
    compute_pass.set_bind_group(
        0, bind_group, [], 0, 999999
    )  # last 2 elements not used
    compute_pass.insert_debug_marker("dispatch!")
    compute_pass.dispatch(nx, ny, nz)
    compute_pass.pop_debug_group()
    compute_pass.end_pass()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture2, "mip_level": 0, "origin": (0, 0, 0)},
        {
            "buffer": buffer,
            "offset": 0,
            "bytes_per_row": bpp * nx,
            "rows_per_image": ny,
        },
        (nx, ny, nz),
    )
    device.queue.submit([command_encoder.finish()])

    # Read the current data of the output buffer
    data2 = data1.__class__.from_buffer(device.queue.read_buffer(buffer))

    # Numpy arrays are easier to work with
    a1 = np.ctypeslib.as_array(data1).reshape(nz, ny, nx, nc)
    a2 = np.ctypeslib.as_array(data2).reshape(nz, ny, nx, nc)

    # Validate!
    for x in range(nx):
        assert np.all(a2[:, :, x, 0] == a1[:, :, x, 0] + x)
    if nc >= 2:
        assert np.all(a2[:, :, :, 1] == a1[:, :, :, 1] + 1)
    if nc >= 3:
        assert np.all(a2[:, :, :, 2] == a1[:, :, :, 2] * 2)
    if nc >= 4:
        assert np.all(a2[:, :, :, 3] == a1[:, :, :, 3])


if __name__ == "__main__":
    run_tests(globals())
