import random
import ctypes

import python_shader
from python_shader import python2shader, ivec3
import wgpu.backends.rs  # noqa

from pytest import mark
from testutils import can_use_wgpu_lib, get_default_device
import numpy as np

# todo: use "image" instead of ""texture" to communicate usage as storage?
# todo: maybe specify sampling in type description??


# %% 1D


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_1d_rgba8uint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex: ("texture", 0, "1d rgba8ui"),
    ):
        color = tex.read(index.x)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex.write(index.x, color)

    # Generate data
    nx, ny, nz, nc = 7, 1, 1, 4
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


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_1d_rg16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "1d rg16i"),
    ):
        color = tex.read(index.x)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex.write(index.x, color)

    # Generate data
    nx, ny, nz, nc = 7, 1, 1, 2
    data1 = (ctypes.c_int16 * nc * nx)()
    for x in range(nx):
        for c in range(nc):
            data1[x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.rg16sint,
        wgpu.TextureDimension.d1,
        (nx, ny, nz, nc),
        data1,
    )


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_1d_r16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "1d r16i"),
    ):
        color = tex.read(index.x)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex.write(index.x, color)

    # Generate data
    nx, ny, nz, nc = 7, 1, 1, 1
    data1 = (ctypes.c_int16 * nc * nx)()
    for x in range(nx):
        for c in range(nc):
            data1[x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.r16sint,
        wgpu.TextureDimension.d1,
        (nx, ny, nz, nc),
        data1,
    )


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_1d_r32float():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "1d r32f"),
    ):
        color = tex.read(index.x)
        color = vec4(color.x + f32(index.x), color.y + 1.0, color.z * 2.0, color.a)
        tex.write(index.x, color)

    # Generate data
    nx, ny, nz, nc = 7, 1, 1, 1
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


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_2d_rgba8uint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex: ("texture", 0, "2d rgba8ui"),
    ):
        color = tex.read(index.xy)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        # tex.write(index.xy, color)  # is syntactic sugar for:
        stdlib.write(tex, index.xy, color)

    # Generate data
    nx, ny, nz, nc = 7, 8, 1, 4
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


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_2d_rg16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "2d rg16i"),
    ):
        color = tex.read(index.xy)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex.write(index.xy, color)

    # Generate data
    nx, ny, nz, nc = 7, 8, 1, 2
    data1 = (ctypes.c_int16 * nc * nx * ny)()
    for y in range(ny):
        for x in range(nx):
            for c in range(nc):
                data1[y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.rg16sint,
        wgpu.TextureDimension.d2,
        (nx, ny, nz, nc),
        data1,
    )


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_2d_r16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "2d r16i"),
    ):
        color = tex.read(index.xy)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex.write(index.xy, color)

    # Generate data
    nx, ny, nz, nc = 7, 8, 1, 1
    data1 = (ctypes.c_int16 * nc * nx * ny)()
    for y in range(ny):
        for x in range(nx):
            for c in range(nc):
                data1[y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.r16sint,
        wgpu.TextureDimension.d2,
        (nx, ny, nz, nc),
        data1,
    )


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_2d_r32float():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "2d r32f"),
    ):
        color = tex.read(index.xy)
        color = vec4(color.x + f32(index.x), color.y + 1.0, color.z * 2.0, color.a)
        tex.write(index.xy, color)

    # Generate data
    nx, ny, nz, nc = 7, 8, 1, 1
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


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_3d_rgba8uint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex: ("texture", 0, "3d rgba8ui"),
    ):
        color = tex.read(index.xyz)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex.write(index.xyz, color)

    # Generate data
    nx, ny, nz, nc = 7, 8, 6, 4
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


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_3d_rg16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "3d rg16i"),
    ):
        color = tex.read(index.xyz)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex.write(index.xyz, color)

    # Generate data
    nx, ny, nz, nc = 7, 8, 6, 2
    data1 = (ctypes.c_int16 * nc * nx * ny * nz)()
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nc):
                    data1[z][y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.rg16sint,
        wgpu.TextureDimension.d3,
        (nx, ny, nz, nc),
        data1,
    )


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex_3d_r16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "3d r16i"),
    ):
        color = tex.read(index.xyz)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex.write(index.xyz, color)

    # Generate data
    nx, ny, nz, nc = 7, 8, 6, 1
    data1 = (ctypes.c_int16 * nc * nx * ny * nz)()
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nc):
                    data1[z][y][x][c] = random.randint(0, 20)

    # Compute and validate
    _compute_texture(
        compute_shader,
        wgpu.TextureFormat.r16sint,
        wgpu.TextureDimension.d3,
        (nx, ny, nz, nc),
        data1,
    )


def test_compute_tex_3d_r32float():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "3d r32f"),
    ):
        color = tex.read(index.xyz)
        color = vec4(color.x + f32(index.x), color.y + 1.0, color.z * 2.0, color.a)
        tex.write(index.xyz, color)

    # Generate data
    nx, ny, nz, nc = 7, 8, 6, 1
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

    python_shader.dev.validate(compute_shader)

    device = get_default_device()
    cshader = device.create_shader_module(code=compute_shader)

    # Create texture and view
    texture = device.create_texture(
        size=(nx, ny, nz),
        dimension=texture_dim,
        format=texture_format,
        usage=wgpu.TextureUsage.STORAGE
        | wgpu.TextureUsage.COPY_DST
        | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_default_view()

    # Create buffer that we need to upload the data
    buffer = device.create_buffer_mapped(
        size=nbytes,
        usage=wgpu.BufferUsage.MAP_READ
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST,
    )
    ctypes.memmove(buffer.mapping, data1, nbytes)
    buffer.unmap()

    # Define bindings
    bindings = [{"binding": 0, "resource": texture_view}]
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "type": wgpu.BindingType.storage_texture,
        }
    ]
    bind_group_layout = device.create_bind_group_layout(bindings=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, bindings=bindings)

    # Create a pipeline and run it
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute_stage={"module": cshader, "entry_point": "main"},
    )
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {"buffer": buffer, "offset": 0, "row_pitch": bpp * nx, "image_height": ny},
        {"texture": texture, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(
        0, bind_group, [], 0, 999999
    )  # last 2 elements not used
    compute_pass.dispatch(nx, ny, nz)
    compute_pass.end_pass()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {"buffer": buffer, "offset": 0, "row_pitch": bpp * nx, "image_height": ny},
        (nx, ny, nz),
    )
    device.default_queue.submit([command_encoder.finish()])

    # Read the current data of the output buffer
    array_uint8 = buffer.map_read()  # slow, can also be done async
    data2 = data1.__class__.from_buffer(array_uint8)

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
    test_compute_tex_1d_rgba8uint()
    test_compute_tex_1d_rg16sint()
    test_compute_tex_1d_r16sint()
    test_compute_tex_1d_r32float()

    test_compute_tex_2d_rgba8uint()
    test_compute_tex_2d_rg16sint()
    test_compute_tex_2d_r16sint()
    test_compute_tex_2d_r32float()

    test_compute_tex_3d_rgba8uint()
    test_compute_tex_3d_rg16sint()
    test_compute_tex_3d_r16sint()
    test_compute_tex_3d_r32float()
