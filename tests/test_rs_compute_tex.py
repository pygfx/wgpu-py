import random
import ctypes

import python_shader
from python_shader import python2shader, ivec3
import wgpu.backends.rs  # noqa
import numpy as np

from pytest import skip
from testutils import can_use_wgpu_lib, get_default_device, can_use_vulkan_sdk
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


def test_compute_tex_1d_rg16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "1d rg16i"),
        tex2: ("texture", 1, "1d rg16i"),
    ):
        color = tex1.read(index.x)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.x, color)

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


def test_compute_tex_1d_r16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "1d r16i"),
        tex2: ("texture", 1, "1d r16i"),
    ):
        color = tex1.read(index.x)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.x, color)

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


def test_compute_tex_2d_rg16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "2d rg16i"),
        tex2: ("texture", 1, "2d rg16i"),
    ):
        color = tex1.read(index.xy)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.xy, color)

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


def test_compute_tex_2d_r16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "2d r16i"),
        tex2: ("texture", 1, "2d r16i"),
    ):
        color = tex1.read(index.xy)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.xy, color)

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


def test_compute_tex_3d_rg16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "3d rg16i"),
        tex2: ("texture", 1, "3d rg16i"),
    ):
        color = tex1.read(index.xyz)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.xyz, color)

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


def test_compute_tex_3d_r16sint():
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "3d r16i"),
        tex2: ("texture", 1, "3d r16i"),
    ):
        color = tex1.read(index.xyz)
        color = ivec4(color.x + index.x, color.y + 1, color.z * 2, color.a)
        tex2.write(index.xyz, color)

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
        index: ("input", "GlobalInvocationId", ivec3),
        tex1: ("texture", 0, "3d r32f"),
        tex2: ("texture", 1, "3d r32f"),
    ):
        color = tex1.read(index.xyz)
        color = vec4(color.x + f32(index.x), color.y + 1.0, color.z * 2.0, color.a)
        tex2.write(index.xyz, color)

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

    if can_use_vulkan_sdk:
        python_shader.dev.validate(compute_shader)

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
    texture_view1 = texture1.create_default_view()
    texture_view2 = texture2.create_default_view()

    # Determine texture component type from the format
    if texture_format.endswith(("norm", "float")):
        texture_component_type = wgpu.TextureComponentType.float
    elif "uint" in texture_format:
        texture_component_type = wgpu.TextureComponentType.uint
    else:
        texture_component_type = wgpu.TextureComponentType.sint

    # Create buffer that we need to upload the data
    buffer_usage = (
        wgpu.BufferUsage.MAP_READ
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST
    )
    buffer = device.create_buffer_mapped(size=nbytes, usage=buffer_usage)
    ctypes.memmove(buffer.mapping, data1, nbytes)
    buffer.unmap()
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
            "type": wgpu.BindingType.readonly_storage_texture,  # <-
            "view_dimension": wgpu.TextureViewDimension.d2,
            "storage_texture_format": texture_format,
            "texture_component_type": texture_component_type,
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "type": wgpu.BindingType.writeonly_storage_texture,  # <-
            "view_dimension": wgpu.TextureViewDimension.d2,
            "storage_texture_format": texture_format,
            "texture_component_type": texture_component_type,
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
        compute_stage={"module": cshader, "entry_point": "main"},
    )
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {
            "buffer": buffer,
            "offset": 0,
            "bytes_per_row": bpp * nx,
            "rows_per_image": ny,
        },
        {"texture": texture1, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
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
        {"texture": texture2, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {
            "buffer": buffer,
            "offset": 0,
            "bytes_per_row": bpp * nx,
            "rows_per_image": ny,
        },
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
