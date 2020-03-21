import random
import ctypes
from ctypes import c_int32, c_ubyte

import python_shader
from python_shader import python2shader, Array, i32, ivec3, u8
import wgpu.backends.rs  # noqa
from wgpu.utils import compute_with_buffers

from pytest import mark
from testutils import can_use_wgpu_lib, iters_equal


@mark.skipif(not can_use_wgpu_lib, reason="Cannot use wgpu lib")
def test_compute_tex2():

    # todo: use "image" instead of ""texture" to communicate usage as storage?
    @python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", ivec3), tex: ("texture", 0, "2d"),
    ):
        color = imageLoad(tex, ivec2(index.xy))
        # color = vec4(color.x, color.y *2.0, color.z * 3.0, color.a * 4.0)
        color = vec4(0.5, 0.5, 0.5, 0.5)
        imageStore(tex, ivec2(index.xy), color)

    python_shader.dev.validate(compute_shader)

    # Generate data
    spirv_format = wgpu.TextureFormat.rgba8uint
    component_type = ctypes.c_uint8
    ny, nx, nc = 7, 8, 4
    data1 = (component_type * nc * nx * ny)()
    for y in range(ny):
        for x in range(nx):
            for c in range(nc):
                data1[y][x][c] = y * 10 + x
    nbytes = ctypes.sizeof(data1)
    bpp = nbytes // (nx * ny)  # bytes per pixel
    # todo: docstrings in wgpu-rs say that BufferCopyView.row_pitch must be multiple of 256

    # Create a device and compile the shader
    adapter = wgpu.request_adapter(power_preference="high-performance")
    device = adapter.request_device(extensions=[], limits={})
    cshader = device.create_shader_module(code=compute_shader)

    # Create texture and view
    texture = device.create_texture(
        size=(nx, ny, 1),
        dimension=wgpu.TextureDimension.d2,
        format=spirv_format,
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
        (nx, ny, 1),
    )
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(
        0, bind_group, [], 0, 999999
    )  # last 2 elements not used
    compute_pass.dispatch(nx, ny, 1)
    compute_pass.end_pass()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {"buffer": buffer, "offset": 0, "row_pitch": bpp * nx, "image_height": ny},
        (nx, ny, 1),
    )
    device.default_queue.submit([command_encoder.finish()])

    # Read the current data of the output buffer
    array_uint8 = buffer.map_read()  # slow, can also be done async
    data2 = data1.__class__.from_buffer(array_uint8)

    flat_t = component_type * (nx * ny * nc)
    data3 = list(flat_t.from_buffer(data1))
    data4 = list(flat_t.from_buffer(data2))

    print(data3)
    print(data4)

    for y in range(ny):
        for x in range(nx):
            assert data1[y][x][0] == data2[y][x][0]
            assert data1[y][x][1] == data2[y][x][1]  # - 1
            assert data1[y][x][2] == data2[y][x][2]  # - 2
            assert data1[y][x][3] == data2[y][x][3]  # - 3


if __name__ == "__main__":
    test_compute_tex2()
