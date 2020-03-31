"""
Test render pipeline, by drawing a whole lot of orange squares ...
"""

import ctypes
import numpy as np

from python_shader import python2shader, f32, vec2, vec4, i32
from python_shader import RES_INPUT, RES_OUTPUT
import wgpu.backends.rs  # noqa
from pytest import skip
from testutils import can_use_wgpu_lib, get_default_device
from renderutils import render_to_texture, render_to_screen  # noqa


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


def test_render_orange_square_indexed():
    """ Render an orange square, using an index buffer.
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

    # Index buffer
    indices = (ctypes.c_int32 * 6)(0, 1, 2, 2, 1, 3)
    ibo = device.create_buffer_mapped(
        size=ctypes.sizeof(indices),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.MAP_WRITE,
    )
    ctypes.memmove(ibo.mapping, indices, ctypes.sizeof(indices))
    ibo.unmap()

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args, topology=wgpu.PrimitiveTopology.triangle_list, ibo=ibo)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        topology=wgpu.PrimitiveTopology.triangle_list,
        ibo=ibo,
    )

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


def test_render_orange_square_vbo():
    """ Render an orange square, using a VBO.
    """

    device = get_default_device()

    @python2shader
    def vertex_shader(
        pos_in: (RES_INPUT, 0, vec2),
        pos: (RES_OUTPUT, "Position", vec4),
        tcoord: (RES_OUTPUT, 0, vec2),
    ):
        pos = vec4(pos_in, 0.0, 1.0)  # noqa

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.5, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(bindings=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, bindings=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Vertex buffer
    pos_data = (ctypes.c_float * 8)(-0.5, -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, +0.5)
    vbo = device.create_buffer_mapped(
        size=ctypes.sizeof(pos_data),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.MAP_WRITE,
    )
    ctypes.memmove(vbo.mapping, pos_data, ctypes.sizeof(pos_data))
    vbo.unmap()

    # Vertex buffer views
    vbo_view = {
        "array_stride": 4 * 2,
        "stepmode": "vertex",
        "attributes": [
            {"format": wgpu.VertexFormat.float2, "offset": 0, "shader_location": 0,},
        ],
    }

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args, vbos=[vbo], vbo_views=[vbo_view])
    a = render_to_texture(*render_args, size=(64, 64), vbos=[vbo], vbo_views=[vbo_view])

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


def test_render_orange_square_color_attachement1():
    """ Render an orange square on a blue background, testing the load_value.
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

    ca = {
        "resolve_target": None,
        "load_value": (0, 0, 0.8, 1),  # LoadOp.load or color
        "store_op": wgpu.StoreOp.store,
    }

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args, color_attachement=ca)
    a = render_to_texture(*render_args, size=(64, 64), color_attachement=ca)

    # Check the blue background
    assert np.all(a[:16, :16, 2] == 204)
    assert np.all(a[:16, -16:, 2] == 204)
    assert np.all(a[-16:, :16, 2] == 204)
    assert np.all(a[-16:, -16:, 2] == 204)

    # Check the square
    sq = a[16:-16, 16:-16, :]
    assert np.all(sq[:, :, 0] == 255)  # red
    assert np.all(sq[:, :, 1] == 127)  # green
    assert np.all(sq[:, :, 2] == 0)  # blue
    assert np.all(sq[:, :, 3] == 255)  # alpha


def test_render_orange_square_color_attachement2():
    """ Render an orange square on a blue background, testing the LoadOp.load,
    though in this case the result is the same as the normal square test.
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

    ca = {
        "resolve_target": None,
        "load_value": wgpu.LoadOp.load,  # LoadOp.load or color
        "store_op": wgpu.StoreOp.store,
    }

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args, color_attachement=ca)
    a = render_to_texture(*render_args, size=(64, 64), color_attachement=ca)

    # Check the background
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


if __name__ == "__main__":
    test_render_orange_square()
    test_render_orange_square_vbo()
    test_render_orange_square_indexed()
    test_render_orange_dots()
    test_render_orange_square_color_attachement1()
    test_render_orange_square_color_attachement2()
