"""
Test render pipeline, by drawing a whole lot of orange squares ...
"""

import ctypes
import numpy as np

from pyshader import python2shader, f32, vec2, vec4, i32
from pyshader import RES_INPUT, RES_OUTPUT
import wgpu.backends.rs  # noqa
from pytest import skip, raises
from testutils import run_tests, can_use_wgpu_lib, get_default_device
from renderutils import render_to_texture, render_to_screen  # noqa


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


@python2shader
def vertex_shader(
    index: (RES_INPUT, "VertexId", i32), pos: (RES_OUTPUT, "Position", vec4),
):
    positions = [
        vec3(-0.5, -0.5, 0.1),
        vec3(-0.5, +0.5, 0.1),
        vec3(+0.5, -0.5, 0.1),
        vec3(+0.5, +0.5, 0.1),
    ]
    p = positions[index]
    pos = vec4(p, 1.0)  # noqa


# %% Simple square


def test_render_orange_square():
    """ Render an orange square and check that there is an orange square.
    """

    device = get_default_device()

    # NOTE: the 0.499 instead of 0.5 is to make sure the resulting value is 127.
    # With 0.5 some drivers would produce 127 and others 128.

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
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


# %% Variations


def test_render_orange_square_indexed():
    """ Render an orange square, using an index buffer.
    """

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Index buffer
    indices = (ctypes.c_int32 * 6)(0, 1, 2, 2, 1, 3)
    ibo = device.create_buffer_with_data(
        data=indices, usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.MAP_WRITE,
    )

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


def test_render_orange_square_indirect():
    """ Render an orange square and check that there is an orange square.
    """

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Buffer with draw parameters for indirect draw call
    params = (ctypes.c_int32 * 4)(4, 1, 0, 0)
    indirect_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.INDIRECT,
    )

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args, indirect_buffer=indirect_buffer)
    a = render_to_texture(*render_args, size=(64, 64), indirect_buffer=indirect_buffer)

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


def test_render_orange_square_indexed_indirect():
    """ Render an orange square, using an index buffer.
    """

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Index buffer
    indices = (ctypes.c_int32 * 6)(0, 1, 2, 2, 1, 3)
    ibo = device.create_buffer_with_data(
        data=indices, usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.MAP_WRITE,
    )

    # Buffer with draw parameters for indirect draw call
    params = (ctypes.c_int32 * 5)(6, 1, 0, 0, 0)
    indirect_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.INDIRECT,
    )

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args, topology=wgpu.PrimitiveTopology.triangle_list, ibo=ibo, indirect_buffer=indirect_buffer)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        topology=wgpu.PrimitiveTopology.triangle_list,
        ibo=ibo,
        indirect_buffer=indirect_buffer,
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
        pos_in: (RES_INPUT, 0, vec2), pos: (RES_OUTPUT, "Position", vec4),
    ):
        pos = vec4(pos_in, 0.0, 1.0)  # noqa

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Vertex buffer
    pos_data = (ctypes.c_float * 8)(-0.5, -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, +0.5)
    vbo = device.create_buffer_with_data(
        data=pos_data, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.MAP_WRITE,
    )

    # Vertex buffer views
    vbo_view = {
        "array_stride": 4 * 2,
        "step_mode": "vertex",
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


def test_render_orange_square_color_attachment1():
    """ Render an orange square on a blue background, testing the load_value.
    """

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
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
    # render_to_screen(*render_args, color_attachment=ca)
    a = render_to_texture(*render_args, size=(64, 64), color_attachment=ca)

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


def test_render_orange_square_color_attachment2():
    """ Render an orange square on a blue background, testing the LoadOp.load,
    though in this case the result is the same as the normal square test.
    """

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
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
    # render_to_screen(*render_args, color_attachment=ca)
    a = render_to_texture(*render_args, size=(64, 64), color_attachment=ca)

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


# %% Viewport and stencil


def test_render_orange_square_viewport():
    """ Render an orange square, in a sub-viewport of the rendered area.
    """

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    def cb(renderpass):
        renderpass.set_viewport(10, 20, 32, 32, 0, 100)

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Fiddled in a small test to covers the raising of an exception
    with raises(TypeError):
        device.create_bind_group(
            layout=bind_group_layout, entries=[{"resource": device}]
        )

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args, renderpass_callback=cb)
    a = render_to_texture(*render_args, size=(64, 64), renderpass_callback=cb)

    # Check that the background is all zero
    bg = a.copy()
    bg[20 + 8 : 52 - 8, 10 + 8 : 42 - 8, :] = 0
    assert np.all(bg == 0)

    # Check the square
    sq = a[20 + 8 : 52 - 8, 10 + 8 : 42 - 8, :]
    assert np.all(sq[:, :, 0] == 255)  # red
    assert np.all(sq[:, :, 1] == 127)  # green
    assert np.all(sq[:, :, 2] == 0)  # blue
    assert np.all(sq[:, :, 3] == 255)  # alpha


def test_render_orange_square_scissor():
    """ Render an orange square, but scissor half the screen away.
    """

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    def cb(renderpass):
        renderpass.set_scissor_rect(0, 0, 32, 32)
        # Alse set blend color. Does not change outout, but covers the call.
        renderpass.set_blend_color((0, 0, 0, 1))

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Render
    render_args = device, vertex_shader, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args, renderpass_callback=cb)
    a = render_to_texture(*render_args, size=(64, 64), renderpass_callback=cb)

    # Check that the background is all zero
    bg = a.copy()
    bg[16:32, 16:32, :] = 0
    assert np.all(bg == 0)

    # Check the square
    sq = a[16:32, 16:32, :]
    assert np.all(sq[:, :, 0] == 255)  # red
    assert np.all(sq[:, :, 1] == 127)  # green
    assert np.all(sq[:, :, 2] == 0)  # blue
    assert np.all(sq[:, :, 3] == 255)  # alpha


def test_render_orange_square_depth():
    """ Render an orange square, but disable half of it using a depth test.
    """

    device = get_default_device()

    @python2shader
    def vertex_shader2(
        index: (RES_INPUT, "VertexId", i32), pos: (RES_OUTPUT, "Position", vec4),
    ):
        positions = [
            vec3(-0.5, -0.5, 0.0),
            vec3(-0.5, +0.5, 0.0),
            vec3(+0.5, -0.5, 0.2),
            vec3(+0.5, +0.5, 0.2),
        ]
        pos = vec4(positions[index], 1.0)  # noqa

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    def cb(renderpass):
        renderpass.set_stencil_reference(42)

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Create dept-stencil texture
    depth_stencil_texture = device.create_texture(
        size=(64, 64, 1),  # when rendering to texture
        # size=(640, 480, 1),  # when rendering to screen
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.depth24plus_stencil8,
        usage=wgpu.TextureUsage.OUTPUT_ATTACHMENT,
    )

    depth_stencil_state = dict(
        format=wgpu.TextureFormat.depth24plus_stencil8,
        depth_write_enabled=True,
        depth_compare=wgpu.CompareFunction.less_equal,
        stencil_front={
            "compare": wgpu.CompareFunction.equal,
            "fail_op": wgpu.StencilOperation.keep,
            "depth_fail_op": wgpu.StencilOperation.keep,
            "pass_op": wgpu.StencilOperation.keep,
        },
        stencil_back={
            "compare": wgpu.CompareFunction.equal,
            "fail_op": wgpu.StencilOperation.keep,
            "depth_fail_op": wgpu.StencilOperation.keep,
            "pass_op": wgpu.StencilOperation.keep,
        },
        stencil_read_mask=0,
        stencil_write_mask=0,
    )

    depth_stencil_attachment = dict(
        attachment=depth_stencil_texture.create_view(),
        depth_load_value=0.1,
        depth_store_op=wgpu.StoreOp.store,
        stencil_load_value=wgpu.LoadOp.load,
        stencil_store_op=wgpu.StoreOp.store,
    )

    # Render
    render_args = device, vertex_shader2, fragment_shader, pipeline_layout, bind_group
    # render_to_screen(*render_args, renderpass_callback=cb, depth_stencil_state=depth_stencil_state, depth_stencil_attachment=depth_stencil_attachment)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        renderpass_callback=cb,
        depth_stencil_state=depth_stencil_state,
        depth_stencil_attachment=depth_stencil_attachment,
    )

    # Check that the background is all zero
    bg = a.copy()
    bg[16:-16, 16:32, :] = 0
    assert np.all(bg == 0)

    # Check the square
    sq = a[16:-16, 16:32, :]
    assert np.all(sq[:, :, 0] == 255)  # red
    assert np.all(sq[:, :, 1] == 127)  # green
    assert np.all(sq[:, :, 2] == 0)  # blue
    assert np.all(sq[:, :, 3] == 255)  # alpha


# %% Not squares


def test_render_orange_dots():
    """ Render four orange dots and check that there are four orange square dots.
    """

    @python2shader
    def vertex_shader(
        index: (RES_INPUT, "VertexId", i32),
        pos: (RES_OUTPUT, "Position", vec4),
        pointsize: (RES_OUTPUT, "PointSize", f32),
    ):
        positions = [
            vec2(-0.5, -0.5),
            vec2(-0.5, +0.5),
            vec2(+0.5, -0.5),
            vec2(+0.5, +0.5),
        ]
        p = positions[index]
        pos = vec4(p, 0.0, 1.0)  # noqa
        pointsize = 16.0  # noqa

    device = get_default_device()

    @python2shader
    def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
        out_color = vec4(1.0, 0.499, 0.0, 1.0)  # noqa

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
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
    run_tests(globals())
