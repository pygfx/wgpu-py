"""
Test render pipeline, by drawing a whole lot of orange squares ...
"""

import ctypes
import numpy as np
import sys

import wgpu.backends.rs  # noqa
from pytest import skip
from testutils import run_tests, can_use_wgpu_lib, is_ci, get_default_device
from renderutils import render_to_texture, render_to_screen  # noqa


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)
elif is_ci and sys.platform == "win32":
    skip("These tests fail on dx12 for some reason", allow_module_level=True)


default_vertex_shader = """
@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4<f32> {
    var positions: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
        vec3<f32>(-0.5, -0.5, 0.1),
        vec3<f32>(-0.5,  0.5, 0.1),
        vec3<f32>( 0.5, -0.5, 0.1),
        vec3<f32>( 0.5,  0.5, 0.1),
    );
    let p: vec3<f32> = positions[vertex_index];
    return vec4<f32>(p, 1.0);
}
"""


# %% Simple square


def test_render_orange_square():
    """Render an orange square and check that there is an orange square."""

    device = get_default_device()

    # NOTE: the 0.499 instead of 0.5 is to make sure the resulting value is 127.
    # With 0.5 some drivers would produce 127 and others 128.

    fragment_shader = """
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """
    shader_source = default_vertex_shader + fragment_shader

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
    """Render an orange square, using an index buffer."""

    device = get_default_device()

    fragment_shader = """
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """
    shader_source = default_vertex_shader + fragment_shader

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Index buffer
    indices = (ctypes.c_int32 * 6)(0, 1, 2, 2, 1, 3)
    ibo = device.create_buffer_with_data(
        data=indices,
        usage=wgpu.BufferUsage.INDEX,
    )

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
    """Render an orange square and check that there is an orange square."""

    device = get_default_device()

    fragment_shader = """
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """
    shader_source = default_vertex_shader + fragment_shader

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Buffer with draw parameters for indirect draw call
    params = (ctypes.c_int32 * 4)(4, 1, 0, 0)
    indirect_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.INDIRECT,
    )

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
    """Render an orange square, using an index buffer."""

    device = get_default_device()

    fragment_shader = """
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """
    shader_source = default_vertex_shader + fragment_shader

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Index buffer
    indices = (ctypes.c_int32 * 6)(0, 1, 2, 2, 1, 3)
    ibo = device.create_buffer_with_data(
        data=indices,
        usage=wgpu.BufferUsage.INDEX,
    )

    # Buffer with draw parameters for indirect draw call
    params = (ctypes.c_int32 * 5)(6, 1, 0, 0, 0)
    indirect_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.INDIRECT,
    )

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
    """Render an orange square, using a VBO."""

    device = get_default_device()

    shader_source = """
        @vertex
        fn vs_main(@location(0) pos : vec2<f32>) -> @builtin(position) vec4<f32> {
            return vec4<f32>(pos, 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Vertex buffer
    pos_data = (ctypes.c_float * 8)(-0.5, -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, +0.5)
    vbo = device.create_buffer_with_data(
        data=pos_data,
        usage=wgpu.BufferUsage.VERTEX,
    )

    # Vertex buffer views
    vbo_view = {
        "array_stride": 4 * 2,
        "step_mode": "vertex",
        "attributes": [
            {
                "format": wgpu.VertexFormat.float32x2,
                "offset": 0,
                "shader_location": 0,
            },
        ],
    }

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
    """Render an orange square on a blue background, testing the load_op."""

    device = get_default_device()

    fragment_shader = """
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """
    shader_source = default_vertex_shader + fragment_shader

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    ca = {
        "resolve_target": None,
        "clear_value": (0, 0, 0.8, 1),
        "load_op": wgpu.LoadOp.clear,
        "store_op": wgpu.StoreOp.store,
    }

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
    """Render an orange square on a blue background, testing the LoadOp.load,
    though in this case the result is the same as the normal square test.
    """

    device = get_default_device()

    fragment_shader = """
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """
    shader_source = default_vertex_shader + fragment_shader

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    ca = {
        "resolve_target": None,
        "load_op": wgpu.LoadOp.load,
        "store_op": wgpu.StoreOp.store,
    }

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
    # render_to_screen(*render_args, color_attachment=ca)
    a = render_to_texture(*render_args, size=(64, 64), color_attachment=ca)

    # Check the background
    bg = a.copy()
    bg[16:-16, 16:-16, :] = 0
    # assert np.all(bg == 0)
    # Actually, it seems unpredictable what the bg is if we dont clear it?

    # Check the square
    sq = a[16:-16, 16:-16, :]
    assert np.all(sq[:, :, 0] == 255)  # red
    assert np.all(sq[:, :, 1] == 127)  # green
    assert np.all(sq[:, :, 2] == 0)  # blue
    assert np.all(sq[:, :, 3] == 255)  # alpha


# %% Viewport and stencil


def test_render_orange_square_viewport():
    """Render an orange square, in a sub-viewport of the rendered area."""

    device = get_default_device()

    fragment_shader = """
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """
    shader_source = default_vertex_shader + fragment_shader

    def cb(renderpass):
        renderpass.set_viewport(10, 20, 32, 32, 0, 1)

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
    """Render an orange square, but scissor half the screen away."""

    device = get_default_device()

    fragment_shader = """
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """
    shader_source = default_vertex_shader + fragment_shader

    def cb(renderpass):
        renderpass.set_scissor_rect(0, 0, 32, 32)
        # Alse set blend color. Does not change outout, but covers the call.
        renderpass.set_blend_constant((0, 0, 0, 1))

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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


def test_render_orange_square_depth16unorm():
    """Render an orange square, but disable half of it using a depth test using 16 bits."""
    _render_orange_square_depth(wgpu.TextureFormat.depth16unorm)


def test_render_orange_square_depth24plus_stencil8():
    """Render an orange square, but disable half of it using a depth test using 24 bits."""
    _render_orange_square_depth(wgpu.TextureFormat.depth24plus_stencil8)


def test_render_orange_square_depth32float():
    """Render an orange square, but disable half of it using a depth test using 32 bits."""
    _render_orange_square_depth(wgpu.TextureFormat.depth32float)


def _render_orange_square_depth(depth_stencil_tex_format):
    device = get_default_device()

    shader_source = """
        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4<f32> {
            var positions: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
                vec3<f32>(-0.5, -0.5, 0.0),
                vec3<f32>(-0.5,  0.5, 0.0),
                vec3<f32>( 0.5, -0.5, 0.2),
                vec3<f32>( 0.5,  0.5, 0.2),
            );
            let p: vec3<f32> = positions[vertex_index];
            return vec4<f32>(p, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """

    def cb(renderpass):
        renderpass.set_stencil_reference(42)

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Create dept-stencil texture
    depth_stencil_texture = device.create_texture(
        size=(64, 64, 1),  # when rendering to texture
        # size=(640, 480, 1),  # when rendering to screen
        dimension=wgpu.TextureDimension.d2,
        format=depth_stencil_tex_format,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )

    depth_stencil_state = dict(
        format=depth_stencil_tex_format,
        depth_write_enabled=True,
        depth_compare=wgpu.CompareFunction.less_equal,
        # stencil_front={
        #     "compare": wgpu.CompareFunction.equal,
        #     "fail_op": wgpu.StencilOperation.keep,
        #     "depth_fail_op": wgpu.StencilOperation.keep,
        #     "pass_op": wgpu.StencilOperation.keep,
        # },
        # stencil_back={
        #     "compare": wgpu.CompareFunction.equal,
        #     "fail_op": wgpu.StencilOperation.keep,
        #     "depth_fail_op": wgpu.StencilOperation.keep,
        #     "pass_op": wgpu.StencilOperation.keep,
        # },
        stencil_read_mask=0,
        stencil_write_mask=0,
        depth_bias=0,
        depth_bias_slope_scale=0.0,
        depth_bias_clamp=0.0,
    )

    depth_stencil_attachment = dict(
        view=depth_stencil_texture.create_view(),
        depth_clear_value=0.1,
        depth_load_op=wgpu.LoadOp.clear,
        depth_store_op=wgpu.StoreOp.store,
        stencil_load_op=wgpu.LoadOp.load,
        stencil_store_op=wgpu.StoreOp.store,
    )

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
    """Render four orange dots and check that there are four orange square dots."""

    device = get_default_device()

    shader_source = """
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            //@builtin(pointSize) point_size: f32,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
            var positions: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
                vec3<f32>(-0.5, -0.5, 0.0),
                vec3<f32>(-0.5,  0.5, 0.0),
                vec3<f32>( 0.5, -0.5, 0.2),
                vec3<f32>( 0.5,  0.5, 0.2),
            );
            var out: VertexOutput;
            out.position =  vec4<f32>(positions[vertex_index], 1.0);
            //out.point_size = 16.0;
            return out;
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.499, 0.0, 1.0);
        }
    """

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
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
    # Ideally we'd want to set the point_size (gl_PointSize) to 16 but
    # this is not supported in WGPU, see https://github.com/gpuweb/gpuweb/issues/332
    # So our points are 1px
    for dot in (
        a[15:16, 15:16, :],
        a[15:16, 47:48, :],
        a[47:48, 15:16, :],
        a[47:48, 47:48, :],
    ):
        assert np.all(dot[:, :, 0] == 255)  # red
        assert np.all(dot[:, :, 1] == 127)  # green
        assert np.all(dot[:, :, 2] == 0)  # blue
        assert np.all(dot[:, :, 3] == 255)  # alpha


if __name__ == "__main__":
    run_tests(globals())
