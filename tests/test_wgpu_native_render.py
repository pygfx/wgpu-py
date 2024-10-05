"""
Test render pipeline, by drawing a whole lot of orange squares ...
"""

import ctypes

import numpy as np
import sys

import pytest

import wgpu
from pytest import skip
from testutils import run_tests, can_use_wgpu_lib, is_ci, get_default_device
from renderutils import render_to_texture
from renderutils import render_to_screen  # noqa: F401 - sometimes used for debugging


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)
elif is_ci and sys.platform == "win32":
    skip("These tests fail on dx12 for some reason", allow_module_level=True)


DEFAULT_SHADER = """
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

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.499, 0.0, 1.0);
}
"""

DEPTH_STENCIL_TEX_FORMATS = [
    wgpu.TextureFormat.depth16unorm,
    wgpu.TextureFormat.depth24plus_stencil8,
    wgpu.TextureFormat.depth32float,
]

# %% Simple square


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_square(use_render_bundle):
    """Render an orange square and check that there is an orange square."""

    device = get_default_device()

    # NOTE: the 0.499 instead of 0.5 is to make sure the resulting value is 127.
    # With 0.5 some drivers would produce 127 and others 128.

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Render
    render_args = device, DEFAULT_SHADER, pipeline_layout, bind_group
    # render_to_screen(*render_args)
    a = render_to_texture(
        *render_args, size=(64, 64), use_render_bundle=use_render_bundle
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


# %% Variations


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_square_indexed(use_render_bundle):
    """Render an orange square, using an index buffer."""

    device = get_default_device()

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
    render_args = device, DEFAULT_SHADER, pipeline_layout, bind_group
    # render_to_screen(*render_args, topology=wgpu.PrimitiveTopology.triangle_list, ibo=ibo)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        topology=wgpu.PrimitiveTopology.triangle_list,
        ibo=ibo,
        use_render_bundle=use_render_bundle,
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


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_square_indirect(use_render_bundle):
    """Render an orange square and check that there is an orange square."""

    device = get_default_device()

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
    render_args = device, DEFAULT_SHADER, pipeline_layout, bind_group
    # render_to_screen(*render_args, indirect_buffer=indirect_buffer)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        indirect_buffer=indirect_buffer,
        use_render_bundle=use_render_bundle,
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


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_square_indexed_indirect(use_render_bundle):
    """Render an orange square, using an index buffer."""

    device = get_default_device()

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
    render_args = device, DEFAULT_SHADER, pipeline_layout, bind_group
    # render_to_screen(*render_args, topology=wgpu.PrimitiveTopology.triangle_list, ibo=ibo, indirect_buffer=indirect_buffer)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        topology=wgpu.PrimitiveTopology.triangle_list,
        ibo=ibo,
        indirect_buffer=indirect_buffer,
        use_render_bundle=use_render_bundle,
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


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_square_vbo(use_render_bundle):
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
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        vbos=[vbo],
        vbo_views=[vbo_view],
        use_render_bundle=use_render_bundle,
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


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_square_color_attachment1(use_render_bundle):
    """Render an orange square on a blue background, testing the load_op."""

    device = get_default_device()

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
    render_args = device, DEFAULT_SHADER, pipeline_layout, bind_group
    # render_to_screen(*render_args, color_attachment=ca)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        color_attachment=ca,
        use_render_bundle=use_render_bundle,
    )

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


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_square_color_attachment2(use_render_bundle):
    """Render an orange square on a blue background, testing the LoadOp.load,
    though in this case the result is the same as the normal square test.
    """

    device = get_default_device()

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    ca = {
        "resolve_target": None,
        "load_op": wgpu.LoadOp.load,
        "store_op": wgpu.StoreOp.store,
    }

    # Render
    render_args = device, DEFAULT_SHADER, pipeline_layout, bind_group
    # render_to_screen(*render_args, color_attachment=ca)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        color_attachment=ca,
        use_render_bundle=use_render_bundle,
    )

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


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_square_viewport(use_render_bundle):
    """Render an orange square, in a sub-viewport of the rendered area."""

    device = get_default_device()

    def cb(renderpass):
        renderpass.set_viewport(10, 20, 32, 32, 0, 1)

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Render
    render_args = device, DEFAULT_SHADER, pipeline_layout, bind_group
    # render_to_screen(*render_args, renderpass_callback=cb)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        renderpass_callback=cb,
        use_render_bundle=use_render_bundle,
    )

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


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_square_scissor(use_render_bundle):
    """Render an orange square, but scissor half the screen away."""

    device = get_default_device()

    def cb(renderpass):
        renderpass.set_scissor_rect(0, 0, 32, 32)
        # Else set blend color. Does not change output, but covers the call.
        renderpass.set_blend_constant((0, 0, 0, 1))

    # Bindings and layout
    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    # Render
    render_args = device, DEFAULT_SHADER, pipeline_layout, bind_group
    # render_to_screen(*render_args, renderpass_callback=cb)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        renderpass_callback=cb,
        use_render_bundle=use_render_bundle,
    )

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


@pytest.mark.parametrize("use_render_bundle", [True, False])
@pytest.mark.parametrize("depth_stencil_tex_format", DEPTH_STENCIL_TEX_FORMATS)
def test_render_orange_square_with_depth(depth_stencil_tex_format, use_render_bundle):
    """Render an orange square, but disable half of it using a depth test using 16 bits."""
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

    # Create depth-stencil texture
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
    )

    depth_stencil_attachment = dict(
        view=depth_stencil_texture.create_view(),
        depth_clear_value=0.1,
        depth_load_op=wgpu.LoadOp.clear,
        depth_store_op=wgpu.StoreOp.store,
    )

    if "stencil" in depth_stencil_tex_format:
        depth_stencil_attachment["stencil_load_op"] = wgpu.LoadOp.load
        depth_stencil_attachment["stencil_store_op"] = wgpu.StoreOp.store
        # The default values of depth_stencil_state are fine.

    # Render
    render_args = device, shader_source, pipeline_layout, bind_group
    # render_to_screen(*render_args, renderpass_callback=cb, depth_stencil_state=depth_stencil_state, depth_stencil_attachment=depth_stencil_attachment)
    a = render_to_texture(
        *render_args,
        size=(64, 64),
        renderpass_callback=cb,
        depth_stencil_state=depth_stencil_state,
        depth_stencil_attachment=depth_stencil_attachment,
        use_render_bundle=use_render_bundle,
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


@pytest.mark.parametrize("use_render_bundle", [True, False])
def test_render_orange_dots(use_render_bundle):
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
    a = render_to_texture(
        *render_args, size=(64, 64), topology=top, use_render_bundle=use_render_bundle
    )

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


@pytest.mark.parametrize("depth_stencil_tex_format", DEPTH_STENCIL_TEX_FORMATS)
def test_stencil_depth_warning(depth_stencil_tex_format, monkeypatch):
    # Verify that when you use stencil_load_op or stencil_store_op, and there is
    # no stencil, you'll get a warning.  But only once.

    # The only way we can see warnings is by monkeypatching wgpu.logger.warning
    warnings = []
    monkeypatch.setattr(wgpu.logger, "warning", lambda msg: warnings.append(msg))

    # We need a new device for each run, since each device keeps track of which warnings
    # it's issued.
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    # Create depth-stencil texture
    depth_stencil_texture = device.create_texture(
        size=(64, 64, 1),  # when rendering to texture
        format=depth_stencil_tex_format,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )

    depth_stencil_attachment = dict(
        view=depth_stencil_texture.create_view(),
        depth_clear_value=0.1,
        depth_load_op=wgpu.LoadOp.clear,
        depth_store_op=wgpu.StoreOp.store,
        stencil_load_op=wgpu.LoadOp.clear,
        stencil_store_op=wgpu.StoreOp.store,
    )

    command_encoder = device.create_command_encoder()
    # This call to begin_render_pass may create warnings.
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[], depth_stencil_attachment=depth_stencil_attachment
    )
    render_pass.end()

    if "stencil" in depth_stencil_tex_format:
        assert not warnings
    else:
        assert len(warnings) == 2
        warnings.sort()
        assert "stencil_load_op" in warnings[0]
        assert "stencil_store_op" in warnings[1]
        warnings.clear()

    # The warnings are issued only once.
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[], depth_stencil_attachment=depth_stencil_attachment
    )
    render_pass.end()
    assert not warnings


if __name__ == "__main__":
    run_tests(globals())
