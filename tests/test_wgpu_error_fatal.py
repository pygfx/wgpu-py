import wgpu
from testutils import run_tests
from renderutils import render_to_texture
from pytest import raises


# from test_wgpu_native_render.py
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

# from test_wgpu_native_errors.py
dedent = lambda s: s.replace("\n        ", "\n").strip()


def test_unreachable():
    # panicked at ~\naga\src\back\spv\block.rs:2401:56:
    # https://github.com/gfx-rs/wgpu/issues/4517
    # real world occurance: https://www.shadertoy.com/view/NsffD2
    # unreachable code due to a pointer and swizzling back.
    # crashes in wgpuDeviceCreateRenderPipeline, not caught in the shader module.
    # passes naga validation and wgsl translation, panics to spv translation (which can be caught via subprocess).
    device = wgpu.utils.get_default_device()

    fragment_shader = """
    fn test(rng: ptr<function, u32>) {}

    fn woops(uv: vec2<u32>) {
        var rngs = vec3<u32>(1, 2, 3);
        test(&rngs.x);
    }

    @fragment
    fn fs_main() -> @location(0) vec4<f32> {
        return vec4<f32>(1.0, 0.499, 0.0, 1.0);
    }
    """
    shader_source = default_vertex_shader + fragment_shader

    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])
    render_args = device, shader_source, pipeline_layout, bind_group

    # would be great to have some kind of error instead of a crash
    with raises(wgpu.GPUError):
        render_to_texture(*render_args, size=(64, 64))


def test_impossible_loop():
    # panicked at src\lib.rs:582:5:
    # should be captured: https://github.com/gfx-rs/wgpu/issues/5926#issuecomment-2216839446
    # some use cases include the increment var being related to a uniform like iTime.
    # other cases are generated code that is just wrong.
    # another case is a translation error from glsl to wgsl where one statement is lost.
    # https://github.com/gfx-rs/wgpu/issues/6208

    device = wgpu.utils.get_default_device()

    fragment_shader = """
    @fragment
    fn fs_main() -> @location(0) vec4<f32> {
        var a = 0.0;
        for (var i = 0; i < 10; i -= 1) {
            a += 0.1;
        }
        return vec4<f32>(1.0, a, 0.0, 1.0);
    }
    """

    # we do see this error logged, but not captured and raised.
    # apparently the device gets lost because it takes too long (infinite loop).
    # Some compilers detect this (DX12) and raise a different error.
    # the python process crashes.
    # the additional backtrace might be a bug upstream.
    expected = """
        Error in wgpuQueueSubmit: Validation Error

    Caused by:
    Parent device is lost
    """

    shader_source = default_vertex_shader + fragment_shader
    shader_source = dedent(shader_source)
    expected = dedent(expected)

    bind_group = None
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])
    render_args = device, shader_source, pipeline_layout, bind_group

    with raises(wgpu.GPUError) as err:
        render_to_texture(*render_args, size=(64, 64))

    error = err.value.message
    assert error == expected, f"Expected:\n\n{expected}"


if __name__ == "__main__":
    run_tests(globals())
