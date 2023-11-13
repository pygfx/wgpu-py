import os

from pytest import fixture, skip
from testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


@fixture(autouse=True, scope="module")
def force_offscreen():
    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
    try:
        yield
    finally:
        del os.environ["WGPU_FORCE_OFFSCREEN"]


def test_shadertoy_wgsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu.utils.shadertoy import Shadertoy  # noqa

    shader_code = """
        fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
            let uv = frag_coord / i_resolution.xy;

            if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }else{
                return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, resolution=(800, 450))
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "wgsl"

    shader._draw_frame()


def test_shadertoy_glsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu.utils.shadertoy import Shadertoy  # noqa

    shader_code = """
        void shader_main(out vec4 fragColor, vec2 frag_coord) {
            vec2 uv = frag_coord / i_resolution.xy;

            if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(i_time * vec3(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, resolution=(800, 450))
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"

    shader._draw_frame()


def test_shadertoy_offscreen():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu.utils.shadertoy import Shadertoy  # noqa

    shader_code = """
        void shader_main(out vec4 fragColor, vec2 frag_coord) {
            vec2 uv = frag_coord / i_resolution.xy;

            if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(i_time * vec3(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, resolution=(800, 450), offscreen=True)
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"
    assert shader._offscreen is True


def test_shadertoy_snapshot():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu.utils.shadertoy import Shadertoy  # noqa

    shader_code = """
        void shader_main(out vec4 fragColor, vec2 frag_coord) {
            vec2 uv = frag_coord / i_resolution.xy;

            if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(i_time * vec3(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, resolution=(800, 450), offscreen=True)
    frame1a = shader.snapshot(
        time_float=0.0,
        mouse_pos=(
            0,
            0,
            0,
            0,
        ),
    )
    frame2a = shader.snapshot(
        time_float=1.2,
        mouse_pos=(
            100,
            200,
            0,
            0,
        ),
    )
    frame1b = shader.snapshot(
        time_float=0.0,
        mouse_pos=(
            0,
            0,
            0,
            0,
        ),
    )
    frame2b = shader.snapshot(
        time_float=1.2,
        mouse_pos=(
            100,
            200,
            0,
            0,
        ),
    )

    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"
    assert shader._offscreen is True
    assert frame1a == frame1b
    assert frame2a == frame2b
