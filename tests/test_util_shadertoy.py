import os

import wgpu.backends.rs  # noqa
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


def test_shadertoy():
    # Import here, because it imports the wgou.gui.auto
    from wgpu.utils.shadertoy import Shadertoy  # noqa

    shader_code = """
        fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
            let uv = frag_coord / i_resolution;

            if ( length(frag_coord - i_mouse) < 20.0 ) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }else{
                return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, resolution=(800, 450))
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code

    shader._draw_frame()
