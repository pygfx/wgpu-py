from wgpu.utils.shadertoy import Shadertoy, ShadertoyChannel

import numpy as np

shader_code_wgsl = """
fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{
    let uv = frag_coord / i_resolution.xy;
    let c0 = textureSample(i_channel0, sampler0, 2.0*uv);
    let c1 = textureSample(i_channel1, sampler1, 3.0*uv);
    return mix(c0,c1,abs(sin(i_time)));
}
"""
diag = np.eye(8, dtype=np.uint8).reshape((8, 8, 1)).repeat(4, axis=2) * 255
gradient = np.linspace(0, 255, 32, dtype=np.uint8).reshape((32, 1, 1)).repeat(32, axis=1).repeat(4, axis=2)

channel0 = ShadertoyChannel(diag, wrap="repeat")
channel1 = ShadertoyChannel(gradient)

shader = Shadertoy(shader_code_wgsl, resolution=(640, 480), inputs=[channel0, channel1])

if __name__ == "__main__":
    shader.show()
