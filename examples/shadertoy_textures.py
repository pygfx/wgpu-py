from wgpu.utils.shadertoy import Shadertoy, ShadertoyChannel

from PIL import Image
import numpy as np

shader_code_wgsl = """
fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{
    let uv = frag_coord / i_resolution.xy;
    let c0 = textureSample(i_channel0, sampler0, uv/(1.0+sin(i_time)));
    return c0;
}
"""
texture_img = Image.open("examples/screenshots/cube.png")
texture_arr = np.array(texture_img)
channel0 = ShadertoyChannel(texture_arr)

shader = Shadertoy(shader_code_wgsl, resolution=(640, 480), inputs=[channel0])

if __name__ == "__main__":
    shader.show()
