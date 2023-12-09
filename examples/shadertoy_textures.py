from wgpu.utils.shadertoy import Shadertoy, ShadertoyChannel

shader_code_wgsl = """
fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{
    let uv = frag_coord / i_resolution.xy;
    let c0 = textureSample(i_channel0, sampler0, 2.0*uv);
    let c1 = textureSample(i_channel1, sampler1, 3.0*uv);
    return mix(c0,c1,abs(sin(i_time)));
}
"""
test_pattern = memoryview(
    bytearray((int(i != k) * 255 for i in range(8) for k in range(8))) * 4
).cast("B", shape=[8, 8, 4])
gradient = memoryview(
    bytearray((i for i in range(0, 255, 8) for _ in range(4))) * 32
).cast("B", shape=[32, 32, 4])

channel0 = ShadertoyChannel(test_pattern, wrap="repeat")
channel1 = ShadertoyChannel(gradient)

shader = Shadertoy(shader_code_wgsl, resolution=(640, 480), inputs=[channel0, channel1])

if __name__ == "__main__":
    shader.show()
