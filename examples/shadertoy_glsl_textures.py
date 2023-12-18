from wgpu.utils.shadertoy import Shadertoy, ShadertoyChannel

shader_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec4 c0 = texture(iChannel0, 2.0*uv);
    vec4 c1 = texture(iChannel1, 3.0*uv);
    fragColor = mix(c0,c1,abs(sin(i_time)));
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

shader = Shadertoy(shader_code, resolution=(640, 480), inputs=[channel0, channel1])

if __name__ == "__main__":
    shader.show()
