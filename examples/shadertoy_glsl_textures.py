from wgpu.utils.shadertoy import Shadertoy, ShadertoyChannel

import numpy as np

shader_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    
    vec4 c0 = texture(iChannel0, 2.0*uv);
    vec4 c1 = texture(iChannel1, 3.0*uv);
    
    fragColor = mix(c0,c1,abs(sin(i_time)));
}

"""
diag = np.eye(8, dtype=np.uint8).reshape((8, 8, 1)).repeat(4, axis=2) * 255
gradient = np.linspace(0, 255, 32, dtype=np.uint8).reshape((32, 1, 1)).repeat(32, axis=1).repeat(4, axis=2)

channel0 = ShadertoyChannel(diag, wrap="repeat")
channel1 = ShadertoyChannel(gradient)

shader = Shadertoy(shader_code, resolution=(640, 480), inputs=[channel0, channel1])

if __name__ == "__main__":
    shader.show()
