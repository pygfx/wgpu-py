from wgpu.utils.shadertoy import Shadertoy, ShadertoyChannel

from PIL import Image
import numpy as np

shader_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    
    vec4 c0 = texture(iChannel0, uv/(1.0+sin(iTime)));
    
    fragColor = c0;
}

"""
texture_img = Image.open("examples/screenshots/cube.png")
texture_arr = np.array(texture_img)
channel0 = ShadertoyChannel(texture_arr)

shader = Shadertoy(shader_code, resolution=(640, 480), inputs=[channel0])

if __name__ == "__main__":
    shader.show()
