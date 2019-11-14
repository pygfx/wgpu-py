
import os

from py2spirv._spirv_constants import *
from py2spirv import Ast2SpirVCompiler, Bytecode2SpirVCompiler

vec3 = "vec3"
vec4 = "vec4"



def vertex_shader():

    return 3 + 4


def vertex_shader():

    fragColor = output(vec3)
    gl_Position = output(vec4)

    # positions = constant(3, vec2)
    positions = [vec2(0.0, -0.4), vec2(0.5, 0.4), vec2(-0.5, 0.5)]
    colors = [vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)]

    def main():
        gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0)
        fragColor = colors[gl_VertexIndex]


def fragment_shader():
    # version 450
    # extension GL_ARB_separate_shader_objects : enable

    fragColor = input(vec3)
    outColor = output(vec4)

    # layout(location = 0) in vec3 fragColor;
    # layout(location = 0) out vec4 outColor;

    def main():
        outColor = vec4(fragColor, 0.5)


c = Ast2SpirVCompiler(fragment_shader)
c.generate()
print(c.to_text())
