"""
Small script to test spirv compiler.
"""

import os

from py2spirv import _spirv_constants as cc
from py2spirv import Ast2SpirVCompiler, Bytecode2SpirVCompiler

# todo: how to declare wether its a vertex or fragment shader
# todo: std


def vertex_shader():

    fragColor = output(vec3)
    gl_Position = output(vec4)

    # positions = constant(3, vec2)
    positions = [vec2(0.0, -0.4), vec2(0.5, 0.4), vec2(-0.5, 0.5)]
    colors = [vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)]

    def main():
        gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0)
        fragColor = colors[gl_VertexIndex]


# def fragment_shader(fragColor: spv.input(vec3, 0),
#                     outColor: spv.output(vec4, 0),
# ):
def fragment_shader():
    # version 450
    # extension GL_ARB_separate_shader_objects : enable
    # layout(location = 0) in vec3 fragColor;
    # layout(location = 0) out vec4 outColor;

    # fragColor = input[0, vec3]
    # outColor = output[0, vec4]
    # outColor = vec4(fragColor, 0.5)  # weird to assign twice

    # output[0, vec4] = vec4(fragColor, 0.5)  # we lost meaning here, what are we outputting??

    outColor = vec4(fragColor, 0.5)

    return outColor


def fragment_shader(input, output):

    input.define("pos", vec3, 12)
    output.define("color", vec4, 0)

    x = vec4(input.pos, 0.5)
    output.color = x


c = Bytecode2SpirVCompiler(fragment_shader)
c.generate()
print(c.disassble())
