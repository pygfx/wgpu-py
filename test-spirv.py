"""
Small script to test spirv compiler.
"""

import os

from py2spirv import _spirv_constants as cc
from py2spirv import Python2SpirVCompiler, WASL2SpirVCompiler
from py2spirv import _types

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
    # layout(location = 1, subdinges=1) in vec2 fragColor2;
    # layout(location = 0) out vec4 outColor;

    # fragColor = input[0, vec3]
    # outColor = output[0, vec4]
    # outColor = vec4(fragColor, 0.5)  # weird to assign twice

    # output[0, vec4] = vec4(fragColor, 0.5)  # we lost meaning here, what are we outputting??

    outColor = vec4(fragColor, 0.5)

    return outColor


def fragment_shader(input, output):
    input.define("index", "VertexId", i32)
    output.define("pos", "Position", vec4)
    output.define("color", 0, vec3)

    positions = Array(
        vec2(+0.0, -0.5),
        vec2(+0.5, +0.5),
        vec2(-0.5, +0.5),
    )

    p = positions[input.index]
    output.pos = vec4(p, 0.0, 1.0)
    output.color = vec3(p, 0.5)


# from sprv import F32, I32, Bool

ffragment_shader = """
fn main (
    inColor: input vec3 0,
    outColor: output vec4 0,
) {
    outColor = vec4(inColor, 1.0)
}
"""

vertex_shader = """
fn main (
    index: input i32 VertexId,  # VertexID or VertexIndex
    pos: output vec4 Position,
) {

    positions = Array(
        vec2(+0.0, -0.5),
        vec2(+0.5, +0.5),
        vec2(-0.5, +0.5),
    )

    pos = vec4(positions[index], 0.0, 1.0)
}
"""

# + testable
# + ziet eruit als Python
# - ziet eruit als Python, maar is het niet echt
# - nieuwe concepten moet je wel leren, is niet echt Python!
# + ja, maar dan ken je de syntax iig al :P


# c = WASL2SpirVCompiler(vertex_shader)
# c = Python2SpirVCompiler(fragment_shader)
c = WASL2SpirVCompiler(ffragment_shader)
c.generate()
print(c.disassble())
c.validate()
