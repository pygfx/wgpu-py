
import os

from py2spirv import _spirv_constants as cc
from py2spirv import Ast2SpirVCompiler, Bytecode2SpirVCompiler

vec3 = "vec3"
vec4 = "vec4"


# todo: how to declare wether its a vertex or fragment shader
# todo: inputs, uniforms, const, outputs
# todo: std

##
class VertexShader:
    pass

    def myhelper(foo: vec3, bar: int):
        return foo * float(bar)  # type derived by tracing


def myhelper(foo: vec3, bar: int):
    return foo * float(bar)  # type derived by tracing


class MyVertexShader(VertexShader):

    # attributes and varyings:
    # layout(location = 0) out vec4 color;
    # layout(location = 0, component = 2) out vec2 arr2[4]
    # uniforms:
    # layout(binding = 0) uniform vec3 color;
    # layout(binding = 0, offset = 4) uniform atomic_uint three;

    def main(self):

        location, type, input/output/uniform, name

        aPos = self.input[0, vec3]
        uColor = self.uniform[0, vec4]

        self.output[0, vec3] = vColor
        self.output["gl_position", vec4] = vPos

        VertexShader.myhelper(xx, yy)
        spir_std.sin(3.4)

# @shader
def main(input, output, uniform):  # can add more args here if needed

    # need 4 things: location, type, input/output/uniform, (name)

    aPos = input[0, vec3]
    uColor = uniform[0, vec4]
    # or
    aPos = input(0, vec3)
    uColor = uniform(0, vec4)

    # Assignment seems most intuitive
    output[0, vec3] = vColor
    output["gl_position", vec4] = vPos
    # or
    output(0, vec3, vColor)
    output("gl_position", vec4, vPos)
    # or ?? --> nah, feels weird to define it like this and then assign to it later
    vColor = output[0, vec3]
    vPos = output["gl_Position", vec4]
    # or ?? --> dito
    vColor = output(0, vec3)
    vPos = output("gl_Position", vec4)

    myhelper(xx, yy)
    spir_std.sin(3.4)

class spv:
    def input(location, type):
        return ("input", location, type)
    def output(location, type):
        return ("output", location, type)


def mainx(aPos: spv.input(0, vec3),
          vPos: spv.output("glPosition", vec4),
):

    vPos = aPos


def main():

    from inputs import vec3_1 as foo
    from output import vec5_3 as vColor
    from uniform import mat44_1 as stuff
    from output import gl_Position


def main():

    with spv.imports:
        foo = vec3



# @shader(aPos:input(vec3, output_xx, etc.)
def myshader():

    foo = vec(2.0, 1.0, 0.0)
    bar = ins[0, vec3] # what type???
    out[1] = foo



##


def vertex_shader():


    v1 = vec(2.1, 3.2, 4.5)
    v2 = vec(v1, v2)
    m1 = mat(v1, v2, v3)
    a1 = arr(v1, v2)
    s1 = struct(pos=v1, speed=v2)

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


def fragment_shader(fragColor: spv.input(vec3, 0),
                    outColor: spv.output(vec4, 0),
):
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
    #
    # x = vec4(input.color, 0.5)
    # output.color = x


c = Bytecode2SpirVCompiler(fragment_shader)
c.generate()
print(c.disassble())
