"""
A collections of ideas on how shaders should be written.
Most importantly, how to access inputs, outputs, and uniforms.

Each io need 4 things:

- whether its input/output/uniform
- location (int)
- type
- (name)

In GLSL this looks like this:

// attributes and varyings:
layout(location = 0) out vec4 color;
layout(location = 0, component = 2) out vec2 arr2[4]
// uniforms:
layout(binding = 0) uniform vec3 color;
layout(binding = 0, offset = 4) uniform atomic_uint three;

"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% types %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def x():

    # --- Dynamic
    v2 = vec(2.1, 4.5)
    v4 = vec(v1, v2)
    m1 = mat(v1, v2, v3)
    a1 = arr(v1, v2)
    s1 = struct(pos=v1, speed=v2)

    vec3 = vec(float, float, float)

    # ouch!
    mat4 = mat(vec(float, float, float), vec(float, float, float), vec(float, float, float))

    # P--- redefined only
    v2 = vec2(2.1, 3.2, 4.5)
    v4 = vec4(v1, v2)

    return 3 + 4


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% io %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %% A shader is a class

# can subclass to add multiple functions
# can use class or instance attributes for io
# can put vertex and fragment shader on single class?

class SomeVertexShader:

    def myhelper(foo: vec3, bar: int):
        return foo * float(bar)  # type derived by tracing


class MyVertexShader(SomeVertexShader):

    def main(self):

        aPos = self.input[0, vec3]
        uColor = self.uniform[0, vec4]

        self.output[0, vec3] = vColor
        self.output["gl_position", vec4] = vPos

        self.myhelper(xx, yy)
        spir_std.sin(3.4)


# %% function style, indexing into io

# @shader
def main(input, output, uniform):  # can add more args here if needed

    # This is ok
    aPos = input[0, vec3]
    uColor = uniform[0, vec4]

    # Not bad, but verbose, and nicer to define location somewhere on top?
    output[0, vec3] = vColor
    output["gl_position", vec4] = vPos

    # weird
    vColor = output[0, vec3]
    vPos = output["gl_Position", vec4]
    vColor = vec3(...)
    vPos = vec4(...)


# %% function style, calling into io

def main(input, output, uniform):  # can add more args here if needed

    # This is also ok
    aPos = input(0, vec3)
    uColor = uniform(0, vec4)

    # Definitely less nice
    output(0, vec3, vColor)
    output("gl_position", vec4, vPos)

    # weird
    vColor = output(0, vec3)
    vPos = output("gl_Position", vec4)
    vColor = vec3(...)
    vPos = vec4(...)


# %% function style, using a define call

def main(input, output):

    input.define("pos", vec3, 0)
    output.define("color", vec4, 1)

    x = vec4(input.color, 0.5)
    output.color = x


# %% function style, using annotations

# Not too bad, really. Is close to how you think it would work.
# The types must actually exist (be imported). Good thing or bad thing, not sure.

class spv:
    def input(location, type):
        return ("input", location, type)
    def output(location, type):
        return ("output", location, type)


def mainx(aPos: spv.input(0, vec3),
          vPos: spv.output("glPosition", vec4),
):

    vPos = aPos



# %% Same, but using decorator

# @shader(aPos:input(vec3, output_xx, etc.)
def myshader():

    foo = vec(2.0, 1.0, 0.0)
    bar = ins[0, vec3] # what type???
    out[1] = foo



# %% function style, using imports

# how to specify location, part of the name, yuk!
def main():

    from inputs import vec3_1 as foo
    from output import vec5_3 as vColor
    from uniform import mat44_1 as stuff
    from output import gl_Position


# %% Using with-statements

# Dito

def main():

    with spv.imports:
        foo = vec3

