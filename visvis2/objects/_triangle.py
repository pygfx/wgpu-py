from ._base import WorldObject

from py2spirv import python2spirv


@python2spirv
def vertex_shader(input, output, uniform):
    input.define("index", "VertexId", i32)
    # input.define("pos", 0, vec2)
    output.define("pos", "Position", vec4)
    output.define(0, color=vec3)

    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

    p = positions[input.index]
    # p = input.pos
    output.pos = vec4(p, 0.0, 1.0)
    output.color = vec3(p, 0.5)


@python2spirv
def fragment_shader(input, output, uniform):
    input.define(0, color=vec3)
    output.define("color", 0, vec4)
    uniform.define("color", 0, vec3)

    output.color = vec4(uniform.color, 0.1)


class Triangle(WorldObject):
    """ Our first WorldObject, a triangle!.
    Forget about sources and materials for now ...
    """

    def __init__(self, pos=(0, 0, 0)):
        self._pos = [float(x) for x in pos]
        assert len(self._pos) == 3  # x, y, z

    def describe_pipeline(self):
        uniforms = [self._pos]
        return {"vertex_shader": vertex_shader, "fragment_shader": fragment_shader, "uniforms": uniforms}
