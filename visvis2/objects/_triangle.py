from ._base import WorldObject

from py2spirv import python2spirv


@python2spirv
def vertex_shader(input, output):
    input.define("index", "VertexId", i32)
    output.define("pos", "Position", vec4)
    output.define("color", 0, vec3)

    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

    p = positions[input.index]
    output.pos = vec4(p, 0.0, 1.0)
    output.color = vec3(p, 0.5)


@python2spirv
def fragment_shader(input, output):
    input.define("color", 0, vec3)
    output.define("color", 0, vec4)

    output.color = vec4(input.color, 1.0)


class Triangle(WorldObject):
    """ Our first WorldObject, a triangle!.
    Forget about sources and materials for now ...
    """

    def __init__(self):
        pass

    def describe_pipeline(self):
        return {"vertex_shader": vertex_shader, "fragment_shader": fragment_shader}
