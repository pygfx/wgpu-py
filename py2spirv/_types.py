
int = int
float = float
bool = bool

# todo: SpirV supports float and int of multiple different bits, must we expose that?

class void:
    pass


class BaseVector:
    _t = None
    _n = 0

    def __init__(self, *args):
        values = []
        for arg in args:
            if isinstance(arg, BaseVector):
                assert arg._t == self._t
                values.extend(arg._values)
            else:
                assert isinstance(arg, self.t)
                values.append(arg)
        if len(values) != self._n:
            raise ValueError(f"Vector {self.__class__.__name__} needs {self._n} elements.")
        self._values = values


class vec2(BaseVector):
    _t = float
    _n = 2


class vec3(BaseVector):
    _t = float
    _n = 3


class vec4(BaseVector):
    _t = float
    _n = 4


class ivec2(BaseVector):
    _t = int
    _n = 2


class ivec3(BaseVector):
    _t = int
    _n = 3


class ivec4(BaseVector):
    _t = int
    _n = 4


class bvec2(BaseVector):
    _t = bool
    _n = 2


class bvec3(BaseVector):
    _t = bool
    _n = 3


class bvec4(BaseVector):
    _t = bool
    _n = 4


class BaseMatrix:
    _t = float
    _ncol = 0
    _nrow = 0

    def __init__(self, *args):
        values = []
        for arg in args:
            assert isinstance(arg, BaseVector)
            assert arg._t == self._t
            if arg._n != self._nrow:
                raise ValueError(f"Matrix {self.__class__.__name__} needs {self._nrow} rows.")
            values.append(arg)
        if len(values) != self._ncol:
            raise ValueError(f"Matrix {self.__class__.__name__} needs {self._ncol} columns.")
        self._values = values


class mat2x2(BaseMatrix):
    _ncol = 2
    _nrow = 2


class mat2x3(BaseMatrix):
    _ncol = 2
    _nrow = 3


class mat2x4(BaseMatrix):
    _ncol = 2
    _nrow = 4


class mat3x2(BaseMatrix):
    _ncol = 3
    _nrow = 2

class mat3x3(BaseMatrix):
    _ncol = 3
    _nrow = 3

class mat3x4(BaseMatrix):
    _ncol = 3
    _nrow = 4

class mat4x2(BaseMatrix):
    _ncol = 4
    _nrow = 2

class mat4x3(BaseMatrix):
    _ncol = 4
    _nrow = 3

class mat4x4(BaseMatrix):
    _ncol = 4
    _nrow = 4


class mat2(mat2x2):
    pass


class mat3(mat3x3):
    pass


class mat4(mat4x4):
    pass


class array:

    def __init__(self, *args):
        self._type = type(args[0])
        for arg in args:
            assert isinstance(arg, self._type)
        values = args

class struct:
    pass


numerical_types = int, float
scalar_types = int, float, bool
vector_types = vec2, vec3, vec4, ivec2, ivec3, ivec4, bvec2, bvec3, bvec4
matrix_types = mat2, mat3, mat4, mat2x2, mat2x3, mat2x4, mat3x2, mat3x3, mat3x4, mat4x2, mat4x3, mat4x4
spirv_types = (void, ) + scalar_types + vector_types + matrix_types + (array,  struct)

spirv_types_map = {}
for _c in spirv_types:
    spirv_types_map[_c.__name__] = _c
