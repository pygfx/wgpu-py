"""
Type info:
* Basic types are boolean, int, float. Latter two are numerics, all three are scalars.
* Vector is two or more values of scalars (float, int, bool). For lengt > 4 need capabilities.
* Matrix is 2, 3, or 4 float vectors (each vector is a column).
* Array is homogeneous collection of non-void-type objects.
* Structure is heterogeneous collection of non-void-type objects.
* imagee, sampler, ...
"""


class ClassPropertyDescriptor(object):
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        # if
        result = self.fget.__get__(obj, klass)()
        # print(result)
        return result


def classproperty(func):
    """ Decorator to turn a class method into a class property.
    """
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return ClassPropertyDescriptor(func)


# todo: SpirV supports float and int of multiple different bits, must we expose that?


def _create_type(name, base, **props):
    """ Create a new type, memoize on name.
    """
    if name not in SpirVType._subtypes:
        SpirVType._subtypes[name] = type(name, (base,), props)
    return SpirVType._subtypes[name]


# %% Abstract types


class SpirVType:
    """ The root base class of all SpirV types.
    """

    _subtypes = {}


class Scalar(SpirVType):
    """ Base class for scalar types (float, int, bool).
    """


class Numeric(Scalar):
    """ Base class for numeric scalars (float, int).
    """


class Float(Numeric):
    """ Base class for float numerics (f16, f32, f64).
    """


class Int(Numeric):
    """ Base class for int numerics (i16, i32, i64).
    """


class Composite(SpirVType):
    """ Base class for composite types (Vector, Matrix, Aggregates).
    """


class Aggregate(Composite):
    """ Base class for Array and Struct types.
    """


class Vector(Composite):
    """ Base class for Vector types. Concrete types are templated based on
    length and subtype.
    """

    length = 0
    subtype = None

    @classproperty
    def f16(cls):
        n = cls.length
        return _create_type(f"vec{n}_f16", Vector, subtype=f16, length=n)

    @classproperty
    def f32(cls):
        n = cls.length
        return _create_type(f"vec{n}_f32", Vector, subtype=f32, length=n)

    @classproperty
    def f64(cls):
        n = cls.length
        return _create_type(f"vec{n}_f64", Vector, subtype=f64, length=n)

    @classproperty
    def i16(cls):
        n = cls.length
        return _create_type(f"vec{n}_i16", Vector, subtype=i16, length=n)

    @classproperty
    def i32(cls):
        n = cls.length
        return _create_type(f"vec{n}_i32", Vector, subtype=i32, length=n)

    @classproperty
    def i64(cls):
        n = cls.length
        return _create_type(f"vec{n}_i64", Vector, subtype=i64, length=n)

    @classproperty
    def boolean(cls):
        n = cls.length
        return _create_type(f"vec{n}_boolean", Vector, subtype=boolean, length=n)


class Matrix(Composite):
    """ Base class for Matrix types. Concrete types are templated based on
    cols, rows and subtype. Subtype can only be Float.
    """

    cols = 0
    rows = 0
    subtype = None

    @classproperty
    def f16(cls):
        cols, rows = cls.cols, cls.rows
        return _create_type(
            f"mat{cols}x{rows}_f16", Matrix, subtype=f16, cols=cols, rows=rows
        )

    @classproperty
    def f32(cls):
        cols, rows = cls.cols, cls.rows
        return _create_type(
            f"mat{cols}x{rows}_f32", Matrix, subtype=f32, cols=cols, rows=rows
        )

    @classproperty
    def f64(cls):
        cols, rows = cls.cols, cls.rows
        return _create_type(
            f"mat{cols}x{rows}_f64", Matrix, subtype=f64, cols=cols, rows=rows
        )


class Array(Aggregate):
    """ Base class for Array types. Concrete types are templated based on
    length and subtype. Subtype can be any SpirVType except void.
    """

    length = 0
    subtype = None

    def __new__(cls, *args):
        if len(args) > 0 and isinstance(args[0], type):
            # Checks
            if len(args) != 2:
                raise ValueError("Creating an array type needs Array(spirv_type, n)")
            subtype, n = args[0], int(args[1])
            if not (issubclass(subtype, SpirVType) and not issubclass(subtype, void)):
                raise TypeError("Array subtype must be a SpirVType (but not void).")
            # todo: check that subtype is a concrete type
            # Create type
            return _create_type(
                f"array{n}_{subtype.__name__}", Array, subtype=subtype, length=n
            )
        else:
            instance = super(Array, cls).__new__(cls, *args)
            return instance


class Struct(Aggregate):
    """ Base class for Struct types. Not implemented.
    """

    def __new__(cls, **kwargs):

        if False:  # if all kwarg values are types, return a new type
            return _create_type(f"struct", Struct)
        else:
            instance = super(Struct, cls).__new__(cls, **kwargs)
            return instance


# %% Concrete types


class void(SpirVType):
    pass


class boolean(Scalar):
    pass


class f16(Float):
    pass


class f32(Float):
    pass


class f64(Float):
    pass


class i16(Int):
    pass


class i32(Int):
    pass


class i64(Int):
    pass


vec2 = _create_type("vec2_f32", Vector, length=2, subtype=f32)
vec3 = _create_type("vec3_f32", Vector, length=3, subtype=f32)
vec4 = _create_type("vec4_f32", Vector, length=4, subtype=f32)

ivec2 = _create_type("vec2_i32", Vector, length=2, subtype=i32)
ivec3 = _create_type("vec3_i32", Vector, length=3, subtype=i32)
ivec4 = _create_type("vec4_i32", Vector, length=4, subtype=i32)

bvec2 = _create_type("vec2_boolean", Vector, length=2, subtype=boolean)
bvec3 = _create_type("vec3_boolean", Vector, length=3, subtype=boolean)
bvec4 = _create_type("vec4_boolean", Vector, length=4, subtype=boolean)


mat2x2 = _create_type("mat2x2_f32", Matrix, col=2, row=2, subtype=f32)
mat2x3 = _create_type("mat2x3_f32", Matrix, col=2, row=3, subtype=f32)
mat2x4 = _create_type("mat2x4_f32", Matrix, col=2, row=4, subtype=f32)

mat3x2 = _create_type("mat3x2_f32", Matrix, col=3, row=2, subtype=f32)
mat3x3 = _create_type("mat3x3_f32", Matrix, col=3, row=3, subtype=f32)
mat3x4 = _create_type("mat3x4_f32", Matrix, col=3, row=4, subtype=f32)

mat4x2 = _create_type("mat4x2_f32", Matrix, col=4, row=2, subtype=f32)
mat4x3 = _create_type("mat4x3_f32", Matrix, col=4, row=3, subtype=f32)
mat4x4 = _create_type("mat4x4_f32", Matrix, col=4, row=4, subtype=f32)

mat2 = mat2x2
mat3 = mat3x3
mat4 = mat4x4


# Types that can be referenced by name. From these types you can create any other type.
spirv_types_map = dict(
    # Scalars
    void=void,
    boolean=boolean,
    f16=f16,
    f32=f32,
    f64=f64,
    i16=i16,
    i32=i32,
    i64=i64,
    # Vectors
    vec2=vec2,
    vec3=vec3,
    vec4=vec4,
    # ivec2=ivec2,
    # ivec3=ivec3,
    # ivec4=ivec4,
    # bvec2=bvec2,
    # bvec3=bvec3,
    # bvec4=bvec4,
    # Matrices
    mat2x2=mat2x2,
    mat2x3=mat2x3,
    mat2x4=mat2x4,
    mat3x2=mat3x2,
    mat3x3=mat3x3,
    mat3x4=mat3x4,
    mat4x2=mat4x2,
    mat4x3=mat4x3,
    mat4x4=mat4x4,
    # Aggregates
    Array=Array,
)

##

# int = int
# float = float
# bool = bool
#
#
#
# class BaseVector:
#     _t = None
#     _n = 0
#
#     def __init__(self, *args):
#         values = []
#         for arg in args:
#             if isinstance(arg, BaseVector):
#                 assert arg._t == self._t
#                 values.extend(arg._values)
#             else:
#                 assert isinstance(arg, self.t)
#                 values.append(arg)
#         if len(values) != self._n:
#             raise ValueError(f"Vector {self.__class__.__name__} needs {self._n} elements.")
#         self._values = values
#
#
# vec2 = _create_type()
# class vec2(BaseVector):
#     _t = float
#     _n = 2
#
#
# class vec3(BaseVector):
#     _t = float
#     _n = 3
#
#
# class vec4(BaseVector):
#     _t = float
#     _n = 4
#
#
# class ivec2(BaseVector):
#     _t = int
#     _n = 2
#
#
# class ivec3(BaseVector):
#     _t = int
#     _n = 3
#
#
# class ivec4(BaseVector):
#     _t = int
#     _n = 4
#
#
# class bvec2(BaseVector):
#     _t = bool
#     _n = 2
#
#
# class bvec3(BaseVector):
#     _t = bool
#     _n = 3
#
#
# class bvec4(BaseVector):
#     _t = bool
#     _n = 4
#
#
# class BaseMatrix:
#     _t = float
#     _ncol = 0
#     _nrow = 0
#
#     def __init__(self, *args):
#         values = []
#         for arg in args:
#             assert isinstance(arg, BaseVector)
#             assert arg._t == self._t
#             if arg._n != self._nrow:
#                 raise ValueError(f"Matrix {self.__class__.__name__} needs {self._nrow} rows.")
#             values.append(arg)
#         if len(values) != self._ncol:
#             raise ValueError(f"Matrix {self.__class__.__name__} needs {self._ncol} columns.")
#         self._values = values
#
#
# class mat2x2(BaseMatrix):
#     _ncol = 2
#     _nrow = 2
#
#
# class mat2x3(BaseMatrix):
#     _ncol = 2
#     _nrow = 3
#
#
# class mat2x4(BaseMatrix):
#     _ncol = 2
#     _nrow = 4
#
#
# class mat3x2(BaseMatrix):
#     _ncol = 3
#     _nrow = 2
#
# class mat3x3(BaseMatrix):
#     _ncol = 3
#     _nrow = 3
#
# class mat3x4(BaseMatrix):
#     _ncol = 3
#     _nrow = 4
#
# class mat4x2(BaseMatrix):
#     _ncol = 4
#     _nrow = 2
#
# class mat4x3(BaseMatrix):
#     _ncol = 4
#     _nrow = 3
#
# class mat4x4(BaseMatrix):
#     _ncol = 4
#     _nrow = 4
#
#
# class mat2(mat2x2):
#     pass
#
#
# class mat3(mat3x3):
#     pass
#
#
# class mat4(mat4x4):
#     pass
#
#
# class array:
#
#     def __init__(self, *args):
#         self._type = type(args[0])
#         for arg in args:
#             assert isinstance(arg, self._type)
#         values = args
#
# class struct:
#     pass
#
#
# numerical_types = int, float
# scalar_types = int, float, bool
# vector_types = vec2, vec3, vec4, ivec2, ivec3, ivec4, bvec2, bvec3, bvec4
# matrix_types = mat2, mat3, mat4, mat2x2, mat2x3, mat2x4, mat3x2, mat3x3, mat3x4, mat4x2, mat4x3, mat4x4
# spirv_types = (void, ) + scalar_types + vector_types + matrix_types + (array,  struct)

# spirv_types_map = {}
# for _c in spirv_types:
#     spirv_types_map[_c.__name__] = _c
