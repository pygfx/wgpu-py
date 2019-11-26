"""
Types in SpirV:

* Basic types are boolean, int, float. Latter two are numerics, all three are scalars.
* Vector is two or more values of scalars (float, int, bool). For lengt > 4 need capabilities.
* Matrix is 2, 3, or 4 float vectors (each vector is a column).
* Array is homogeneous collection of non-void-type objects.
* Structure is heterogeneous collection of non-void-type objects.
* image, sampler, ...

Here we follow the SpirV type hierarchy. We define abstract types, which
can be specialized (made concrete) by calling them. By convention,
abstract types start with a capital letter, concrete types are lowercase.

"""


_subtypes = {}


def _create_type(name, base, props):
    """ Create a new type, memoize on name.
    """
    if name not in _subtypes:
        assert not props.get("is_abstract", True), "can only create concrete types"
        _subtypes[name] = type(name, (base,), props)
    return _subtypes[name]


# %% Abstract types


class SpirVType:
    """ The root base class of all SpirV types.
    """
    is_abstract = True

    def __init__(self):
        if self.is_abstract:
            name = self.__class__.__name__
            raise RuntimeError(f"{name} is an abstract class and cannot be instantiated")



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

    subtype = None
    length = 0

    def __new__(cls, *args):
        if cls.is_abstract:
            if len(args) != 2:
                raise TypeError("Vector specialization needs 2 args: Vector(subtype, n)")
            subtype, n = args
            n = int(n)
            if not isinstance(subtype, type) and issubclass(subtype, Scalar):
                raise TypeError("Vector subtype must be a Scalar type.")
            elif subtype.is_abstract:
                raise TypeError("Vector subtype cannot be an abstract SpirV type.")
            if n < 2 or n > 4:
                raise TypeError("Vector can have 2, 3 or 4 elements.")
            props = dict(subtype=subtype, length=n, is_abstract=False)
            return _create_type(f"vec{n}_{subtype.__name__}", Vector, props)
        else:
            return super().__new__(*args)

    def __init__(self, *args):
        raise NotImplementedError("Instantiation")


class Matrix(Composite):
    """ Base class for Matrix types. Concrete types are templated based on
    cols, rows and subtype. Subtype can only be Float.
    """

    subtype = None
    cols = 0
    rows = 0

    def __new__(cls, *args):
        if cls.is_abstract:
            if len(args) != 3:
                raise TypeError("Matrix specialization needs 3 args: Matrix(subtype, cols, rows)")
            subtype, cols, rows = args
            cols, rows = int(cols), int(rows)
            if not isinstance(subtype, type) and issubclass(subtype, Float):
                raise TypeError("Matrix subtype must be a Float type.")
            elif subtype.is_abstract:
                raise TypeError("Matrix subtype cannot be an abstract SpirV type.")
            if cols < 2 or cols > 4:
                raise TypeError("Matrix can have 2, 3 or 4 columns.")
            if rows < 2 or rows > 4:
                raise TypeError("Matrix can have 2, 3 or 4 rows.")
            props = dict(subtype=subtype, cols=cols, rows=rows, is_abstract=False)
            return _create_type(f"mat{cols}x{rows}_{subtype.__name__}", Matrix, props)
        else:
            return super().__new__(*args)

    def __init__(self, *args):
        raise NotImplementedError("Instantiation")


class Array(Aggregate):
    """ Base class for Array types. Concrete types are templated based on
    length and subtype. Subtype can be any SpirVType except void.
    """

    subtype = None
    length = 0

    def __new__(cls, *args):
        if cls.is_abstract:
            if len(args) != 2:
                raise TypeError("Array specialization needs 2 args: Array(subtype, n)")
            subtype, n = args
            n = int(n)
            if not isinstance(subtype, type) and issubclass(subtype, SpirVType):
                raise TypeError("Array subtype must be a SpirV type.")
            elif issubclass(subtype, void):
                raise TypeError("Array subtype cannot be void.")
            elif subtype.is_abstract:
                raise TypeError("Array subtype cannot be an abstract SpirV type.")
            if n < 1:
                raise TypeError("Array must have at least 1 element.")

            props = dict(subtype=subtype, length=n, is_abstract=False)
            return _create_type(f"array{n}_{subtype.__name__}", Array, props)
        else:
            return super().__new__(*args)

    def __init__(self, *args):
        raise NotImplementedError("Instantiation")


class Struct(Aggregate):
    """ Base class for Struct types. Not implemented.
    """

    def __new__(cls, **kwargs):
        if cls.is_abstract:
            raise NotImplementedError("Specialization")
        else:
            return super().__new__(**kwargs)

    def __init__(self, **kwargs):
        raise NotImplementedError("Instantiation")


# %% Concrete types


class void(SpirVType):
    is_abstract = False


class boolean(Scalar):
    is_abstract = False


class f16(Float):
    is_abstract = False


class f32(Float):
    is_abstract = False


class f64(Float):
    is_abstract = False


class i16(Int):
    is_abstract = False


class i32(Int):
    is_abstract = False


class i64(Int):
    is_abstract = False


# %% Convenient concrete types

vec2 = Vector(f32, 2)
vec3 = Vector(f32, 3)
vec4 = Vector(f32, 4)

ivec2 = Vector(i32, 2)
ivec3 = Vector(i32, 3)
ivec4 = Vector(i32, 4)

mat2 = Matrix(f32, 2, 2)
mat3 = Matrix(f32, 3, 3)
mat4 = Matrix(f32, 4, 4)


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
    mat2=mat2,
    # mat2x3=mat2x3,
    # mat2x4=mat2x4,
    # mat3x2=mat3x2,
    mat3=mat3,
    # mat3x4=mat3x4,
    # mat4x2=mat4x2,
    # mat4x3=mat4x3,
    mat4=mat4,
    # Aggregates
    Array=Array,  # todo: only concrete types here?
    # Struct=Struct,
)
