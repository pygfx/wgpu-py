"""
These flags are defined in ``wgpu.flags``, but are also available from the root wgpu namespace.

Flags are bitmasks; zero or multiple fields can be set at the same time.
Flags are integer bitmasks, but can also be passed as strings, so instead of
``wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST``,
one can also write ``"MAP_READ|COPY_DIST"``.
"""

from ._coreutils import BaseEnum as _BaseEnum


class Flags(_BaseEnum):
    """Base flags class for wgpu."""

    pass


# CODE BELOW THIS POINT IS AUTOGENERATED - DO NOT EDIT


# There are 5 flags

__all__ = [
    "BufferUsage",
    "MapMode",
    "TextureUsage",
    "ShaderStage",
    "ColorWrite",
]


class BufferUsage(Flags):
    MAP_READ = 1
    MAP_WRITE = 2
    COPY_SRC = 4
    COPY_DST = 8
    INDEX = 16
    VERTEX = 32
    UNIFORM = 64
    STORAGE = 128
    INDIRECT = 256
    QUERY_RESOLVE = 512


class MapMode(Flags):
    READ = 1
    WRITE = 2


class TextureUsage(Flags):
    COPY_SRC = 1
    COPY_DST = 2
    TEXTURE_BINDING = 4
    STORAGE_BINDING = 8
    RENDER_ATTACHMENT = 16


class ShaderStage(Flags):
    VERTEX = 1
    FRAGMENT = 2
    COMPUTE = 4


class ColorWrite(Flags):
    RED = 1
    GREEN = 2
    BLUE = 4
    ALPHA = 8
    ALL = 15
