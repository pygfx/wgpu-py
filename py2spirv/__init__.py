"""
A Python to SpirV compiler.
"""

__version__ = "0.1.0"


from ._compiler import BaseSpirVCompiler
from ._compiler_bytecode import Bytecode2SpirVCompiler
from ._compiler_pybytecode import Python2SpirVCompiler
from ._compiler_wasl import WASL2SpirVCompiler
from ._types import vec2, vec3, vec4  # todo: and more
