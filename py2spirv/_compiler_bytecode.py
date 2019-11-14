
from ._compiler import BaseSpirVCompiler, str_to_words, STORAGE_CLASSES
from . import _spirv_constants as cc


class Bytecode2SpirVCompiler(BaseSpirVCompiler):
    """ WIP Python 2 SpirV Compiler that parses Python bytecode to generate
    SpirV code.
    """

    def _prepare(self):
        self._co = self._py_func.__code__


    def _generate(self):
        pass
