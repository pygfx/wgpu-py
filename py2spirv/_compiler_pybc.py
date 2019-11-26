import inspect
from dis import dis as pprint_bytecode

from ._module import SpirVModule
from ._generator_bc import Bytecode2SpirVGenerator
from . import _generator_bc as bc
from ._dis import dis


def python2spirv(func, shader_type=None):
    """ Compile a Python function to SpirV and return as a SpirVModule object.

    This function takes the bytecode of the given function, converts it to
    a more standardized (and SpirV specific) bytecode, and then generates
    the SpirV from that. All in dependency-free pure Python.
    """

    if not inspect.isfunction(func):
        raise TypeError("python2spirv expects a Python function.")

    converter = PyBytecode2Bytecode()
    converter.convert(func)
    bytecode = converter.dump()

    generator = Bytecode2SpirVGenerator()
    generator.generate(bytecode, shader_type)
    bb = generator.to_bytes()

    m = SpirVModule(func, bb, "compiled from a Python function")
    m.gen = generator
    return m


class PyBytecode2Bytecode:
    """ Convert Python bytecode to our own well-defined bytecode.
    Python bytecode depends on other variables on the code object, and differs
    between Python functions. This class converts this, so that the next step
    of code generation becomes simpler.
    """

    def convert(self, py_func):
        self._py_func = py_func
        self._co = co = self._py_func.__code__

        self._opcodes = []
        self._convert()

    def emit(self, opcode, arg):
        self._opcodes.append((opcode, arg))

    def dump(self):
        return self._opcodes

    def _convert(self):

        # co.co_code  # bytes
        #
        # co.co_name
        # co.co_filename
        # co.co_firstlineno
        #
        # co.co_argcount
        # co.co_kwonlyargcount
        # co.co_nlocals
        # co.co_consts
        # co.co_varnames
        # co.co_names  # nonlocal names
        # co.co_cellvars
        # co.co_freevars
        #
        # co.co_stacksize  # the maximum depth the stack can reach while executing the code
        # co.co_flags  # flags if this code object has nested scopes/generators/etc.
        # co.co_lnotab  # line number table  https://svn.python.org/projects/python/branches/pep-0384/Objects/lnotab_notes.txt

        # Pointer in the bytecode stream
        self._pointer = 0

        # Bytecode is a stack machine. The stack has both Python objects, for stuff
        # that has not yet been resolved. And IdInt objects representing SpirV objects.
        self._stack = []

        # Keep track of shade io
        self._input = {}
        self._output = {}
        self._uniforms = {}
        self._constants = {}  # ?

        # Python variable names -> (SpirV object id, type_id)
        # self._aliases = {}

        # Parse
        while self._pointer < len(self._co.co_code):
            opcode = self._next()
            opname = dis.opname[opcode]
            method_name = "_op_" + opname.lower()
            method = getattr(self, method_name, None)
            if method is None:
                pprint_bytecode(self._co)
                raise RuntimeError(f"Cannot parse {opname} yet (no {method_name}()).")
            else:
                method()

    def _next(self):
        res = self._co.co_code[self._pointer]
        self._pointer += 1
        return res

    def _define(self, kind, name, location, type):
        if kind == "input":
            self.emit(bc.CO_INPUT, ("input." + name, location, type))
            self._input[name] = type
        elif kind == "output":
           self.emit(bc.CO_OUTPUT, ("output." + name, location, type))
           self._output[name] = type

    # %%

    def _op_pop_top(self):
        self._stack.pop()
        self._next()  # todo: why need pointer advance?
        self.emit(bc.CO_POP_TOP, ())

    def _op_return_value(self):
        result = self._stack.pop()
        assert result is None
        # for now, there is no return in our-bytecode
        self._next() # todo: why need pointer advance?

    def _op_load_fast(self):
        # store a variable that is used in an inner scope.
        i = self._next()
        name = self._co.co_varnames[i]
        if name == "input":
            self._stack.append("input")
        elif name == "output":
            self._stack.append("output")
        else:
            self.emit(bc.CO_LOAD, name)
            self._stack.append(name)  # todo: euhm, do we still need a stack?

    def _op_load_const(self):
        i = self._next()
        ob = self._co.co_consts[i]
        if isinstance(ob, str):
            # We use strings in e.g. input.define(), mmm
            self._stack.append(ob)
        elif isinstance(ob, (float, int, bool)):
            self.emit(bc.CO_LOAD_CONSTANT, ob)
            self._stack.append(ob)
        elif ob is None:
            self._stack.append(None)  # todo: for the final return ...
        else:
            raise NotImplementedError()

    def _op_load_global(self):
        i = self._next()
        name = self._co.co_names[i]
        self.emit(bc.CO_LOAD, name)
        self._stack.append(name)

    def _op_load_attr(self):
        i = self._next()
        name = self._co.co_names[i]
        ob = self._stack.pop()

        if ob == "input":
            if name not in self._input:
                raise NameError(f"No input {name} defined.")
            self.emit(bc.CO_LOAD, "input." + name)
        elif ob == "output":
            raise AttributeError("Cannot read from output.")
        else:
            raise NotImplementedError()

        self._stack.append(None)

    def _op_store_attr(self):
        i = self._next()
        name = self._co.co_names[i]
        ob = self._stack.pop()
        value = self._stack.pop()
        # assert isinstance(value, IdInt)

        if ob == "input":
            raise AttributeError("Cannot assign to input.")
        elif ob == "output":
            if name not in self._output:
                raise NameError(f"No output {name} defined.")
            self.emit(bc.CO_STORE, "output." + name)
        else:
            raise NotImplementedError()

    def _op_store_fast(self):
        i = self._next()
        name = self._co.co_varnames[i]
        ob = self._stack.pop()
        self.emit(bc.CO_STORE, name)

    def _op_load_method(self):  # new in Python 3.7
        i = self._next()
        method_name = self._co.co_names[i]
        ob = self._stack.pop()
        if ob == "input" or ob == "output":
            if method_name == "define":
                func = self._define
            else:
                raise RuntimeError(f"Can only define() on {ob} ojects.")
        else:
            raise NotImplementedError()

        self._stack.append(func)
        self._stack.append(ob)

    def _op_call_method(self): # new in Python 3.7
        nargs = self._next()
        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        ob = self._stack.pop()
        if ob in ("input", "output"):
            func = self._stack.pop()
            result = func(ob, *args)
            self._stack.append(result)
        else:
            self.emit(bc.CO_CALL, nargs)
            self._stack.append(None)

    def _op_call_function(self):
        nargs = self._next()
        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        func = self._stack.pop()
        assert isinstance(func, str)

        self.emit(bc.CO_CALL, nargs)
        self._stack.append(None)

    def _op_binary_subscr(self):
        self._next()  # because always 1 arg even if dummy
        index = self._stack.pop()
        ob = self._stack.pop()
        self.emit(bc.CO_INDEX, None)
        self._stack.append(None)
