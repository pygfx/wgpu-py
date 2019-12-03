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

    if not shader_type:
        # Try to auto-detect
        if "vert" in func.__name__ and not "frag" in func.__name__:
            shader_type = "vertex"
        elif "frag" in func.__name__ and not "vert" in func.__name__:
            shader_type = "fragment"

    converter = PyBytecode2Bytecode()
    converter.convert(func)
    bytecode = converter.dump()

    generator = Bytecode2SpirVGenerator()
    generator.generate(bytecode, shader_type)
    bb = generator.to_bytes()

    m = SpirVModule(func, bb, f"compiled from Pyfunc {func.__name__}")
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
        self._uniform = {}
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

    def _peak_next(self):
        return self._co.co_code[self._pointer]

    def _define(self, kind, location, **variables):
        COS = {"input": bc.CO_INPUT, "output": bc.CO_OUTPUT, "uniform": bc.CO_UNIFORM}
        DICTS = {"input": self._input, "output": self._output, "uniform": self._uniform}
        co = COS[kind]
        d = DICTS[kind]
        args = [location]
        for name, type in variables.items():
            args.extend([kind + "." + name, type])
            d[name] = type
        self.emit(co, tuple(args))

    # %%

    def _op_pop_top(self):
        self._stack.pop()
        self._next()  # todo: why need pointer advance?
        self.emit(bc.CO_POP_TOP, ())

    def _op_return_value(self):
        result = self._stack.pop()
        assert result is None
        # for now, there is no return in our-bytecode
        self._next()  # todo: why need pointer advance?

    def _op_load_fast(self):
        # store a variable that is used in an inner scope.
        i = self._next()
        name = self._co.co_varnames[i]
        if name in ("input", "output", "uniform"):
            self._stack.append(name)
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
        elif isinstance(ob, tuple):
            self._stack.append(ob)  # may be needed for kwargs in define()
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

        if name == "define" and ob in ("input", "output", "uniform"):
            self._stack.append((self._define, ob))
        elif ob == "input":
            if name not in self._input:
                raise NameError(f"No input {name} defined.")
            self.emit(bc.CO_LOAD, "input." + name)
            self._stack.append("input." + name)
        elif ob == "uniform":
            if name not in self._uniform:
                raise NameError(f"No uniform {name} defined.")
            self.emit(bc.CO_LOAD, "uniform." + name)
            self._stack.append("uniform." + name)
        elif ob == "output":
            raise AttributeError("Cannot read from output.")
        else:
            raise NotImplementedError()

    def _op_store_attr(self):
        i = self._next()
        name = self._co.co_names[i]
        ob = self._stack.pop()
        value = self._stack.pop()
        # assert isinstance(value, IdInt)

        if ob == "input":
            raise AttributeError("Cannot assign to input.")
        elif ob == "uniform":
            raise AttributeError("Cannot assign to uniform.")
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
        if ob in ("input", "output", "uniform"):
            if method_name == "define":
                func = self._define
            else:
                raise RuntimeError(f"Can only define() on {ob} ojects.")
        else:
            raise NotImplementedError()

        self._stack.append(func)
        self._stack.append(ob)

    def _op_call_method(self):  # new in Python 3.7
        nargs = self._next()
        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        ob = self._stack.pop()
        if ob in ("input", "output", "uniform"):
            name, location, type = args
            func = self._stack.pop()
            result = func(ob, location, **{name: type})
            self._stack.append(result)
        else:
            self.emit(bc.CO_CALL, nargs)
            self._stack.append(None)

    def _op_call_function_kw(self):
        nargs = self._next()
        kwarg_names = self._stack.pop()
        n_kwargs = len(kwarg_names)
        n_pargs = nargs - n_kwargs

        args = self._stack[-nargs:]
        self._stack[-nargs:] = []

        func = self._stack.pop()
        assert isinstance(func, tuple) and func[0].__func__.__name__ == "_define"
        func_define, what = func

        pargs = args[:n_pargs]
        kwargs = {kwarg_names[i]: args[i+n_pargs] for i in range(n_kwargs)}
        func_define(what, *pargs, **kwargs)
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
        if isinstance(index, tuple):
            self.emit(bc.CO_INDEX, len(index))
        else:
            self.emit(bc.CO_INDEX, 1)
        self._stack.append(None)

    def _op_build_tuple(self):
        raise SyntaxError("No tuples in SpirV-ish Python")

        n = self._next()
        res = [self._stack.pop() for i in range(n)]
        res = tuple(reversed(res))

        if dis.opname[self._peak_next()] == "BINARY_SUBSCR":
            self._stack.append(res)
            # No emit, in the SpirV bytecode we pop the subscript indices off the stack.
        else:
            raise NotImplementedError("Tuples are not supported.")

    def _op_build_list(self):
        n = self._next()
        res = [self._stack.pop() for i in range(n)]
        res = list(reversed(res))
        self._stack.append(res)
        self.emit(bc.CO_BUILD_ARRAY, n)
