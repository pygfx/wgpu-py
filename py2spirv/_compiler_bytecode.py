import struct
from dis import dis as pprint_bytecode

from ._compiler import IdInt, BaseSpirVCompiler, str_to_words, STORAGE_CLASSES
from . import _spirv_constants as cc
from ._dis import dis
from . import _types


class Bytecode2SpirVCompiler(BaseSpirVCompiler):
    """ WIP Python 2 SpirV Compiler that parses Python bytecode to generate
    SpirV code.
    """

    def _prepare(self):
        self._co = co = self._py_func.__code__

        co.co_code  # bytes

        co.co_name
        co.co_filename
        co.co_firstlineno

        co.co_argcount
        co.co_kwonlyargcount
        co.co_nlocals
        co.co_consts
        co.co_varnames
        co.co_names  # nonlocal names
        co.co_cellvars
        co.co_freevars

        co.co_stacksize  # the maximum depth the stack can reach while executing the code
        co.co_flags  # flags if this code object has nested scopes/generators/etc.
        co.co_lnotab  # line number table  https://svn.python.org/projects/python/branches/pep-0384/Objects/lnotab_notes.txt

    def _generate(self):

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
        self._aliases = {}

        # Declare funcion
        return_type_id = self.get_type_id(_types.void)
        func_type_id = self.create_id("func_declaration")
        self.gen_instruction("types", cc.OpTypeFunction, func_type_id, return_type_id)  # 0 args

        # Start function definition
        func_id = self._entry_point_id
        func_control = 0  # can specify whether it should inline, etc.
        self.gen_func_instruction(cc.OpFunction, return_type_id, func_id, func_control, func_type_id)
        self.gen_func_instruction(cc.OpLabel, self.create_id("label"))

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

        # End function definition
        self.gen_func_instruction(cc.OpFunctionEnd)

    def _next(self):
        res = self._co.co_code[self._pointer]
        self._pointer += 1
        return res

    def _define(self, kind, name, type, location):
        if kind == "input":
            var_id, type_id = self.create_object(type)  # todo: or "input." + name)
            pointer_id = self.create_id("pointer")

            self._input[name] = type, var_id  # the code only needs var_id

            self.gen_instruction("annotations", cc.OpDecorate, var_id, cc.Decoration_Location, location)
            self.gen_instruction("types", cc.OpTypePointer, pointer_id, cc.StorageClass_Input, type_id)
            self.gen_instruction("types", cc.OpVariable, pointer_id, var_id, cc.StorageClass_Input)

        elif kind == "output":
            var_id, type_id = self.create_object(type)  # todo: or "output." + name)
            pointer_id = self.create_id("pointer")

            self._output[name] = type, var_id

            self.gen_instruction("annotations", cc.OpDecorate, var_id, cc.Decoration_Location, location)
            self.gen_instruction("types", cc.OpTypePointer, pointer_id, cc.StorageClass_Output, type_id)
            self.gen_instruction("types", cc.OpVariable, pointer_id, var_id, cc.StorageClass_Output)

    # %%

    def _op_pop_top(self):
        self._stack.pop()
        self._next()  # todo: why need pointer advance?

    def _op_return_value(self):
        result = self._stack.pop()
        assert result is None
        self.gen_func_instruction(cc.OpReturn)
        self._next() # todo: why need pointer advance?

    def _op_load_fast(self):
        # store a variable that is used in an inner scope.
        i = self._next()
        name = self._co.co_varnames[i]
        if name == "input":
            ob = "input"
        elif name == "output":
            ob = "output"
        elif name in self._aliases:
            ob = self._aliases[name]
        else:
            raise NameError(f"Using invalid variable: {name}")
        self._stack.append(ob)

    def _op_load_const(self):
        i = self._next()
        ob = self._co.co_consts[i]
        if isinstance(ob, str):
            # We use strings in e.g. input.define(), mmm
            self._stack.append(ob)
        elif isinstance(ob, (float, int, bool)):
            id, type_id = self.create_object(type(ob))
            if isinstance(ob, float):
                bb = struct.pack("<f", ob)
            elif isinstance(ob, int):
                bb = struct.pack("<i", ob)
            elif isinstance(ob, bool):
                bb = struct.pack("<I", 0xffffffff if ob else 0)
            self.gen_func_instruction(cc.OpConstant, type_id, id, bb)
            self._stack.append(id)
        elif ob is None:
            self._stack.append(None)  # todo: for the final return ...
        else:
            raise NotImplementedError()

    def _op_load_global(self):
        i = self._next()
        name = self._co.co_names[i]
        if name in _types.spirv_types_map:
            ob = _types.spirv_types_map[name]
        else:
            raise NotImplementedError(f"Cannot yet load global {name}")
        self._stack.append(ob)

    def _op_load_attr(self):
        i = self._next()
        name = self._co.co_names[i]
        ob = self._stack.pop()

        sub_ob = "TODO: index of spirv object"

        if ob == "input":
            if name not in self._input:
                raise NameError(f"No input {name} defined.")
            type, var_id = self._input[name]
            id, type_id = self.create_object(type)
            self.gen_func_instruction(cc.OpLoad, type_id, id, var_id)
            sub_ob = id
        elif ob == "output":
            raise AttributeError("Cannot read from output.")
        else:
            raise NotImplementedError()

        self._stack.append(sub_ob)

    def _op_store_attr(self):
        i = self._next()
        name = self._co.co_names[i]
        ob = self._stack.pop()
        value = self._stack.pop()
        assert isinstance(value, IdInt)

        if ob == "input":
            raise AttributeError("Cannot assign to input.")
        elif ob == "output":
            if name not in self._output:
                raise NameError(f"No output {name} defined.")
            type, var_id = self._output[name]
            self.gen_func_instruction(cc.OpStore, var_id, value)
        else:
            raise NotImplementedError()

    def _op_store_fast(self):
        i = self._next()
        name = self._co.co_varnames[i]
        ob = self._stack.pop()
        self._aliases[name] = ob

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
        func = self._stack.pop()
        result = func(ob, *args)
        self._stack.append(result)

    def _op_call_function(self):
        nargs = self._next()
        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        func = self._stack.pop()

        if issubclass(func, _types.BaseVector):
            result = self._vector_packing(func, args)
            self._stack.append(result)

        else:
            raise NotImplementedError()

    def _vector_packing(self, vector_type, args):

        n, t = vector_type._n, vector_type._t  # noqa
        type_id = self.get_type_id(t)
        composite_ids = []

        # Deconstruct
        for arg in args:
            if not isinstance(arg, IdInt):
                raise RuntimeError("Expected a SpirV object")
            element_type = self.get_type_from_id(arg)
            if element_type in (float, int, bool):
                assert element_type is t, "vector type mismatch"
                composite_ids.append(arg)
            elif issubclass(element_type, _types.BaseVector):
                assert element_type._t is t, "vector type mismatch"
                for i in range(element_type._n):
                    comp_id = self.create_id("composite")
                    self.gen_func_instruction(
                        cc.OpCompositeExtract, type_id, comp_id, arg, i
                    )
                    composite_ids.append(comp_id)
            else:
                raise TypeError(f"Invalid type to compose vector: {element_type}")

        # Check the length
        if len(composite_ids) != n:
            raise TypeError(
                f"{vector_type} did not expect {len(composite_ids)} elements"
            )

        # Construct
        result_id, vector_type_id = self.create_object(vector_type)
        self.gen_func_instruction(
            cc.OpCompositeConstruct, vector_type_id, result_id, *composite_ids
        )
        return result_id
