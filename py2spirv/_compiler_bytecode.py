from dis import dis as pprint_bytecode

from ._compiler import BaseSpirVCompiler, str_to_words, STORAGE_CLASSES
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
        else:
            raise NotImplementedError()
        self._stack.append(ob)

    def _op_load_const(self):
        i = self._next()
        ob = self._co.co_consts[i]
        self._stack.append(ob)
        # todo: in some cases we need to generate OpConstant ...

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
            # todo: generate OpLoad
            name, id, type, type_id = self.get_variable_info("input:" + name)
            load_id = self.create_id("WUUUUUUT?")
            self.gen_func_instruction(cc.OpLoad, type_id, load_id, id)
            sub_ob = load_id
        elif ob == "output":
            if name not in self._output:
                raise NameError(f"No output {name} defined.")
            # todo: generate OpStore
            1/0
        else:
            raise NotImplementedError()

        self._stack.append(sub_ob)

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

        if func is vec4:
            1/0

        raise NotImplementedError()
