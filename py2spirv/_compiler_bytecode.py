import struct

from ._compiler import BaseSpirVCompiler
from ._compiler import IdInt, BaseSpirVCompiler, str_to_words
from . import _spirv_constants as cc
from . import _types


CO_INPUT = "CO_INPUT"
CO_ASSIGN = "CO_ASSIGN"
CO_LOAD_CONSTANT = "CO_LOAD_CONSTANT"
CO_LOAD = "CO_LOAD"
CO_BINARY_OP = "CO_BINARY_OP"
CO_STORE = "CO_STORE"
CO_CALL = "CO_CALL"
CO_INDEX = "CO_INDEX"


class BytecodeSpirVCompiler(BaseSpirVCompiler):
    """ A compiler that operates on our own well-defined bytecode.
    Use directly with bytecode input, or subclass to accept other kinds of
    input, convert that input to bytecode and then feed that into this base class.
    """

    def __init__(self, bytecode, input, output):
        self._bytecode = bytecode
        self._input = input
        self._output = output

    def _generate_io(self):
        for name in list(self._input.keys()):
            type_str, location = self._input[name]
            type = _types.spirv_types_map[type_str]

            var_id, type_id = self.create_object(type)  # todo: or "input." + name)
            pointer_id = self.create_id("pointer")

            self._input[name] = type, var_id  # the code only needs var_id

            # Define location
            assert isinstance(location, (int, str))
            if isinstance(location, int):
                self.gen_instruction("annotations", cc.OpDecorate, var_id, cc.Decoration_Location, location)
            else:
                try:
                    location = cc.builtins[location]
                except KeyError:
                    raise NameError(f"Not a known builtin io variable: {location}")
                self.gen_instruction("annotations", cc.OpDecorate, var_id, cc.Decoration_BuiltIn, location)

            # Create a variable (via a pointer)
            self.gen_instruction("types", cc.OpTypePointer, pointer_id, cc.StorageClass_Input, type_id)
            self.gen_instruction("types", cc.OpVariable, pointer_id, var_id, cc.StorageClass_Input)

        for name in list(self._output.keys()):
            type_str, location = self._output[name]
            type = _types.spirv_types_map[type_str]

            var_id, type_id = self.create_object(type)  # todo: or "output." + name)
            pointer_id = self.create_id("pointer")

            self._output[name] = type, var_id

            # Define location
            assert isinstance(location, (int, str))
            if isinstance(location, int):
                self.gen_instruction("annotations", cc.OpDecorate, var_id, cc.Decoration_Location, location)
            else:
                try:
                    location = cc.builtins[location]
                except KeyError:
                    raise NameError(f"Not a known builtin io variable: {location}")
                self.gen_instruction("annotations", cc.OpDecorate, var_id, cc.Decoration_BuiltIn, location)

            # Create a variable (via a pointer)
            self.gen_instruction("types", cc.OpTypePointer, pointer_id, cc.StorageClass_Output, type_id)
            self.gen_instruction("types", cc.OpVariable, pointer_id, var_id, cc.StorageClass_Output)

    def _generate(self):

        self._stack = []
        self._aliases = {}

        self._generate_io()

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
        for opcode, arg in self._bytecode:
            method_name = "_op_" + opcode[3:].lower()
            method = getattr(self, method_name, None)
            if method is None:
                # pprint_bytecode(self._co)
                raise RuntimeError(f"Cannot parse {opcode} yet (no {method_name}()).")
            else:
                method(arg)

        # End function definition
        self.gen_func_instruction(cc.OpReturn)
        self.gen_func_instruction(cc.OpFunctionEnd)

    def _op_load(self, name):
        # store a variable that is used in an inner scope.
        if name in self._aliases:
            ob = self._aliases[name]
        elif name in self._input:
            type, var_id = self._input[name]
            id, type_id = self.create_object(type)
            self.gen_func_instruction(cc.OpLoad, type_id, id, var_id)
            ob = id
        else:
            raise NameError(f"Using invalid variable: {name}")
        self._stack.append(ob)

    def _op_load_constant(self, ob):
        if isinstance(ob, (float, int, bool)):
            if isinstance(ob, float):
                id, type_id = self.create_object(_types.f32)
                bb = struct.pack("<f", ob)
                self.gen_instruction("types", cc.OpConstant, type_id, id, bb)
                # bb = struct.pack("<d", ob)
                # self.gen_instruction("types", cc.OpConstant, type_id, id, bb[:4], bb[4:])
            elif isinstance(ob, int):
                id, type_id = self.create_object(_types.i32)
                bb = struct.pack("<i", ob)
                self.gen_instruction("types", cc.OpConstant, type_id, id, bb)
            elif isinstance(ob, bool):
                id, type_id = self.create_object(_types.boolean)
                op = cc.OpConstantTrue if ob else cc.OpConstantFalse
                self.gen_instruction("types", op, type_id, id)
            else:
                raise NotImplementedError()
            self._stack.append(id)
        else:
            raise NotImplementedError()

        # Also see OpConstantNull OpConstantSampler OpConstantComposite

    def _op_load_global(self):
        raise NotImplementedError()

    def _op_store(self, name):
        ob = self._stack.pop()
        if name in self._output:
            type, var_id = self._output[name]
            self.gen_func_instruction(cc.OpStore, var_id, ob)
        self._aliases[name] = ob

    def _op_call(self, arg):
        name, nargs = arg

        args = self._stack[-nargs:]
        self._stack[-nargs:] = []

        if name in _types.spirv_types_map:
            type = _types.spirv_types_map[name]
            if issubclass(type, _types.Vector):
                result = self._vector_packing(type, args)
            elif issubclass(type, _types.Array):
                result = self._array_packing(args)
            else:
                raise NotImplementedError()
            self._stack.append(result)
        else:
            raise NotImplementedError()

    def _vector_packing(self, vector_type, args):

        n, t = vector_type.length, vector_type.subtype  # noqa
        type_id = self.get_type_id(t)
        composite_ids = []

        # Deconstruct
        for arg in args:
            if not isinstance(arg, IdInt):
                raise RuntimeError("Expected a SpirV object")
            element_type = self.get_type_from_id(arg)
            if issubclass(element_type, _types.Scalar):
                assert element_type is t, "vector type mismatch"
                composite_ids.append(arg)
            elif issubclass(element_type, _types.Vector):
                # todo: a contiguous subset of the scalars consumed can be represented by a vector operand instead!
                # -> I think this means we can simply do composite_ids.append(arg)
                assert element_type.subtype is t, "vector type mismatch"
                for i in range(element_type.length):
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

        assert len(composite_ids) >= 2, "When constructing a vector, there must be at least two Constituent operands."

        # Construct
        result_id, vector_type_id = self.create_object(vector_type)
        self.gen_func_instruction(
            cc.OpCompositeConstruct, vector_type_id, result_id, *composite_ids
        )
        # todo: or OpConstantComposite
        return result_id

    def _array_packing(self, args):
        n = len(args)
        if n == 0:
            raise IndexError("No support for zero-sized arrays.")

        # Check that all args have the same type
        element_type = self.get_type_from_id(args[0])
        composite_ids = args
        for arg in args:
            assert self.get_type_from_id(arg) is element_type, "array type mismatch"

        # Create array class
        array_type = _types.Array(element_type, n)

        result_id, type_id = self.create_object(array_type)
        self.gen_func_instruction(cc.OpCompositeConstruct, type_id, result_id, *composite_ids)
        # todo: or OpConstantComposite

        return result_id


    def _op_binary_op(self, op):
        right = self._stack.pop()
        left = self._stack.pop()
        right_type = self.get_type_from_id(right)
        left_type = self.get_type_from_id(left)

        assert left_type is _types.vec3
        assert issubclass(right_type, _types.Float)

        if op == "*":
            id, type_id = self.create_object(left_type)
            self.gen_func_instruction(cc.OpVectorTimesScalar, type_id, id, left, right)
        elif op == "/":
            1/0
        elif op == "+":
            1/0
        elif op == "-":
            1/0
        else:
            raise NotImplementedError(f"Wut is {op}??")
        self._stack.append(id)

    def _op_index(self, name):

        # Select object
        if name in self._aliases:
            container_id = self._aliases[name]
        else:
            raise NameError(f"Unknown variable {name}.")

        # Get type of object and index
        container_type = self.get_type_from_id(container_id)
        element_type = container_type.subtype
        container_type_id = self.get_type_id(container_type)
        index = self._stack.pop()

        # assert self.get_type_from_id(index) is int

        if issubclass(container_type, _types.Array):

            # todo: maybe ... the variable for a constant should be created only once ... instead of every time it gets indexed
            # Put the array into a variable
            container_variable = self.create_id("variable")
            container_variable_type = self.create_id("pointer_type")
            self.gen_instruction("types", cc.OpTypePointer, container_variable_type, cc.StorageClass_Function, container_type_id)
            self.gen_func_instruction(cc.OpVariable, container_variable_type, container_variable, cc.StorageClass_Function)
            self.gen_func_instruction(cc.OpStore, container_variable, container_id)

            # Prepare result id and type
            result_id, result_type_id = self.create_object(element_type)

            # Create pointer into the array
            pointer1 = self.create_id("pointer")
            pointer2 = self.create_id("pointer")
            self.gen_instruction("types", cc.OpTypePointer, pointer1, cc.StorageClass_Function, result_type_id)
            self.gen_func_instruction(cc.OpInBoundsAccessChain, pointer1, pointer2, container_variable, index)

            # Load the element from the array
            self.gen_func_instruction(cc.OpLoad, result_type_id, result_id, pointer2)
        else:
            raise NotImplementedError()

        self._stack.append(result_id)

        # OpAccessChain: Create a pointer into a composite object that can be used with OpLoad and OpStore.

        # OpVectorExtractDynamic: Extract a single, dynamically selected, component of a vector.
        # OpVectorInsertDynamic: Make a copy of a vector, with a single, variably selected, component modified.
        # OpVectorShuffle: Select arbitrary components from two vectors to make a new vector.
        # OpCompositeInsert: Make a copy of a composite object, while modifying one part of it. (updating an element)


    def _op_if(self):
        raise NotImplementedError()
        # OpSelect
