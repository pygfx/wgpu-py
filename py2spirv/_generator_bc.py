import struct

from ._generator_base import IdInt, BaseSpirVGenerator
from . import _spirv_constants as cc
from . import _types


CO_INPUT = "CO_INPUT"
CO_OUTPUT = "CO_OUTPUT"
CO_ASSIGN = "CO_ASSIGN"
CO_LOAD_CONSTANT = "CO_LOAD_CONSTANT"
CO_LOAD = "CO_LOAD"
CO_BINARY_OP = "CO_BINARY_OP"
CO_STORE = "CO_STORE"
CO_CALL = "CO_CALL"
CO_INDEX = "CO_INDEX"
CO_POP_TOP = "CO_POP_TOP"


class Bytecode2SpirVGenerator(BaseSpirVGenerator):
    """ A generator that operates on our own well-defined bytecode.

    Bytecode describing a stack machine is a pretty nice representation to generate
    SpirV code, because the code gets visited in a flow, making it easier to
    do type inference. By implementing our own bytecode, we can implement a single
    generator based on that, and use the bytecode as a target for different source
    languages. Also, we can target the bytecode a bit towards SpirV, making this
    class relatively simple. In other words, it separates concerns very well.
    """

    def _generate(self, bytecode):

        self._stack = []
        self._input = {}
        self._output = {}
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
        for opcode, arg in bytecode:
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

    def _op_pop_top(self, arg):
        self._stack.pop()

    def _op_input(self, name_location_type):
        name, location, type_str = name_location_type

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

    def _op_output(self, name_location_type):
        name, location, type_str = name_location_type
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

    def _op_load(self, name):
        # store a variable that is used in an inner scope.
        if name in self._aliases:
            ob = self._aliases[name]
        elif name in self._input:
            type, var_id = self._input[name]
            id, type_id = self.create_object(type)
            self.gen_func_instruction(cc.OpLoad, type_id, id, var_id)
            ob = id
        elif name in _types.spirv_types_map:
            ob = _types.spirv_types_map[name]
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

    def _op_call(self, nargs):

        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        func = self._stack.pop()

        if isinstance(func, type):
            if issubclass(func, _types.Vector):
                result = self._vector_packing(func, args)
            elif issubclass(func, _types.Array):
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

    def _op_index(self, arg):

        index = self._stack.pop()
        container_id = self._stack.pop()

        # Get type of object and index
        container_type = self.get_type_from_id(container_id)
        element_type = container_type.subtype
        container_type_id = self.get_type_id(container_type)

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
