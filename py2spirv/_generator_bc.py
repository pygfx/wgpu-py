import struct

from ._generator_base import IdInt, BaseSpirVGenerator
from . import _spirv_constants as cc
from . import _types


CO_INPUT = "CO_INPUT"
CO_OUTPUT = "CO_OUTPUT"
CO_UNIFORM = "CO_UNIFORM"
CO_ASSIGN = "CO_ASSIGN"
CO_LOAD_CONSTANT = "CO_LOAD_CONSTANT"
CO_LOAD = "CO_LOAD"
CO_BINARY_OP = "CO_BINARY_OP"
CO_STORE = "CO_STORE"
CO_CALL = "CO_CALL"
CO_INDEX = "CO_INDEX"
CO_POP_TOP = "CO_POP_TOP"
CO_BUILD_ARRAY = "CO_BUILD_ARRAY"


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
        self._uniform = {}
        self._aliases = {}

        # Declare funcion
        return_type_id = self.get_type_id(_types.void)
        func_type_id = self.create_id("func_declaration")
        self.gen_instruction(
            "types", cc.OpTypeFunction, func_type_id, return_type_id
        )  # 0 args

        # Start function definition
        func_id = self._entry_point_id
        func_control = 0  # can specify whether it should inline, etc.
        self.gen_func_instruction(
            cc.OpFunction, return_type_id, func_id, func_control, func_type_id
        )
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

    def _op_input(self, args):
        location, *name_type_pairs = args
        self._setup_io_variable("input", location, name_type_pairs)

    def _op_output(self, args):
        location, *name_type_pairs = args
        self._setup_io_variable("output", location, name_type_pairs)

    def _op_uniform(self, args):
        binding, *name_type_pairs = args
        self._setup_io_variable("uniform", binding, name_type_pairs)

    def _setup_io_variable(self, kind, location, name_type_pairs):

        n_names = len(name_type_pairs) / 2
        singleton_mode = n_names ==1 and kind != "uniform"

        # Triage over input kind
        if kind == "input":
            storage_class, iodict = cc.StorageClass_Input, self._input
        elif kind == "output":
            storage_class, iodict = cc.StorageClass_Output, self._output
        elif kind == "uniform":  # location == binding
            storage_class, iodict = cc.StorageClass_Uniform, self._uniform
        else:
            raise RuntimeError(f"Invalid IO kind {kind}")

        # Get the root variable
        if singleton_mode:
            # Singleton (not allowed for Uniform)
            name, type_str = name_type_pairs
            var_type = _types.spirv_types_map[type_str]
            var_id, var_type_id = self.create_object(var_type)  # todo: or f"{kind}.{name}"
        else:
            # todo: TBH I am not sure if this is allowed for non-uniforms :D
            assert kind == "uniform", f"euhm, I dont know if you can use block {kind}s"
            # Block - the variable is a struct
            subtypes = {}
            for i in range(0, len(name_type_pairs), 2):
                key, subtype_str = name_type_pairs[i], name_type_pairs[i+1]
                subtypes[key] = _types.spirv_types_map[subtype_str]
            var_type = _types.Struct(**subtypes)
            var_id, var_type_id = self.create_object(var_type)
            # Define Variable as block
            self.gen_instruction(
                "annotations", cc.OpDecorate, var_id, cc.Decoration_Block
            )

        # Define location of variable
        if isinstance(location, int):
            self.gen_instruction(
                "annotations", cc.OpDecorate, var_id, cc.Decoration_Location, location
            )
        elif isinstance(location, str):
            try:
                location = cc.builtins[location]
            except KeyError:
                raise NameError(f"Not a known builtin io variable: {location}")
            self.gen_instruction(
                "annotations", cc.OpDecorate, var_id, cc.Decoration_BuiltIn, location
            )

        # Create a variable (via a pointer)
        var_pointer_id = self.create_id("pointer")
        self.gen_instruction(
            "types", cc.OpTypePointer, var_pointer_id, storage_class, var_type_id
        )
        self.gen_instruction(
            "types", cc.OpVariable, var_pointer_id, var_id, storage_class
        )

        # Store internal info to derefererence the variables
        if singleton_mode:
            if name in iodict:
                raise NameError(f"{kind} {name} already exists")
            iodict[name] = var_type, var_id  # the code only needs var_id
        else:
            for i, subname in enumerate(subtypes):
                subtype = subtypes[subname]
                sub_pointer_id, subtype_id = self.create_object(subtype)
                self.gen_instruction(
                    "types", cc.OpTypePointer, sub_pointer_id, storage_class, subtype_id
                )
                index_id, index_type_id = self.create_object(_types.i32)
                # todo: can re-use constants!
                self.gen_instruction(
                    "types", cc.OpConstant, index_type_id, index_id, struct.pack("<i", i)
                )
                if subname in iodict:
                    raise NameError(f"{kind} {subname} already exists")
                iodict[subname] = subtype, sub_pointer_id, var_id, index_id

    def _op_load(self, name):
        # store a variable that is used in an inner scope.
        if name in self._aliases:
            ob = self._aliases[name]
        elif name in self._input:
            io_args = self._input[name]
            if len(io_args) == 4:
                type, pointer_id, var_id, index_id = io_args
                temp_id = self.create_id("struct-field")
                self.gen_func_instruction(cc.OpAccessChain, pointer_id, temp_id, var_id, index_id)
                id, type_id = self.create_object(type)
                self.gen_func_instruction(cc.OpLoad, type_id, id, temp_id)
            else:
                type, pointer_id = io_args
                id, type_id = self.create_object(type)
                self.gen_func_instruction(cc.OpLoad, type_id, id, pointer_id)
            ob = id
        elif name in self._uniform:
            type, pointer_id, var_id, index_id = self._uniform[name]
            temp_id = self.create_id("struct-field")
            self.gen_func_instruction(cc.OpAccessChain, pointer_id, temp_id, var_id, index_id)
            id, type_id = self.create_object(type)
            self.gen_func_instruction(cc.OpLoad, type_id, id, temp_id)
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
            io_args = self._output[name]
            if len(io_args) == 4:  # Struct
                type, pointer_id, var_id, index_id = io_args
                # type_id = self.get_type_id(type)
                id = self.create_id("struct-field")
                self.gen_func_instruction(cc.OpAccessChain, pointer_id, id, var_id, index_id)
                self.gen_func_instruction(cc.OpStore, id, ob)
            else:  # Simple
                type, pointer_id = io_args  # pointer is a Variable
                self.gen_func_instruction(cc.OpStore, pointer_id, ob)

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

    def _op_build_array(self, nargs):
        # Literal array
        args = self._stack[-nargs:]
        self._stack[-nargs:] = []
        result = self._array_packing(args)
        self._stack.append(result)

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

        assert (
            len(composite_ids) >= 2
        ), "When constructing a vector, there must be at least two Constituent operands."

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
        array_type = _types.Array(n, element_type)

        result_id, type_id = self.create_object(array_type)
        self.gen_func_instruction(
            cc.OpCompositeConstruct, type_id, result_id, *composite_ids
        )
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
            1 / 0
        elif op == "+":
            1 / 0
        elif op == "-":
            1 / 0
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
            self.gen_instruction(
                "types",
                cc.OpTypePointer,
                container_variable_type,
                cc.StorageClass_Function,
                container_type_id,
            )
            self.gen_func_instruction(
                cc.OpVariable,
                container_variable_type,
                container_variable,
                cc.StorageClass_Function,
            )
            self.gen_func_instruction(cc.OpStore, container_variable, container_id)

            # Prepare result id and type
            result_id, result_type_id = self.create_object(element_type)

            # Create pointer into the array
            pointer1 = self.create_id("pointer")
            pointer2 = self.create_id("pointer")
            self.gen_instruction(
                "types",
                cc.OpTypePointer,
                pointer1,
                cc.StorageClass_Function,
                result_type_id,
            )
            self.gen_func_instruction(
                cc.OpInBoundsAccessChain, pointer1, pointer2, container_variable, index
            )

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
