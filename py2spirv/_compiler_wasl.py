import struct
from textx import metamodel_from_str

from ._compiler import IdInt, BaseSpirVCompiler, str_to_words
from . import _spirv_constants as cc
from . import _types


grammar = """
Program: Procedure;
Comment: /#.*$/;
Procedure: 'fn' name=ID '(' params+=IOParameter[',']  ','? ')' '{' body=Body '}';
IOParameter: name=ID ':' mode=ID type=ID location=Location;
Location: INT | ID;
Parameter: name=ID ':' type=ID;
Body: expressions+=Statement;
Statement: Assignment | Expression;
Expression: CallExpr | Sum;
CallExpr: name=ID '(' args+=Expression[','] ','? ')';
Assignment: lhs=ID '=' rhs=Expression;
Sum: lhs=Term rhs*=SumRHS;
SumRHS: op=AddOp value=Term;
Term: lhs=Factor rhs*=TermRHS;
TermRHS: op=MulOp value=Factor;
Factor: IdentifierIndexed | Identifier | Number;
MulOp: '*' | '/';
AddOp: '+' | '-';
Number: value=FLOAT;
Identifier: name=ID;
IdentifierIndexed: name=ID '[' index=Expression ']';
""".lstrip()


meta_model = metamodel_from_str(grammar, classes=[])


CO_INPUT = "CO_INPUT"
CO_ASSIGN = "CO_ASSIGN"
CO_LOAD_CONSTANT = "CO_LOAD_CONSTANT"
CO_LOAD = "CO_LOAD"
CO_BINARY_OP = "CO_BINARY_OP"
CO_STORE = "CO_STORE"
CO_CALL = "CO_CALL"
CO_INDEX = "CO_INDEX"


class Ast2Bytecode:

    def __init__(self):
        self._opcodes = []
        self._input = {}
        self._output = {}

    def emit(self, opcode, arg):
        self._opcodes.append((opcode, arg))

    def dump(self):
        return self._opcodes, self._input, self._output

    def visit(self, node):

        method_name = "visit_" + node.__class__.__name__
        getattr(self, method_name)(node)

    def visit_Procedure(self, node):
        name = node.name
        for param in node.params:
            if param.mode == "input":
                self._input[param.name] = param.type, param.location
            elif param.mode == "output":
                self._output[param.name] = param.type, param.location
            elif param.mode == "uniform":
                raise NotImplementedError()
            else:
                raise TypeError(f"Funcion argument {param.name} must be input, output or uniform, not {param.mode}.")

        for node in node.body.expressions:
            self.visit(node)

    def visit_Assignment(self, node):
        self.visit(node.rhs)
        self.emit(CO_STORE, node.lhs)

    def visit_Sum(self, node):
        self.visit(node.lhs)
        for term in node.rhs:
            self.visit(term)
            1/0

    def visit_Term(self, node):
        self.visit(node.lhs)
        for term_rhs in node.rhs:
            self.visit(term_rhs.value)
            self.emit(CO_BINARY_OP, term_rhs.op)

    def visit_Identifier(self, node):
        self.emit(CO_LOAD, node.name)

    def visit_IdentifierIndexed(self, node):
        self.visit(node.index)
        self.emit(CO_INDEX, node.name)

    def visit_Number(self, node):
        self.emit(CO_LOAD_CONSTANT, node.value)

    def visit_CallExpr(self, node):
        for arg in node.args:
            self.visit(arg)
        self.emit(CO_CALL, (node.name, len(node.args)))


class WASL2SpirVCompiler(BaseSpirVCompiler):

    def __init__(self, s):

        model = meta_model.model_from_str(s)
        dinges = Ast2Bytecode()
        dinges.visit(model)
        self._bytecode, self._input, self._output = dinges.dump()

    def _generate_io(self):
        for name in list(self._input.keys()):
            type_str, location = self._input[name]
            type = _types.spirv_types_map[type_str]

            var_id, type_id = self.create_object(type)  # todo: or "input." + name)
            pointer_id = self.create_id("pointer")

            self._input[name] = type, var_id  # the code only needs var_id

            self.gen_instruction("annotations", cc.OpDecorate, var_id, cc.Decoration_Location, location)
            self.gen_instruction("types", cc.OpTypePointer, pointer_id, cc.StorageClass_Input, type_id)
            self.gen_instruction("types", cc.OpVariable, pointer_id, var_id, cc.StorageClass_Input)

        for name in list(self._output.keys()):
            type_str, location = self._output[name]
            type = _types.spirv_types_map[type_str]

            var_id, type_id = self.create_object(type)  # todo: or "output." + name)
            pointer_id = self.create_id("pointer")

            self._output[name] = type, var_id

            self.gen_instruction("annotations", cc.OpDecorate, var_id, cc.Decoration_Location, location)
            self.gen_instruction("types", cc.OpTypePointer, pointer_id, cc.StorageClass_Output, type_id)
            self.gen_instruction("types", cc.OpVariable, pointer_id, var_id, cc.StorageClass_Output)

    def _generate(self):

        self._stack = []
        self._aliases = {}

        self._generate_io()

        # Parse
        for opcode, arg in self._bytecode:
            method_name = "_op_" + opcode[3:].lower()
            method = getattr(self, method_name, None)
            if method is None:
                # pprint_bytecode(self._co)
                raise RuntimeError(f"Cannot parse {opcode} yet (no {method_name}()).")
            else:
                method(arg)

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
            id, type_id = self.create_object(type(ob))
            if isinstance(ob, float):
                bb = struct.pack("<f", ob)
            elif isinstance(ob, int):
                bb = struct.pack("<i", ob)
            elif isinstance(ob, bool):
                bb = struct.pack("<I", 0xffffffff if ob else 0)
            self.gen_func_instruction(cc.OpConstant, type_id, id, bb)
            self._stack.append(id)
        else:
            raise NotImplementedError()

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
            if issubclass(type, _types.BaseVector):
                result = self._vector_packing(type, args)
            elif type is _types.array:
                result = self._array_packing(type, args)
            else:
                raise NotImplementedError()
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

    def _array_packing(self, array_type, args):
        1/0

    def _op_binary_op(self, op):
        right = self._stack.pop()
        left = self._stack.pop()
        right_type = self.get_type_from_id(right)
        left_type = self.get_type_from_id(left)

        assert left_type is _types.vec3
        assert right_type is _types.float

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
