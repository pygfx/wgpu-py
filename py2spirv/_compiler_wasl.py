from textx import metamodel_from_str

from ._module import SpirVModule
from ._generator_bc import Bytecode2SpirVGenerator
from . import _generator_bc as bc


grammar = """
Program: Procedure;
Comment: /#.*$/;
Procedure: 'fn' name=ID '(' params*=IOParameter[',']  ','? ')' '{' body=Body '}';
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


def wasl2spirv(code, shader_type=None):
    """ Compile WASL code to SpirV and return as a SpirVModule object.

    WASL is our own defined domain specific language (DSL) to write shaders.
    It is highly experimental. The code is parsed using textx, the resulting
    AST is converted to bytecode, from which the SpirV is generated.
    """
    if not isinstance(code, str):
        raise TypeError("wasl2spirv expects a string.")

    ast = meta_model.model_from_str(code)

    converter = Wasl2Bytecode()
    converter.convert(ast)
    bytecode = converter.dump()

    generator = Bytecode2SpirVGenerator()
    generator.generate(bytecode, shader_type)
    bb = generator.to_bytes()

    m = SpirVModule(code, bb, "compiled from WASL")
    m.gen = generator
    return m


class Wasl2Bytecode:
    """ Compile WASL AST to bytecode.
    """

    def convert(self, ast):
        self._opcodes = []
        self.visit(ast)

    def dump(self):
        return self._opcodes

    def emit(self, opcode, arg):
        self._opcodes.append((opcode, arg))

    def visit(self, node):

        method_name = "visit_" + node.__class__.__name__
        getattr(self, method_name)(node)

    def visit_Procedure(self, node):
        name = node.name
        for param in node.params:
            if param.mode == "input":
                self.emit(bc.CO_INPUT, (param.name, param.location, param.type))
            elif param.mode == "output":
                self.emit(bc.CO_OUTPUT, (param.name, param.location, param.type))
            elif param.mode == "uniform":
                raise NotImplementedError()
            else:
                raise TypeError(
                    f"Funcion argument {param.name} must be input, output or uniform, not {param.mode}."
                )

        for node in node.body.expressions:
            self.visit(node)

    def visit_Assignment(self, node):
        self.visit(node.rhs)
        self.emit(bc.CO_STORE, node.lhs)

    def visit_Sum(self, node):
        self.visit(node.lhs)
        for term in node.rhs:
            self.visit(term)
            1 / 0

    def visit_Term(self, node):
        self.visit(node.lhs)
        for term_rhs in node.rhs:
            self.visit(term_rhs.value)
            self.emit(bc.CO_BINARY_OP, term_rhs.op)

    def visit_Identifier(self, node):
        self.emit(bc.CO_LOAD, node.name)

    def visit_IdentifierIndexed(self, node):
        self.emit(bc.CO_LOAD, node.name)
        self.visit(node.index)
        self.emit(bc.CO_INDEX, None)

    def visit_Number(self, node):
        self.emit(bc.CO_LOAD_CONSTANT, node.value)

    def visit_CallExpr(self, node):
        self.emit(bc.CO_LOAD, node.name)
        for arg in node.args:
            self.visit(arg)
        self.emit(bc.CO_CALL, len(node.args))
