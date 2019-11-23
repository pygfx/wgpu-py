
from textx import metamodel_from_str

from . import _compiler_bytecode as bc


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


class WASL2SpirVCompiler(bc.BytecodeSpirVCompiler):
    """ Compiles a string with WASL shader code to SpirV.
    """

    def __init__(self, s):
        converter = Wasl2Bytecode()
        converter.convert(s)
        bytecode, input, output = converter.dump()
        super().__init__(bytecode, input, output)


class Wasl2Bytecode:
    """ Compile WASL AST to bytecode.
    """

    def convert(self, s):
        self._opcodes = []
        self._input = {}
        self._output = {}
        ast = meta_model.model_from_str(s)
        self.visit(ast)

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
        self.emit(bc.CO_STORE, node.lhs)

    def visit_Sum(self, node):
        self.visit(node.lhs)
        for term in node.rhs:
            self.visit(term)
            1/0

    def visit_Term(self, node):
        self.visit(node.lhs)
        for term_rhs in node.rhs:
            self.visit(term_rhs.value)
            self.emit(bc.CO_BINARY_OP, term_rhs.op)

    def visit_Identifier(self, node):
        self.emit(bc.CO_LOAD, node.name)

    def visit_IdentifierIndexed(self, node):
        self.visit(node.index)
        self.emit(bc.CO_INDEX, node.name)

    def visit_Number(self, node):
        self.emit(bc.CO_LOAD_CONSTANT, node.value)

    def visit_CallExpr(self, node):
        for arg in node.args:
            self.visit(arg)
        self.emit(bc.CO_CALL, (node.name, len(node.args)))
