from textx import metamodel_from_str, export


example = """
fn main (foo: input vec3 12,
         bar: output vec3 13
) {

    # js bvjsvb
    bar = foo * 3
}
"""







grammar = """
Program: Procedure;
Comment: /#.*$/;
Procedure: 'fn' name=ID '(' params+=Parameter[','] ')' '{' body=Body '}';
Parameter: name=ID ':' mode=ID type=ID location=INT;
Body: expressions+=Expression;
Expression:
    AssignmentExpr;
AssignmentExpr: lhs=ID rhs*=AssignmentExprRHS;
AssignmentExprRHS: '=' Sum;
Sum: lhs=Term rhs*=SumRHS;
SumRHS: op=AddOp value=Term;
Term: lhs=Factor rhs*=TermRHS;
TermRHS: op=MulOp value=Factor;
Factor: Identifier | Number;
MulOp: '*' | '/';
AddOp: '+' | '-';
Number: value=FLOAT;
Identifier: name=ID;
""".lstrip()

mm = metamodel_from_str(grammar, classes=[])

model = mm.model_from_str(example)
export.model_export(model, "bla.dot")
