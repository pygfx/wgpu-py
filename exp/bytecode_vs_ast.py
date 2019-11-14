"""
Little experiment that gets bytecode from a function, and the AST.
For the latter, the source is needed, which we obtain via inspect.
When frozen, PyInstaller compiles everything to pyc (or pyo) files
and the source is lost. The bytecode survives though!
"""

import ast
import dis
import inspect

from bytecode_vs_ast2 import bar


def foo(a:int, x) -> float:
    b = a + 1
    print(b)
    return b * 3


def parse_bytecode(func):
    co = func.__code__
    print(f"Bytecode for {co.co_name} in {co.co_filename} at line {co.co_firstlineno}")
    for op in co.co_code:
        print(op, dis.opname[op])


def parse_ast(func):
    pycode = inspect.getsource(func)
    print(f"AST of {pycode.splitlines()[0].strip()}")
    print(pycode.strip())
    print(ast.parse(pycode))


def parse_types(func):
    print(func.__name__, func.__annotations__)


print("========== types in main")
try:
    parse_types(foo)
except Exception as err:
    print(err)
print("========== bytecode in main")
try:
    parse_bytecode(foo)
except Exception as err:
    print(err)
print("==========  ast in main")
try:
    parse_ast(foo)
except Exception as err:
    print(err)


print("========== types in module")
try:
    parse_types(bar)
except Exception as err:
    print(err)
print("========== bytecode in other module")
try:
    parse_bytecode(bar)
except Exception as err:
    print(err)
print("==========  ast in other module")
try:
    parse_ast(bar)
except Exception as err:
    print(err)
