import dis


def foo(a:int, x):
    b = a + 1
    print(b)
    a = vec3(1, 2, 3)
    return b * 3


def parse(func):

    co = func.__code__

    co.co_code

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
    print(f"Bytecode for {co.co_name} in {co.co_filename} at line {co.co_firstlineno}")

    # https://docs.python.org/3.6/library/dis.html#python-bytecode-instructions
    for op in co.co_code:
        print(op, dis.opname[op])

    # lineno = addr = 0
    # for addr_incr, line_incr in co_lnotab:
    #     addr += addr_incr
    #     if addr > A:
    #         return lineno
    #     lineno += line_incr

parse(foo)
print(dis.dis(foo))