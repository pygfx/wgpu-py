"""
Strive for full coverage of the codegen utils module.
"""

from codegen.utils import (
    remove_c_comments,
    blacken,
    Patcher,
    to_snake_case,
    to_camel_case,
)

from pytest import raises


def dedent(code):
    return code.replace("\n    ", "\n")


def test_to_snake_case():
    assert to_snake_case("foo_bar_spam") == "foo_bar_spam"
    assert to_snake_case("_foo_bar_spam") == "_foo_bar_spam"
    assert to_snake_case("fooBarSpam") == "foo_bar_spam"
    assert to_snake_case("_fooBarSpam") == "_foo_bar_spam"
    assert to_snake_case("maxTextureDimension1D") == "max_texture_dimension1d"


def test_to_camel_case():
    assert to_camel_case("foo_bar_spam") == "fooBarSpam"
    assert to_camel_case("_foo_bar_spam") == "_fooBarSpam"
    assert to_camel_case("fooBarSpam") == "fooBarSpam"
    assert to_camel_case("_fooBarSpam") == "_fooBarSpam"
    assert to_camel_case("max_texture_dimension1d") == "maxTextureDimension1D"


def test_remove_c_comments():
    code1 = """
    x1 hello// comment
    // comment
    x2 hello/* comment */
    x3/* comment */ hello
    x4 /* comment
    comment
    */hello
    """

    code3 = """
    x1 hello

    x2 hello
    x3 hello
    x4 hello
    """

    code1, code3 = dedent(code1), dedent(code3)

    code2 = remove_c_comments(code1)

    assert code2 == code3


def test_blacken_singleline():
    code1 = """
    def foo():
        pass
    def foo(
    ):
        pass
    def foo(
        a1, a2, a3
    ):
        pass
    def foo(
        a1, a2, a3,
    ):
        pass
    def foo(
        a1,
        a2,
        a3,
    ):
        pass
    """

    code2 = """
    def foo():
        pass
    def foo():
        pass
    def foo(a1, a2, a3):
        pass
    def foo(a1, a2, a3):
        pass
    def foo(a1, a2, a3):
        pass
    """

    code1 = dedent(code1).strip()
    code2 = dedent(code2).strip()

    code3 = blacken(code1, True)
    code3 = code3.replace("\n\n", "\n").replace("\n\n", "\n").strip()

    assert code3 == code2

    # Also test simply long lines
    code = "foo = 1" + " + 1" * 100
    assert len(code) > 300
    assert code.count("\n") == 0
    assert blacken(code, False).strip().count("\n") > 3
    assert blacken(code, True).strip().count("\n") == 0


def test_blacken_comments():
    code1 = """
    def foo():  # hi
        pass
    def foo(
        a1, # hi
        a2, # ha
        a3,
    ):  # ho
        pass
    """

    code2 = """
    def foo():  # hi
        pass
    def foo(a1, a2, a3):  # hi ha ho
        pass
    """

    code1 = dedent(code1).strip()
    code2 = dedent(code2).strip()

    code3 = blacken(code1, True)
    code3 = code3.replace("\n\n", "\n").replace("\n\n", "\n").strip()

    assert code3 == code2


def test_patcher():
    code = """
    class Foo1:
        def bar1(self):
            pass
        def bar2(self):
            pass
        @property
        def bar3(self):
            pass

    class Foo2:
        def bar1(self):
            pass
        @property
        def bar2(self):
            pass
        def bar3(self):
            pass
    """

    code = blacken(dedent(code))
    p = Patcher(code)

    # Dump before doing anything, should yield original
    assert p.dumps() == code

    # Check iter_lines
    lines = []
    for line, i in p.iter_lines():
        assert isinstance(line, str)
        assert isinstance(i, int)
        lines.append(line)
    assert "\n".join(lines).strip() == code.strip()

    # Check iter_properties
    names = []
    for classname, i1, i2 in p.iter_classes():
        for funcname, j1, j2 in p.iter_properties(i1 + 1):
            names.append(classname + "." + funcname)
    assert names == ["Foo1.bar3", "Foo2.bar2"]

    # Check iter_methods
    names = []
    for classname, i1, i2 in p.iter_classes():
        for funcname, j1, j2 in p.iter_methods(i1 + 1):
            names.append(classname + "." + funcname)
    assert names == ["Foo1.bar1", "Foo1.bar2", "Foo2.bar1", "Foo2.bar3"]

    # Check insert_line (can insert into same line multiple times
    p = Patcher(code)
    for classname, i1, i2 in p.iter_classes():
        p.insert_line(i1, "# a class")
        p.insert_line(i1, "# a class")
    code2 = p.dumps()
    assert code2.count("# a class") == 4

    # Check replace_line (can only replace one time per line)
    p = Patcher(code2)
    for line, i in p.iter_lines():
        if line.lstrip().startswith("#"):
            p.replace_line(i, "# comment")
            with raises(Exception):
                p.replace_line(i, "# comment")
    code2 = p.dumps()
    assert code2.count("#") == 4
    assert code2.count("# comment") == 4

    # Remove comments
    p = Patcher(code2)
    for line, i in p.iter_lines():
        if line.lstrip().startswith("#"):
            p.remove_line(i)
    code2 = p.dumps()
    assert code2.count("#") == 0

    # We should be back to where we started
    assert code2 == code


def test_patcher2():
    code = """
    class Foo1:
        def bar1(self):
            pass
        @property
        def bar2(self):
            pass
    """

    p = Patcher(dedent(code))

    # Check property line indices
    for classname, i1, i2 in p.iter_classes():
        for funcname, j1, j2 in p.iter_properties(i1 + 1):
            line = p.lines[j1].lstrip()
            assert line.startswith("def")
            assert funcname in line
            assert "pass" in p.lines[j2]

    # Check method line indices
    for classname, i1, i2 in p.iter_classes():
        for funcname, j1, j2 in p.iter_methods(i1 + 1):
            line = p.lines[j1].lstrip()
            assert line.startswith("def")
            assert funcname in line
            assert "pass" in p.lines[j2]


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
