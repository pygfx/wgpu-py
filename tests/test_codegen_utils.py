from testutils import run_tests
from wgpu.codegen.utils import blacken, Patcher, to_snake_case, to_camel_case


def dedent(code):
    return code.replace("\n    ", "\n")


def test_to_snake_case():
    assert to_snake_case("foo_bar_spam") == "foo_bar_spam"
    assert to_snake_case("_foo_bar_spam") == "_foo_bar_spam"
    assert to_snake_case("fooBarSpam") == "foo_bar_spam"
    assert to_snake_case("_fooBarSpam") == "_foo_bar_spam"


def test_to_camel_case():
    assert to_camel_case("foo_bar_spam") == "fooBarSpam"
    assert to_camel_case("_foo_bar_spam") == "_fooBarSpam"
    assert to_camel_case("fooBarSpam") == "fooBarSpam"
    assert to_camel_case("_fooBarSpam") == "_fooBarSpam"


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

    lines = []
    for line, i in p.iter_lines():
        assert isinstance(line, str)
        assert isinstance(i, int)
        lines.append(line)
    assert "\n".join(lines).strip() == code.strip()

    names = []
    for classname, i1, i2 in p.iter_classes():
        for funcname, j1, j2 in p.iter_properties(i1 + 1):
            names.append(classname + "." + funcname)
    assert names == ["Foo1.bar3", "Foo2.bar2"]

    names = []
    for classname, i1, i2 in p.iter_classes():
        for funcname, j1, j2 in p.iter_methods(i1 + 1):
            names.append(classname + "." + funcname)
    assert names == ["Foo1.bar1", "Foo1.bar2", "Foo2.bar1", "Foo2.bar3"]


if __name__ == "__main__":
    run_tests(globals())
