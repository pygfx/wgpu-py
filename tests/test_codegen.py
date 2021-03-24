from testutils import run_tests
from codegen.utils import blacken, Patcher


def dedent(code):
    return code.replace("\n    ", "\n")


def test_patcher():

    code = """
    class Foo1:
        def bar1(self):
            pass
        def bar2(self):
            pass

    class Foo2:
        def bar1(self):
            pass
        def bar2(self):
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
        for funcname, j1, j2 in p.iter_methods(i1 + 1):
            names.append(classname + "." + funcname)
    assert names == ["Foo1.bar1", "Foo1.bar2", "Foo2.bar1", "Foo2.bar2"]


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


if __name__ == "__main__":
    run_tests(globals())

