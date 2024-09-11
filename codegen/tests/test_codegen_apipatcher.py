""" Test some parts of apipatcher.py, and Implicitly tests idlparser.py.
"""

from codegen.utils import blacken
from codegen.apipatcher import CommentRemover, AbstractCommentInjector


def dedent(code):
    return code.replace("\n    ", "\n")


def test_comment_remover():
    code = """
    #
    # a comment
    # IDL: some idl spec
    # FIXME: unknown api method
    # FIXME: unknown api property
    # FIXME: unknown api class
    # FIXME: new method - only user should remove
    # FIXME: was changed - only user should remove
    """

    p = CommentRemover()
    p.apply(dedent(code))
    code = p.dumps()

    assert code.count("#") == 4

    assert "IDL" not in code  # IDL is auto-added by the codegen
    assert "unknown" not in code  # these are also auto-added

    assert "new" in code  # user should remove these
    assert "was changed" in code  # user should remove these


class MyCommentInjector(AbstractCommentInjector):
    def class_is_known(self, classname):
        return True

    def prop_is_known(self, classname, propname):
        return True

    def method_is_known(self, classname, methodname):
        return True

    def get_class_comment(self, classname):
        return "# this is a class"

    def get_prop_comment(self, classname, propname):
        return "# this is a property"

    def get_method_comment(self, classname, methodname):
        return "# this is a method"


def test_comment_injector():
    code1 = """
    class X:
        'x'

        def foo(self):
            pass

        @whatever
        def bar(self):
            pass

        @property
        def spam(self):
            pass

        @property
        # valid Python, but we want comments above decorators
        def eggs(self):
            pass
    """

    code3 = """
    # this is a class
    class X:
        'x'

        # this is a method
        def foo(self):
            pass

        # this is a method
        @whatever
        def bar(self):
            pass

        # this is a property
        @property
        def spam(self):
            pass

        # valid Python, but we want comments above decorators
        # this is a property
        @property
        def eggs(self):
            pass
    """
    code3 = blacken(dedent(code3)).strip()

    p = MyCommentInjector()
    p.apply(dedent(code1))
    code2 = p.dumps().strip()

    assert code2 == code3


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
