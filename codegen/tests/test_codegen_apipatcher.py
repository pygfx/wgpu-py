""" Test some parts of apipatcher.py, and Implicitly tests idlparser.py.
"""

from codegen.utils import blacken
from codegen.apipatcher import CommentRemover, AbstractCommentInjector, IdlPatcherMixin


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


def test_async_api_logic():

    class Object(object):
        pass

    class OtherIdlPatcherMixin(IdlPatcherMixin):
        def __init__(self):
            cls = Object()
            cls.attributes = {
                "prop1": "x prop1 bla",
                "prop2": "Promise<x> prop2 bla",
            }
            cls.functions = {
                "method1": "x method1 bla",
                "method2": "Promise<x> method2 bla",
                "method3Async": "Promise<x> method3 bla",
                "method3": "x method3 bla",
            }

            self.idl = Object()
            self.idl.classes = {"Foo": cls}

    patcher = OtherIdlPatcherMixin()
    patcher.detect_async_props_and_methods()

    # Normal prop
    assert patcher.name2idl("Foo", "prop1") == "prop1"
    assert patcher.name2idl("Foo", "prop1_sync") == "prop1InvalidVariant"
    assert patcher.name2idl("Foo", "prop1_async") == "prop1InvalidVariant"

    # Unknow prop, name still works
    assert patcher.name2idl("Foo", "prop_unknown") == "propUnknown"

    # Async prop
    assert patcher.name2idl("Foo", "prop2_async") == "prop2"
    assert patcher.name2idl("Foo", "prop2_sync") == "prop2"
    assert patcher.name2idl("Foo", "prop2") == "prop2InvalidVariant"

    # Normal method
    assert patcher.name2idl("Foo", "method1") == "method1"
    assert patcher.name2idl("Foo", "method1_sync") == "method1InvalidVariant"
    assert patcher.name2idl("Foo", "method1_async") == "method1InvalidVariant"

    # Async method
    assert patcher.name2idl("Foo", "method2_async") == "method2"
    assert patcher.name2idl("Foo", "method2_sync") == "method2"
    assert patcher.name2idl("Foo", "method2") == "method2InvalidVariant"

    # Async method that also has sync variant in JS
    assert patcher.name2idl("Foo", "method3_async") == "method3Async"
    assert patcher.name2idl("Foo", "method3") == "method3"
    assert patcher.name2idl("Foo", "method3_sync") == "method3InvalidVariant"


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
