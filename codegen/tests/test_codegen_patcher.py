"""Test the patchers in codegen_patcher.py."""

from codegen_patcher import (
    ApiPatcher,
    CommentRemover,
    StructValidationChecker,
    detect_implemented_properties_and_methods,
)
from codegen_utils import format_code


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
    # FIXME: missing check_struct
    # FIXME: new method - only user should remove
    # FIXME: was changed - only user should remove
    """

    p = CommentRemover()
    p.apply(dedent(code))
    code = p.dumps()

    assert code.count("#") == 5

    # These are auto-added by the codegen, so are also auto-removed
    assert "unknown" not in code
    assert "missing check_struct" not in code

    # These are not auto-added, so only the user should remove these
    assert "a comment" in code
    assert "IDL" in code
    assert "new" in code
    assert "was changed" in code


def test_detect_implemented_properties_and_methods():
    code = """
    class Foo:
        @property
        def prop1(self):
            return 1

        @property
        def prop2(self):
            raise NotImplementedError()

        def meth1(self):
            return 1

        def meth2(self):
            raise NotImplementedError()
    """

    code = format_code(dedent(code))
    implemented = detect_implemented_properties_and_methods(code)
    assert implemented == {"Foo.prop1", "Foo.meth1"}


def make_api_dict():
    return {
        "classes": {
            "Foo": {
                "bases": ["object"],
                "doc": "Doc for Foo.",
                "properties": {
                    "label": {"type": "str", "doc": "The label."},
                    "extra": {"type": "int", "doc": "An extra prop."},
                },
                "methods": {
                    "bar": {
                        "doc": "Do bar.",
                        "args": {"x": {"type": "int", "default": ""}},
                        "returns": {"type": "None", "doc": ""},
                    },
                    "from_struct": {
                        "doc": "",
                        "args": {
                            "desc": {
                                "type": "structs.BufferDescriptor | None",
                                "default": "None",
                            }
                        },
                        "returns": {"type": "None", "doc": ""},
                    },
                    "cook": {
                        "doc": "",
                        "args": {"temp": {"type": "int", "default": "100"}},
                        "returns": {"type": "int", "doc": ""},
                    },
                },
            },
            "NewClass": {
                "bases": [],
                "doc": "",
                "properties": {},
                "methods": {
                    "spam": {"doc": "", "args": {}, "returns": {"type": "str"}},
                },
            },
        },
        "structs": {
            "BufferDescriptor": {
                "fields": {
                    "size": {"type": "int", "default": ""},
                    "usage": {"type": "int", "default": "5"},
                    "more": {"type": "str", "default": ""},
                }
            },
        },
    }


def test_api_patcher():
    code = """
    class Foo:
        def spam(self):
            pass
        def _private(self):
            pass
        @property
        def label(self):
            return self._label

    class GPUPromise:
        def whatever(self):
            pass

    class UnknownClass:
        pass
    """

    api = make_api_dict()
    p = ApiPatcher(api, True)
    p.apply(format_code(dedent(code)))
    code2 = p.dumps()

    # The class def is updated with the bases
    assert "# FIXME: was Foo:" in code2
    assert "class Foo(object):" in code2

    # The known prop has its signature updated
    assert "def label(self) -> str:" in code2

    # The unknown method gets a fixme, the private method does not
    assert "# FIXME: unknown api: method Foo.spam" in code2
    assert "_private" in code2
    assert "unknown api: method Foo._private" not in code2

    # Methods missing in the code are added, with correct signatures
    assert "# FIXME: new method to implement" in code2
    assert "def bar(self, x: int) -> None:" in code2
    assert "def cook(self, temp: int = 100) -> int:" in code2
    # A single arg that is a struct is exploded into its fields
    assert (
        "def from_struct(self, *, size: int, usage: int = 5, more: str) -> None:"
        in code2
    )
    assert "raise NotImplementedError()" in code2

    # A prop missing in the code is added
    assert "# FIXME: new prop to implement" in code2

    # The external class is left alone
    assert "class GPUPromise:" in code2
    assert "def whatever(self):" in code2
    assert not [
        line for line in code2.splitlines() if "GPUPromise" in line and "FIXME" in line
    ]

    # An unknown class gets a fixme, and missing classes are added
    assert "# FIXME: unknown api: class UnknownClass" in code2
    assert "# FIXME: new class to implement" in code2
    assert "class NewClass:" in code2
    assert "def spam(self) -> str:" in code2

    # The result is valid Python
    compile(code2, "test", "exec")


def test_api_patcher_implemented_on_base_class():
    api = make_api_dict()
    p = ApiPatcher(api, False, implemented_on_base_class={"Foo.cook"})
    p.apply(format_code("class Foo:\n    pass\n"))
    code2 = p.dumps()

    # The class now inherits from the generated base class
    assert "class Foo(classes.Foo):" in code2
    # The method that is implemented on the base class is not added
    assert "def cook" not in code2


def test_struct_validation_checker(monkeypatch):
    struct_names = [f"S{i}" for i in range(12)]
    api = {
        "classes": {"Foo": {"doc": ""}},
        "structs": {name: {"fields": {}} for name in struct_names},
    }
    code = """
    class Foo:
        def bar(self, x: structs.S1, y: structs.S2, z: structs.S3):
            return x

        def solo(self, x: structs.S4, y: structs.S5, z: structs.S6, w: structs.S7, v: structs.S8):
            return x

        def defer_async(self, x: structs.S9, y: structs.S10, z: structs.S11):
            return self._defer_async(x)

    class OtherClass:
        def whatever(self, x: structs.S12):
            return x
    """

    # The check map for the (classes, methods) in the code above. Note that
    # (Foo, solo) is missing, and OtherClass is not in the api at all.
    checks = {
        ("Foo", "bar"): {"S1", "S2", "S3"},
        ("Foo", "defer_async"): {"S9", "S10", "S11"},
    }
    monkeypatch.setattr(
        StructValidationChecker,
        "_get_structure_checks",
        staticmethod(lambda _c: checks),
    )

    p = StructValidationChecker(api)
    p.apply(format_code(dedent(code)))
    code2 = p.dumps()

    # A method whose structs are all checked gets no fixme
    assert "missing check_struct in bar" not in code2
    # A method that is not in the check-map gets a fixme
    assert "# FIXME: missing check_struct in solo" in code2
    # Methods that defer to an async variant are excused
    assert "defer_async" in code2
    assert "missing check_struct in defer_async" not in code2
    # Classes that are not in the api are skipped
    assert "whatever" in code2
    assert "missing check_struct in whatever" not in code2

    # The result is valid Python
    compile(code2, "test", "exec")


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
