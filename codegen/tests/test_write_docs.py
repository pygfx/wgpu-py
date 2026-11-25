"""Test the RST generation of API diffs in write_docs.py."""

from codegen_defs import read_definition_file
from codegen_utils import read_file
from write_docs import diff_api_to_rst, insert_diff_text, write_docs

from pytest import raises


def make_base_api():
    return {
        "structs": {
            "Point": {
                "fields": {
                    "x": {"type": "float", "default": "0.0"},
                    "y": {"type": "float", "default": "0.0"},
                }
            }
        },
        "classes": {
            "Foo": {
                "bases": [],
                "methods": {
                    "bar": {
                        "args": {"x": {"type": "int", "default": ""}},
                        "returns": {"type": "None"},
                    },
                },
            }
        },
    }


def make_diff_api():
    return {
        "meta": {"filename": "test.toml", "description": "", "notes": ""},
        "structs": {
            "Point": {
                "fields": {
                    "x": {"type": "int", "default": "0"},
                    "y": {"type": "float", "default": "0.0"},
                    "z": {"type": "int", "default": ""},
                    "w": {"diff": "REMOVE"},
                }
            },
            "Box": {"diff": "REMOVE"},
        },
        "classes": {
            "Foo": {
                "methods": {
                    "bar": {
                        "args": {"x": {"type": "int | None", "default": "None"}},
                        "returns": {"type": "int"},
                    },
                    "baz": {"doc": "", "args": {}, "returns": {"type": "None"}},
                }
            },
            "Spam": {"doc": "", "bases": [], "properties": {}, "methods": {}},
            "Old": {"diff": "REMOVE"},
        },
    }


def test_diff_api_to_rst():
    text = diff_api_to_rst(make_diff_api(), make_base_api())
    expected_lines = [
        "* Changed: the ``x`` field of the ``Point`` struct is now ``int``, defaults to ``0``.",
        "* Added: the ``z`` field of the ``Point`` struct.",
        "* Removed: the ``w`` field of the ``Point`` struct.",
        "* Removed: the ``Box`` struct.",
        "* Changed: the ``x`` argument of the ``Foo.bar`` method is now ``int | None``, defaults to ``None``.",
        "* Changed: the return value of the ``Foo.bar`` method is now ``int``.",
        "* Added: the ``Foo.baz`` method.",
        "* Added: the ``Spam`` class.",
        "* Removed: the ``Old`` class.",
    ]
    assert text == "\n".join(expected_lines) + "\n"


def test_diff_api_to_rst_of_the_real_diff():
    diff_api = read_definition_file("diff_base.toml")
    base_api = read_definition_file("webgpu.toml")
    text = diff_api_to_rst(diff_api, base_api)

    lines = text.splitlines()
    assert len(lines) > 20
    prefixes = ("* Removed: ", "* Added: ", "* Changed: ")
    assert all(line.startswith(prefixes) for line in lines)
    # Every bullet ends with a period
    assert all(line.endswith(".") for line in lines)
    # Sanity check of some entries that we know are there
    assert "* Removed: the ``NavigatorGPU`` class." in lines
    assert "* Added: the ``DrawCancelled`` class." in lines


def test_insert_diff_text():
    code = "\n".join(
        [
            "Before",
            "",
            ".. api-difference-start",
            "",
            "old text",
            "",
            ".. api-difference-end",
            "",
            "After",
        ]
    )
    result = insert_diff_text(code, "* Removed: something.")
    assert result == "\n".join(
        [
            "Before",
            "",
            ".. api-difference-start",
            "",
            "* Removed: something.",
            "",
            ".. api-difference-end",
            "",
            "After",
        ]
    )

    # The markers must be present
    with raises(RuntimeError):
        insert_diff_text("no markers here", "* bullet")


def test_write_docs(monkeypatch):
    import codegen_utils

    written = {}

    def mock_write_file(text, filename):
        written[str(filename)] = text

    monkeypatch.setattr(codegen_utils, "_write_file", mock_write_file)

    # Note that docs/wgpu.rst on disk is already up-to-date (the docs are
    # generated with this very code), so this also tests idempotency.
    write_docs()
    new_code = written.popitem()[1]

    assert new_code == read_file("docs", "wgpu.rst")
    assert "* Removed: the ``NavigatorGPU`` class." in new_code
    assert ".. api-difference-start" in new_code
    assert ".. api-difference-end" in new_code


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
