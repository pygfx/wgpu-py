"""Test the definition file reading/writing/checking,
and the json/toml converters in codegen_defs.py.
"""

import json
import tomllib

import codegen_utils
from codegen_defs import (
    _toml_key,
    _value_to_toml,
    check_definition_structure,
    enum_name_to_value,
    read_definition_file,
    structure_to_json,
    structure_to_toml,
    write_definition_file,
)

from pytest import raises


def make_structure():
    return {
        "meta": {
            "filename": "test.toml",
            "description": "A test structure.",
            "notes": "Covers all sections.",
        },
        "flags": {
            "Fruit": {
                "doc": "Fruit flags.\n\nMultiline doc!",
                "fields": {
                    "APPLE": {"value": 1, "doc": "an apple"},
                    "PEAR": {"value": 2, "doc": "a pear"},
                },
            }
        },
        "enums": {
            "Color": {
                "doc": "A color.",
                "fields": {
                    "red": {"value": "red", "doc": "the color red"},
                    "dark_green": {"value": "dark-green", "doc": "a dark green"},
                },
            }
        },
        "structs": {
            "Point": {
                "doc": "A point.",
                "fields": {
                    "x": {"type": "int", "default": "0", "doc": ""},
                    "label": {"type": "str", "default": "", "doc": ""},
                },
            }
        },
        "classes": {
            "Spam": {
                "doc": "Doc for class Spam.",
                "bases": ["Eggs"],
                "properties": {"size": {"type": "int", "doc": "size doc"}},
                "methods": {
                    "cook": {
                        "doc": "cook doc",
                        "returns": {"type": "int", "doc": ""},
                        "args": {
                            "temp": {"type": "int", "default": "100", "doc": ""},
                            "keep": {"type": "bool", "default": "", "diff": "REMOVE"},
                        },
                    },
                    "d1_foo": {"doc": "", "returns": {"type": "None"}, "args": {}},
                },
            }
        },
    }


def test_enum_name_to_value():
    assert enum_name_to_value("low_power") == "low-power"
    assert enum_name_to_value("bgra8unorm") == "bgra8unorm"
    assert enum_name_to_value("texture_dimension_d1") == "texture-dimension-1d"
    assert enum_name_to_value("point_d3") == "point-3d"
    assert enum_name_to_value("d2") == "2d"


def test_read_definition_file():
    # Reading a toml file, also runs check_definition_structure()
    s = read_definition_file("diff_base.toml")
    assert "meta" in s
    assert "classes" in s

    # Reading a json file, checks the filename in the meta
    s = read_definition_file("wgpu_api.json")
    assert s["meta"]["filename"] == "wgpu_api.json"

    # Invalid extension
    with raises(RuntimeError):
        read_definition_file("webgpu.idl")


def test_write_definition_file(monkeypatch):
    structure = make_structure()

    written = {}

    def mock_write_file(text, filename):
        written[str(filename)] = text

    monkeypatch.setattr(codegen_utils, "_write_file", mock_write_file)

    # The toml version parses back to the same structure (all sections filled)
    write_definition_file(structure, "test.toml")
    s2 = tomllib.loads(written.popitem()[1])
    expected = dict(structure)
    check_definition_structure(s2)
    assert s2 == expected

    # The json version parses back to the same structure too. Note that
    # the meta has the json filename
    structure = make_structure()
    expected2 = make_structure()
    expected2["meta"]["filename"] = "test.json"
    write_definition_file(structure, "test.json")
    s3 = json.loads(written.popitem()[1])
    assert s3 == expected2

    # An unknown section is an error
    structure = make_structure()
    structure["bogus"] = {}
    with raises(RuntimeError):
        write_definition_file(structure, "test.toml")

    # An invalid extension is an error
    with raises(RuntimeError):
        write_definition_file(make_structure(), "test.txt")


def test_check_definition_structure():
    # A valid structure passes
    check_definition_structure(make_structure())

    # An unknown section
    with raises(AssertionError):
        check_definition_structure({"bogus": {}})

    # Meta with an unknown key, or a non-str value
    with raises(AssertionError):
        check_definition_structure({"meta": {"bogus": "x"}})
    with raises(AssertionError):
        check_definition_structure({"meta": {"notes": 3}})

    # Flags: bad name, bad attr, bad field name, bad field key
    with raises(AssertionError):
        check_definition_structure({"flags": {"Bad_Name": {}}})
    with raises(RuntimeError):
        check_definition_structure({"flags": {"F": {"bogus": 1}}})
    with raises(AssertionError):
        check_definition_structure({"flags": {"F": {"fields": {"lower": {}}}}})
    with raises(RuntimeError):
        check_definition_structure({"flags": {"F": {"fields": {"A": {"bogus": 1}}}}})
    with raises(AssertionError):
        check_definition_structure({"flags": {"F": {"diff": "bogus"}}})

    # Enums: bad name, bad attr, bad field name, mismatching value, bad key
    with raises(AssertionError):
        check_definition_structure({"enums": {"Bad_Name": {}}})
    with raises(RuntimeError):
        check_definition_structure({"enums": {"E": {"bogus": 1}}})
    with raises(AssertionError):
        check_definition_structure({"enums": {"E": {"fields": {"UPPER": {}}}}})
    with raises(AssertionError):
        check_definition_structure(
            {"enums": {"E": {"fields": {"foo": {"value": "nope"}}}}}
        )
    with raises(RuntimeError):
        check_definition_structure({"enums": {"E": {"fields": {"foo": {"bogus": 1}}}}})

    # Structs: bad name, bad attr, bad field name, bad field key
    with raises(AssertionError):
        check_definition_structure({"structs": {"Bad_Name": {}}})
    with raises(RuntimeError):
        check_definition_structure({"structs": {"St": {"bogus": 1}}})
    with raises(AssertionError):
        check_definition_structure({"structs": {"St": {"fields": {"UPPER": {}}}}})
    with raises(RuntimeError):
        check_definition_structure(
            {"structs": {"St": {"fields": {"foo": {"bogus": 1}}}}}
        )

    # Classes: bad name, bad attr, bad prop, bad method, bad returns, bad arg
    with raises(AssertionError):
        check_definition_structure({"classes": {"Bad_Name": {}}})
    with raises(RuntimeError):
        check_definition_structure({"classes": {"C": {"bogus": 1}}})
    with raises(AssertionError):
        check_definition_structure({"classes": {"C": {"bases": "not-a-list"}}})
    with raises(AssertionError):
        check_definition_structure(
            {"classes": {"C": {"properties": {"UPPER": {"type": "int"}}}}}
        )
    with raises(RuntimeError):
        check_definition_structure(
            {"classes": {"C": {"properties": {"ok": {"bogus": "int"}}}}}
        )
    with raises(AssertionError):
        check_definition_structure(
            {"classes": {"C": {"methods": {"UPPER": {"doc": ""}}}}}
        )
    with raises(RuntimeError):
        check_definition_structure(
            {"classes": {"C": {"methods": {"ok": {"bogus": 1}}}}}
        )
    with raises(RuntimeError):
        check_definition_structure(
            {"classes": {"C": {"methods": {"ok": {"returns": {"bogus": "int"}}}}}}
        )
    with raises(RuntimeError):
        check_definition_structure(
            {"classes": {"C": {"methods": {"ok": {"args": {"x": {"bogus": 1}}}}}}}
        )
    with raises(AssertionError):
        check_definition_structure(
            {"classes": {"C": {"methods": {"ok": {"args": {"X": {"type": "int"}}}}}}}
        )


def test_structure_to_json():
    # Scalars
    assert structure_to_json(42) == "42"
    assert structure_to_json("spam") == '"spam"'
    assert structure_to_json(None) == "null"
    assert structure_to_json(True) == "true"

    # Small leaf dicts and lists are single-lined
    assert structure_to_json({"a": 1, "b": 2}) == '{ "a": 1, "b": 2 }'
    assert structure_to_json({"a": 1}) == '{ "a": 1 }'
    assert structure_to_json([1, 2, 3]) == "[ 1, 2, 3 ]"
    assert structure_to_json(["a", "b"]) == '[ "a", "b" ]'

    # Empty dicts and lists
    assert structure_to_json({}) == "{  }"
    assert structure_to_json([]) == "[  ]"

    # A dict with a single large value is single-lined
    d = {"key": "x" * 150}
    assert structure_to_json(d) == '{ "key": ' + json.dumps("x" * 150) + " }"

    # A large dict becomes multiline, with indentation
    d = {"key": "x" * 150, "foo": {"bar": 3}}
    text = structure_to_json(d)
    assert text.startswith("{\n")
    assert text.endswith("\n}")
    assert '"foo": { "bar": 3 }' in text
    assert '"key"' in text

    # A list with dicts inside becomes multiline too
    text = structure_to_json([{"a": 1, "big": "y" * 150}])
    assert text.startswith("[\n")
    assert text.endswith("\n]")
    assert '"big"' in text


def test_toml_key():
    assert _toml_key("foo_bar") == "foo_bar"
    assert _toml_key("S1") == "S1"
    assert _toml_key("a.b") == '"a.b"'
    assert _toml_key("a b") == '"a b"'


def test_value_to_toml():
    assert _value_to_toml(True) == "true"
    assert _value_to_toml(False) == "false"
    assert _value_to_toml(3) == "3"
    assert _value_to_toml(1.5) == "1.5"
    assert _value_to_toml(["a", "b"]) == "['a', 'b']"
    assert _value_to_toml("simple") == "'simple'"
    assert _value_to_toml("it's") == '"it\'s"'
    assert _value_to_toml("line1\nline2") == "'''\nline1\nline2'''"
    # Strings that end with a quote cannot be literal multiline
    assert _value_to_toml("ends with '") == json.dumps("ends with '")
    # Only leaf values are supported
    with raises(RuntimeError):
        _value_to_toml(object())


def test_structure_to_toml(monkeypatch):
    structure = make_structure()

    # Full roundtrip: the toml parses back into the same structure
    text = structure_to_toml(structure)
    parsed = tomllib.loads(text)
    check_definition_structure(parsed)
    assert parsed == structure

    # The meta section is a table at the top
    assert text.startswith("[meta]\nfilename = 'test.toml'\n")
    # Sections are separated by banners, at least 100 chars wide
    assert "## flags" not in text
    for line in text.splitlines():
        if line.startswith("# ===") or line.startswith("# ----"):
            assert len(line) >= 100
    # Entities become tables, with dotted keys for the fields
    assert "[flags.Fruit]" in text
    assert "fields.APPLE.value = 1" in text
    assert "[enums.Color]" in text
    assert "fields.dark_green.value = 'dark-green'" in text
    assert "[classes.Spam.properties.size]" in text
    assert "[classes.Spam.methods.cook]" in text
    assert "args.keep.diff = 'REMOVE'" in text
    # Classes get a banner of their own
    assert any(line.startswith("# ---- Spam ") for line in text.splitlines())
    # Methods are tables too
    assert "[classes.Spam.methods.d1_foo]" in text
    # Multi-line docs are written as literal strings
    assert "'''" in text

    # The actual writes also produce a matching file (via the patched _write_file)
    written = {}

    def mock_write_file(text, filename):
        written[str(filename)] = text

    monkeypatch.setattr(codegen_utils, "_write_file", mock_write_file)
    structure["meta"]["filename"] = "test.toml"
    write_definition_file(structure, "test.toml")
    text2 = written.popitem()[1]
    assert tomllib.loads(text2) == structure
    assert text2 == text


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
