import sys
import json
import re
import tomllib
import subprocess
from pathlib import Path

DEFINITIONS_DIR = Path(__file__).parent.absolute() / "definitions"
ROOT_DIR = Path(__file__).parents[1].absolute()


def read_file(*fname):
    """Read a file from disk using the relative filename. Line endings are normalized."""
    filename = ROOT_DIR.joinpath(*fname)
    with open(filename, "rb") as f:
        return (
            f.read()
            .decode()
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\\\n", "")  # Macros escaping a newline
        )


def write_file(text: str, *fname):
    """Write the given text to disk, using the relative filename. Line endings are matched to what's used for __init__.py."""
    filename = ROOT_DIR.joinpath(*fname)
    if not text.endswith("\n"):
        text = text + "\n"
    with open(filename, "wb") as f:
        f.write(text.replace("\n", LE).encode())


def _get_line_endings(text):
    """Detect whether the line endings in use is CR LF or CRLF."""
    # Count how many line ending chars there are
    crlf_count = text.count("\r\n")
    lf_count = text.count("\n") - crlf_count
    cr_count = text.count("\r") - crlf_count
    assert lf_count + cr_count + crlf_count >= 4
    # Check what's used the most, or whether it's a combination.
    if lf_count > cr_count and lf_count > crlf_count:
        return "\n"
    elif cr_count > lf_count and cr_count > crlf_count:
        return "\r"
    else:
        return "\r\n"


LE = _get_line_endings((ROOT_DIR / "apigen" / "__init__.py").read_bytes().decode())


def enum_name_to_value(name):
    parts = name.split("_")
    for i in range(len(parts)):
        if parts[i] in ("d1", "d2", "d3"):
            parts[i] = parts[i][::-1]
    return "-".join(parts)


def dict_deep_merge(a: dict, b: dict, allow_add: bool = True) -> dict:
    """Return a new dict: b merged into a, recursing into nested dicts."""
    result = dict(a)
    for key, val in b.items():
        if isinstance(val, dict):
            if key in result and isinstance(result[key], dict):
                if "diff" in val:
                    assert val["diff"] == "REMOVE"
                    result.pop(key, None)
                else:
                    result[key] = dict_deep_merge(result[key], val, allow_add)
            elif allow_add:
                result[key] = val
        else:
            result[key] = val

    return result


def indent_code(text: str, indent=4) -> str:
    text = text.strip()
    lines = []
    for line in text.split("\n"):
        lines.append(" " * indent + line.rstrip())
    lines.append("")
    return "\n".join(lines)


def write_definition_file(structure: dict, fname: str) -> None:
    """Write an API definition to disk. The structure is serialized to json, indented, but with singular and small leaf nodes single-lined."""

    unknown = [k for k in structure if k not in ("meta", *_section_names)]
    if unknown:
        raise RuntimeError(f"structure_to_toml(): unknown section(s): {unknown!r}")

    # Put sections in predefined order, with an ordered meta at the top
    meta = {"filename": fname}
    for meta_key in ["description", "notes"]:
        meta[meta_key] = structure["meta"][meta_key]
    s = {"meta": meta}
    for section in _section_names:
        s[section] = structure.get(section, {})

    if fname.endswith(".json"):
        text = structure_to_json(s)
    elif fname.endswith(".toml"):
        text = structure_to_toml(s)
    else:
        raise RuntimeError(
            f"write_definition_file(): invalid definition file extension for {fname!r}."
        )
    write_file(text, "apigen", "definitions", fname)


def _is_json_leaf(obj):
    values = obj.values() if isinstance(obj, dict) else obj
    return not any(isinstance(v, (dict, list)) for v in values)


def structure_to_json(obj, indent=2, level=0, max_len=100, **kw):
    if not isinstance(obj, (dict, list)):
        return json.dumps(obj, **kw)

    if _is_json_leaf(obj):
        flat = json.dumps(obj, **kw).strip("{}")
        if len(obj) <= 1 or len(flat) < max_len:
            if isinstance(obj, list):
                return "[ " + flat.strip("[]") + " ]"
            else:
                return "{ " + flat.strip("{}") + " }"

    pad = " " * (indent * (level + 1))
    end = " " * (indent * level)
    sub = dict(indent=indent, level=level + 1, max_len=max_len, **kw)

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        body = ",\n".join(
            f"{pad}{json.dumps(k, **kw)}: {structure_to_json(v, **sub)}"
            for k, v in obj.items()
        )
        return "{\n" + body + "\n" + end + "}"
    else:  # list
        if not obj:
            return "[]"
        body = ",\n".join(pad + structure_to_json(v, **sub) for v in obj)
        return "[\n" + body + "\n" + end + "]"


def _toml_key(key: str) -> str:
    """Represent a dict key in TOML syntax, quoting it when it cannot be a bare key."""
    if re.match(r"^[A-Za-z0-9_-]+$", key):
        return key
    else:
        return json.dumps(key)


def _value_to_toml(val):
    """Represent a leaf value (or a list of leaf values) in TOML syntax.
    Multi-line strings are written as literal (triple-single-quoted) strings.
    """
    if isinstance(val, bool):
        return ["false", "true"][val]
    elif isinstance(val, (int, float)):
        return repr(val)
    elif isinstance(val, list):
        return "[" + ", ".join(_value_to_toml(v) for v in val) + "]"
    elif isinstance(val, str):
        if "\n" in val and "'''" not in val and not val.endswith("'"):
            return "'''\n" + val + "'''"
        elif "'" in val:
            return json.dumps(val)  # basic string, i.e. with escapes
        else:
            return f"'{val}'"
    else:
        raise RuntimeError(f"Only expected leaf values, not {val.__class__.__name__}")


_scalar_key_order = ("doc", "diff", "value", "type", "default", "bases")
_section_names = ("flags", "enums", "structs", "classes")
_child_table_keys = ("methods", "properties")
_banner_width = 100


def _banner_lines(title: str, double: bool = False) -> list:
    """Create a comment banner, at least 100 chars wide. Sections get a
    double-row banner (with "="), classes a single-line one (with "-").
    """
    char = "=" if double else "-"
    line = f"# {char * 4} {title} "
    line = line + char * max(_banner_width - len(line), 4)
    result = [line]
    if double:
        result.append("# " + char * max(_banner_width - 2, 4))
    return result


def structure_to_toml(structure: dict) -> str:
    """Generate a nicely readable TOML representation of the given structure
    (the shape checked by check_definition_structure()).

    The "meta" section is written as a table first, so it survives a
    roundtrip. Each named entity becomes a table: flags, enums, structs,
    classes (e.g. "[flags.TextureUsageFlags]"), and the methods and
    properties of a class (e.g. "[classes.GPU.methods.foo]"). Deeper
    sub-dicts (fields, method args, returns) are written as dotted keys
    inside the entity's table, e.g. "args.width.type = 'int'", with blank
    lines between groups. Empty sub-dicts are written as inline tables
    (e.g. "args = {}"), so that the TOML parses back into the exact same
    structure.
    """

    lines = []

    # Write the meta section as a table, in a fixed key order
    meta = structure.get("meta", {})
    if meta:
        meta = {
            key: meta[key]
            for key in ("filename", "description", "notes")
            if key in meta
        }
        _write_entity_table(lines, ["meta"], meta)

    # Write each section, separated by a banner
    for section, section_dict in structure.items():
        if section == "meta" or not section_dict:
            continue
        lines.append("")
        lines.extend(_banner_lines(section, double=True))
        for name, item_dict in section_dict.items():
            if section == "classes":
                lines.append("")
                lines.extend(_banner_lines(name))
            _write_entity_table(lines, [section, name], item_dict)

    return "\n".join(lines).lstrip() + "\n"


def _write_entity_table(lines: list, path: list, d: dict) -> None:
    """Write the given dict as a TOML table, e.g. "[classes.GPU.methods.foo]".
    Non-dict values become plain keys, deeper sub-dicts become dotted keys,
    and the keys in _child_table_keys become tables of their own. The header
    is only written when the table has content of its own.
    """
    scalar_keys = [k for k in d if not isinstance(d[k], dict)]
    empty_keys = [k for k in d if isinstance(d[k], dict) and not d[k]]
    dotted_dict = {
        k: v
        for k, v in d.items()
        if isinstance(v, dict) and v and k not in _child_table_keys
    }
    child_keys = [k for k in d if k in _child_table_keys and d[k]]

    if scalar_keys or empty_keys or dotted_dict or not d:
        lines.append("")
        lines.append("[" + ".".join(_toml_key(p) for p in path) + "]")
        scalar_keys = [k for k in _scalar_key_order if k in scalar_keys] + [
            k for k in scalar_keys if k not in _scalar_key_order
        ]
        for key in scalar_keys:
            lines.append(f"{_toml_key(key)} = {_value_to_toml(d[key])}")
        for key in empty_keys:
            lines.append(f"{_toml_key(key)} = {{}}")

    if dotted_dict:
        _write_dotted_dict(lines, [], dotted_dict)

    for key in child_keys:
        for sub_name, sub_dict in d[key].items():
            _write_entity_table(lines, [*path, key, sub_name], sub_dict)


def _write_dotted_dict(lines: list, path: list, d: dict) -> None:
    """Write the given dict as dotted keys under the given path, e.g.
    "args.width.type = 'int'". Sub-dict groups are separated by blank lines.
    """
    scalar_keys = [k for k in d if not isinstance(d[k], dict) or not d[k]]
    scalar_keys = [k for k in _scalar_key_order if k in scalar_keys] + [
        k for k in scalar_keys if k not in _scalar_key_order
    ]
    for key in scalar_keys:
        prefix = ".".join([*path, _toml_key(key)]) if path else _toml_key(key)
        val = d[key]
        if isinstance(val, dict):  # an empty dict, written inline
            lines.append(f"{prefix} = {{}}")
        else:
            lines.append(f"{prefix} = {_value_to_toml(val)}")

    first = True
    for key, val in d.items():
        if isinstance(val, dict) and val:
            if not first:
                lines.append("")
            first = False
            _write_dotted_dict(lines, [*path, key], val)


def read_definition_file(fname: str) -> dict:
    """Read an API definition from disk and check that the structure is correct.
    Returns the structure as a dict.
    """
    source = read_file("apigen", "definitions", fname)

    if fname.endswith(".json"):
        structure = json.loads(source)
        assert (
            "meta" in structure
            and "filename" in structure["meta"]
            and fname == structure["meta"]["filename"]
        )
    elif fname.endswith(".toml"):
        structure = tomllib.loads(source)
    else:
        raise RuntimeError(
            f"read_definition_file(): invalid definition file extension for {fname!r}."
        )

    check_definition_structure(structure)
    return structure


def check_definition_structure(structure):
    diff_actions = ("REMOVE",)

    for section, section_dict in structure.items():
        assert section in ("meta", "flags", "enums", "structs", "classes")
        assert isinstance(section_dict, dict)

        if section == "meta":
            for k, v in section_dict.items():
                assert k in ("filename", "description", "notes")
                assert isinstance(v, str)

        if section == "flags":
            for flag_name, flag_dict in structure[section].items():
                assert "_" not in flag_name
                for flag_attr_key, flag_attr_value in flag_dict.items():
                    if flag_attr_key == "doc":
                        assert isinstance(flag_attr_value, str)
                    elif flag_attr_key == "diff":
                        assert flag_attr_value in diff_actions
                    elif flag_attr_key == "fields":
                        assert isinstance(flag_attr_value, dict)
                        for field_name, field_dict in flag_attr_value.items():
                            assert field_name.upper() == field_name
                            assert isinstance(field_dict, dict)
                            for field_dict_key, field_dict_val in field_dict.items():
                                if field_dict_key == "doc":
                                    assert isinstance(field_dict_val, str)
                                elif field_dict_key == "diff":
                                    assert field_dict_val in diff_actions
                                elif field_dict_key == "value":
                                    assert isinstance(field_dict_val, int)
                                else:
                                    raise RuntimeError(
                                        f"Invalid flag field dict key {field_dict_key!r}"
                                    )
                    else:
                        raise RuntimeError(f"Invalid flag attr {flag_attr_key!r}")

        if section == "enums":
            for enum_name, enum_dict in structure[section].items():
                assert "_" not in enum_name
                for enum_attr_key, enum_attr_value in enum_dict.items():
                    if enum_attr_key == "doc":
                        assert isinstance(enum_attr_value, str)
                    elif enum_attr_key == "diff":
                        assert enum_attr_value in diff_actions
                    elif enum_attr_key == "fields":
                        assert isinstance(enum_attr_value, dict)
                        for field_name, field_dict in enum_attr_value.items():
                            assert field_name.lower() == field_name
                            assert isinstance(field_dict, dict)
                            for field_dict_key, field_dict_val in field_dict.items():
                                if field_dict_key == "doc":
                                    assert isinstance(field_dict_val, str)
                                elif field_dict_key == "diff":
                                    assert field_dict_val in diff_actions
                                elif field_dict_key == "value":
                                    assert isinstance(field_dict_val, str)
                                    assert field_dict_val == enum_name_to_value(
                                        field_name
                                    )
                                else:
                                    raise RuntimeError(
                                        f"Invalid enum field dict key {field_dict_key!r}"
                                    )
                    else:
                        raise RuntimeError(f"Invalid enum attr {enum_attr_key!r}")

        if section == "structs":
            for struct_name, struct_dict in structure[section].items():
                assert "_" not in struct_name
                for struct_attr_key, struct_attr_value in struct_dict.items():
                    if struct_attr_key == "doc":
                        assert isinstance(struct_attr_value, str)
                    elif struct_attr_key == "diff":
                        assert struct_attr_value in diff_actions
                    elif struct_attr_key == "fields":
                        assert isinstance(struct_attr_value, dict)
                        for field_name, field_dict in struct_attr_value.items():
                            assert field_name.lower() == field_name
                            assert isinstance(field_dict, dict)
                            for field_dict_key, field_dict_val in field_dict.items():
                                if field_dict_key == "doc":
                                    assert isinstance(field_dict_val, str)
                                elif field_dict_key == "diff":
                                    assert field_dict_val in diff_actions
                                elif field_dict_key == "type":
                                    assert isinstance(field_dict_val, str)
                                elif field_dict_key == "default":
                                    assert isinstance(field_dict_val, str)
                                else:
                                    raise RuntimeError(
                                        f"Invalid struct field dict key {field_dict_key!r}"
                                    )
                    else:
                        raise RuntimeError(f"Invalid struct attr {struct_attr_key!r}")

        if section == "classes":
            for class_name, class_dict in structure[section].items():
                assert "_" not in class_name
                for class_attr_key, class_attr_value in class_dict.items():
                    if class_attr_key == "doc":
                        assert isinstance(class_attr_value, str)
                    elif class_attr_key == "diff":
                        assert class_attr_value in diff_actions
                    elif class_attr_key == "bases":
                        assert isinstance(class_attr_value, list)
                        assert all(isinstance(x, str) for x in class_attr_value)
                    elif class_attr_key == "properties":
                        assert isinstance(class_attr_value, dict)
                        for prop_name, prop_dict in class_attr_value.items():
                            assert prop_name.lower() == prop_name
                            assert isinstance(prop_dict, dict)
                            for prop_dict_key, prop_dict_val in prop_dict.items():
                                if prop_dict_key == "doc":
                                    assert isinstance(prop_dict_val, str)
                                elif prop_dict_key == "diff":
                                    assert prop_dict_val in diff_actions
                                elif prop_dict_key == "type":
                                    assert isinstance(prop_dict_val, str)
                                else:
                                    raise RuntimeError(
                                        f"Invalid property dict key {prop_dict_key!r}"
                                    )
                    elif class_attr_key == "methods":
                        assert isinstance(class_attr_value, dict)
                        for method_name, method_dict in class_attr_value.items():
                            assert method_name.lower() == method_name
                            assert isinstance(method_dict, dict)
                            for method_dict_key, method_dict_val in method_dict.items():
                                if method_dict_key == "doc":
                                    assert isinstance(method_dict_val, str)
                                elif method_dict_key == "diff":
                                    assert method_dict_val in diff_actions
                                elif method_dict_key == "returns":
                                    assert isinstance(method_dict_val, dict)
                                    for (
                                        return_dict_key,
                                        return_dict_val,
                                    ) in method_dict_val.items():
                                        if return_dict_key == "doc":
                                            assert isinstance(return_dict_val, str)
                                        elif return_dict_key == "type":
                                            assert isinstance(return_dict_val, str)
                                        else:
                                            raise RuntimeError(
                                                f"Invalid method returns dict key {return_dict_key!r}"
                                            )
                                elif method_dict_key == "args":
                                    assert isinstance(method_dict_val, dict)
                                    for arg_name, arg_dict in method_dict_val.items():
                                        assert arg_name.lower() == arg_name
                                        assert isinstance(arg_dict, dict)
                                        for (
                                            arg_dict_key,
                                            arg_dict_val,
                                        ) in arg_dict.items():
                                            if arg_dict_key == "doc":
                                                assert isinstance(arg_dict_val, str)
                                            elif arg_dict_key == "diff":
                                                assert arg_dict_val in diff_actions
                                            elif arg_dict_key == "type":
                                                assert isinstance(arg_dict_val, str)
                                            elif arg_dict_key == "default":
                                                assert isinstance(arg_dict_val, str)
                                            else:
                                                raise RuntimeError(
                                                    f"Invalid method arg dict key {arg_dict_key!r}"
                                                )
                                else:
                                    raise RuntimeError(
                                        f"Invalid method dict key {method_dict_key!r}"
                                    )
                    else:
                        raise RuntimeError(f"Invalid class attr {class_attr_key!r}")


class FormatError(Exception):
    pass


def format_code(src, singleline=False):
    """Format the given src string. If singleline is True, all function
    signatures become single-line, so they can be parsed and updated.
    """
    line_length = 320 if singleline else 88
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "format",
        "--line-length",
        str(line_length),
        "--no-cache",
        "--stdin-filename",
        "tmp.py",
    ]

    p = subprocess.run(cmd, input=src.encode(), capture_output=True)

    if p.returncode:
        stdout = p.stdout.decode(errors="ignore")
        stderr = p.stderr.decode(errors="ignore")
        raise FormatError(
            f"Could not format source ({p.returncode}).\n\nstdout: "
            + stdout
            + "\n\nstderr: "
            + stderr
        )
    else:
        result = p.stdout.decode(errors="ignore")

    # Make defs single-line. You'd think that setting the line length
    # to a very high number would do the trick, but it does not.
    if singleline:
        result = _make_sigs_singleline(result)

    return result


def _make_sigs_singleline(code):
    lines1 = code.splitlines()
    lines2 = []

    sig_state = 0
    sig_brace_depth = 0
    sig_line = ""
    sig_comment = ""

    for line in lines1:
        # Check to enter in signature-retrieval mode
        if not sig_state:
            if line.lstrip().startswith(("def ", "async def", "class ")):
                sig_state = 1
                sig_line = ""
                sig_comment = ""
                sig_brace_depth = 0
                if line.lstrip().startswith("class ") and "(" not in line:
                    sig_state = 3  # search for closing colon directly
            else:
                lines2.append(line)
                continue

        # If we get here, we're in a signature

        # Handle comment
        line, _, c = line.partition("#")
        line = line.rstrip()
        c = c.strip()
        if c:
            sig_comment += " " + c.strip()

        if sig_state == 1:
            # Find the first opening brace
            i = line.find("(")
            if i >= 0:
                i += 1
                sig_brace_depth = 1
                sig_line += line[:i]
                line = line[i:]
                sig_state = 2

        line = line.lstrip()
        if sig_line.endswith(","):
            line = " " + line

        if sig_state == 2:
            # Resolve braces until we find the closing brace
            while True:
                i1 = line.find("(")
                i2 = line.find(")")
                if i1 >= 0 and i1 < i2:
                    i = i1 + 1
                    sig_brace_depth += 1
                    sig_line += line[:i]
                    line = line[i:]
                elif i2 >= 0:
                    i = i2 + 1
                    sig_brace_depth -= 1
                    sig_line += line[:i]
                    line = line[i:]
                else:
                    break
                if sig_brace_depth == 0:
                    sig_state = 3
                    break

        if sig_state == 3:
            # Find the closing colon
            i = line.find(":")
            if i >= 0:
                i += 1
                sig_line += line[:i]
                line = line[i:]
                # Finish the signature line
                sig_line = sig_line.replace(", )", ")")
                if sig_comment:
                    sig_line += "  #" + sig_comment
                lines2.append(sig_line)
                if line.strip():
                    lines2.append(" " * 12 + line.strip())
                # End the search
                sig_state = 0
                line = ""

        sig_line += line

    lines2.append("")
    return "\n".join(lines2)
