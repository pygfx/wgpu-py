import sys
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


def write_file(text:str, *fname):
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

LE = _get_line_endings((ROOT_DIR / 'apigen' / "__init__.py").read_bytes().decode())


def to_snake_case(name, separator="_"):
    """Convert a name from camelCase to snake_case. Names that already are
    snake_case remain the same.
    """
    name2 = ""
    for c in name:
        c2 = c.lower()
        if c2 != c and len(name2) > 0:
            prev = name2[-1]
            if c2 == "d" and prev in "123":
                name2 = name2[:-1] + separator + prev
            elif prev != separator:
                name2 += separator
        name2 += c2
    return name2


def dict_deep_merge(a:dict, b:dict, allow_add: bool=True) -> dict:
    """Return a new dict: b merged into a, recursing into nested dicts."""
    result = dict(a)
    for key, val in b.items():
        if isinstance(val, dict):
            if key in result and isinstance(result[key], dict):
                if "diff" in val:
                    assert val['diff'] == "REMOVE"
                    result.pop(key, None)
                else:
                    result[key] = dict_deep_merge(result[key], val, allow_add)
            elif allow_add:
                result[key] = val
        else:
            result[key] = val

    return result


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
        result = _make_sigs_singline(result)

    return result


def indent(text: str, indent=4) -> str:
    text = text.strip()
    lines = []
    for line in text.split("\n"):
        lines.append(" " * indent + line.rstrip())
    lines.append("")
    return "\n".join(lines)


def read_definition_file(fname: str) -> dict:
    """Read a TOML API definition file and check that the structure is correct.
    Returns the structure as a dict.
    """
    source = read_file("apigen", "definitions", fname)
    structure = tomllib.loads(source)

    diff_actions = ("REMOVE", )

    # The code below is just a tedious traversing of the structure that checks each leaf node.

    for section, section_dict in structure.items():
        assert section in ("flags", "enums", "structs", "classes")
        assert isinstance(section_dict, dict)

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
                                    raise RuntimeError(f"Invalid flag field dict key {field_dict_key!r}")
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
                                     assert field_dict_val == enum_name_to_value(field_name)
                                else:
                                    raise RuntimeError(f"Invalid enum field dict key {field_dict_key!r}")
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
                                elif field_dict_key == "required":
                                     assert isinstance(field_dict_val, bool)
                                else:
                                    raise RuntimeError(f"Invalid struct field dict key {field_dict_key!r}")
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
                                    raise RuntimeError(f"Invalid property dict key {prop_dict_key!r}")
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
                                    for return_dict_key, return_dict_val in method_dict_val.items():
                                        if return_dict_key == "doc":
                                            assert isinstance(return_dict_val, str)
                                        elif return_dict_key == "type":
                                            assert isinstance(return_dict_val, str)
                                        else:
                                            raise RuntimeError(f"Invalid method returns dict key {return_dict_key!r}")
                                elif method_dict_key == "args":
                                    assert isinstance(method_dict_val, dict)
                                    for arg_name, arg_dict in method_dict_val.items():
                                        assert arg_name.lower() == arg_name
                                        assert isinstance(arg_dict, dict)
                                        for arg_dict_key, arg_dict_val in arg_dict.items():
                                            if arg_dict_key == "doc":
                                                assert isinstance(arg_dict_val, str)
                                            elif arg_dict_key == "diff":
                                                assert arg_dict_val in diff_actions
                                            elif arg_dict_key == "type":
                                                    assert isinstance(arg_dict_val, str)
                                            elif arg_dict_key == "default":
                                                    assert isinstance(arg_dict_val, str)
                                            elif arg_dict_key == "required":
                                                    assert isinstance(arg_dict_val, bool)
                                            else:
                                                raise RuntimeError(f"Invalid method arg dict key {arg_dict_key!r}")
                                else:
                                    raise RuntimeError(f"Invalid method dict key {method_dict_key!r}")
                    else:
                        raise RuntimeError(f"Invalid class attr {class_attr_key!r}")

    return structure


def enum_name_to_value(name):
    parts = name.split("_")
    for i in range(len(parts)):
        if parts[i] in ("d1", "d2", "d3"):
            parts[i] = parts[i][::-1]
    return "-".join(parts)
