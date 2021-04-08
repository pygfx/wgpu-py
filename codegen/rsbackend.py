"""
Apply codegen to rs backend.
"""

import os

from codegen.utils import lib_dir, blacken, Patcher
from codegen.hparser import HParser
from codegen.idlparser import IdlParser


mappings_preamble = '''
""" Mappings for the rs backend. """

# THIS CODE IS AUTOGENERATED - DO NOT EDIT

# flake8: noqa
'''.lstrip()


# todo: this is WIP


def write_mappings():

    ip = IdlParser()
    ip.parse()
    hp = HParser()
    hp.parse()

    # Create enummap, which allows the rs backend to resolve enum field names
    # to the corresponding integer value.
    enummap = {}
    for name in ip.enums:
        if name not in hp.enums:
            print(f"Enum {name} missing in wgpu.h")
            continue
        for ikey in ip.enums[name].values():
            hkey = ikey
            hkey = hkey.replace("1d", "D1").replace("2d", "D2").replace("3d", "D3")
            hkey = hkey.replace("-", " ").title().replace(" ", "")
            if hkey in hp.enums[name]:
                enummap[name + "." + ikey] = hp.enums[name][hkey]
            else:
                print(f"Enum field {name}.{ikey} missing in .h")

    # Some structs have fields that are enum values. The rs backend
    # must be able to resolve these too.
    cstructfield2enum = {}
    for structname, struct in hp.structs.items():
        for key, val in struct.items():
            if isinstance(val, str) and val.startswith("WGPU"):
                enumname = val[4:]
                if enumname in ip.enums:
                    cstructfield2enum[f"{structname[4:]}.{key}"] = enumname
                else:
                    pass  # a struct

    # Generate code
    pylines = [mappings_preamble]

    pylines.append(f"# There are {len(enummap)} enum mappings\n")
    pylines.append("enummap = {")
    for key in sorted(enummap.keys()):
        pylines.append(f'    "{key}": {enummap[key]!r},')
    pylines.append("}\n")

    pylines.append(f"# There are {len(cstructfield2enum)} struct-field enum mappings\n")
    pylines.append("cstructfield2enum = {")
    for key in sorted(cstructfield2enum.keys()):
        pylines.append(f'    "{key}": {cstructfield2enum[key]!r},')
    pylines.append("}\n")

    # Write
    code = blacken("\n".join(pylines))  # just in case; code is already black
    with open(os.path.join(lib_dir, "backends", "rs_mappings.py"), "wb") as f:
        f.write(code.encode())
    print("Written to rs_mappings.py")


def patch_structs():

    hp = HParser()
    hp.parse()

    filename = os.path.join(lib_dir, "backends", "rs.py")
    with open(filename, "rb") as f:
        source = f.read().decode()

    p = Patcher(source)

    line_index = None
    brace_depth = 0
    struct_lines = []

    for line, i in p.iter_lines():
        if line.lstrip().startswith(
            ("# FIXME: unknown", "FIXME: invalid", "# fields:")
        ):
            p.remove_line(i)

        if "new_struct_p(" in line or "new_struct(" in line:
            if line.lstrip().startswith("def "):
                continue  # Implementation
            if "_new_struct" in line:
                continue  # Implementation
            if "new_struct_p()" in line or "new_struct()" in line:
                continue  # Comments or docs
            line_index = i
            j = line.index("new_struct")
            line = line[j:]  # start brace searching from right pos
            brace_depth = 0
            struct_lines = []

        if line_index:
            for c in line:
                if c == "#":
                    break
                elif c == "(":
                    brace_depth += 1
                elif c == ")":
                    brace_depth -= 1
                    assert brace_depth >= 0
                    if brace_depth == 0:
                        _validate_struct(hp, p, line_index, i)
                        line_index = None
                        break

    with open(filename, "wb") as f:
        source = f.write(p.dumps().encode())


def _validate_struct(hp, p, i1, i2):

    lines = p.lines[
        i1 : i2 + 1
    ]  # note: i2 is the line index where the closing brace is
    indent = " " * (len(lines[-1]) - len(lines[-1].lstrip()))

    if len(lines) == 1:
        # Single line - add a comma before the closing brace
        print("Making a struct multiline. Rerun codegen to validate the struct.")
        line = lines[0]
        i = line.rindex(")")
        line = line[:i] + "," + line[i:]
        p.replace_line(i1, line)
        return
    elif len(lines) == 3 and lines[1].count("="):
        # Triplet - add a comma after the last element
        print("Making a struct multiline. Rerun codegen to validate the struct.")
        p.replace_line(i1 + 1, p.lines[i1 + 1] + ",")
        return

    # We can assume that the struct is multi-line and formatted by Black!
    assert len(lines) >= 3

    # Get struct name, and verify
    name = lines[1].strip().strip(',"')
    struct_name = name.strip(" *")
    if name.endswith("*"):
        if "new_struct_p" not in lines[0]:
            p.insert_line(i1, indent + f"# FIXME: invalid, use new_struct_p()")
    else:
        if "new_struct_p" in lines[0]:
            p.insert_line(i1, indent + f"# FIXME: invalid, use new_struct()")

    # Get struct object and create annotation line
    if struct_name not in hp.structs:
        print(f"Unknown struct {struct_name}")
        p.insert_line(i1, indent + f"# FIXME: unknown struct {struct_name}")
        return
    else:
        struct = hp.structs[struct_name]
        fields = ", ".join(f"{key}: {val}" for key, val in struct.items())
        p.insert_line(i1, indent + f"# fields: " + fields)

    # Check keys
    keys_found = []
    for j in range(2, len(lines) - 1):
        line = lines[j]
        key = line.split("=")[0].strip()
        if key.startswith("# not used:"):
            key = key.split(":")[1].split("=")[0].strip()
        elif key.startswith("#"):
            continue
        keys_found.append(key)
        if key not in struct:
            p.insert_line(i1 + j, indent + f"    # FIXME: unknown field {key}")
            print(f"Struct {struct_name} does not have key {key}")

    # Insert comments for unused keys
    more_lines = []
    for key in struct:
        if key not in keys_found:
            more_lines.append(indent + f"    # not used: {key}")
    if more_lines:
        p.insert_line(i2, "\n".join(more_lines))


if __name__ == "__main__":
    write_mappings()
    patch_structs()
