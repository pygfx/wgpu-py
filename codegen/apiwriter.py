"""
Writes the parts of the API that are simple: flags, enums, structs.
"""

import re

from codegen.utils import print, blacken, to_snake_case
from codegen.idlparser import get_idl_parser
from codegen.files import file_cache


ref_pattern = re.compile(r"\W((GPU|flags\.|enums\.|structs\.)\w+?)\W", re.MULTILINE)


def resolve_crossrefs(text):
    # Similar code as in docs/conf.py
    text += " "
    i2 = 0
    while True:
        m = ref_pattern.search(text, i2)
        if not m:
            break
        i1, i2 = m.start(1), m.end(1)
        prefix = m.group(2)
        ref_indicator = ":obj:" if prefix.lower() == prefix else ":class:"
        name = m.group(1)
        if name.startswith("structs."):
            link = name.split(".")[1]
        else:
            link = "wgpu." + name
        insertion = f"{ref_indicator}`{name} <{link}>`"
        text = text[:i1] + insertion + text[i2:]
        i2 += len(insertion) - len(name)
    return text.rstrip()


def write_flags():
    # Get preamble
    pylines = []
    for line in file_cache.read("flags.py").splitlines():
        pylines.append(line)
        if "AUTOGENERATED" in line:
            pylines += ["", ""]
            break
    # Prepare
    idl = get_idl_parser()
    n = len(idl.flags)
    # List'm
    pylines.append(f"# There are {n} flags\n")
    pylines.append("__all__ = [")
    for name in idl.flags.keys():
        pylines.append(f'    "{name}",')
    pylines.append("]\n\n")
    # The flags definitions
    for name, d in idl.flags.items():
        # Generate Code
        pylines.append(f"class {name}(Flags):\n")
        for key, val in d.items():
            pylines.append(f"    {key} = {val!r}")  # note: can add docs using "#: "
        pylines.append("\n")
    # Write
    code = blacken("\n".join(pylines))
    file_cache.write("flags.py", code)
    print(f"Wrote {n} flags to flags.py")


def write_enums():
    # Get preamble
    pylines = []
    for line in file_cache.read("enums.py").splitlines():
        pylines.append(line)
        if "AUTOGENERATED" in line:
            pylines += ["", ""]
            break
    # Prepare
    idl = get_idl_parser()
    n = len(idl.enums)
    # List'm
    pylines.append(f"# There are {n} enums\n")
    pylines.append("__all__ = [")
    for name in idl.enums.keys():
        pylines.append(f'    "{name}",')
    pylines.append("]\n\n")
    for name, d in idl.enums.items():
        # Generate Code
        pylines.append(f"class {name}(Enum):\n")
        for key, val in d.items():
            pylines.append(f"    {key} = None")  # note: can add docs using "#: "
        pylines.append("\n")
    # Write
    code = blacken("\n".join(pylines))
    file_cache.write("enums.py", code)
    print(f"Wrote {n} enums to enums.py")


def write_structs():
    # Get preamble
    pylines = []
    for line in file_cache.read("structs.py").splitlines():
        pylines.append(line)
        if "AUTOGENERATED" in line:
            pylines += ["", ""]
            break
    # Prepare
    idl = get_idl_parser()
    n = len(idl.structs)
    ignore = ["ImageCopyTextureTagged"]
    pylines.append(f"# There are {n} structs\n")
    # List'm
    pylines.append("__all__ = [")
    for name in idl.structs.keys():
        if name not in ignore:
            pylines.append(f'    "{name}",')
    pylines.append("]\n\n")
    for name, d in idl.structs.items():
        if name in ignore:
            continue
        # Object-docstring as a comment
        for field in d.values():
            tp = idl.resolve_type(field.typename).strip("'")
            if field.default is not None:
                pylines.append(
                    resolve_crossrefs(f"#: * {field.name} :: {tp} = {field.default}")
                )
            else:
                pylines.append(resolve_crossrefs(f"#: * {field.name} :: {tp}"))
        # Generate Code
        pylines.append(f'{name} = Struct(\n    "{name}",')
        for field in d.values():
            key = to_snake_case(field.name)
            val = idl.resolve_type(field.typename)
            if not val.startswith(("'", '"')):
                val = f"'{val}'"
            pylines.append(f"    {key}={val},")
        pylines.append(")\n")

    # Write
    code = blacken("\n".join(pylines))
    file_cache.write("structs.py", code)
    print(f"Wrote {n} structs to structs.py")
