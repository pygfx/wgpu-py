"""
Utility functions.
"""

import os
import inspect

from . import _api, _constants


def get_resource_dir():
    # todo: use setuptools package data thingy
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")


def help(*searches, dev=False):
    """ Print constants, enums, structs, and functions that contain the given searches.
    If dev is True, will also print info from the definitions in .idl and .h, which
    can be useful during debugging and dev.
    """

    # Strip prefixes used in .idl and .h
    name_parts = []
    for name_part in searches:
        if name_part.lower().startswith("wgpu"):
            name_part = name_part[4:]
        elif name_part.lower().startswith("gpu"):
            name_part = name_part[3:]
        name_parts.append(name_part)

    # Prepare
    quoted = [f"{name_part!r}" for name_part in name_parts]
    print(f"Searching for {' and '.join(quoted)} ...")
    name_parts = [name_part.lower() for name_part in name_parts]
    all_lines = []  # list of (title, lines_list)

    if dev:
        from ._parsers import IdlParser, HParser

        filename = os.path.join(get_resource_dir(), "webgpu.idl")
        idl_parser = IdlParser(open(filename, "rb").read().decode())
        idl_parser.parse()
        filename = os.path.join(get_resource_dir(), "wgpu.h")
        h_parser = HParser(open(filename, "rb").read().decode())
        h_parser.parse()

    # Find flags
    lines = []
    all_lines.append(("flags", lines))
    for name_part in name_parts:
        for name, val in _constants.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(val, _constants.Flags):
                if name_part in name.lower():
                    lines.append(name + ": " + ", ".join(val))
                else:
                    for option in val:
                        if name_part in option.lower():
                            lines.append(name + "." + option)

    if dev:
        lines = []
        all_lines.append(("flags in .idl", lines))
        for name_part in name_parts:
            for name, d in idl_parser.flags.items():
                if name_part in name.lower():
                    lines.append(name + ": " + ", ".join(f"{key}={d[key]}" for key in d.keys()))
                else:
                    for key in d.keys():
                        if name_part in key.lower():
                            lines.append(f"{name}.{key} = {d[key]}")
        lines = []
        all_lines.append(("flags in .h", lines))
        for name_part in name_parts:
            for name, d in h_parser.flags.items():
                if name_part in name.lower():
                    lines.append(name + ": " + ", ".join(f"{key}={d[key]}" for key in d.keys()))
                else:
                    for key in d.keys():
                        if name_part in key.lower():
                            lines.append(f"{name}.{key} = {d[key]}")

    # Find enums
    lines = []
    all_lines.append(("enums", lines))
    for name_part in name_parts:
        for name, val in _constants.__dict__.items():
            if name.startswith("_"):
                continue
            elif isinstance(val, _constants.Enum):
                if name_part in name.lower():
                    lines.append(name + ": " + ", ".join(f"'{x}'" for x in val))
                else:
                    for option in val:
                        if name_part in option.lower():
                            lines.append(name + "." + option)

    if dev:
        lines = []
        all_lines.append(("enums in .idl", lines))
        for name_part in name_parts:
            for name, d in idl_parser.enums.items():
                if name_part in name.lower():
                    lines.append(name + ": " + ", ".join(f"{key}='{d[key]}'" for key in d.keys()))
                else:
                    for key, val in d.items():
                        if name_part in key.lower() or name_part in val.lower():
                            lines.append(f"{name}.{key} = '{val}'")
        lines = []
        all_lines.append(("enums in .h", lines))
        for name_part in name_parts:
            for name, d in h_parser.enums.items():
                if name_part in name.lower():
                    lines.append(name + ": " + ", ".join(f"{key}={d[key]}" for key in d.keys()))
                else:
                    for key, val in d.items():
                        if name_part in key.lower():
                            lines.append(f"{name}.{key} = {val}")


    # Find structs
    lines = []
    all_lines.append(("structs", lines))
    for name_part in name_parts:
        for name, val in _constants.__dict__.items():
            if name.startswith("_"):
                continue
            elif name.startswith("make"):  # struct
                # todo: also check fields
                name = name[4:]
                if name_part in name.lower():
                    lines.append(name)

    if dev:
        lines = []
        all_lines.append(("structs in .idl", lines))
        for name_part in name_parts:
            for name, d in idl_parser.structs.items():
                if name_part in name.lower():
                    x = "GPU" + name + " {"
                    for field in d.values():
                        x += "\n        " + field.line
                    x += "\n    }"
                    lines.append(x)
        lines = []
        all_lines.append(("structs in .h", lines))
        for name_part in name_parts:
            for name, d in h_parser.structs.items():
                if name_part in name.lower():
                    x = "WGPU" + name + " {"
                    for field in d.values():
                        x += "\n        " + field.line
                    x += "\n    }"
                    lines.append(x)
            # else:
            #     for field in d.values():
            #         if name_part in field.name.lower() or name_part in field.typename.lower():
            #             items["structs"].append(name + "." + field.py_arg())


    # Find functions
    lines = []
    all_lines.append(("functions", lines))
    for name_part in name_parts:
        name_part_f = name_part.replace("_", "").replace(".", "")
        for cls in _api.__dict__.values():
            if isinstance(cls, type):
                for attr_name, attr in cls.__dict__.items():
                    if attr_name.startswith("_") or not callable(attr):
                        continue
                    funcname = cls.__name__ + "." + attr_name
                    func_id = funcname.replace(".", "").lower()
                    if name_part_f in func_id:
                        sig = str(inspect.signature(attr)).replace("self, ", "")
                        lines.append(funcname + sig)

    if dev:
        lines = []
        all_lines.append(("functions in .idl", lines))
        for name_part in name_parts:
            name_part_f = name_part.replace("_", "").replace(".", "")
            for func_id, line in idl_parser.functions.items():
                if name_part_f in func_id:
                    lines.append(line.strip())
        lines = []
        all_lines.append(("functions in .h", lines))
        for name_part in name_parts:
            name_part_f = name_part.replace("_", "").replace(".", "")
            for func_id, line in h_parser.functions.items():
                if name_part_f in func_id:
                    lines.append(line.strip())

    # Display
    for title, lines in all_lines:
        if " in " not in title:
            print(f"\n--- {len(lines)} {title} ".ljust(80, "-"))
        elif lines:
            print(f"--- {len(lines)} {title} ---")
        if lines:
            print("\n".join("    " + line for line in lines))
