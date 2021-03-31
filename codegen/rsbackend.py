"""
Apply codegen to rs backend.
"""

import os

from codegen.utils import lib_dir, blacken
from codegen.hparser import HParser
from codegen.idlparser import IdlParser


mappings_preamble = '''
""" Mappings for the rs backend. """

# THIS CODE IS AUTOGENERATED - DO NOT EDIT
"""
# flake8: noqa
'''.lstrip()


# todo: this is WIP


def write_mappings():

    ip = IdlParser()
    ip.parse()
    hp = HParser()
    hp.parse()

    # Create enummap
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

    # Generate code
    pylines = [mappings_preamble]
    pylines.append(f"# There are {len(ip.enums)} mappings\n")
    pylines.append("enummap = {")
    for key, val in enummap.items():
        pylines.append(f'    "{key}": {val!r},')
    pylines.append("}\n")

    # todo: the enummap is fine. The stuff below is more complex.
    # Also, I'm not sure whether this should change with the rest of the rs backend codegen
    pylines.append("cstructfield2enum = {")
    for structname, struct in hp.structs.items():
        for field in struct.values():
            if field.typename.startswith("WGPU"):
                enumname = field.typename[4:]
                if enumname in ip.enums:
                    pylines.append(f'    "{structname}.{field.name}": "{enumname}",')
    pylines.append("}\n")

    # Write
    code = blacken("\n".join(pylines))  # just in case; code is already black
    with open(os.path.join(lib_dir, "backends", "rs_mappings.py"), "wb") as f:
        f.write(code.encode())
    print("Written to rs_mappings.py")


write_mappings()
