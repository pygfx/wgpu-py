import os

from cffi import FFI
from cffi.model import EnumType

from codegen.utils import print, lib_dir, remove_c_comments


_parser = None


def get_h_parser(*, allow_cache=True):
    """Get the global HParser object."""

    # Singleton pattern
    global _parser
    if _parser and allow_cache:
        return _parser

    # Get source
    lines = []
    with open(os.path.join(lib_dir, "resources", "wgpu.h")) as f:
        for line in f.readlines():
            if line.startswith(
                (
                    "#include ",
                    "#define WGPU_LOCAL",
                    "#define WGPUColor",
                    "#define WGPUOrigin3d_ZERO",
                    "#if defined",
                    "#endif",
                )
            ):
                continue
            elif line.startswith("#define ") and "(" in line and ")" in line:
                i1, i2 = line.index("("), line.index(")")
                line = line[:i1] + line[i2 + 1 :]
            lines.append(line)
    source = "".join(lines)

    # Create parser
    hp = HParser(source)
    hp.parse()
    _parser = hp
    return hp


class HParser:
    """Object to parse the wgpu.h header file, by letting cffi do the heavy lifting."""

    def __init__(self, source):
        self.source = source

    def parse(self, verbose=True):
        self.flags = {}
        self.enums = {}
        self.structs = {}
        self.functions = {}

        self._parse_from_h()
        self._parse_from_cffi()

        if verbose:
            print(f"The wgpu.h defines {len(self.functions)} functions")
            keys = "flags", "enums", "structs"
            stats = ", ".join(f"{len(getattr(self, key))} {key}" for key in keys)
            print("The wgpu.h defines " + stats)

    def _parse_from_h(self):
        code = self.source

        # Collect structs
        i1 = i2 = i3 = i4 = 0
        while True:
            # Find struct
            i1 = code.find("typedef struct", i4)
            i2 = code.find("{", i1)
            i3 = code.find("}", i2)
            i4 = code.find(";", i3)
            if i1 < 0:
                break
            # Only do simple structs, not Unions
            if 0 < code.find("{", i2 + 1) < i3:
                continue
            # Decompose
            name = code[i3 + 1 : i4].strip()
            self.structs[name] = struct = {}
            for f in code[i2 + 1 : i3].strip().strip(";").split(";"):
                parts = remove_c_comments(f).strip().split()
                key = parts[-1].strip("*")
                struct[key] = " ".join(parts[:-1])

        # Collect functions
        i1 = i2 = i3 = 0
        while True:
            # Find function
            i1 = code.find("wgpu_", i3)
            i2 = code.find("(", i1)
            i3 = code.find(");", i2)
            if i1 < 0:
                break
            # Extract name, and check whether we found something real
            name = code[i1:i2]
            if not (name and name.isidentifier()):
                i3 = i1 + 5
                continue
            # Decompose further
            i1 = code.rfind("\n", 0, i1)
            line = code[i1 : i3 + 2]
            line = " ".join(line.split())  # effective way to put on one line
            self.functions[name] = line

    def _parse_from_cffi(self):

        self.ffi = ffi = FFI()
        ffi.cdef(self.source)

        # Collect structs. We iterate over all types. Some will resolve
        # to C types, the rest are structs. The types for the struct
        # fields are reduced to the C primitives, making it less useful
        # for annotations. We update the structs that we've found by
        # parsing wgpu.h directly.
        for names in ffi.list_types():
            for name in names:
                # name = ffi.getctype(name) - no, keep original
                if name.startswith("WGPU"):
                    t = ffi.typeof(name)
                    if not hasattr(t, "fields"):
                        continue  # probably an enum
                    elif not t.fields:
                        continue  # base struct / alias
                    s = ffi.new(f"{name} *")
                    # Construct struct
                    struct = {}
                    for key, field in t.fields:
                        typename = field.type.cname
                        # typename = ffi.getctype(typename)
                        if typename.startswith("WGPU"):
                            val = typename  # Enum or struct
                        else:
                            val = type(getattr(s, key)).__name__
                        struct[key] = val
                    # Update
                    if name not in self.structs:
                        self.structs[name] = struct
                    else:
                        ori_struct = self.structs[name]
                        assert set(struct) == set(ori_struct)
                        for key, val in struct.items():
                            if ori_struct[key] != val:
                                if val.startswith("_"):  # _CDataBase
                                    pass
                                elif ori_struct[key].startswith("WGPU"):
                                    if "/" not in ori_struct[key]:
                                        ori_struct[key] += "/" + val
                                else:
                                    ori_struct[key] = val
                    # Make copies
                    alt_name = name
                    while alt_name != ffi.getctype(alt_name):
                        alt_name = ffi.getctype(alt_name)
                        self.structs[alt_name] = self.structs[name]

        # Collect enums. Warning: we access private ffi
        # stuff here. It seems its either this or load the lib.
        for key, (tp, _) in ffi._parser._declarations.items():
            tag, name = key.split(" ", 1)
            if isinstance(tp, EnumType):
                self.enums[name[4:]] = fields = {}
                for enumname, val in zip(tp.enumerators, tp.enumvalues):
                    fields[enumname[len(name) + 1 :]] = val
            elif tag == "function":
                # We don't process these, because the type info has
                # been reduced to primitive types, but for annotations
                # the higher-level type names are more useful. We
                # extract these by parsing wgpu.h directly
                pass

        # Collect flags by iterating over constants that are not enums.
        for key, value in ffi._parser._int_constants.items():
            if key.startswith("WGPU"):
                if key.upper() == key or "_" not in key:
                    continue
                name, _, field = key.partition("_")
                name = name[4:]  # strip "WGPU"
                if name in self.enums:
                    continue
                fields = self.flags.setdefault(name, {})
                fields[field] = value
