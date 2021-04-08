import os

from cffi import FFI
from cffi.model import EnumType

from codegen.utils import lib_dir


class HParser:
    """Object to parse the wgpu.h header file, by letting cffi do the heavy lifting."""

    def __init__(self):
        pass

    def _get_wgpu_h(self):
        lines = []
        with open(os.path.join(lib_dir, "resources", "wgpu.h")) as f:
            for line in f.readlines():
                if not line.startswith(
                    (
                        "#include ",
                        "#define WGPU_LOCAL",
                        "#define WGPUColor",
                        "#define WGPUOrigin3d_ZERO",
                        "#if defined",
                        "#endif",
                    )
                ):
                    lines.append(line)
        return "".join(lines)

    def parse(self):
        self.flags = {}
        self.enums = {}
        self.structs = {}

        self._parse_from_h()
        self._parse_from_cffi()

    def _parse_from_h(self):
        code = self._get_wgpu_h()

        # Parsing structs is not that hard
        i1 = i2 = i3 = i4 = 0
        while True:
            # Find struct
            i1 = code.find("typedef struct", i4)
            if i1 < 0:
                break
            i2 = code.find("{", i1)
            i3 = code.find("}", i2)
            i4 = code.find(";", i3)
            if 0 < code.find("{", i2 + 1) < i3:
                continue  # Union ... not going there
            # Decompose
            name = code[i3 + 1 : i4].strip()
            self.structs[name] = struct = {}
            for f in code[i2 + 1 : i3].strip().strip(";").split(";"):
                parts = f.strip().split()
                key = parts[-1].strip("*")
                struct[key] = " ".join(parts[:-1])

    def _parse_from_cffi(self):

        self.ffi = ffi = FFI()
        ffi.cdef(self._get_wgpu_h())

        # Iterate over all types. Some will resolve to C types, the rest are structs.
        for names in ffi.list_types():
            for name in names:
                # name = ffi.getctype(name) - no, keep original
                if name.startswith("WGPU"):
                    t = ffi.typeof(name)
                    if not hasattr(t, "fields"):
                        continue  # probably an enum
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
                                    ori_struct[key] += "/" + val
                                else:
                                    ori_struct[key] = val

        # Collect enums and functions. Warning: we access private ffi
        # stuff here. It seems its either this or load the lib.
        for key, (tp, _) in ffi._parser._declarations.items():
            tag, name = key.split(" ", 1)
            if isinstance(tp, EnumType):
                self.enums[name[4:]] = fields = {}
                for enumname, val in zip(tp.enumerators, tp.enumvalues):
                    fields[enumname[len(name) + 1 :]] = val
            elif tag == "function":
                pass  # print("function", name)

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
